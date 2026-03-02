using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Options for audio embedding pooling post-processing.
/// </summary>
public class AudioEmbeddingPoolingOptions
{
    /// <summary>Name of the input column containing raw ONNX scores (VBuffer&lt;float&gt;).</summary>
    public string InputColumnName { get; set; } = "Scores";

    /// <summary>Name of the output column for pooled embeddings (VBuffer&lt;float&gt;).</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>Pooling strategy to reduce sequence dimension.</summary>
    public AudioPoolingStrategy Pooling { get; set; } = AudioPoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embedding.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Hidden dimension of the model output. Auto-populated from scorer.</summary>
    public int HiddenDim { get; set; }

    /// <summary>Whether the model output is already pooled (single vector per sample).</summary>
    public bool IsPrePooled { get; set; }
}

/// <summary>
/// Estimator for audio embedding pooling (Stage 3).
/// </summary>
public sealed class AudioEmbeddingPoolingEstimator : IEstimator<AudioEmbeddingPoolingTransformer>
{
    private readonly IHostEnvironment _env;
    private readonly AudioEmbeddingPoolingOptions _options;

    public AudioEmbeddingPoolingEstimator(IHostEnvironment env, AudioEmbeddingPoolingOptions options)
    {
        _env = env;
        _options = options;
    }

    public AudioEmbeddingPoolingTransformer Fit(IDataView input)
    {
        var effectiveOptions = _options;

        // Auto-discover dimensions from scorer column annotations when not explicitly set
        if (effectiveOptions.HiddenDim <= 0)
        {
            var col = input.Schema.GetColumnOrNull(effectiveOptions.InputColumnName);
            if (col.HasValue)
            {
                var annotations = col.Value.Annotations;
                var hiddenDimCol = annotations.Schema.GetColumnOrNull("HiddenDim");
                if (hiddenDimCol.HasValue)
                {
                    int hiddenDim = 0;
                    annotations.GetValue("HiddenDim", ref hiddenDim);

                    bool hasPooledOutput = false;
                    var pooledCol = annotations.Schema.GetColumnOrNull("HasPooledOutput");
                    if (pooledCol.HasValue)
                        annotations.GetValue("HasPooledOutput", ref hasPooledOutput);

                    effectiveOptions = new AudioEmbeddingPoolingOptions
                    {
                        InputColumnName = _options.InputColumnName,
                        OutputColumnName = _options.OutputColumnName,
                        Pooling = _options.Pooling,
                        Normalize = _options.Normalize,
                        HiddenDim = hiddenDim,
                        IsPrePooled = hasPooledOutput
                    };
                }
            }
        }

        return new AudioEmbeddingPoolingTransformer(_env, effectiveOptions);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.OutputColumnName, SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Single, false, null]);

        return new SchemaShape(columns.Values);
    }
}

/// <summary>
/// Stage 3 sub-transform for embeddings: pooling (mean/CLS/max) + optional L2 normalization.
/// </summary>
public sealed class AudioEmbeddingPoolingTransformer : ITransformer, IDisposable
{
    private readonly AudioEmbeddingPoolingOptions _options;
    internal AudioEmbeddingPoolingOptions Options => _options;

    public bool IsRowToRowMapper => true;

    internal AudioEmbeddingPoolingTransformer(IHostEnvironment env, AudioEmbeddingPoolingOptions options)
    {
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new PoolerDataView(input, this);
    }

    /// <summary>
    /// Direct API: pool raw ONNX output into a fixed embedding vector.
    /// </summary>
    internal float[] Pool(float[] rawOutput)
    {
        float[] embedding;

        if (_options.IsPrePooled || _options.HiddenDim <= 0)
        {
            // Already pooled — just use as-is
            embedding = rawOutput;
        }
        else
        {
            int seqLen = rawOutput.Length / _options.HiddenDim;
            embedding = _options.Pooling switch
            {
                AudioPoolingStrategy.MeanPooling => MeanPool(rawOutput, seqLen, _options.HiddenDim),
                AudioPoolingStrategy.MaxPooling => MaxPool(rawOutput, seqLen, _options.HiddenDim),
                AudioPoolingStrategy.ClsToken => rawOutput[.._options.HiddenDim],
                _ => MeanPool(rawOutput, seqLen, _options.HiddenDim)
            };
        }

        if (_options.Normalize)
        {
            float norm = TensorPrimitives.Norm(embedding);
            if (norm > 0)
                TensorPrimitives.Divide(embedding, norm, embedding);
        }

        return embedding;
    }

    private static float[] MeanPool(float[] data, int seqLen, int hiddenDim)
    {
        var result = new float[hiddenDim];
        for (int s = 0; s < seqLen; s++)
        {
            var span = data.AsSpan(s * hiddenDim, hiddenDim);
            TensorPrimitives.Add(result, span, result);
        }
        TensorPrimitives.Divide(result, seqLen, result);
        return result;
    }

    private static float[] MaxPool(float[] data, int seqLen, int hiddenDim)
    {
        var result = new float[hiddenDim];
        Array.Fill(result, float.MinValue);
        for (int s = 0; s < seqLen; s++)
        {
            var span = data.AsSpan(s * hiddenDim, hiddenDim);
            TensorPrimitives.Max(result, span, result);
        }
        return result;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        int dim = _options.HiddenDim > 0 ? _options.HiddenDim : 0;
        builder.AddColumn(_options.OutputColumnName,
            dim > 0 ? new VectorDataViewType(NumberDataViewType.Single, dim)
                     : new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Audio embedding pooler transforms cannot be saved.");

    public void Dispose() { }

    // --- Lazy IDataView wrapper ---

    private sealed class PoolerDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly AudioEmbeddingPoolingTransformer _transformer;

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        internal PoolerDataView(IDataView input, AudioEmbeddingPoolingTransformer transformer)
        {
            _input = input;
            _transformer = transformer;
            Schema = transformer.GetOutputSchema(input.Schema);
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            var inputCol = _input.Schema.GetColumnOrNull(_transformer._options.InputColumnName)
                ?? throw new InvalidOperationException(
                    $"Input column '{_transformer._options.InputColumnName}' not found.");

            var upstreamCols = columnsNeeded
                .Select(c => _input.Schema.GetColumnOrNull(c.Name))
                .Where(c => c != null)
                .Select(c => c!.Value)
                .Append(inputCol)
                .Distinct()
                .ToList();

            var inputCursor = _input.GetRowCursor(upstreamCols, rand);
            return new PoolerCursor(this, inputCursor, _transformer);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };
    }

    private sealed class PoolerCursor : DataViewRowCursor
    {
        private readonly PoolerDataView _dataView;
        private readonly DataViewRowCursor _inputCursor;
        private readonly AudioEmbeddingPoolingTransformer _transformer;
        private readonly int _outputColIndex;

        private float[]? _currentEmbedding;
        private long _position = -1;
        private bool _disposed;

        internal PoolerCursor(
            PoolerDataView dataView,
            DataViewRowCursor inputCursor,
            AudioEmbeddingPoolingTransformer transformer)
        {
            _dataView = dataView;
            _inputCursor = inputCursor;
            _transformer = transformer;
            _outputColIndex = dataView.Schema.GetColumnOrNull(transformer._options.OutputColumnName)?.Index
                ?? throw new InvalidOperationException("Output column not found.");
        }

        public override DataViewSchema Schema => _dataView.Schema;
        public override long Position => _position;
        public override long Batch => _inputCursor.Batch;

        public override bool MoveNext()
        {
            if (!_inputCursor.MoveNext())
                return false;

            _position++;

            var inputCol = _inputCursor.Schema.GetColumnOrNull(_transformer._options.InputColumnName);
            if (inputCol == null) return false;

            var getter = _inputCursor.GetGetter<VBuffer<float>>(inputCol.Value);
            var buffer = default(VBuffer<float>);
            getter(ref buffer);

            var rawOutput = buffer.DenseValues().ToArray();
            _currentEmbedding = _transformer.Pool(rawOutput);

            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Index == _outputColIndex)
            {
                ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                {
                    var data = _currentEmbedding ?? Array.Empty<float>();
                    var editor = VBufferEditor.Create(ref value, data.Length);
                    data.AsSpan().CopyTo(editor.Values);
                    value = editor.Commit();
                };
                return (ValueGetter<TValue>)(object)getter;
            }

            var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
            if (inputCol != null)
                return _inputCursor.GetGetter<TValue>(inputCol.Value);

            throw new InvalidOperationException($"Column '{column.Name}' not found.");
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => _inputCursor.GetIdGetter();

        public override bool IsColumnActive(DataViewSchema.Column column)
            => column.Index == _outputColIndex || _inputCursor.IsColumnActive(
                _inputCursor.Schema.GetColumnOrNull(column.Name) ?? column);

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                    _inputCursor.Dispose();
                _disposed = true;
            }
            base.Dispose(disposing);
        }
    }
}
