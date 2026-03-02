using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Options for audio classification post-processing.
/// </summary>
public class AudioClassificationPostProcessingOptions
{
    /// <summary>Class labels for classification.</summary>
    public required string[] Labels { get; set; }

    /// <summary>Name of the input column containing raw scores (VBuffer&lt;float&gt;).</summary>
    public string InputColumnName { get; set; } = "Scores";

    /// <summary>Name of the output column for predicted label.</summary>
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";

    /// <summary>Name of the output column for probabilities.</summary>
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";

    /// <summary>Name of the output column for top score.</summary>
    public string ScoreColumnName { get; set; } = "Score";
}

/// <summary>
/// Estimator for audio classification post-processing (Stage 3).
/// </summary>
public sealed class AudioClassificationPostProcessingEstimator : IEstimator<AudioClassificationPostProcessingTransformer>
{
    private readonly IHostEnvironment _env;
    private readonly AudioClassificationPostProcessingOptions _options;

    public AudioClassificationPostProcessingEstimator(IHostEnvironment env, AudioClassificationPostProcessingOptions options)
    {
        _env = env;
        _options = options;
    }

    public AudioClassificationPostProcessingTransformer Fit(IDataView input)
    {
        return new AudioClassificationPostProcessingTransformer(_env, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.PredictedLabelColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.PredictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
             TextDataViewType.Instance, false, null]);
        columns[_options.ScoreColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.ScoreColumnName, SchemaShape.Column.VectorKind.Scalar,
             NumberDataViewType.Single, false, null]);
        columns[_options.ProbabilitiesColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.ProbabilitiesColumnName, SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Single, false, null]);

        return new SchemaShape(columns.Values);
    }
}

/// <summary>
/// Stage 3 sub-transform for classification: softmax + argmax + label mapping.
/// </summary>
public sealed class AudioClassificationPostProcessingTransformer : ITransformer, IDisposable
{
    private readonly AudioClassificationPostProcessingOptions _options;
    internal AudioClassificationPostProcessingOptions Options => _options;

    public bool IsRowToRowMapper => true;

    internal AudioClassificationPostProcessingTransformer(IHostEnvironment env, AudioClassificationPostProcessingOptions options)
    {
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new ClassificationDataView(input, this);
    }

    /// <summary>
    /// Direct API: post-process raw scores into classification results.
    /// </summary>
    internal AudioClassificationResult PostProcess(float[] rawScores)
    {
        var probabilities = new float[rawScores.Length];
        TensorPrimitives.SoftMax(rawScores, probabilities);

        int bestIdx = TensorPrimitives.IndexOfMax(probabilities);
        string label = bestIdx < _options.Labels.Length ? _options.Labels[bestIdx] : $"Class_{bestIdx}";

        return new AudioClassificationResult
        {
            PredictedLabel = label,
            Score = probabilities[bestIdx],
            Probabilities = probabilities
        };
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        builder.AddColumn(_options.ScoreColumnName, NumberDataViewType.Single);
        builder.AddColumn(_options.ProbabilitiesColumnName,
            new VectorDataViewType(NumberDataViewType.Single, _options.Labels.Length));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Audio classification post-process transforms cannot be saved.");

    public void Dispose() { }

    // --- Lazy IDataView wrapper ---

    private sealed class ClassificationDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly AudioClassificationPostProcessingTransformer _transformer;

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        internal ClassificationDataView(IDataView input, AudioClassificationPostProcessingTransformer transformer)
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
            return new ClassificationCursor(this, inputCursor, _transformer);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };
    }

    private sealed class ClassificationCursor : DataViewRowCursor
    {
        private readonly ClassificationDataView _dataView;
        private readonly DataViewRowCursor _inputCursor;
        private readonly AudioClassificationPostProcessingTransformer _transformer;
        private readonly int _labelColIndex;
        private readonly int _scoreColIndex;
        private readonly int _probsColIndex;

        private AudioClassificationResult? _currentResult;
        private long _position = -1;
        private bool _disposed;

        internal ClassificationCursor(
            ClassificationDataView dataView,
            DataViewRowCursor inputCursor,
            AudioClassificationPostProcessingTransformer transformer)
        {
            _dataView = dataView;
            _inputCursor = inputCursor;
            _transformer = transformer;
            _labelColIndex = dataView.Schema[transformer._options.PredictedLabelColumnName].Index;
            _scoreColIndex = dataView.Schema[transformer._options.ScoreColumnName].Index;
            _probsColIndex = dataView.Schema[transformer._options.ProbabilitiesColumnName].Index;
        }

        public override DataViewSchema Schema => _dataView.Schema;
        public override long Position => _position;
        public override long Batch => _inputCursor.Batch;

        public override bool MoveNext()
        {
            if (!_inputCursor.MoveNext())
                return false;

            _position++;

            // Read raw scores from upstream
            var scoresCol = _inputCursor.Schema.GetColumnOrNull(_transformer._options.InputColumnName);
            if (scoresCol == null) return false;

            var getter = _inputCursor.GetGetter<VBuffer<float>>(scoresCol.Value);
            var buffer = default(VBuffer<float>);
            getter(ref buffer);

            var rawScores = buffer.DenseValues().ToArray();
            _currentResult = _transformer.PostProcess(rawScores);

            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Index == _labelColIndex)
            {
                ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                {
                    value = (_currentResult?.PredictedLabel ?? "").AsMemory();
                };
                return (ValueGetter<TValue>)(object)getter;
            }

            if (column.Index == _scoreColIndex)
            {
                ValueGetter<float> getter = (ref float value) =>
                {
                    value = _currentResult?.Score ?? 0f;
                };
                return (ValueGetter<TValue>)(object)getter;
            }

            if (column.Index == _probsColIndex)
            {
                ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                {
                    var probs = _currentResult?.Probabilities ?? Array.Empty<float>();
                    var editor = VBufferEditor.Create(ref value, probs.Length);
                    probs.AsSpan().CopyTo(editor.Values);
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
            => column.Index == _labelColIndex || column.Index == _scoreColIndex
            || column.Index == _probsColIndex || _inputCursor.IsColumnActive(
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

