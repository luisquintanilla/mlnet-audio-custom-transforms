using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Runtime;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Stage 2 sub-transform: runs ONNX model inference on extracted features.
/// Produces raw model output scores as VBuffer&lt;float&gt;.
/// </summary>
public sealed class OnnxAudioScoringTransformer : ITransformer, IDisposable
{
    private readonly OnnxAudioScoringOptions _options;
    private readonly InferenceSession _session;
    private readonly string _inputTensorName;
    private readonly string _outputTensorName;
    private bool _disposed;

    internal OnnxAudioScoringOptions Options => _options;

    /// <summary>Hidden dimension discovered from the ONNX model output shape.</summary>
    public int HiddenDim { get; }

    /// <summary>Whether the model has a pooled output (single vector per sample).</summary>
    public bool HasPooledOutput { get; }

    public bool IsRowToRowMapper => true;

    internal OnnxAudioScoringTransformer(IHostEnvironment env, OnnxAudioScoringOptions options)
    {
        _options = options;

        var sessionOptions = new SessionOptions();
        if (options.GpuDeviceId != null)
            sessionOptions.AppendExecutionProvider_DML(options.GpuDeviceId.Value);

        _session = new InferenceSession(options.ModelPath, sessionOptions);

        // Auto-detect tensor names
        _inputTensorName = options.InputTensorName
            ?? _session.InputMetadata.Keys.First();
        _outputTensorName = options.OutputTensorName
            ?? _session.OutputMetadata.Keys.First();

        // Discover output dimensions
        var outputMeta = _session.OutputMetadata[_outputTensorName];
        var shape = outputMeta.Dimensions;
        if (shape.Length >= 2)
        {
            HiddenDim = shape[^1]; // Last dimension is hidden dim
            HasPooledOutput = shape.Length == 2; // [batch, hidden] vs [batch, seq, hidden]
        }
    }

    public IDataView Transform(IDataView input)
    {
        return new AudioScorerDataView(input, this);
    }

    /// <summary>
    /// Direct API: score features through ONNX model.
    /// Input is flattened feature array [frames * bins]. Returns raw model output.
    /// Automatically pads/truncates to match fixed-input models (CLAP, AST).
    /// </summary>
    internal float[] Score(float[] flatFeatures, int frames, int featureDim)
    {
        var modelDims = _session.InputMetadata[_inputTensorName].Dimensions.ToArray();

        // Detect expected frame count from model metadata
        int expectedFrames = modelDims.Length == 3 && modelDims[1] > 0 ? modelDims[1] :
                             modelDims.Length == 4 && modelDims[2] > 0 ? modelDims[2] :
                             -1;

        // Pad or truncate if model expects fixed-size input
        bool wasTruncated = false;
        if (expectedFrames > 0 && frames != expectedFrames)
        {
            wasTruncated = frames > expectedFrames;
            var padded = new float[expectedFrames * featureDim];
            int copyLength = Math.Min(frames, expectedFrames) * featureDim;
            Array.Copy(flatFeatures, padded, copyLength);
            flatFeatures = padded;
            frames = expectedFrames;
        }

        // Build input shape
        int[] inputShape;
        if (modelDims.Length == 3)
            inputShape = [1, frames, featureDim];
        else if (modelDims.Length == 4)
            inputShape = [1, 1, frames, featureDim];
        else
            inputShape = [1, flatFeatures.Length];

        var tensor = new DenseTensor<float>(flatFeatures, inputShape);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputTensorName, tensor)
        };

        // Handle CLAP's is_longer secondary input (bool tensor indicating truncation)
        if (_session.InputMetadata.ContainsKey("is_longer"))
        {
            var isLongerTensor = new DenseTensor<bool>([1, 1]);
            isLongerTensor[0, 0] = wasTruncated;
            inputs.Add(NamedOnnxValue.CreateFromTensor("is_longer", isLongerTensor));
        }

        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == _outputTensorName);
        return output.AsEnumerable<float>().ToArray();
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single),
            BuildScoreAnnotations());
        return builder.ToSchema();
    }

    /// <summary>
    /// Builds column annotations that communicate model dimensions to downstream stages.
    /// This enables auto-discovery in composed pipelines (e.g., pooler reads HiddenDim via .Append()).
    /// </summary>
    private DataViewSchema.Annotations BuildScoreAnnotations()
    {
        int hiddenDim = HiddenDim;
        bool hasPooledOutput = HasPooledOutput;
        var metaBuilder = new DataViewSchema.Annotations.Builder();
        metaBuilder.Add<int>("HiddenDim", NumberDataViewType.Int32,
            (ref int value) => value = hiddenDim);
        metaBuilder.Add<bool>("HasPooledOutput", BooleanDataViewType.Instance,
            (ref bool value) => value = hasPooledOutput);
        return metaBuilder.ToAnnotations();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Audio ONNX scorer transforms cannot be saved.");

    public void Dispose()
    {
        if (!_disposed)
        {
            _session.Dispose();
            _disposed = true;
        }
    }

    // --- Lazy IDataView wrapper ---

    private sealed class AudioScorerDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly OnnxAudioScoringTransformer _scorer;

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        internal AudioScorerDataView(IDataView input, OnnxAudioScoringTransformer scorer)
        {
            _input = input;
            _scorer = scorer;

            var builder = new DataViewSchema.Builder();
            builder.AddColumns(input.Schema);
            builder.AddColumn(scorer._options.OutputColumnName,
                new VectorDataViewType(NumberDataViewType.Single),
                scorer.BuildScoreAnnotations());
            Schema = builder.ToSchema();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            var inputCol = _input.Schema.GetColumnOrNull(_scorer._options.InputColumnName)
                ?? throw new InvalidOperationException(
                    $"Input column '{_scorer._options.InputColumnName}' not found.");

            var upstreamCols = columnsNeeded
                .Select(c => _input.Schema.GetColumnOrNull(c.Name))
                .Where(c => c != null)
                .Select(c => c!.Value)
                .Append(inputCol)
                .Distinct()
                .ToList();

            var inputCursor = _input.GetRowCursor(upstreamCols, rand);
            return new AudioScorerCursor(this, inputCursor, _scorer);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };
    }

    private sealed class AudioScorerCursor : DataViewRowCursor
    {
        private readonly AudioScorerDataView _dataView;
        private readonly DataViewRowCursor _inputCursor;
        private readonly OnnxAudioScoringTransformer _scorer;
        private readonly int _outputColIndex;

        private float[]? _currentScores;
        private long _position = -1;
        private bool _disposed;

        internal AudioScorerCursor(
            AudioScorerDataView dataView,
            DataViewRowCursor inputCursor,
            OnnxAudioScoringTransformer scorer)
        {
            _dataView = dataView;
            _inputCursor = inputCursor;
            _scorer = scorer;
            _outputColIndex = dataView.Schema.GetColumnOrNull(scorer._options.OutputColumnName)?.Index
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

            // Read features from upstream
            var inputCol = _inputCursor.Schema.GetColumnOrNull(_scorer._options.InputColumnName);
            if (inputCol == null) return false;

            var getter = _inputCursor.GetGetter<VBuffer<float>>(inputCol.Value);
            var buffer = default(VBuffer<float>);
            getter(ref buffer);

            var flatFeatures = buffer.DenseValues().ToArray();

            // Infer frame/feature dimensions from features column
            // Features are flattened [frames * bins] — we need to reconstruct dims
            // The feature extractor sets LastFrameCount/LastFeatureDim but we may not have access.
            // Use ONNX model metadata to determine expected input shape.
            var inputMeta = _scorer._session.InputMetadata[_scorer._inputTensorName];
            var dims = inputMeta.Dimensions;
            int frames, featureDim;

            if (dims.Length == 3)
            {
                // [batch, frames, features] — features dim is known, derive frames
                featureDim = dims[2] > 0 ? dims[2] : 80; // fallback to 80 mel bins
                frames = flatFeatures.Length / featureDim;
            }
            else if (dims.Length == 4)
            {
                featureDim = dims[3] > 0 ? dims[3] : 80;
                frames = flatFeatures.Length / featureDim;
            }
            else
            {
                featureDim = flatFeatures.Length;
                frames = 1;
            }

            _currentScores = _scorer.Score(flatFeatures, frames, featureDim);
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Index == _outputColIndex)
            {
                ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                {
                    var data = _currentScores ?? Array.Empty<float>();
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
