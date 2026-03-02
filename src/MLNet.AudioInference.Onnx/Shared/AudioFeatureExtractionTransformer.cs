using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Stage 1 sub-transform: extracts audio features (mel spectrogram) from raw PCM audio.
/// Wraps an AudioFeatureExtractor and produces a "Features" column as flattened VBuffer&lt;float&gt;.
/// </summary>
public sealed class AudioFeatureExtractionTransformer : ITransformer, IDisposable
{
    private readonly AudioFeatureExtractionOptions _options;
    internal AudioFeatureExtractionOptions Options => _options;

    // Cached dimensions from last extraction (set during Transform)
    internal int LastFrameCount { get; private set; }
    internal int LastFeatureDim { get; private set; }

    public bool IsRowToRowMapper => true;

    internal AudioFeatureExtractionTransformer(IHostEnvironment env, AudioFeatureExtractionOptions options)
    {
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new FeatureExtractionDataView(input, this);
    }

    /// <summary>
    /// Direct API: extract features from AudioData instances.
    /// </summary>
    public float[][] ExtractFeatures(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new float[audioInputs.Count][];
        for (int i = 0; i < audioInputs.Count; i++)
        {
            var features = _options.FeatureExtractor.Extract(audioInputs[i]);
            int frames = features.GetLength(0);
            int bins = features.GetLength(1);
            LastFrameCount = frames;
            LastFeatureDim = bins;

            // Flatten [frames, bins] → float[]
            var flat = new float[frames * bins];
            Buffer.BlockCopy(features, 0, flat, 0, flat.Length * sizeof(float));
            results[i] = flat;
        }
        return results;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Audio feature extraction transforms cannot be saved.");

    public void Dispose() { }

    // --- Lazy IDataView wrapper ---

    private sealed class FeatureExtractionDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly AudioFeatureExtractionTransformer _transformer;

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        internal FeatureExtractionDataView(IDataView input, AudioFeatureExtractionTransformer transformer)
        {
            _input = input;
            _transformer = transformer;

            var builder = new DataViewSchema.Builder();
            builder.AddColumns(input.Schema);
            builder.AddColumn(transformer._options.OutputColumnName,
                new VectorDataViewType(NumberDataViewType.Single));
            Schema = builder.ToSchema();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            // Always need the audio input column
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
            return new FeatureExtractionCursor(this, inputCursor, _transformer);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };
    }

    private sealed class FeatureExtractionCursor : DataViewRowCursor
    {
        private readonly FeatureExtractionDataView _dataView;
        private readonly DataViewRowCursor _inputCursor;
        private readonly AudioFeatureExtractionTransformer _transformer;
        private readonly int _outputColIndex;

        private float[]? _currentFeatures;
        private long _position = -1;
        private bool _disposed;

        internal FeatureExtractionCursor(
            FeatureExtractionDataView dataView,
            DataViewRowCursor inputCursor,
            AudioFeatureExtractionTransformer transformer)
        {
            _dataView = dataView;
            _inputCursor = inputCursor;
            _transformer = transformer;
            _outputColIndex = dataView.Schema.GetColumnOrNull(transformer._options.OutputColumnName)?.Index
                ?? throw new InvalidOperationException("Output column not found in schema.");
        }

        public override DataViewSchema Schema => _dataView.Schema;
        public override long Position => _position;
        public override long Batch => _inputCursor.Batch;

        public override bool MoveNext()
        {
            if (!_inputCursor.MoveNext())
                return false;

            _position++;

            // Read audio samples from input
            var audioCol = _dataView.Schema.GetColumnOrNull(_transformer._options.InputColumnName);
            if (audioCol == null) return false;

            // Get from upstream cursor using the input schema column
            var inputAudioCol = _inputCursor.Schema.GetColumnOrNull(_transformer._options.InputColumnName);
            if (inputAudioCol == null) return false;

            var getter = _inputCursor.GetGetter<VBuffer<float>>(inputAudioCol.Value);
            var buffer = default(VBuffer<float>);
            getter(ref buffer);

            var samples = buffer.DenseValues().ToArray();
            var audio = new AudioData(samples, _transformer._options.SampleRate);

            // Extract features
            var features = _transformer._options.FeatureExtractor.Extract(audio);
            int frames = features.GetLength(0);
            int bins = features.GetLength(1);
            _transformer.LastFrameCount = frames;
            _transformer.LastFeatureDim = bins;

            // Flatten [frames, bins] → float[]
            _currentFeatures = new float[frames * bins];
            Buffer.BlockCopy(features, 0, _currentFeatures, 0, _currentFeatures.Length * sizeof(float));

            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Index == _outputColIndex)
            {
                ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                {
                    var data = _currentFeatures ?? Array.Empty<float>();
                    var editor = VBufferEditor.Create(ref value, data.Length);
                    data.AsSpan().CopyTo(editor.Values);
                    value = editor.Commit();
                };
                return (ValueGetter<TValue>)(object)getter;
            }

            // Passthrough: delegate to upstream cursor
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
