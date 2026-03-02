using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using MLNet.Audio.Core;
using System.Numerics.Tensors;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer for audio classification using ONNX models.
/// Handles the entire pipeline: audio → feature extraction → ONNX scoring → softmax → labels.
/// Uses eager evaluation (same pattern as ChatClientTransformer / OnnxTextGenerationTransformer).
/// </summary>
public sealed class OnnxAudioClassificationTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioClassificationOptions _options;
    private readonly InferenceSession _session;
    private readonly string _inputTensorName;
    private readonly string _outputTensorName;
    private readonly int[] _inputShape;

    public bool IsRowToRowMapper => true;

    internal OnnxAudioClassificationTransformer(MLContext mlContext, OnnxAudioClassificationOptions options)
    {
        _mlContext = mlContext;
        _options = options;

        var sessionOptions = new SessionOptions();
        if (options.GpuDeviceId.HasValue)
        {
            try { sessionOptions.AppendExecutionProvider_CUDA(options.GpuDeviceId.Value); }
            catch { /* fallback to CPU */ }
        }

        _session = new InferenceSession(options.ModelPath, sessionOptions);

        // Auto-discover tensor names
        _inputTensorName = options.InputTensorName
            ?? _session.InputNames.FirstOrDefault(n => n.Contains("input"))
            ?? _session.InputNames[0];

        _outputTensorName = options.OutputTensorName
            ?? _session.OutputNames.FirstOrDefault(n => n.Contains("logits"))
            ?? _session.OutputNames[0];

        // Discover input shape for feature validation
        var inputMeta = _session.InputMetadata[_inputTensorName];
        _inputShape = inputMeta.Dimensions;
    }

    /// <summary>
    /// Classify audio samples directly (outside ML.NET pipeline).
    /// </summary>
    public AudioClassificationResult[] Classify(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new AudioClassificationResult[audioInputs.Count];

        for (int i = 0; i < audioInputs.Count; i++)
        {
            var audio = audioInputs[i];

            // Stage 1: Feature extraction
            var features = _options.FeatureExtractor.Extract(audio);
            int frames = features.GetLength(0);
            int featureDim = features.GetLength(1);

            // Flatten to 1D for tensor construction: [1, frames, featureDim]
            var flatFeatures = new float[frames * featureDim];
            Buffer.BlockCopy(features, 0, flatFeatures, 0, flatFeatures.Length * sizeof(float));

            // Stage 2: ONNX scoring
            using var ortValue = OrtValue.CreateTensorValueFromMemory(
                flatFeatures,
                [1, frames, featureDim]);

            var inputs = new Dictionary<string, OrtValue> { [_inputTensorName] = ortValue };

            using var runResults = _session.Run(new RunOptions(), inputs, [_outputTensorName]);
            var outputTensor = runResults[0].GetTensorDataAsSpan<float>();

            // Stage 3: Softmax + label assignment
            int numClasses = _options.Labels.Length;
            var logits = outputTensor.Slice(0, numClasses).ToArray();
            var probabilities = Softmax(logits);

            int bestIdx = 0;
            float bestScore = probabilities[0];
            for (int j = 1; j < probabilities.Length; j++)
            {
                if (probabilities[j] > bestScore)
                {
                    bestIdx = j;
                    bestScore = probabilities[j];
                }
            }

            results[i] = new AudioClassificationResult
            {
                PredictedLabel = _options.Labels[bestIdx],
                Score = bestScore,
                Probabilities = probabilities,
                Labels = _options.Labels
            };
        }

        return results;
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation. Reads all audio, classifies, returns results.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var audioInputs = audioSamples.Select(s => new AudioData(s, _options.SampleRate)).ToList();
        var results = Classify(audioInputs);

        var outputRows = results.Select(r => new AudioClassificationOutput
        {
            PredictedLabel = r.PredictedLabel,
            Score = r.Score,
            Probabilities = r.Probabilities
        }).ToArray();

        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        builder.AddColumn(_options.ScoreColumnName, NumberDataViewType.Single);
        builder.AddColumn(_options.ProbabilitiesColumnName,
            new VectorDataViewType(NumberDataViewType.Single, _options.Labels.Length));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use ModelPackager.Save() instead.");

    public void Dispose()
    {
        _session?.Dispose();
    }

    private List<float[]> ReadAudioColumn(IDataView data)
    {
        var result = new List<float[]>();
        var col = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' not found in schema.");

        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        var buffer = default(VBuffer<float>);

        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.DenseValues().ToArray());
        }

        return result;
    }

    private static float[] Softmax(float[] logits)
    {
        var result = new float[logits.Length];
        TensorPrimitives.SoftMax(logits, result);
        return result;
    }
}

/// <summary>
/// Result of audio classification for a single audio input.
/// </summary>
public class AudioClassificationResult
{
    public required string PredictedLabel { get; init; }
    public required float Score { get; init; }
    public required float[] Probabilities { get; init; }
    public required string[] Labels { get; init; }
}

/// <summary>
/// Output row for ML.NET IDataView integration.
/// </summary>
internal class AudioClassificationOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";

    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probabilities")]
    [VectorType]
    public float[] Probabilities { get; set; } = [];
}
