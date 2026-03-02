using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using MLNet.Audio.Core;
using System.Numerics.Tensors;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer for audio embeddings using ONNX models.
/// Handles: audio → feature extraction → ONNX scoring → pooling → normalization → float[] embedding.
/// Uses eager evaluation (same pattern as OnnxTextEmbeddingTransformer).
/// </summary>
public sealed class OnnxAudioEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioEmbeddingOptions _options;
    private readonly InferenceSession _session;
    private readonly string _inputTensorName;
    private readonly string _outputTensorName;

    /// <summary>
    /// Embedding dimension discovered from the ONNX model.
    /// </summary>
    public int EmbeddingDimension { get; }

    public bool IsRowToRowMapper => true;

    internal OnnxAudioEmbeddingTransformer(MLContext mlContext, OnnxAudioEmbeddingOptions options)
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

        _inputTensorName = options.InputTensorName
            ?? _session.InputNames.FirstOrDefault(n => n.Contains("input"))
            ?? _session.InputNames[0];

        // Prefer pre-pooled output names, fall back to hidden states
        string[] preferredOutputNames = ["sentence_embedding", "pooler_output", "embeddings",
            "last_hidden_state", "output"];
        _outputTensorName = options.OutputTensorName
            ?? _session.OutputNames.FirstOrDefault(n => preferredOutputNames.Contains(n))
            ?? _session.OutputNames[0];

        // Discover embedding dimension from output metadata
        var outputMeta = _session.OutputMetadata[_outputTensorName];
        var dims = outputMeta.Dimensions;
        EmbeddingDimension = dims.Length > 0 ? dims[^1] : 0;
    }

    /// <summary>
    /// Generate embeddings directly (outside ML.NET pipeline).
    /// </summary>
    public float[][] GenerateEmbeddings(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new float[audioInputs.Count][];

        for (int i = 0; i < audioInputs.Count; i++)
        {
            var audio = audioInputs[i];

            // Stage 1: Feature extraction
            var features = _options.FeatureExtractor.Extract(audio);
            int frames = features.GetLength(0);
            int featureDim = features.GetLength(1);

            var flatFeatures = new float[frames * featureDim];
            Buffer.BlockCopy(features, 0, flatFeatures, 0, flatFeatures.Length * sizeof(float));

            // Stage 2: ONNX scoring
            using var ortValue = OrtValue.CreateTensorValueFromMemory(
                flatFeatures, [1, frames, featureDim]);

            var inputs = new Dictionary<string, OrtValue> { [_inputTensorName] = ortValue };

            using var runResults = _session.Run(new RunOptions(), inputs, [_outputTensorName]);
            var outputSpan = runResults[0].GetTensorDataAsSpan<float>();
            var outputDims = runResults[0].GetTensorTypeAndShape().Shape;

            // Stage 3: Pooling
            float[] embedding;
            if (outputDims.Length == 2)
            {
                // Pre-pooled: [batch, dim] — just take the embedding directly
                int dim = (int)outputDims[1];
                embedding = outputSpan.Slice(0, dim).ToArray();
            }
            else if (outputDims.Length == 3)
            {
                // Hidden states: [batch, seq, dim] — apply pooling
                int seqLen = (int)outputDims[1];
                int dim = (int)outputDims[2];
                embedding = Pool(outputSpan, seqLen, dim);
            }
            else
            {
                embedding = outputSpan.ToArray();
            }

            // Stage 4: Normalization
            if (_options.Normalize)
                L2Normalize(embedding);

            results[i] = embedding;
        }

        return results;
    }

    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var audioInputs = audioSamples.Select(s => new AudioData(s, _options.SampleRate)).ToList();
        var embeddings = GenerateEmbeddings(audioInputs);

        var outputRows = embeddings.Select(e => new AudioEmbeddingOutput { Embedding = e }).ToArray();
        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, EmbeddingDimension > 0 ? EmbeddingDimension : 0));
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

    private float[] Pool(ReadOnlySpan<float> hiddenStates, int seqLen, int dim)
    {
        var result = new float[dim];

        switch (_options.Pooling)
        {
            case AudioPoolingStrategy.ClsToken:
                hiddenStates.Slice(0, dim).CopyTo(result);
                break;

            case AudioPoolingStrategy.MaxPooling:
                Array.Fill(result, float.MinValue);
                for (int s = 0; s < seqLen; s++)
                    TensorPrimitives.Max(result, hiddenStates.Slice(s * dim, dim), result);
                break;

            case AudioPoolingStrategy.MeanPooling:
            default:
                for (int s = 0; s < seqLen; s++)
                    TensorPrimitives.Add(result, hiddenStates.Slice(s * dim, dim), result);
                TensorPrimitives.Divide(result, (float)seqLen, result);
                break;
        }

        return result;
    }

    private static void L2Normalize(float[] vector)
    {
        float norm = TensorPrimitives.Norm(vector.AsSpan());
        if (norm > 0)
            TensorPrimitives.Divide(vector, norm, vector);
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
}

internal class AudioEmbeddingOutput
{
    [ColumnName("Embedding")]
    [VectorType]
    public float[] Embedding { get; set; } = [];
}
