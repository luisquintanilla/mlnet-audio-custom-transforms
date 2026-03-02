using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer for audio embeddings using ONNX models.
/// Composes 3 sub-transforms: feature extraction → ONNX scoring → pooling + normalization.
/// Provides both Transform(IDataView) for ML.NET pipelines and GenerateEmbeddings() for direct use.
/// </summary>
public sealed class OnnxAudioEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioEmbeddingOptions _options;
    private readonly AudioFeatureExtractionTransformer _featureTransformer;
    private readonly OnnxAudioScoringTransformer _scorerTransformer;
    private readonly AudioEmbeddingPoolingTransformer _poolerTransformer;

    /// <summary>
    /// Embedding dimension discovered from the ONNX model.
    /// </summary>
    public int EmbeddingDimension => _scorerTransformer.HiddenDim;

    public bool IsRowToRowMapper => true;

    internal OnnxAudioEmbeddingTransformer(
        MLContext mlContext,
        OnnxAudioEmbeddingOptions options,
        AudioFeatureExtractionTransformer featureTransformer,
        OnnxAudioScoringTransformer scorerTransformer,
        AudioEmbeddingPoolingTransformer poolerTransformer)
    {
        _mlContext = mlContext;
        _options = options;
        _featureTransformer = featureTransformer;
        _scorerTransformer = scorerTransformer;
        _poolerTransformer = poolerTransformer;
    }

    /// <summary>
    /// ML.NET Transform — chains 3 sub-transforms via lazy IDataView.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var features = _featureTransformer.Transform(input);       // Stage 1: audio → mel features
        var scores = _scorerTransformer.Transform(features);        // Stage 2: features → ONNX scores
        var embeddings = _poolerTransformer.Transform(scores);      // Stage 3: scores → pooled embeddings
        return embeddings;
    }

    /// <summary>
    /// Direct convenience API: generate embeddings (bypasses IDataView).
    /// </summary>
    public float[][] GenerateEmbeddings(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new float[audioInputs.Count][];

        // Use sub-transform direct APIs
        var features = _featureTransformer.ExtractFeatures(audioInputs);
        for (int i = 0; i < audioInputs.Count; i++)
        {
            int featureDim = _featureTransformer.LastFeatureDim;
            int frames = _featureTransformer.LastFrameCount;

            var rawScores = _scorerTransformer.Score(features[i], frames, featureDim);
            results[i] = _poolerTransformer.Pool(rawScores);
        }

        return results;
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
        _featureTransformer?.Dispose();
        _scorerTransformer?.Dispose();
        _poolerTransformer?.Dispose();
    }
}

internal class AudioEmbeddingOutput
{
    [ColumnName("Embedding")]
    [VectorType]
    public float[] Embedding { get; set; } = [];
}
