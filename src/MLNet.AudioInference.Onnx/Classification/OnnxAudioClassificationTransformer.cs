using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using MLNet.Audio.Core;
using System.Numerics.Tensors;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer for audio classification using ONNX models.
/// Composes 3 sub-transforms: feature extraction → ONNX scoring → softmax + labels.
/// Provides both Transform(IDataView) for ML.NET pipelines and Classify() for direct use.
/// </summary>
public sealed class OnnxAudioClassificationTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioClassificationOptions _options;
    private readonly AudioFeatureExtractionTransformer _featureTransformer;
    private readonly OnnxAudioScorerTransformer _scorerTransformer;
    private readonly AudioClassificationPostProcessTransformer _postProcessTransformer;

    public bool IsRowToRowMapper => true;

    internal OnnxAudioClassificationTransformer(
        MLContext mlContext,
        OnnxAudioClassificationOptions options,
        AudioFeatureExtractionTransformer featureTransformer,
        OnnxAudioScorerTransformer scorerTransformer,
        AudioClassificationPostProcessTransformer postProcessTransformer)
    {
        _mlContext = mlContext;
        _options = options;
        _featureTransformer = featureTransformer;
        _scorerTransformer = scorerTransformer;
        _postProcessTransformer = postProcessTransformer;
    }

    /// <summary>
    /// ML.NET Transform — chains 3 sub-transforms via lazy IDataView.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var features = _featureTransformer.Transform(input);       // Stage 1: audio → mel features
        var scores = _scorerTransformer.Transform(features);        // Stage 2: features → ONNX scores
        var results = _postProcessTransformer.Transform(scores);    // Stage 3: scores → labels + probs
        return results;
    }

    /// <summary>
    /// Direct convenience API: classify audio samples (bypasses IDataView).
    /// </summary>
    public AudioClassificationResult[] Classify(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new AudioClassificationResult[audioInputs.Count];

        // Use sub-transform direct APIs
        var features = _featureTransformer.ExtractFeatures(audioInputs);
        for (int i = 0; i < audioInputs.Count; i++)
        {
            var featureExtractor = _options.FeatureExtractor;
            int featureDim = _featureTransformer.LastFeatureDim;
            int frames = _featureTransformer.LastFrameCount;

            var rawScores = _scorerTransformer.Score(features[i], frames, featureDim);
            results[i] = _postProcessTransformer.PostProcess(rawScores);
            results[i].Labels = _options.Labels;
        }

        return results;
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
        _featureTransformer?.Dispose();
        _scorerTransformer?.Dispose();
        _postProcessTransformer?.Dispose();
    }
}

/// <summary>
/// Result of audio classification for a single audio input.
/// </summary>
public class AudioClassificationResult
{
    public required string PredictedLabel { get; set; }
    public required float Score { get; set; }
    public required float[] Probabilities { get; set; }
    public string[]? Labels { get; set; }
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
