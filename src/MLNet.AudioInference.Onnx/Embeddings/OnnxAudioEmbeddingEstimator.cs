using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for audio embeddings using ONNX models.
/// Composes 3 sub-transforms: feature extraction → ONNX scoring → embedding pooling.
/// </summary>
public sealed class OnnxAudioEmbeddingEstimator : IEstimator<OnnxAudioEmbeddingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioEmbeddingOptions _options;

    public OnnxAudioEmbeddingEstimator(MLContext mlContext, OnnxAudioEmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");

        _mlContext = mlContext;
        _options = options;
    }

    public OnnxAudioEmbeddingTransformer Fit(IDataView input)
    {
        var env = (IHostEnvironment)_mlContext;

        // Stage 1: Feature extraction
        var featureOptions = new AudioFeatureExtractionOptions
        {
            FeatureExtractor = _options.FeatureExtractor,
            InputColumnName = _options.InputColumnName,
            OutputColumnName = "Features",
            SampleRate = _options.SampleRate
        };
        var featureEstimator = new AudioFeatureExtractionEstimator(env, featureOptions);
        var featureTransformer = featureEstimator.Fit(input);

        // Stage 2: ONNX scoring
        var featuredData = featureTransformer.Transform(input);
        var scorerOptions = new OnnxAudioScorerOptions
        {
            ModelPath = _options.ModelPath,
            InputColumnName = "Features",
            OutputColumnName = "Scores",
            InputTensorName = _options.InputTensorName,
            OutputTensorName = _options.OutputTensorName,
            GpuDeviceId = _options.GpuDeviceId
        };
        var scorerEstimator = new OnnxAudioScorerEstimator(env, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(featuredData);

        // Stage 3: Embedding pooling + normalization
        var scoredData = scorerTransformer.Transform(featuredData);
        var poolerOptions = new AudioEmbeddingPoolerOptions
        {
            InputColumnName = "Scores",
            OutputColumnName = _options.OutputColumnName,
            Pooling = _options.Pooling,
            Normalize = _options.Normalize,
            HiddenDim = scorerTransformer.HiddenDim,
            IsPrePooled = scorerTransformer.HasPooledOutput
        };
        var poolerEstimator = new AudioEmbeddingPoolerEstimator(env, poolerOptions);
        var poolerTransformer = poolerEstimator.Fit(scoredData);

        return new OnnxAudioEmbeddingTransformer(
            _mlContext, _options, featureTransformer, scorerTransformer, poolerTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}
