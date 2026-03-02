using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for audio classification using ONNX models.
/// Composes 3 sub-transforms: feature extraction → ONNX scoring → classification post-processing.
/// </summary>
public sealed class OnnxAudioClassificationEstimator : IEstimator<OnnxAudioClassificationTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxAudioClassificationOptions _options;

    public OnnxAudioClassificationEstimator(MLContext mlContext, OnnxAudioClassificationOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (options.Labels is null || options.Labels.Length == 0)
            throw new ArgumentException("Labels must be provided and non-empty.");

        _mlContext = mlContext;
        _options = options;
    }

    public OnnxAudioClassificationTransformer Fit(IDataView input)
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

        // Stage 3: Classification post-processing
        var scoredData = scorerTransformer.Transform(featuredData);
        var postProcessOptions = new AudioClassificationPostProcessOptions
        {
            Labels = _options.Labels,
            InputColumnName = "Scores",
            PredictedLabelColumnName = _options.PredictedLabelColumnName,
            ProbabilitiesColumnName = _options.ProbabilitiesColumnName,
            ScoreColumnName = _options.ScoreColumnName
        };
        var postProcessEstimator = new AudioClassificationPostProcessEstimator(env, postProcessOptions);
        var postProcessTransformer = postProcessEstimator.Fit(scoredData);

        return new OnnxAudioClassificationTransformer(
            _mlContext, _options, featureTransformer, scorerTransformer, postProcessTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        // Use reflection to construct SchemaShape.Column (internal constructor)
        var colCtor = typeof(SchemaShape.Column).GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.PredictedLabelColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.PredictedLabelColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);

        columns[_options.ScoreColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.ScoreColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        columns[_options.ProbabilitiesColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.ProbabilitiesColumnName,
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}
