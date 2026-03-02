using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for audio classification using ONNX models.
/// Implements the IEstimator pattern — call Fit() to produce a transformer.
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
        return new OnnxAudioClassificationTransformer(_mlContext, _options);
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
