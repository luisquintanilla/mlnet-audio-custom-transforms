using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for audio embeddings using ONNX models.
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
        return new OnnxAudioEmbeddingTransformer(_mlContext, _options);
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
