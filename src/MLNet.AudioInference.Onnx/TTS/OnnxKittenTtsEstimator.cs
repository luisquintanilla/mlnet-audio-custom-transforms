using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// IEstimator wrapper for OnnxKittenTtsTransformer.
/// Fit() creates the transformer immediately (no training needed).
/// </summary>
public sealed class OnnxKittenTtsEstimator : IEstimator<OnnxKittenTtsTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxKittenTtsOptions _options;

    public OnnxKittenTtsEstimator(MLContext mlContext, OnnxKittenTtsOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public OnnxKittenTtsTransformer Fit(IDataView input)
        => new(_mlContext, _options);

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.VariableVector,
            (DataViewType)NumberDataViewType.Single,
            false,
            null
        ]);

        return new SchemaShape(columns.Values);
    }
}
