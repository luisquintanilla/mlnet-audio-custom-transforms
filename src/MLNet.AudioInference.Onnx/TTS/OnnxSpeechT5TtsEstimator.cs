using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// IEstimator wrapper for OnnxSpeechT5TtsTransformer.
/// Fit() creates the transformer immediately (no training needed).
/// </summary>
public sealed class OnnxSpeechT5TtsEstimator : IEstimator<OnnxSpeechT5TtsTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxSpeechT5Options _options;

    public OnnxSpeechT5TtsEstimator(MLContext mlContext, OnnxSpeechT5Options options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public OnnxSpeechT5TtsTransformer Fit(IDataView input)
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
