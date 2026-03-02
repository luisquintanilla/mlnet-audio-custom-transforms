using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for raw ONNX Whisper speech-to-text.
/// Creates an OnnxWhisperTransformer that handles the full audio-to-text pipeline
/// using standard ONNX Runtime (no ORT GenAI dependency).
/// </summary>
public class OnnxWhisperEstimator : IEstimator<OnnxWhisperTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxWhisperOptions _options;

    public OnnxWhisperEstimator(MLContext mlContext, OnnxWhisperOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public OnnxWhisperTransformer Fit(IDataView input)
    {
        return new OnnxWhisperTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}
