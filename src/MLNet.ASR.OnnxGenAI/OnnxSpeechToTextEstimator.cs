using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// ML.NET estimator for local Whisper speech-to-text via ORT GenAI.
/// Creates an OnnxSpeechToTextTransformer that handles the full audio-to-text pipeline.
/// </summary>
public class OnnxSpeechToTextEstimator : IEstimator<OnnxSpeechToTextTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxSpeechToTextOptions _options;

    public OnnxSpeechToTextEstimator(MLContext mlContext, OnnxSpeechToTextOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public OnnxSpeechToTextTransformer Fit(IDataView input)
    {
        return new OnnxSpeechToTextTransformer(_mlContext, _options);
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
