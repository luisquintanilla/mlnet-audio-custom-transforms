using System.Reflection;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator that wraps an ISpeechToTextClient for provider-agnostic speech-to-text.
/// Mirrors the ChatClientEstimator pattern from text inference transforms.
/// </summary>
public sealed class SpeechToTextClientEstimator : IEstimator<SpeechToTextClientTransformer>
{
    private readonly MLContext _mlContext;
    private readonly ISpeechToTextClient _client;
    private readonly SpeechToTextClientOptions _options;

    public SpeechToTextClientEstimator(
        MLContext mlContext,
        ISpeechToTextClient client,
        SpeechToTextClientOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(client);

        _mlContext = mlContext;
        _client = client;
        _options = options ?? new SpeechToTextClientOptions();
    }

    public SpeechToTextClientTransformer Fit(IDataView input)
    {
        return new SpeechToTextClientTransformer(_mlContext, _client, _options);
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
