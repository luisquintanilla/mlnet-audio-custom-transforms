using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer that wraps an ISpeechToTextClient for provider-agnostic ASR.
/// Mirrors the ChatClientTransformer pattern — any MEAI-compliant STT provider
/// (Azure, OpenAI, local Whisper, etc.) can be used as an ML.NET pipeline step.
/// Uses eager evaluation: reads all audio rows, transcribes, returns results.
/// </summary>
public sealed class SpeechToTextClientTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly ISpeechToTextClient _client;
    private readonly SpeechToTextClientOptions _options;

    public bool IsRowToRowMapper => true;

    internal SpeechToTextClientTransformer(
        MLContext mlContext,
        ISpeechToTextClient client,
        SpeechToTextClientOptions options)
    {
        _mlContext = mlContext;
        _client = client;
        _options = options;
    }

    /// <summary>
    /// Transcribe audio samples directly (outside ML.NET pipeline).
    /// </summary>
    public string[] Transcribe(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new string[audioInputs.Count];

        for (int i = 0; i < audioInputs.Count; i++)
        {
            var audio = audioInputs[i];

            // Convert AudioData to WAV stream for the ISpeechToTextClient
            using var stream = new MemoryStream();
            AudioIO.SaveWav(stream, audio);
            stream.Position = 0;

            var sttOptions = new SpeechToTextOptions
            {
                SpeechSampleRate = _options.SampleRate,
            };
            if (_options.SpeechLanguage is not null)
                sttOptions.SpeechLanguage = _options.SpeechLanguage;
            if (_options.TextLanguage is not null)
                sttOptions.TextLanguage = _options.TextLanguage;

            // Sync-over-async bridge (same approach as ChatClientTransformer)
            var response = _client.GetTextAsync(stream, sttOptions)
                .GetAwaiter().GetResult();

            results[i] = response?.Text ?? string.Empty;
        }

        return results;
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation. Reads all audio, transcribes, returns results.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var audioInputs = audioSamples
            .Select(s => new AudioData(s, _options.SampleRate))
            .ToList();
        var transcriptions = Transcribe(audioInputs);

        var outputRows = transcriptions
            .Select(t => new SpeechToTextOutput { Text = t })
            .ToArray();

        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "SpeechToTextClientTransformer wraps a runtime ISpeechToTextClient and cannot be serialized. " +
            "For serializable ASR pipelines, use an ONNX-native transformer.");

    public void Dispose()
    {
        _client?.Dispose();
    }

    private List<float[]> ReadAudioColumn(IDataView data)
    {
        var result = new List<float[]>();
        var col = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' not found in schema.");

        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        var buffer = default(VBuffer<float>);

        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.DenseValues().ToArray());
        }

        return result;
    }
}

internal class SpeechToTextOutput
{
    [ColumnName("Text")]
    public string Text { get; set; } = "";
}
