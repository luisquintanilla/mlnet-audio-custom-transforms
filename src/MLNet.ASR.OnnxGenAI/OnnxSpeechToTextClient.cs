using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// Local ISpeechToTextClient backed by Whisper via ORT GenAI.
/// Plugs into the MEAI ecosystem — usable via DI, middleware, etc.
/// </summary>
public sealed class OnnxSpeechToTextClient : ISpeechToTextClient
{
    private readonly OnnxSpeechToTextTransformer _transformer;
    private readonly OnnxSpeechToTextOptions _options;

    public OnnxSpeechToTextClient(OnnxSpeechToTextOptions options)
    {
        _options = options;
        _transformer = new OnnxSpeechToTextTransformer(new Microsoft.ML.MLContext(), options);
    }

    public SpeechToTextClientMetadata Metadata => new(
        providerName: "OnnxGenAI-Whisper",
        providerUri: null,
        defaultModelId: System.IO.Path.GetFileName(_options.ModelPath));

    public Task<SpeechToTextResponse> GetTextAsync(
        Stream audioStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Load WAV audio from the stream
        var audio = AudioIO.LoadWav(audioStream);

        // Resample if needed
        if (audio.SampleRate != _options.SampleRate)
            audio = AudioIO.Resample(audio, _options.SampleRate);

        var results = _transformer.TranscribeWithTimestamps([audio]);
        var result = results[0];

        var response = new SpeechToTextResponse(result.Text);

        return Task.FromResult(response);
    }

    public IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(
        Stream audioStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // For now, return the full result as a single update.
        // Future: implement true streaming with chunked audio processing.
        return StreamingImpl(audioStream, options, cancellationToken);
    }

    private async IAsyncEnumerable<SpeechToTextResponseUpdate> StreamingImpl(
        Stream audioStream,
        SpeechToTextOptions? options,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var response = await GetTextAsync(audioStream, options, cancellationToken);

        yield return new SpeechToTextResponseUpdate(response.Text)
        {
            Kind = SpeechToTextResponseUpdateKind.TextUpdated
        };
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is null && serviceType.IsAssignableFrom(GetType()))
            return this;
        return null;
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
