using System.Runtime.CompilerServices;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ITextToSpeechClient implementation wrapping OnnxSpeechT5TtsTransformer.
/// Provides the MEAI-style abstraction for local SpeechT5 TTS.
///
/// Usage:
///   using var client = new OnnxTextToSpeechClient(options);
///   var response = await client.GetAudioAsync("Hello, world!");
///   AudioIO.SaveWav("output.wav", response.Audio);
/// </summary>
public sealed class OnnxTextToSpeechClient : ITextToSpeechClient
{
    private readonly OnnxSpeechT5TtsTransformer _transformer;

    public TextToSpeechClientMetadata Metadata { get; }

    public OnnxTextToSpeechClient(OnnxSpeechT5Options options)
    {
        var mlContext = new Microsoft.ML.MLContext();
        _transformer = new OnnxSpeechT5TtsTransformer(mlContext, options);

        var modelId = Path.GetFileName(Path.GetDirectoryName(options.EncoderModelPath)) ?? "speecht5";
        Metadata = new TextToSpeechClientMetadata(
            "OnnxSpeechT5",
            new Uri("https://huggingface.co/microsoft/speecht5_tts"),
            modelId);
    }

    public Task<TextToSpeechResponse> GetAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var speakerEmbedding = options?.SpeakerEmbedding;
        var audio = _transformer.Synthesize(text, speakerEmbedding);

        var response = new TextToSpeechResponse
        {
            Audio = audio,
            Voice = options?.Voice ?? "default"
        };

        return Task.FromResult(response);
    }

    public async IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // For now, single chunk (full synthesis then yield).
        // Future: chunk vocoder output for real-time streaming.
        var response = await GetAudioAsync(text, options, cancellationToken);

        yield return new TextToSpeechResponseUpdate
        {
            Audio = response.Audio,
            IsFinal = true
        };
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
