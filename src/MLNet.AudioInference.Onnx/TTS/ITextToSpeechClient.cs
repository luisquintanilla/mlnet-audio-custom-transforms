using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Provider-agnostic text-to-speech client interface.
///
/// MEAI (Microsoft.Extensions.AI) does not yet define ITextToSpeechClient as of 10.x.
/// This is our prototype interface following MEAI patterns (mirrors ISpeechToTextClient).
/// When MEAI adds an official ITextToSpeechClient, this can be replaced.
///
/// Design follows MEAI conventions:
///   - GetAudioAsync for single-shot synthesis
///   - GetStreamingAudioAsync for chunked streaming output
///   - Metadata for provider/model info
///   - Options for per-request configuration
/// </summary>
public interface ITextToSpeechClient : IDisposable
{
    /// <summary>
    /// Synthesizes speech from text and returns the complete audio result.
    /// </summary>
    Task<TextToSpeechResponse> GetAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech from text and streams audio chunks as they are generated.
    /// </summary>
    IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Metadata about the TTS provider and model.
    /// </summary>
    TextToSpeechClientMetadata Metadata { get; }
}

/// <summary>
/// Response from a text-to-speech synthesis request.
/// </summary>
public class TextToSpeechResponse
{
    /// <summary>The synthesized audio.</summary>
    public required AudioData Audio { get; init; }

    /// <summary>The voice/speaker used for synthesis.</summary>
    public string? Voice { get; init; }

    /// <summary>Duration of the generated audio.</summary>
    public TimeSpan Duration => Audio.Duration;
}

/// <summary>
/// A streaming chunk of synthesized audio.
/// </summary>
public class TextToSpeechResponseUpdate
{
    /// <summary>Chunk of audio samples.</summary>
    public required AudioData Audio { get; init; }

    /// <summary>Whether this is the final chunk.</summary>
    public bool IsFinal { get; init; }
}

/// <summary>
/// Options for a text-to-speech request.
/// </summary>
public class TextToSpeechOptions
{
    /// <summary>Voice/speaker identifier or name.</summary>
    public string? Voice { get; set; }

    /// <summary>Speech speed multiplier (1.0 = normal).</summary>
    public float Speed { get; set; } = 1.0f;

    /// <summary>Language code (e.g., "en").</summary>
    public string? Language { get; set; }

    /// <summary>Speaker embedding for custom voice (e.g., x-vector from reference audio).</summary>
    public float[]? SpeakerEmbedding { get; set; }
}

/// <summary>
/// Metadata about a text-to-speech client.
/// </summary>
public class TextToSpeechClientMetadata
{
    public string ProviderName { get; }
    public Uri? ProviderUri { get; }
    public string? DefaultModelId { get; }

    public TextToSpeechClientMetadata(string providerName, Uri? providerUri = null, string? defaultModelId = null)
    {
        ProviderName = providerName;
        ProviderUri = providerUri;
        DefaultModelId = defaultModelId;
    }
}
