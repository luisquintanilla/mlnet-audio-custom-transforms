using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Internal interface bridging any ONNX-based TTS transformer to <see cref="OnnxTextToSpeechClient"/>.
/// Implementations extract model-specific parameters from <see cref="TextToSpeechOptions"/>
/// and delegate to their own synthesis pipeline.
/// </summary>
internal interface IOnnxTtsSynthesizer : IDisposable
{
    /// <summary>Synthesize speech from text using the given options.</summary>
    AudioData Synthesize(string text, TextToSpeechOptions? options);

    /// <summary>Provider name for metadata (e.g., "OnnxSpeechT5", "OnnxKittenTts").</summary>
    string ProviderName { get; }

    /// <summary>Provider URI for metadata.</summary>
    Uri? ProviderUri { get; }

    /// <summary>Default model identifier.</summary>
    string? ModelId { get; }
}
