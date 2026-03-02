namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Represents a detected speech segment with timestamps and confidence.
/// </summary>
public record SpeechSegment(TimeSpan Start, TimeSpan End, float Confidence)
{
    /// <summary>Duration of the speech segment.</summary>
    public TimeSpan Duration => End - Start;
}

/// <summary>
/// Interface for voice activity detection.
/// Detects speech vs. silence segments in audio streams.
/// </summary>
public interface IVoiceActivityDetector : IDisposable
{
    /// <summary>
    /// Detect speech segments in an audio stream.
    /// </summary>
    IAsyncEnumerable<SpeechSegment> DetectSpeechAsync(
        Stream audioStream,
        VadOptions? options = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for Voice Activity Detection.
/// </summary>
public class VadOptions
{
    /// <summary>Speech probability threshold. Default: 0.5.</summary>
    public float Threshold { get; set; } = 0.5f;

    /// <summary>Minimum speech segment duration. Default: 250ms.</summary>
    public TimeSpan MinSpeechDuration { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>Minimum silence duration to split segments. Default: 100ms.</summary>
    public TimeSpan MinSilenceDuration { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>Padding added to start and end of segments. Default: 30ms.</summary>
    public TimeSpan SpeechPad { get; set; } = TimeSpan.FromMilliseconds(30);

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;
}
