using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for the ONNX VAD (Voice Activity Detection) transform.
/// Designed to work with Silero VAD v5 ONNX model.
/// </summary>
public class OnnxVadOptions
{
    /// <summary>Path to the ONNX VAD model file (e.g., silero_vad.onnx).</summary>
    public required string ModelPath { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for speech segment data. Default: "SpeechSegments".</summary>
    public string OutputColumnName { get; set; } = "SpeechSegments";

    /// <summary>Speech probability threshold. Default: 0.5.</summary>
    public float Threshold { get; set; } = 0.5f;

    /// <summary>Minimum speech duration to keep a segment. Default: 250ms.</summary>
    public TimeSpan MinSpeechDuration { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>Minimum silence duration to split segments. Default: 100ms.</summary>
    public TimeSpan MinSilenceDuration { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>Padding added around detected speech. Default: 30ms.</summary>
    public TimeSpan SpeechPad { get; set; } = TimeSpan.FromMilliseconds(30);

    /// <summary>
    /// Frame/window size in samples for VAD processing.
    /// Silero VAD uses 512 samples (32ms at 16kHz) or 256 (16ms).
    /// Default: 512.
    /// </summary>
    public int WindowSize { get; set; } = 512;

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>GPU device ID. Null = CPU.</summary>
    public int? GpuDeviceId { get; set; }
}
