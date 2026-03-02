namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for the SpeechToTextClient ML.NET transform.
/// Provider-agnostic — wraps any ISpeechToTextClient implementation.
/// </summary>
public class SpeechToTextClientOptions
{
    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for the transcribed text. Default: "Text".</summary>
    public string OutputColumnName { get; set; } = "Text";

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Speech language (e.g., "en", "es", "fr"). Null = auto-detect.</summary>
    public string? SpeechLanguage { get; set; }

    /// <summary>Target text language for translation. Null = same as speech language.</summary>
    public string? TextLanguage { get; set; }
}
