namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for raw ONNX Whisper speech-to-text (no ORT GenAI dependency).
/// Uses standard HuggingFace optimum-exported ONNX models.
/// </summary>
public class OnnxWhisperOptions
{
    /// <summary>Path to encoder_model.onnx.</summary>
    public required string EncoderModelPath { get; set; }

    /// <summary>
    /// Path to decoder_model_merged.onnx (handles both prefill and decode-with-past).
    /// This is the recommended model from HuggingFace optimum export.
    /// </summary>
    public required string DecoderModelPath { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for the transcribed text. Default: "Text".</summary>
    public string OutputColumnName { get; set; } = "Text";

    /// <summary>Maximum number of tokens to generate. Default: 256.</summary>
    public int MaxTokens { get; set; } = 256;

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Speech language code (e.g., "en", "es", "fr"). Null = auto-detect.</summary>
    public string? Language { get; set; }

    /// <summary>Whether to translate to English. Default: false.</summary>
    public bool Translate { get; set; } = false;

    /// <summary>Whether the model is multilingual. Default: true.</summary>
    public bool IsMultilingual { get; set; } = true;

    /// <summary>Number of mel bins. Default: 80 (Whisper v1-v2). Use 128 for v3.</summary>
    public int NumMelBins { get; set; } = 80;

    /// <summary>Number of decoder layers. Auto-detected from model if 0.</summary>
    public int NumDecoderLayers { get; set; } = 0;

    /// <summary>Number of attention heads. Auto-detected from model if 0.</summary>
    public int NumAttentionHeads { get; set; } = 0;

    /// <summary>
    /// Sampling temperature. 0 = greedy (argmax). Default: 0 (greedy).
    /// </summary>
    public float Temperature { get; set; } = 0f;
}
