using Microsoft.ML.Tokenizers;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for KittenTTS text-to-speech using raw ONNX Runtime.
/// Uses models from https://github.com/KittenML/KittenTTS (HuggingFace: KittenML/kitten-tts-*).
///
/// Model files:
///   - model.onnx    — single-pass TTS model (phoneme IDs + voice embedding → waveform)
///   - voices.npz    — NumPy compressed archive of voice embeddings (8 voices)
///
/// Available voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
///
/// Available models (HuggingFace):
///   - KittenML/kitten-tts-mini-0.8  (80M params, ~80 MB)
///   - KittenML/kitten-tts-micro-0.8 (40M params, ~41 MB)
///   - KittenML/kitten-tts-nano-0.8  (15M params, ~56 MB)
/// </summary>
public class OnnxKittenTtsOptions
{
    /// <summary>Path to the KittenTTS ONNX model (model.onnx).</summary>
    public required string ModelPath { get; set; }

    /// <summary>Path to the voices NPZ file (voices.npz). If null, looks in the model directory.</summary>
    public string? VoicesPath { get; set; }

    /// <summary>
    /// Path to espeak-ng executable. If null, auto-detects from PATH.
    /// Required for phonemization (text → IPA phonemes).
    /// </summary>
    public string? EspeakPath { get; set; }

    /// <summary>Default voice name. Default: "Jasper".</summary>
    public string DefaultVoice { get; set; } = "Jasper";

    /// <summary>Default speech speed multiplier. Default: 1.0.</summary>
    public float DefaultSpeed { get; set; } = 1.0f;

    /// <summary>Output audio sample rate in Hz. Default: 24000 (KittenTTS standard).</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Maximum characters per text chunk for synthesis. Default: 400.</summary>
    [Obsolete("Use MaxTokensPerChunk with a Tokenizer for token-aware chunking.")]
    public int MaxChunkLength { get; set; } = 400;

    /// <summary>Optional tokenizer for IPA symbol encoding. Defaults to KittenTtsTokenizer when not specified.</summary>
    public Tokenizer? Tokenizer { get; set; }

    /// <summary>
    /// Maximum tokens per chunk when using token-aware chunking. Used when a Tokenizer is available.
    /// When null, falls back to MaxChunkLength for backward compatibility.
    /// </summary>
    public int? MaxTokensPerChunk { get; set; }

    /// <summary>Number of samples to trim from end of generated audio. Default: 5000.</summary>
    public int TrimEndSamples { get; set; } = 5000;

    /// <summary>Name of the input column containing text (string). Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output column for audio samples (float[]). Default: "Audio".</summary>
    public string OutputColumnName { get; set; } = "Audio";
}
