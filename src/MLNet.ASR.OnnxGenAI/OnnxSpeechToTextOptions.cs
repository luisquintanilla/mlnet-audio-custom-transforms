using MLNet.Audio.Core;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// Configuration for local Whisper speech-to-text via ORT GenAI.
/// </summary>
public class OnnxSpeechToTextOptions
{
    /// <summary>
    /// Path to the ORT GenAI Whisper model directory.
    /// Must contain genai_config.json, encoder/decoder ONNX files, and tokenizer files.
    /// </summary>
    public required string ModelPath { get; set; }

    /// <summary>
    /// Audio feature extractor for preprocessing. Default: WhisperFeatureExtractor (80 mel bins).
    /// If null, attempts to use ORT GenAI's built-in processor.
    /// </summary>
    public AudioFeatureExtractor? FeatureExtractor { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for the transcribed text. Default: "Text".</summary>
    public string OutputColumnName { get; set; } = "Text";

    /// <summary>Maximum number of tokens to generate. Default: 256.</summary>
    public int MaxLength { get; set; } = 256;

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Speech language code (e.g., "en", "es", "fr"). Null = auto-detect.</summary>
    public string? Language { get; set; }

    /// <summary>Whether to translate to English. Default: false (transcribe in source language).</summary>
    public bool Translate { get; set; } = false;

    /// <summary>
    /// Whether the model is multilingual. Default: true.
    /// English-only models (whisper-tiny.en, whisper-base.en) should set this to false.
    /// </summary>
    public bool IsMultilingual { get; set; } = true;

    /// <summary>Number of mel bins for feature extraction. Default: 80 (Whisper v1-v2). Use 128 for v3.</summary>
    public int NumMelBins { get; set; } = 80;
}
