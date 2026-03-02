namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for SpeechT5 text-to-speech using raw ONNX Runtime.
/// Uses HuggingFace optimum-exported ONNX models (NeuML/txtai-speecht5-onnx format).
///
/// Model files:
///   - encoder_model.onnx            — text → encoder hidden states
///   - decoder_model_merged.onnx     — autoregressive mel spectrogram generation (with KV cache)
///   - decoder_postnet_and_vocoder.onnx — mel → waveform (postnet + HiFi-GAN)
///   - spm_char.model                — SentencePiece tokenizer (character-level)
///   - speaker.npy (optional)        — default speaker x-vector embedding
/// </summary>
public class OnnxSpeechT5Options
{
    /// <summary>Path to encoder_model.onnx.</summary>
    public required string EncoderModelPath { get; set; }

    /// <summary>Path to decoder_model_merged.onnx (handles both prefill and decode-with-past).</summary>
    public required string DecoderModelPath { get; set; }

    /// <summary>Path to decoder_postnet_and_vocoder.onnx (postnet + HiFi-GAN vocoder).</summary>
    public required string VocoderModelPath { get; set; }

    /// <summary>
    /// Path to the SentencePiece tokenizer model (spm_char.model).
    /// If null, attempts to find it in the same directory as the encoder model.
    /// </summary>
    public string? TokenizerModelPath { get; set; }

    /// <summary>
    /// Path to a speaker embedding file (.npy format, float32 x-vector).
    /// If null, attempts to find speaker.npy in the model directory.
    /// The embedding is typically 512-dim from an x-vector or ECAPA-TDNN extractor.
    /// </summary>
    public string? SpeakerEmbeddingPath { get; set; }

    /// <summary>Name of the input column containing text (string). Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output column for audio samples (float[]). Default: "Audio".</summary>
    public string OutputColumnName { get; set; } = "Audio";

    /// <summary>Maximum number of mel spectrogram frames to generate. Default: 1000 (~15s at 16kHz).</summary>
    public int MaxMelFrames { get; set; } = 1000;

    /// <summary>
    /// Threshold for the stop token probability. When the decoder's stop head
    /// predicts above this value, generation stops. Default: 0.5.
    /// </summary>
    public float StopThreshold { get; set; } = 0.5f;

    /// <summary>Number of mel spectrogram bins. Default: 80 (SpeechT5 standard).</summary>
    public int NumMelBins { get; set; } = 80;

    /// <summary>Output audio sample rate in Hz. Default: 16000 (SpeechT5 standard).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Number of decoder layers. Auto-detected from model if 0.</summary>
    public int NumDecoderLayers { get; set; } = 0;

    /// <summary>Number of attention heads. Auto-detected from model if 0.</summary>
    public int NumAttentionHeads { get; set; } = 0;
}
