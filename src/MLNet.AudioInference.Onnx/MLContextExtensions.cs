using Microsoft.Extensions.AI;
using Microsoft.ML;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Extension methods for MLContext to provide audio transform entry points.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Creates an audio classification estimator using an ONNX model.
    /// </summary>
    public static OnnxAudioClassificationEstimator OnnxAudioClassification(
        this TransformsCatalog catalog,
        OnnxAudioClassificationOptions options)
    {
        var mlContext = GetMLContext(catalog);
        return new OnnxAudioClassificationEstimator(mlContext, options);
    }

    /// <summary>
    /// Creates an audio embedding estimator using an ONNX model.
    /// </summary>
    public static OnnxAudioEmbeddingEstimator OnnxAudioEmbedding(
        this TransformsCatalog catalog,
        OnnxAudioEmbeddingOptions options)
    {
        var mlContext = GetMLContext(catalog);
        return new OnnxAudioEmbeddingEstimator(mlContext, options);
    }

    /// <summary>
    /// Creates a speech-to-text estimator wrapping any ISpeechToTextClient provider.
    /// Provider-agnostic — works with Azure, OpenAI, local Whisper, etc.
    /// </summary>
    public static SpeechToTextClientEstimator SpeechToText(
        this TransformsCatalog catalog,
        ISpeechToTextClient client,
        SpeechToTextClientOptions? options = null)
    {
        var mlContext = GetMLContext(catalog);
        return new SpeechToTextClientEstimator(mlContext, client, options);
    }

    /// <summary>
    /// Creates a Voice Activity Detection estimator using an ONNX model (e.g., Silero VAD).
    /// </summary>
    public static OnnxVadEstimator OnnxVad(
        this TransformsCatalog catalog,
        OnnxVadOptions options)
    {
        var mlContext = GetMLContext(catalog);
        return new OnnxVadEstimator(mlContext, options);
    }

    /// <summary>
    /// Creates a raw ONNX Whisper speech-to-text estimator.
    /// Uses standard HuggingFace optimum-exported ONNX models (no ORT GenAI needed).
    /// Full control over the encoder-decoder pipeline with KV cache management.
    /// </summary>
    public static OnnxWhisperEstimator OnnxWhisper(
        this TransformsCatalog catalog,
        OnnxWhisperOptions options)
    {
        var mlContext = GetMLContext(catalog);
        return new OnnxWhisperEstimator(mlContext, options);
    }

    /// <summary>
    /// Creates a raw ONNX Whisper speech-to-text estimator wrapped as an ISpeechToTextClient.
    /// Combines the full-control raw ONNX pipeline with MEAI ecosystem support
    /// (DI, middleware, logging, telemetry via SpeechToTextClientBuilder).
    /// </summary>
    public static SpeechToTextClientEstimator OnnxWhisperSpeechToText(
        this TransformsCatalog catalog,
        OnnxWhisperOptions options)
    {
        var mlContext = GetMLContext(catalog);
        ISpeechToTextClient client = new OnnxWhisperSpeechToTextClient(options);
        return new SpeechToTextClientEstimator(mlContext, client);
    }

    /// <summary>
    /// Creates a SpeechT5 text-to-speech estimator using raw ONNX Runtime.
    /// Uses HuggingFace optimum-exported ONNX models (NeuML/txtai-speecht5-onnx format).
    /// Full encoder-decoder-vocoder pipeline: text → mel spectrogram → waveform.
    /// </summary>
    public static OnnxSpeechT5TtsEstimator SpeechT5Tts(
        this TransformsCatalog catalog,
        OnnxSpeechT5Options options)
    {
        var mlContext = GetMLContext(catalog);
        return new OnnxSpeechT5TtsEstimator(mlContext, options);
    }

    private static MLContext GetMLContext(TransformsCatalog catalog)
    {
        // Use reflection to get MLContext from TransformsCatalog
        // (same approach as mlnet-text-inference-custom-transforms)
        var envProp = typeof(TransformsCatalog).GetProperty("Environment",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var env = envProp?.GetValue(catalog);

        if (env is MLContext mlContext)
            return mlContext;

        // Fallback: create a new MLContext
        return new MLContext();
    }
}
