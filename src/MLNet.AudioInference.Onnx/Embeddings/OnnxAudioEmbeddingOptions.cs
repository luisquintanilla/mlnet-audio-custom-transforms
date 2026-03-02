using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for the ONNX audio embedding transform.
/// </summary>
public class OnnxAudioEmbeddingOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Audio feature extractor to use for preprocessing.</summary>
    public required AudioFeatureExtractor FeatureExtractor { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for the embedding vector. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>Pooling strategy for aggregating model output. Default: MeanPooling.</summary>
    public AudioPoolingStrategy Pooling { get; set; } = AudioPoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embedding. Default: true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>ONNX input tensor name. Null = auto-detect.</summary>
    public string? InputTensorName { get; set; }

    /// <summary>ONNX output tensor name. Null = auto-detect.</summary>
    public string? OutputTensorName { get; set; }

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>GPU device ID. Null = CPU.</summary>
    public int? GpuDeviceId { get; set; }
}

/// <summary>
/// Pooling strategy for audio embeddings.
/// </summary>
public enum AudioPoolingStrategy
{
    MeanPooling,
    ClsToken,
    MaxPooling
}
