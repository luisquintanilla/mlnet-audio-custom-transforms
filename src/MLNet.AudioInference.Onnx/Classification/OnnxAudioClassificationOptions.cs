using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Configuration for the ONNX audio classification transform.
/// </summary>
public class OnnxAudioClassificationOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Audio feature extractor to use for preprocessing.</summary>
    public required AudioFeatureExtractor FeatureExtractor { get; set; }

    /// <summary>Class labels for the classification output.</summary>
    public required string[] Labels { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for the predicted label. Default: "PredictedLabel".</summary>
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";

    /// <summary>Name of the output column for class probabilities. Default: "Probabilities".</summary>
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";

    /// <summary>Name of the output column for the confidence score. Default: "Score".</summary>
    public string ScoreColumnName { get; set; } = "Score";

    /// <summary>ONNX input tensor name. Null = auto-detect.</summary>
    public string? InputTensorName { get; set; }

    /// <summary>ONNX output tensor name. Null = auto-detect.</summary>
    public string? OutputTensorName { get; set; }

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>GPU device ID. Null = CPU.</summary>
    public int? GpuDeviceId { get; set; }
}
