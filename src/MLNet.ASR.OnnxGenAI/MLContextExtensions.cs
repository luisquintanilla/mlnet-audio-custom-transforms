using Microsoft.ML;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// MLContext extension methods for the ORT GenAI Whisper ASR package.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Create a local Whisper speech-to-text estimator using ORT GenAI.
    /// </summary>
    /// <example>
    /// <code>
    /// var estimator = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
    /// {
    ///     ModelPath = "models/whisper-base",
    ///     Language = "en"
    /// });
    /// </code>
    /// </example>
    public static OnnxSpeechToTextEstimator OnnxSpeechToText(
        this TransformsCatalog transforms,
        OnnxSpeechToTextOptions options)
    {
        var mlContext = (MLContext)typeof(TransformsCatalog)
            .GetProperty("Environment", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .GetValue(transforms)!;

        return new OnnxSpeechToTextEstimator(mlContext, options);
    }
}
