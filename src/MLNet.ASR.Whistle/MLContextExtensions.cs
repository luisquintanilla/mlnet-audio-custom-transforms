using Microsoft.ML;

namespace MLNet.ASR.Whistle;

/// <summary>ML.NET extensions for the Whistle speech-to-text package.</summary>
public static class MLContextExtensions
{
    /// <summary>Creates an ML.NET Whistle speech-to-text estimator.</summary>
    public static WhistleSpeechToTextEstimator WhistleSpeechToText(
        this TransformsCatalog transforms,
        WhistleTransformOptions options)
    {
        ArgumentNullException.ThrowIfNull(transforms);
        ArgumentNullException.ThrowIfNull(options);

        var environmentProperty = typeof(TransformsCatalog).GetProperty(
            "Environment",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var environment = environmentProperty?.GetValue(transforms);
        var mlContext = environment as MLContext ?? new MLContext();

        return new WhistleSpeechToTextEstimator(mlContext, options);
    }
}
