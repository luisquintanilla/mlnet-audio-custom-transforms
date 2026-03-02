using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Options for the audio feature extraction sub-transform.
/// </summary>
public class AudioFeatureExtractionOptions
{
    /// <summary>Feature extractor to use (MelSpectrogramExtractor, WhisperFeatureExtractor, etc.).</summary>
    public required AudioFeatureExtractor FeatureExtractor { get; set; }

    /// <summary>Name of the input column containing audio samples (VBuffer&lt;float&gt;).</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column for extracted features (VBuffer&lt;float&gt;, flattened [frames × bins]).</summary>
    public string OutputColumnName { get; set; } = "Features";

    /// <summary>Sample rate of input audio.</summary>
    public int SampleRate { get; set; } = 16000;
}

/// <summary>
/// Estimator that creates an AudioFeatureExtractionTransformer.
/// Stage 1 of the composed audio pipeline: AudioData → mel spectrogram features.
/// </summary>
public sealed class AudioFeatureExtractionEstimator : IEstimator<AudioFeatureExtractionTransformer>
{
    private readonly IHostEnvironment _env;
    private readonly AudioFeatureExtractionOptions _options;

    public AudioFeatureExtractionEstimator(IHostEnvironment env, AudioFeatureExtractionOptions options)
    {
        _env = env;
        _options = options;
    }

    public AudioFeatureExtractionTransformer Fit(IDataView input)
    {
        return new AudioFeatureExtractionTransformer(_env, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        // Add features output column
        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.OutputColumnName, SchemaShape.Column.VectorKind.VariableVector,
             NumberDataViewType.Single, false, null]);

        return new SchemaShape(columns.Values);
    }
}
