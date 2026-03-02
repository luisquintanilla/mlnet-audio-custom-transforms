namespace MLNet.Audio.Core;

/// <summary>
/// Base class for audio feature extractors. Converts raw PCM audio into
/// feature tensors suitable for ONNX model input.
/// This is the audio equivalent of Microsoft.ML.Tokenizers.Tokenizer for text.
/// </summary>
public abstract class AudioFeatureExtractor
{
    /// <summary>
    /// Expected sample rate for input audio. Audio will be resampled if needed.
    /// </summary>
    public abstract int SampleRate { get; }

    /// <summary>
    /// Extracts features from audio data, returning a 2D float array [frames x features].
    /// </summary>
    public float[,] Extract(AudioData audio)
    {
        var resampled = audio.SampleRate != SampleRate
            ? AudioIO.Resample(audio, SampleRate)
            : audio;

        var mono = resampled.Channels > 1
            ? AudioIO.ToMono(resampled)
            : resampled;

        return ExtractFeatures(mono.Samples);
    }

    /// <summary>
    /// Core feature extraction on mono PCM samples at the correct sample rate.
    /// Returns [frames x features] float array.
    /// </summary>
    protected abstract float[,] ExtractFeatures(float[] samples);
}
