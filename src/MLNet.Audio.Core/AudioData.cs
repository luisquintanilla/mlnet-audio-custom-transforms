namespace MLNet.Audio.Core;

/// <summary>
/// Represents a segment of audio data as PCM samples.
/// This is the fundamental audio type used throughout the audio transforms pipeline.
/// </summary>
public class AudioData
{
    /// <summary>
    /// PCM audio samples (mono, normalized to [-1.0, 1.0]).
    /// </summary>
    public float[] Samples { get; }

    /// <summary>
    /// Sample rate in Hz (e.g., 16000 for most speech models).
    /// </summary>
    public int SampleRate { get; }

    /// <summary>
    /// Number of audio channels. Always 1 (mono) after loading — stereo is mixed down.
    /// </summary>
    public int Channels { get; }

    /// <summary>
    /// Duration of the audio.
    /// </summary>
    public TimeSpan Duration => TimeSpan.FromSeconds((double)Samples.Length / SampleRate);

    public AudioData(float[] samples, int sampleRate, int channels = 1)
    {
        ArgumentNullException.ThrowIfNull(samples);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);

        Samples = samples;
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <summary>
    /// Creates an empty AudioData with the specified sample rate.
    /// </summary>
    public static AudioData Empty(int sampleRate = 16000) => new([], sampleRate);
}
