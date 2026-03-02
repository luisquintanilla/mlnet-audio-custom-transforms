namespace MLNet.Audio.Core;

/// <summary>
/// Feature extractor matching the HuggingFace WhisperFeatureExtractor behavior.
/// Produces log-mel spectrogram features with 80 or 128 mel bins, padded/truncated
/// to exactly 30 seconds (3000 frames at 10ms hop).
/// </summary>
public class WhisperFeatureExtractor : MelSpectrogramExtractor
{
    /// <summary>
    /// Maximum number of frames (30 seconds at 10ms hop = 3000 frames).
    /// </summary>
    public int MaxFrames { get; init; } = 3000;

    /// <summary>
    /// Whether to pad shorter audio to MaxFrames with zeros. Default true.
    /// </summary>
    public bool PadToMaxFrames { get; init; } = true;

    public WhisperFeatureExtractor(int numMelBins = 80)
        : base(sampleRate: 16000)
    {
        NumMelBins = numMelBins;
        FftSize = 400;      // 25ms window at 16kHz
        HopLength = 160;    // 10ms hop at 16kHz
        LogScale = true;
        LowFrequency = 0f;
    }

    /// <summary>
    /// Extracts features from AudioData, producing a [frames x mel_bins] array
    /// padded/truncated to MaxFrames.
    /// </summary>
    public new float[,] Extract(AudioData audio)
    {
        var raw = base.Extract(audio);
        int rawFrames = raw.GetLength(0);
        int melBins = raw.GetLength(1);

        int targetFrames = PadToMaxFrames ? MaxFrames : Math.Min(rawFrames, MaxFrames);
        var result = new float[targetFrames, melBins];

        int framesToCopy = Math.Min(rawFrames, targetFrames);
        for (int i = 0; i < framesToCopy; i++)
            for (int j = 0; j < melBins; j++)
                result[i, j] = raw[i, j];

        // Remaining frames are already zero (padded) if PadToMaxFrames is true

        return result;
    }

    /// <summary>
    /// Splits long audio into 30-second chunks with overlap for processing.
    /// Returns a list of feature arrays, one per chunk.
    /// </summary>
    public List<float[,]> ExtractChunked(AudioData audio, int overlapFrames = 0)
    {
        var chunks = new List<float[,]>();
        int samplesPerChunk = MaxFrames * HopLength;  // 3000 * 160 = 480000 samples = 30s
        int overlapSamples = overlapFrames * HopLength;
        int stepSamples = samplesPerChunk - overlapSamples;

        var samples = audio.Samples;
        int offset = 0;

        while (offset < samples.Length)
        {
            int remaining = samples.Length - offset;
            int chunkLength = Math.Min(samplesPerChunk, remaining);
            var chunkSamples = new float[chunkLength];
            Array.Copy(samples, offset, chunkSamples, 0, chunkLength);

            var chunkAudio = new AudioData(chunkSamples, audio.SampleRate, audio.Channels);
            chunks.Add(Extract(chunkAudio));

            offset += stepSamples;
            if (chunkLength < samplesPerChunk)
                break; // Last partial chunk
        }

        return chunks;
    }
}
