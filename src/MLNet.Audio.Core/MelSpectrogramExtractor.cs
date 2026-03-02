using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Signals;
using NWaves.Windows;
using System.Numerics.Tensors;

namespace MLNet.Audio.Core;

/// <summary>
/// Extracts log-mel spectrogram features from audio.
/// Uses NWaves for FFT and mel filter bank computation.
/// This is the primary feature extractor for models like Whisper, AST, and CLAP.
/// </summary>
public class MelSpectrogramExtractor : AudioFeatureExtractor
{
    private readonly int _sampleRate;

    /// <summary>
    /// Number of mel frequency bins (e.g., 80 for Whisper, 128 for AST).
    /// </summary>
    public int NumMelBins { get; init; } = 80;

    /// <summary>
    /// FFT window size in samples. Default 400 (25ms at 16kHz).
    /// </summary>
    public int FftSize { get; init; } = 400;

    /// <summary>
    /// Hop length between frames in samples. Default 160 (10ms at 16kHz).
    /// </summary>
    public int HopLength { get; init; } = 160;

    /// <summary>
    /// Lower edge of the mel filter bank in Hz.
    /// </summary>
    public float LowFrequency { get; init; } = 0f;

    /// <summary>
    /// Upper edge of the mel filter bank in Hz. Defaults to SampleRate / 2.
    /// </summary>
    public float? HighFrequency { get; init; }

    /// <summary>
    /// Whether to apply log scaling to the mel spectrogram. Default true.
    /// </summary>
    public bool LogScale { get; init; } = true;

    /// <summary>
    /// Window function to apply. Default is Hann.
    /// </summary>
    public WindowType Window { get; init; } = WindowType.Hann;

    public override int SampleRate => _sampleRate;

    public MelSpectrogramExtractor(int sampleRate = 16000)
    {
        _sampleRate = sampleRate;
    }

    protected override float[,] ExtractFeatures(float[] samples)
    {
        if (samples.Length == 0)
            return new float[0, NumMelBins];

        var signal = new DiscreteSignal(_sampleRate, samples);

        var options = new FilterbankOptions
        {
            SamplingRate = _sampleRate,
            FrameSize = FftSize,
            HopSize = HopLength,
            FilterBankSize = NumMelBins,
            LowFrequency = LowFrequency,
            HighFrequency = HighFrequency ?? _sampleRate / 2.0,
            Window = Window,
            LogFloor = 1e-10f
        };

        var extractor = new FilterbankExtractor(options);
        var featureVectors = extractor.ComputeFrom(signal);

        int numFrames = featureVectors.Count;
        var result = new float[numFrames, NumMelBins];

        for (int i = 0; i < numFrames; i++)
        {
            var frame = featureVectors[i];

            if (LogScale)
            {
                // SIMD: clamp to floor then log
                Span<float> row = stackalloc float[NumMelBins];
                frame.AsSpan(0, NumMelBins).CopyTo(row);
                TensorPrimitives.Max(row, 1e-10f, row);
                TensorPrimitives.Log(row, row);
                for (int j = 0; j < NumMelBins; j++)
                    result[i, j] = row[j];
            }
            else
            {
                for (int j = 0; j < NumMelBins; j++)
                    result[i, j] = frame[j];
            }
        }

        return result;
    }
}
