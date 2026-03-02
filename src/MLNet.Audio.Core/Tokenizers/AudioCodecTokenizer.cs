namespace MLNet.Audio.Core;

/// <summary>
/// Abstract base class for audio codec tokenizers — the audio equivalent of text tokenizers.
///
/// Text tokenizers convert strings → discrete token IDs.
/// Audio codec tokenizers convert audio waveforms → discrete codec codes via neural codecs
/// (EnCodec, DAC, SpeechTokenizer, Mimi).
///
/// This mirrors Microsoft.ML.Tokenizers.Tokenizer conceptually but operates on audio:
///   Text:  string → Tokenizer.Encode() → int[]     → Tokenizer.Decode() → string
///   Audio: float[] → AudioCodecTokenizer.Encode() → int[][] → AudioCodecTokenizer.Decode() → float[]
///
/// The multi-codebook output (int[][]) comes from Residual Vector Quantization (RVQ),
/// where each codebook captures progressively finer audio detail.
///
/// DESIGN NOTE: This class demonstrates that Microsoft.ML.Tokenizers.Tokenizer should be
/// generic/modality-agnostic. The current Tokenizer base class assumes string input.
/// A future design could be:
///   Tokenizer&lt;TInput, TToken&gt;
///     ├── TextTokenizer : Tokenizer&lt;string, EncodedToken&gt;  (what exists today)
///     └── AudioCodecTokenizer : Tokenizer&lt;AudioData, AudioCodeToken&gt;  (what this prototypes)
/// </summary>
public abstract class AudioCodecTokenizer : IDisposable
{
    /// <summary>
    /// Number of codebooks used by this codec (e.g., 8 for EnCodec at 6kbps).
    /// More codebooks = higher quality = higher bitrate.
    /// </summary>
    public abstract int NumCodebooks { get; }

    /// <summary>
    /// Size of each codebook (number of unique codes). Typically 1024.
    /// </summary>
    public abstract int CodebookSize { get; }

    /// <summary>
    /// Sample rate expected by the codec.
    /// </summary>
    public abstract int SampleRate { get; }

    /// <summary>
    /// Number of audio samples per codec frame (stride).
    /// E.g., EnCodec at 24kHz with 75Hz frame rate = 320 samples per frame.
    /// </summary>
    public abstract int HopLength { get; }

    /// <summary>
    /// Encode audio waveform to discrete codec codes.
    /// Returns codes[codebook][frame] — each codebook produces a stream of integer codes.
    /// </summary>
    /// <param name="audio">Input audio (will be resampled to SampleRate if needed).</param>
    /// <returns>
    /// 2D array where codes[c][f] is the code from codebook c at frame f.
    /// Shape: [NumCodebooks, numFrames].
    /// </returns>
    public abstract int[][] Encode(AudioData audio);

    /// <summary>
    /// Decode discrete codec codes back to audio waveform.
    /// </summary>
    /// <param name="codes">Codec codes with shape [NumCodebooks, numFrames].</param>
    /// <returns>Reconstructed audio.</returns>
    public abstract AudioData Decode(int[][] codes);

    /// <summary>
    /// Convenience: encode and return as a flat sequence of tokens for language model consumption.
    /// Interleaves codebook codes: [cb0_f0, cb1_f0, ..., cbN_f0, cb0_f1, cb1_f1, ...].
    /// Offset each codebook by codebook_index * CodebookSize for unique token IDs.
    /// </summary>
    public virtual int[] EncodeFlat(AudioData audio)
    {
        var codes = Encode(audio);
        int numFrames = codes[0].Length;
        var flat = new int[NumCodebooks * numFrames];

        for (int f = 0; f < numFrames; f++)
        {
            for (int c = 0; c < NumCodebooks; c++)
            {
                // Offset each codebook so tokens are unique across codebooks
                flat[f * NumCodebooks + c] = codes[c][f] + (c * CodebookSize);
            }
        }

        return flat;
    }

    /// <summary>
    /// Total vocabulary size across all codebooks.
    /// </summary>
    public int VocabularySize => NumCodebooks * CodebookSize;

    /// <summary>
    /// Compute the number of codec frames for a given audio duration.
    /// </summary>
    public int GetFrameCount(TimeSpan duration)
    {
        int totalSamples = (int)(duration.TotalSeconds * SampleRate);
        return totalSamples / HopLength;
    }

    public virtual void Dispose() { }
}

/// <summary>
/// A single audio codec token — the audio equivalent of EncodedToken for text.
/// </summary>
/// <param name="CodebookValues">Code value from each codebook at this time step.</param>
/// <param name="Timestamp">Time position of this frame in the original audio.</param>
/// <param name="FrameIndex">Frame index in the codec output sequence.</param>
public record AudioCodeToken(int[] CodebookValues, TimeSpan Timestamp, int FrameIndex);
