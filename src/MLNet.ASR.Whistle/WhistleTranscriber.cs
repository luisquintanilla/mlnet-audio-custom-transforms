using System.Text.Json;
using System.Text.Json.Serialization;
using MLNet.ASR.Whistle.Native;
using MLNet.Audio.Core;

namespace MLNet.ASR.Whistle;

/// <summary>
/// Direct Whistle transcription API backed by the Cactus Needle native runtime.
/// </summary>
public sealed class WhistleTranscriber : IDisposable
{
    private static readonly JsonSerializerOptions NativeJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private readonly ValidatedWhistleOptions _options;
    private readonly Func<string?, IWhistleNativeApi> _nativeFactory;
    private bool _disposed;

    /// <summary>Creates a transcriber using the production native C ABI.</summary>
    public WhistleTranscriber(WhistleOptions options)
        : this(options, CactusWhistleNativeApi.Create)
    {
    }

    internal WhistleTranscriber(
        WhistleOptions options,
        Func<string?, IWhistleNativeApi> nativeFactory)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(nativeFactory);
        _options = options.Validate();
        _nativeFactory = nativeFactory;
    }

    /// <summary>Transcribes one audio clip after mono/16 kHz normalization.</summary>
    public WhistleTranscriptionResult Transcribe(
        AudioData audio,
        WhistleTranscriptionOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(audio);
        options ??= new WhistleTranscriptionOptions();

        var language = WhistleLanguages.Normalize(options.Language ?? _options.Language);
        var keywords = options.Keywords is null
            ? _options.Keywords
            : options.Keywords
                .Where(k => !string.IsNullOrWhiteSpace(k))
                .Select(k => k.Trim())
                .ToArray();
        var wordTimestamps = options.WordTimestamps ?? _options.WordTimestamps;

        var normalized = NormalizeAudio(audio);
        var keywordText = keywords.Length == 0 ? null : string.Join('\n', keywords);

        using var lease = WhistleRuntime.Acquire(_options, _nativeFactory);
        var nativeResult = lease.Api.Transcribe(
            normalized.Samples,
            language,
            keywordText,
            wordTimestamps,
            _options.OutputBufferSize);

        return ParseResult(nativeResult.Json);
    }

    /// <summary>Transcribes a collection of clips in order.</summary>
    public WhistleTranscriptionResult[] Transcribe(
        IReadOnlyList<AudioData> audio,
        WhistleTranscriptionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var results = new WhistleTranscriptionResult[audio.Count];
        for (var i = 0; i < audio.Count; i++)
            results[i] = Transcribe(audio[i], options);

        return results;
    }

    /// <summary>Transcribes a WAV stream without taking ownership of the stream.</summary>
    public WhistleTranscriptionResult Transcribe(
        Stream audioStream,
        WhistleTranscriptionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(audioStream);

        using var bufferedAudio = new MemoryStream();
        audioStream.CopyTo(bufferedAudio);
        bufferedAudio.Position = 0;
        return Transcribe(AudioIO.LoadWav(bufferedAudio), options);
    }

    /// <summary>Transcribes a WAV file from disk.</summary>
    public WhistleTranscriptionResult TranscribeFile(
        string audioPath,
        WhistleTranscriptionOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(audioPath);

        using var audioStream = File.OpenRead(audioPath);
        return Transcribe(audioStream, options);
    }

    /// <summary>Marks this transcriber as unavailable for further requests.</summary>
    public void Dispose() => _disposed = true;

    private AudioData NormalizeAudio(AudioData audio)
    {
        if (audio.Samples.Length == 0)
            throw new ArgumentException("Whistle requires at least one audio sample.", nameof(audio));

        var normalized = audio.Channels == 1 ? audio : AudioIO.ToMono(audio);
        if (normalized.SampleRate != _options.SampleRate)
            normalized = AudioIO.Resample(normalized, _options.SampleRate);

        if (normalized.Samples.Length == 0)
            throw new ArgumentException("Whistle requires at least one audio sample.", nameof(audio));

        float[]? clampedSamples = null;
        for (var i = 0; i < normalized.Samples.Length; i++)
        {
            var sample = normalized.Samples[i];
            if (!float.IsFinite(sample))
                throw new ArgumentException("Whistle audio samples must be finite.", nameof(audio));

            if (sample is < -1f or > 1f)
            {
                clampedSamples ??= normalized.Samples.ToArray();
                clampedSamples[i] = Math.Clamp(sample, -1f, 1f);
            }
        }

        if (clampedSamples is not null)
            normalized = new AudioData(clampedSamples, normalized.SampleRate, channels: 1);

        if (normalized.Duration > _options.MaxDuration)
            throw new ArgumentOutOfRangeException(
                nameof(audio),
                $"Whistle accepts clips up to {_options.MaxDuration.TotalSeconds:0.#} seconds.");

        return normalized;
    }

    private static WhistleTranscriptionResult ParseResult(string json)
    {
        try
        {
            var native = JsonSerializer.Deserialize<NativeResult>(json, NativeJsonOptions)
                ?? throw new InvalidDataException("Whistle returned an empty transcription result.");

            if (native.Text is null ||
                native.Language is null ||
                native.TtftMs is null ||
                native.DecodeTps is null)
            {
                throw new InvalidDataException(
                    "Whistle returned a transcription result missing required fields.");
            }

            return new WhistleTranscriptionResult
            {
                Text = native.Text,
                Language = string.IsNullOrWhiteSpace(native.Language) ? null : native.Language,
                TimeToFirstToken = TimeSpan.FromMilliseconds(native.TtftMs.Value),
                DecodeTokensPerSecond = native.DecodeTps.Value,
                Words = native.Words?
                    .Select(word => new WhistleWord
                    {
                        Word = word.Word ?? string.Empty,
                        Start = TimeSpan.FromSeconds(word.Start),
                        End = TimeSpan.FromSeconds(word.End),
                        Probability = word.Probability
                    })
                    .ToArray() ?? []
            };
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException("Whistle returned malformed transcription JSON.", ex);
        }
    }

    private sealed class NativeResult
    {
        public string? Text { get; set; }
        public string? Language { get; set; }

        [JsonPropertyName("ttft_ms")]
        public double? TtftMs { get; set; }

        [JsonPropertyName("decode_tps")]
        public double? DecodeTps { get; set; }

        public NativeWord[]? Words { get; set; }
    }

    private sealed class NativeWord
    {
        public string? Word { get; set; }
        public double Start { get; set; }
        public double End { get; set; }
        public double Probability { get; set; }
    }
}
