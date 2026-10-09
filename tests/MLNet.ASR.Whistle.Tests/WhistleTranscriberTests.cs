using Microsoft.Extensions.AI;
using MLNet.ASR.Whistle.Native;
using MLNet.Audio.Core;
using Xunit;

namespace MLNet.ASR.Whistle.Tests;

[Collection("Whistle runtime")]
public sealed class WhistleTranscriberTests : IDisposable
{
    private readonly string _modelPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.cact");

    public WhistleTranscriberTests()
    {
        File.WriteAllBytes(_modelPath, [1, 2, 3]);
    }

    [Fact]
    public void Transcribe_normalizes_audio_and_parses_native_json()
    {
        var api = new FakeNativeApi();
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath, WordTimestamps = true },
            _ => api);

        var result = transcriber.Transcribe(
            new AudioData([1f, 0f, 0f, 1f], sampleRate: 8_000, channels: 2));

        Assert.Equal("hello world", result.Text);
        Assert.Equal("en", result.Language);
        Assert.Equal(TimeSpan.FromMilliseconds(12.5), result.TimeToFirstToken);
        Assert.Equal(4.5, result.DecodeTokensPerSecond);
        Assert.Single(result.Words);
        Assert.Equal("hello", result.Words[0].Word);
        Assert.Equal(TimeSpan.FromMilliseconds(100), result.Words[0].Start);
        Assert.Equal(0.98, result.Words[0].Probability);
        Assert.Equal(4, api.Samples.Length);
        Assert.All(api.Samples, sample => Assert.Equal(0.5f, sample));
    }

    [Fact]
    public void Options_reject_unsupported_language()
    {
        Assert.Throws<ArgumentException>(() =>
            new WhistleTranscriber(
                new WhistleOptions { ModelPath = _modelPath, Language = "ja" },
                _ => new FakeNativeApi()));
    }

    [Fact]
    public void Transcribe_rejects_empty_audio()
    {
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => new FakeNativeApi());

        Assert.Throws<ArgumentException>(() => transcriber.Transcribe(AudioData.Empty()));
    }

    [Fact]
    public void Transcribe_rejects_malformed_native_json()
    {
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => new FakeNativeApi { Json = "{not-json" });

        Assert.Throws<InvalidDataException>(() =>
            transcriber.Transcribe(new AudioData([0.1f], 16_000)));
    }

    [Fact]
    public void Transcribe_rejects_native_json_missing_required_fields()
    {
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => new FakeNativeApi { Json = """{"text":"hello"}""" });

        Assert.Throws<InvalidDataException>(() =>
            transcriber.Transcribe(new AudioData([0.1f], 16_000)));
    }

    [Fact]
    public void Transcribe_maps_empty_native_language_to_unknown()
    {
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => new FakeNativeApi
            {
                Json = """{"text":"","language":"","ttft_ms":0,"decode_tps":0}"""
            });

        var result = transcriber.Transcribe(new AudioData([0.1f], 16_000));

        Assert.Null(result.Language);
    }

    [Fact]
    public void Transcribe_wav_stream_does_not_close_caller_stream()
    {
        using var wav = new MemoryStream();
        AudioIO.SaveWav(wav, new AudioData([0.25f], 16_000));
        wav.Position = 0;

        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => new FakeNativeApi());

        var result = transcriber.Transcribe(wav);

        Assert.Equal("hello world", result.Text);
        Assert.True(wav.CanRead);
    }

    [Fact]
    public async Task Speech_to_text_client_returns_metadata_and_stream_lifecycle()
    {
        using var client = new WhistleSpeechToTextClient(
            new WhistleOptions { ModelPath = _modelPath, WordTimestamps = true },
            _ => new FakeNativeApi());

        using var wav = new MemoryStream();
        AudioIO.SaveWav(wav, new AudioData([0.25f], 16_000));
        wav.Position = 0;

        var response = await client.GetTextAsync(wav);

        Assert.Equal("hello world", response.Text);
        Assert.Equal("en", response.AdditionalProperties!["language"]);
        Assert.Equal(TimeSpan.FromMilliseconds(100), response.StartTime);
        Assert.Equal(TimeSpan.FromMilliseconds(400), response.EndTime);
        Assert.True(wav.CanRead);

        wav.Position = 0;
        var updates = new List<SpeechToTextResponseUpdate>();
        await foreach (var update in client.GetStreamingTextAsync(wav))
            updates.Add(update);

        Assert.Equal(
            [
                SpeechToTextResponseUpdateKind.SessionOpen,
                SpeechToTextResponseUpdateKind.TextUpdated,
                SpeechToTextResponseUpdateKind.SessionClose
            ],
            updates.Select(update => update.Kind));
        Assert.Equal("hello world", updates[1].Text);
    }

    [Fact]
    public void Runtime_rejects_a_different_model_after_first_load()
    {
        var api = new FakeNativeApi();
        using var first = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => api);
        first.Transcribe(new AudioData([0.1f], 16_000));

        var secondModel = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.cact");
        File.WriteAllBytes(secondModel, [4, 5, 6]);
        try
        {
            using var second = new WhistleTranscriber(
                new WhistleOptions { ModelPath = secondModel },
                _ => new FakeNativeApi());

            Assert.Throws<InvalidOperationException>(() =>
                second.Transcribe(new AudioData([0.1f], 16_000)));
        }
        finally
        {
            File.Delete(secondModel);
        }
    }

    [Fact]
    public async Task Runtime_serializes_native_calls()
    {
        var api = new FakeNativeApi { Delay = true };
        using var transcriber = new WhistleTranscriber(
            new WhistleOptions { ModelPath = _modelPath },
            _ => api);

        await Task.WhenAll(
            Task.Run(() => transcriber.Transcribe(new AudioData([0.1f], 16_000))),
            Task.Run(() => transcriber.Transcribe(new AudioData([0.2f], 16_000))));

        Assert.Equal(1, api.MaximumConcurrentCalls);
    }

    public void Dispose()
    {
        WhistleRuntime.ResetForTests();
        File.Delete(_modelPath);
    }

    private sealed class FakeNativeApi : IWhistleNativeApi
    {
        public float[] Samples { get; private set; } = [];
        public bool Delay { get; init; }
        public int MaximumConcurrentCalls { get; private set; }
        public string Json { get; init; } =
            """{"text":"hello world","language":"en","ttft_ms":12.5,"decode_tps":4.5,"words":[{"word":"hello","start":0.1,"end":0.4,"probability":0.98}]}""";
        private int _activeCalls;

        public void Load(ReadOnlyMemory<byte> model)
        {
        }

        public NativeTranscriptionResult Transcribe(
            ReadOnlyMemory<float> samples,
            string? language,
            string? keywords,
            bool wordTimestamps,
            int outputBufferSize)
        {
            var activeCalls = Interlocked.Increment(ref _activeCalls);
            MaximumConcurrentCalls = Math.Max(MaximumConcurrentCalls, activeCalls);
            if (Delay)
                Thread.Sleep(20);

            Samples = samples.ToArray();
            Interlocked.Decrement(ref _activeCalls);
            return new NativeTranscriptionResult(2, Json);
        }

        public void Dispose()
        {
        }
    }
}

[CollectionDefinition("Whistle runtime", DisableParallelization = true)]
public sealed class WhistleRuntimeCollection : ICollectionFixture<object>
{
}
