using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ISpeechToTextClient implementation wrapping OnnxWhisperTransformer (raw ONNX, no ORT GenAI).
/// Brings the raw ONNX Whisper path into the MEAI ecosystem — usable via DI, middleware
/// (logging, telemetry), and the SpeechToTextClientBuilder pipeline.
///
/// Contrast with:
///   - OnnxSpeechToTextClient (MLNet.ASR.OnnxGenAI): Whisper via ORT GenAI
///   - SpeechToTextClientTransformer: wraps any ISpeechToTextClient in ML.NET
/// </summary>
public sealed class OnnxWhisperSpeechToTextClient : ISpeechToTextClient
{
    private readonly OnnxWhisperTransformer _transformer;
    private readonly OnnxWhisperOptions _options;

    public OnnxWhisperSpeechToTextClient(OnnxWhisperOptions options)
    {
        _options = options;
        _transformer = new OnnxWhisperTransformer(new Microsoft.ML.MLContext(), options);
    }

    public SpeechToTextClientMetadata Metadata => new(
        providerName: "OnnxWhisper-RawOnnx",
        providerUri: null,
        defaultModelId: Path.GetFileName(Path.GetDirectoryName(_options.EncoderModelPath)));

    public Task<SpeechToTextResponse> GetTextAsync(
        Stream audioSpeechStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var audio = LoadAndResample(audioSpeechStream, options);
        var results = _transformer.TranscribeWithTimestamps([audio]);
        var result = results[0];

        var response = new SpeechToTextResponse(result.Text)
        {
            ModelId = options?.ModelId ?? Metadata.DefaultModelId,
        };

        // Populate timestamps from Whisper segments
        if (result.Segments.Length > 0)
        {
            response.StartTime = result.Segments[0].Start;
            response.EndTime = result.Segments[^1].End;

            response.AdditionalProperties = new AdditionalPropertiesDictionary
            {
                ["segments"] = result.Segments.Select(s => new
                {
                    start = s.Start.TotalSeconds,
                    end = s.End.TotalSeconds,
                    text = s.Text
                }).ToArray(),
                ["language"] = result.Language ?? _options.Language,
                ["tokenCount"] = result.TokenIds.Length,
            };
        }

        return Task.FromResult(response);
    }

    public IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(
        Stream audioSpeechStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return StreamingImpl(audioSpeechStream, options, cancellationToken);
    }

    private async IAsyncEnumerable<SpeechToTextResponseUpdate> StreamingImpl(
        Stream audioSpeechStream,
        SpeechToTextOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // Session open
        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionOpen
        };

        cancellationToken.ThrowIfCancellationRequested();

        var audio = LoadAndResample(audioSpeechStream, options);
        var results = _transformer.TranscribeWithTimestamps([audio]);
        var result = results[0];

        // Yield per-segment updates with timestamps
        if (result.Segments.Length > 0)
        {
            foreach (var segment in result.Segments)
            {
                cancellationToken.ThrowIfCancellationRequested();

                yield return new SpeechToTextResponseUpdate(segment.Text)
                {
                    Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                    StartTime = segment.Start,
                    EndTime = segment.End,
                    ModelId = options?.ModelId ?? Metadata.DefaultModelId,
                };
            }
        }
        else
        {
            // No segments — yield full text as a single update
            yield return new SpeechToTextResponseUpdate(result.Text)
            {
                Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                ModelId = options?.ModelId ?? Metadata.DefaultModelId,
            };
        }

        // Session close
        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionClose
        };

        await Task.CompletedTask;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is not null)
            return null;

        if (serviceType.IsAssignableFrom(GetType()))
            return this;

        if (serviceType == typeof(SpeechToTextClientMetadata))
            return Metadata;

        return null;
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }

    /// <summary>
    /// Loads WAV audio from stream and resamples based on options or defaults.
    /// </summary>
    private AudioData LoadAndResample(Stream audioStream, SpeechToTextOptions? options)
    {
        var audio = AudioIO.LoadWav(audioStream);

        var targetRate = options?.SpeechSampleRate ?? _options.SampleRate;
        if (audio.SampleRate != targetRate)
            audio = AudioIO.Resample(audio, targetRate);

        return audio;
    }
}
