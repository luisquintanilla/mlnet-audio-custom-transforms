using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// Local ISpeechToTextClient backed by Whisper via ORT GenAI.
/// Plugs into the MEAI ecosystem — usable via DI, middleware
/// (logging, telemetry), and the SpeechToTextClientBuilder pipeline.
/// </summary>
public sealed class OnnxSpeechToTextClient : ISpeechToTextClient
{
    private readonly OnnxSpeechToTextTransformer _transformer;
    private readonly OnnxSpeechToTextOptions _options;

    public OnnxSpeechToTextClient(OnnxSpeechToTextOptions options)
    {
        _options = options;
        _transformer = new OnnxSpeechToTextTransformer(new Microsoft.ML.MLContext(), options);
    }

    public SpeechToTextClientMetadata Metadata => new(
        providerName: "OnnxGenAI-Whisper",
        providerUri: null,
        defaultModelId: System.IO.Path.GetFileName(_options.ModelPath));

    public Task<SpeechToTextResponse> GetTextAsync(
        Stream audioStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var audio = LoadAndResample(audioStream, options);

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
        Stream audioStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return StreamingImpl(audioStream, options, cancellationToken);
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators
    private async IAsyncEnumerable<SpeechToTextResponseUpdate> StreamingImpl(
        Stream audioStream,
        SpeechToTextOptions? options,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionOpen
        };

        // Pre-compute updates in a list so we can use try/catch
        // (C# does not allow yield return inside try blocks with catch clauses).
        var updates = new List<SpeechToTextResponseUpdate>();
        SpeechToTextResponseUpdate? errorUpdate = null;

        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            var audio = LoadAndResample(audioStream, options);
            var results = _transformer.TranscribeWithTimestamps([audio]);
            var result = results[0];

            if (result.Segments.Length > 0)
            {
                foreach (var segment in result.Segments)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    updates.Add(new SpeechToTextResponseUpdate(segment.Text)
                    {
                        Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                        StartTime = segment.Start,
                        EndTime = segment.End,
                        ModelId = options?.ModelId ?? Metadata.DefaultModelId,
                    });
                }
            }
            else
            {
                updates.Add(new SpeechToTextResponseUpdate(result.Text)
                {
                    Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                    ModelId = options?.ModelId ?? Metadata.DefaultModelId,
                });
            }
        }
        catch (Exception ex)
        {
            errorUpdate = new SpeechToTextResponseUpdate(ex.Message)
            {
                Kind = SpeechToTextResponseUpdateKind.Error,
                ModelId = options?.ModelId ?? Metadata.DefaultModelId,
            };
        }

        foreach (var update in updates)
            yield return update;

        if (errorUpdate is not null)
            yield return errorUpdate;

        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionClose
        };
    }
#pragma warning restore CS1998

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
