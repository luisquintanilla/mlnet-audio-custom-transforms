using Microsoft.Extensions.AI;
using MLNet.ASR.Whistle.Native;

namespace MLNet.ASR.Whistle;

/// <summary>
/// Microsoft.Extensions.AI speech-to-text client backed by Cactus Whistle.
/// </summary>
public sealed class WhistleSpeechToTextClient : ISpeechToTextClient
{
    private readonly WhistleOptions _options;
    private readonly WhistleTranscriber _transcriber;

    /// <summary>Creates a client backed by the supplied Whistle options.</summary>
    public WhistleSpeechToTextClient(WhistleOptions options)
        : this(options, CactusWhistleNativeApi.Create)
    {
    }

    internal WhistleSpeechToTextClient(
        WhistleOptions options,
        Func<string?, IWhistleNativeApi> nativeFactory)
    {
        _options = options;
        _transcriber = new WhistleTranscriber(options, nativeFactory);
    }

    /// <summary>Metadata describing the local Whistle speech-to-text provider.</summary>
    public SpeechToTextClientMetadata Metadata => new(
        providerName: "Whistle-Native",
        providerUri: new Uri("https://huggingface.co/Cactus-Compute/whistle"),
        defaultModelId: Path.GetFileNameWithoutExtension(_options.ModelPath));

    /// <inheritdoc />
    public async Task<SpeechToTextResponse> GetTextAsync(
        Stream audioStream,
        SpeechToTextOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audioStream);
        cancellationToken.ThrowIfCancellationRequested();

        await using var bufferedAudio = new MemoryStream();
        await audioStream.CopyToAsync(bufferedAudio, cancellationToken);
        bufferedAudio.Position = 0;
        var audio = MLNet.Audio.Core.AudioIO.LoadWav(bufferedAudio);
        var result = _transcriber.Transcribe(audio, new WhistleTranscriptionOptions
        {
            Language = options?.SpeechLanguage,
            WordTimestamps = _options.WordTimestamps
        });

        var response = new SpeechToTextResponse(result.Text)
        {
            ModelId = options?.ModelId ?? Metadata.DefaultModelId,
            StartTime = result.Words.Count > 0 ? result.Words[0].Start : null,
            EndTime = result.Words.Count > 0 ? result.Words[^1].End : null,
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                ["language"] = result.Language ?? string.Empty,
                ["ttftMs"] = result.TimeToFirstToken.TotalMilliseconds,
                ["decodeTokensPerSecond"] = result.DecodeTokensPerSecond,
                ["words"] = result.Words.Select(w => new
                {
                    word = w.Word,
                    start = w.Start.TotalSeconds,
                    end = w.End.TotalSeconds,
                    probability = w.Probability
                }).ToArray()
            }
        };

        return response;
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(
        Stream audioStream,
        SpeechToTextOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionOpen,
            ModelId = options?.ModelId ?? Metadata.DefaultModelId
        };

        SpeechToTextResponseUpdate update;
        try
        {
            var response = await GetTextAsync(audioStream, options, cancellationToken);
            update = new SpeechToTextResponseUpdate(response.Text)
            {
                Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                ModelId = response.ModelId,
                StartTime = response.StartTime,
                EndTime = response.EndTime
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            update = new SpeechToTextResponseUpdate(ex.Message)
            {
                Kind = SpeechToTextResponseUpdateKind.Error,
                ModelId = options?.ModelId ?? Metadata.DefaultModelId
            };
        }

        yield return update;
        yield return new SpeechToTextResponseUpdate
        {
            Kind = SpeechToTextResponseUpdateKind.SessionClose,
            ModelId = options?.ModelId ?? Metadata.DefaultModelId
        };
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
    public void Dispose() => _transcriber.Dispose();
}
