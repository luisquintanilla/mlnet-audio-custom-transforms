using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// <see cref="ITextToSpeechClient"/> implementation that delegates to any
/// <see cref="IOnnxTtsSynthesizer"/> (SpeechT5, KittenTTS, etc.).
///
/// Usage:
///   using var client = new OnnxTextToSpeechClient(speecht5Options);
///   var response = await client.GetAudioAsync("Hello, world!");
///   var audio = response.Contents.OfType&lt;DataContent&gt;().First();
///   File.WriteAllBytes("output.wav", audio.Data!.Value.ToArray());
/// </summary>
#pragma warning disable AIEXP001 // ITextToSpeechClient is experimental
public sealed class OnnxTextToSpeechClient : ITextToSpeechClient
#pragma warning restore AIEXP001
{
    private readonly IOnnxTtsSynthesizer _synthesizer;
    private readonly TextToSpeechClientMetadata _metadata;

    /// <summary>Creates a TTS client backed by the SpeechT5 ONNX pipeline.</summary>
    public OnnxTextToSpeechClient(OnnxSpeechT5Options options)
    {
        var mlContext = new Microsoft.ML.MLContext();
        _synthesizer = new OnnxSpeechT5TtsTransformer(mlContext, options);
        _metadata = new TextToSpeechClientMetadata(
            _synthesizer.ProviderName,
            _synthesizer.ProviderUri,
            _synthesizer.ModelId);
    }

    /// <summary>Creates a TTS client backed by the KittenTTS ONNX pipeline.</summary>
    public OnnxTextToSpeechClient(OnnxKittenTtsOptions options)
    {
        var mlContext = new Microsoft.ML.MLContext();
        _synthesizer = new OnnxKittenTtsTransformer(mlContext, options);
        _metadata = new TextToSpeechClientMetadata(
            _synthesizer.ProviderName,
            _synthesizer.ProviderUri,
            _synthesizer.ModelId);
    }

    /// <inheritdoc/>
#pragma warning disable AIEXP001
    public Task<TextToSpeechResponse> GetAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default)
#pragma warning restore AIEXP001
    {
        cancellationToken.ThrowIfCancellationRequested();

        var audio = _synthesizer.Synthesize(text, options);
        var wavBytes = ToWavBytes(audio);

        var response = new TextToSpeechResponse
        {
            Contents = [new DataContent(wavBytes, "audio/wav")],
            ModelId = _synthesizer.ModelId
        };

        return Task.FromResult(response);
    }

    /// <inheritdoc/>
#pragma warning disable AIEXP001, CS1998
    public async IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
#pragma warning restore AIEXP001, CS1998
    {
        yield return new TextToSpeechResponseUpdate
        {
            Kind = TextToSpeechResponseUpdateKind.SessionOpen,
            ModelId = _synthesizer.ModelId
        };

        List<TextToSpeechResponseUpdate>? updates = null;
        TextToSpeechResponseUpdate? errorUpdate = null;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            var audio = _synthesizer.Synthesize(text, options);
            var wavBytes = ToWavBytes(audio);

            updates =
            [
                new TextToSpeechResponseUpdate
                {
                    Kind = TextToSpeechResponseUpdateKind.AudioUpdated,
                    Contents = [new DataContent(wavBytes, "audio/wav")],
                    ModelId = _synthesizer.ModelId
                }
            ];
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            errorUpdate = new TextToSpeechResponseUpdate
            {
                Kind = TextToSpeechResponseUpdateKind.Error,
                Contents = [new TextContent(ex.Message)],
                ModelId = _synthesizer.ModelId
            };
        }

        if (errorUpdate is not null)
            yield return errorUpdate;

        if (updates is not null)
        {
            foreach (var update in updates)
                yield return update;
        }

        yield return new TextToSpeechResponseUpdate
        {
            Kind = TextToSpeechResponseUpdateKind.SessionClose,
            ModelId = _synthesizer.ModelId
        };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is not null)
            return null;

        if (serviceType.IsAssignableFrom(GetType()))
            return this;

#pragma warning disable AIEXP001
        if (serviceType == typeof(TextToSpeechClientMetadata))
            return _metadata;
#pragma warning restore AIEXP001

        return null;
    }

    public void Dispose()
    {
        _synthesizer?.Dispose();
    }

    private static byte[] ToWavBytes(AudioData audio)
    {
        using var ms = new MemoryStream();
        AudioIO.SaveWav(ms, audio);
        return ms.ToArray();
    }
}
