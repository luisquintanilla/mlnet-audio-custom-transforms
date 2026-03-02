using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// IEmbeddingGenerator implementation for audio embeddings via ONNX.
/// Bridges the ML.NET audio embedding transformer to the Microsoft.Extensions.AI abstraction.
/// </summary>
public sealed class OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>
{
    private readonly OnnxAudioEmbeddingTransformer _transformer;

    public EmbeddingGeneratorMetadata Metadata { get; }

    public OnnxAudioEmbeddingGenerator(OnnxAudioEmbeddingTransformer transformer, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        _transformer = transformer;

        Metadata = new EmbeddingGeneratorMetadata(
            providerName: "MLNet.AudioInference.Onnx",
            defaultModelId: modelId,
            defaultModelDimensions: transformer.EmbeddingDimension);
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<AudioData> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var audioList = values.ToList();
        var embeddings = _transformer.GenerateEmbeddings(audioList);

        var result = new GeneratedEmbeddings<Embedding<float>>(
            embeddings.Select(e => new Embedding<float>(e)).ToList());

        return Task.FromResult(result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(OnnxAudioEmbeddingGenerator))
            return this;
        return null;
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
