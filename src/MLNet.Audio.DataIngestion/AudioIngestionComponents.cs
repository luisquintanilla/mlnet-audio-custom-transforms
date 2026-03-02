using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using MLNet.Audio.Core;

namespace MLNet.Audio.DataIngestion;

/// <summary>
/// Reads WAV audio files into IngestionDocument objects.
/// Stores the decoded AudioData in the first section's Metadata["audio"]
/// for downstream chunkers to access.
/// </summary>
public class AudioDocumentReader : IngestionDocumentReader
{
    private readonly int _targetSampleRate;

    /// <param name="targetSampleRate">Resample all audio to this rate (default: 16000 Hz).</param>
    public AudioDocumentReader(int targetSampleRate = 16000)
    {
        _targetSampleRate = targetSampleRate;
    }

    public override Task<IngestionDocument> ReadAsync(
        Stream stream, string name, string mediaType, CancellationToken cancellationToken = default)
    {
        var audio = AudioIO.LoadWav(stream);
        if (audio.SampleRate != _targetSampleRate)
            audio = AudioIO.Resample(audio, _targetSampleRate);

        var doc = new IngestionDocument(name);

        var section = new IngestionDocumentSection($"audio:{name}");
        section.Text = $"Audio file: {name}, Duration: {audio.Duration.TotalSeconds:F2}s, " +
                       $"SampleRate: {audio.SampleRate}Hz, Samples: {audio.Samples.Length}";
        section.Metadata["audio"] = audio;
        doc.Sections.Add(section);

        return Task.FromResult(doc);
    }
}

/// <summary>
/// Chunks audio documents into fixed time-window segments.
/// Each chunk's Content is the AudioData segment itself,
/// with timing metadata stored in chunk.Metadata.
/// </summary>
public class AudioSegmentChunker : IngestionChunker<AudioData>
{
    private readonly TimeSpan _segmentDuration;

    /// <param name="segmentDuration">Duration of each chunk (default: 2 seconds).</param>
    public AudioSegmentChunker(TimeSpan? segmentDuration = null)
    {
        _segmentDuration = segmentDuration ?? TimeSpan.FromSeconds(2);
    }

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IngestionDocument doc, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Retrieve AudioData stored by AudioDocumentReader
        AudioData? audio = null;
        foreach (var section in doc.Sections)
        {
            if (section.Metadata.TryGetValue("audio", out var obj) && obj is AudioData a)
            {
                audio = a;
                break;
            }
        }

        if (audio is null)
            yield break;

        int samplesPerSegment = (int)(_segmentDuration.TotalSeconds * audio.SampleRate);
        int totalSamples = audio.Samples.Length;
        int segmentIndex = 0;

        for (int offset = 0; offset < totalSamples; offset += samplesPerSegment)
        {
            int length = Math.Min(samplesPerSegment, totalSamples - offset);
            var segmentSamples = audio.Samples[offset..(offset + length)];
            var segmentAudio = new AudioData(segmentSamples, audio.SampleRate);

            var startTime = TimeSpan.FromSeconds((double)offset / audio.SampleRate);
            var endTime = TimeSpan.FromSeconds((double)(offset + length) / audio.SampleRate);

            var chunk = new IngestionChunk<AudioData>(segmentAudio, doc, $"segment-{segmentIndex}");
            chunk.Metadata["startTime"] = startTime.TotalSeconds.ToString("F2");
            chunk.Metadata["endTime"] = endTime.TotalSeconds.ToString("F2");
            chunk.Metadata["segmentIndex"] = segmentIndex.ToString();
            chunk.Metadata["sourceFile"] = doc.Identifier;

            yield return chunk;
            segmentIndex++;
        }

        await Task.CompletedTask;
    }
}

/// <summary>
/// Enriches audio chunks with embeddings from an IEmbeddingGenerator.
/// Stores the embedding vector in chunk.Metadata["embedding"] as float[].
/// This connects Layer 3 (DataIngestion) → Layer 2 (MEAI) → Layer 1 (ML.NET).
/// </summary>
public class AudioEmbeddingChunkProcessor : IngestionChunkProcessor<AudioData>
{
    private readonly IEmbeddingGenerator<AudioData, Embedding<float>> _generator;

    public AudioEmbeddingChunkProcessor(
        IEmbeddingGenerator<AudioData, Embedding<float>> generator)
    {
        _generator = generator;
    }

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<AudioData>> chunks,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in chunks.WithCancellation(cancellationToken))
        {
            var embeddings = await _generator.GenerateAsync(
                [chunk.Content], cancellationToken: cancellationToken);

            if (embeddings.Count > 0)
                chunk.Metadata["embedding"] = embeddings[0].Vector.ToArray();

            yield return chunk;
        }
    }
}
