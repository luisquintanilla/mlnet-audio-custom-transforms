using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.Audio.DataIngestion;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Audio DataIngestion Sample — End-to-End Audio Pipeline
// ======================================================================
// Demonstrates Microsoft.Extensions.DataIngestion with audio:
//
//   Layer 3 (DataIngestion): AudioDocumentReader → AudioSegmentChunker → AudioEmbeddingChunkProcessor
//   Layer 2 (MEAI):          IEmbeddingGenerator<AudioData, Embedding<float>>
//   Layer 1 (ML.NET):        AudioFeatureExtraction → OnnxScoring → EmbeddingPooling
//
// This shows that DataIngestion is modality-agnostic — not just for text/PDF.
// The same Reader → Chunker → Processor pattern works for audio.
//
// Before running:
//   huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap
// ======================================================================

var modelPath = args.Length > 0 ? args[0] : "models/clap/onnx/model.onnx";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine("Download a CLAP ONNX model from HuggingFace, e.g.:");
    Console.WriteLine("  huggingface-cli download laion/clap-htsat-unfused --include \"onnx/*\" --local-dir models/clap");
    Console.WriteLine("\nRunning with synthetic demo (no model) instead...\n");
    await RunSyntheticDemo();
    return;
}

// --- Step 0: Generate synthetic audio files ---
Console.WriteLine("=== Generating Synthetic Audio Files ===\n");

var audioDir = "audio_samples";
Directory.CreateDirectory(audioDir);

GenerateTone(Path.Combine(audioDir, "music_440hz.wav"), 440, 16000, 4);
GenerateTone(Path.Combine(audioDir, "music_880hz.wav"), 880, 16000, 4);
GenerateNoise(Path.Combine(audioDir, "noise.wav"), 16000, 4);
GenerateChirp(Path.Combine(audioDir, "chirp.wav"), 200, 2000, 16000, 4);

var wavFiles = Directory.GetFiles(audioDir, "*.wav");

// --- Step 1: Create ML.NET model → MEAI embedding generator ---
Console.WriteLine("\n=== Setting Up Embedding Pipeline (ML.NET → MEAI) ===\n");

var mlContext = new MLContext();
var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = modelPath,
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
    {
        NumMelBins = 64,
        FftSize = 512,
        HopLength = 160
    },
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true,
    SampleRate = 16000
};

var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
using var transformer = estimator.Fit(emptyData);
using IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

Console.WriteLine("  ML.NET model loaded → MEAI IEmbeddingGenerator ready");

// --- Step 2: Create DataIngestion pipeline ---
Console.WriteLine("\n=== DataIngestion Pipeline: Read → Chunk → Embed ===\n");

var reader = new AudioDocumentReader(targetSampleRate: 16000);
var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2));
var processor = new AudioEmbeddingChunkProcessor(generator);

// Process all audio files through the pipeline
var allChunks = new List<(string File, string Segment, float[] Embedding)>();

foreach (var wavFile in wavFiles)
{
    Console.WriteLine($"  Processing: {Path.GetFileName(wavFile)}");

    // Layer 3: DataIngestion — Read
    using var stream = File.OpenRead(wavFile);
    var doc = await reader.ReadAsync(stream, Path.GetFileName(wavFile), "audio/wav");
    Console.WriteLine($"    Read: {doc.Sections[0].Text}");

    // Layer 3: DataIngestion — Chunk
    var chunks = chunker.ProcessAsync(doc);

    // Layer 3: DataIngestion — Process (embed via MEAI → ML.NET)
    var enrichedChunks = processor.ProcessAsync(chunks);

    await foreach (var chunk in enrichedChunks)
    {
        if (chunk.Metadata.TryGetValue("embedding", out var embObj) && embObj is float[] embedding)
        {
            var start = chunk.Metadata["startTime"];
            var end = chunk.Metadata["endTime"];
            var label = $"[{start}s → {end}s]";
            allChunks.Add((Path.GetFileName(wavFile), label, embedding));
            Console.WriteLine($"    Chunk {label}: [{embedding.Length}]-dim embedding");
        }
    }
}

Console.WriteLine($"\n  Total chunks embedded: {allChunks.Count}");

// --- Step 3: Similarity search over embedded chunks ---
Console.WriteLine("\n=== Audio Similarity Search ===\n");

// Use the first chunk as query
if (allChunks.Count >= 2)
{
    var query = allChunks[0];
    Console.WriteLine($"  Query: {query.File} {query.Segment}");
    Console.WriteLine($"  Results (sorted by similarity):\n");

    var results = allChunks
        .Skip(1)
        .Select(c => new
        {
            c.File,
            c.Segment,
            Similarity = TensorPrimitives.CosineSimilarity(query.Embedding, c.Embedding)
        })
        .OrderByDescending(r => r.Similarity)
        .ToList();

    foreach (var r in results)
    {
        Console.WriteLine($"    {r.Similarity:F4}  {r.File} {r.Segment}");
    }
}

// Cleanup
Directory.Delete(audioDir, true);
Console.WriteLine("\nDone!");
return;

// === Helpers ===

static void GenerateTone(string path, float freq, int sr, int seconds)
{
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * freq * i / sr) * 0.5f;
    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  {Path.GetFileName(path)}: {seconds}s tone at {freq}Hz");
}

static void GenerateNoise(string path, int sr, int seconds)
{
    var rng = new Random(42);
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f;
    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  {Path.GetFileName(path)}: {seconds}s white noise");
}

static void GenerateChirp(string path, float startFreq, float endFreq, int sr, int seconds)
{
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
    {
        float t = (float)i / sr;
        float freq = startFreq + (endFreq - startFreq) * (t / seconds);
        samples[i] = MathF.Sin(2 * MathF.PI * freq * t) * 0.5f;
    }
    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  {Path.GetFileName(path)}: {seconds}s chirp {startFreq}→{endFreq}Hz");
}

static async Task RunSyntheticDemo()
{
    Console.WriteLine("=== Synthetic Demo (DataIngestion without model) ===\n");
    Console.WriteLine("Demonstrates the DataIngestion pipeline components:\n");

    // Generate test audio
    var dir = "synthetic_audio";
    Directory.CreateDirectory(dir);

    int sr = 16000;
    var samples = new float[sr * 4];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sr) * 0.5f;
    AudioIO.SaveWav(Path.Combine(dir, "test.wav"), new AudioData(samples, sr));

    // Reader
    var reader = new AudioDocumentReader(targetSampleRate: 16000);
    using var stream = File.OpenRead(Path.Combine(dir, "test.wav"));
    var doc = await reader.ReadAsync(stream, "test.wav", "audio/wav");
    Console.WriteLine($"  AudioDocumentReader: {doc.Sections[0].Text}");

    // Chunker
    var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(1));
    int chunkCount = 0;
    await foreach (var chunk in chunker.ProcessAsync(doc))
    {
        Console.WriteLine($"  AudioSegmentChunker: chunk [{chunk.Metadata["startTime"]}s → {chunk.Metadata["endTime"]}s], " +
                          $"{chunk.Content.Duration.TotalSeconds:F2}s of audio");
        chunkCount++;
    }
    Console.WriteLine($"\n  Total chunks: {chunkCount}");
    Console.WriteLine("\n  (AudioEmbeddingChunkProcessor skipped — requires ONNX model)");

    Directory.Delete(dir, true);
    Console.WriteLine("\nSynthetic demo complete!");
}

class AudioInput
{
    public float[] Audio { get; set; } = [];
}
