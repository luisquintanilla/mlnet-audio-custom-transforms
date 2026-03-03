using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Audio Embeddings Sample — Generate Vector Embeddings from Audio
// ======================================================================
// This sample demonstrates audio embedding generation using ML.NET custom
// transforms with a local ONNX model. Works with any audio encoder model
// that produces embeddings (CLAP, Wav2Vec2, HuBERT, etc.).
//
// Shows both:
// - Direct API (transformer.GenerateEmbeddings)
// - MEAI interface (IEmbeddingGenerator<AudioData, Embedding<float>>)
//
// Before running:
// 1. Download a CLAP or audio encoder ONNX model from HuggingFace
// 2. Place test WAV files in the sample directory
// ======================================================================

var modelPath = args.Length > 0 ? args[0] : "models/clap/onnx/model.onnx";
var audioDir = args.Length > 1 ? args[1] : ".";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine("Download a CLAP ONNX model from HuggingFace, e.g.:");
    Console.WriteLine("  huggingface-cli download lquint/clap-htsat-unfused-onnx --local-dir models/clap");
    Console.WriteLine("\nRunning with synthetic demo instead...\n");
    RunSyntheticDemo();
    return;
}

var wavFiles = Directory.GetFiles(audioDir, "*.wav");
if (wavFiles.Length == 0)
{
    Console.WriteLine("No WAV files found. Generating test files...");
    GenerateTestFiles(audioDir);
    wavFiles = Directory.GetFiles(audioDir, "*.wav");
}

// Setup
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

// --- Approach 1: ML.NET Pipeline (Fit / Transform) ---
Console.WriteLine("=== ML.NET Pipeline (Fit / Transform) ===");

var audioInputs = wavFiles.Select(f =>
{
    var audio = AudioIO.LoadWav(f);
    if (audio.SampleRate != 16000)
        audio = AudioIO.Resample(audio, 16000);
    return audio;
}).ToList();

var data = mlContext.Data.LoadFromEnumerable(
    audioInputs.Select(a => new AudioInput { Audio = a.Samples }).ToArray());

var model = estimator.Fit(data);
var output = model.Transform(data);

var outputRows = mlContext.Data.CreateEnumerable<EmbeddingOutput>(output, reuseRowObject: false).ToList();
for (int i = 0; i < outputRows.Count; i++)
{
    var emb = outputRows[i].Embedding;
    Console.WriteLine($"  {Path.GetFileName(wavFiles[i])}: [{emb.Length}]-dim embedding");
    Console.WriteLine($"    First 5 values: [{string.Join(", ", emb.Take(5).Select(v => v.ToString("F4")))}...]");
}

// --- Approach 2: Composed Pipeline (individual sub-transforms with .Append()) ---
Console.WriteLine("\n=== Composed Pipeline (.Append()) ===");

// Compose individual stages: feature extraction → ONNX scoring → embedding pooling
// The pooler auto-discovers HiddenDim from scorer column annotations — no manual config needed
var composedPipeline = new AudioFeatureExtractionEstimator(mlContext,
        new AudioFeatureExtractionOptions
        {
            FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000) { NumMelBins = 64, FftSize = 512, HopLength = 160 },
            SampleRate = 16000
        })
    .Append(new OnnxAudioScoringEstimator(mlContext,
        new OnnxAudioScoringOptions { ModelPath = modelPath }))
    .Append(new AudioEmbeddingPoolingEstimator(mlContext,
        new AudioEmbeddingPoolingOptions
        {
            Pooling = AudioPoolingStrategy.MeanPooling,
            Normalize = true
        }));

var composedModel = composedPipeline.Fit(data);
var composedOutput = composedModel.Transform(data);

var composedRows = mlContext.Data.CreateEnumerable<EmbeddingOutput>(composedOutput, reuseRowObject: false).ToList();
for (int i = 0; i < composedRows.Count; i++)
{
    var emb = composedRows[i].Embedding;
    Console.WriteLine($"  {Path.GetFileName(wavFiles[i])}: [{emb.Length}]-dim embedding");
}

// --- Approach 3: Direct API ---
Console.WriteLine("\n=== Direct Embedding API ===");

var embeddings = transformer.GenerateEmbeddings(audioInputs);

for (int i = 0; i < wavFiles.Length; i++)
{
    Console.WriteLine($"  {Path.GetFileName(wavFiles[i])}: [{embeddings[i].Length}]-dim embedding");
    Console.WriteLine($"    First 5 values: [{string.Join(", ", embeddings[i].Take(5).Select(v => v.ToString("F4")))}...]");
}

// --- Approach 4: MEAI IEmbeddingGenerator ---
Console.WriteLine("\n=== MEAI IEmbeddingGenerator<AudioData, Embedding<float>> ===");

using IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

var meaiEmbeddings = await generator.GenerateAsync(audioInputs);
Console.WriteLine($"  Generated {meaiEmbeddings.Count} embeddings");

// --- Cosine similarity demo ---
if (embeddings.Length >= 2)
{
    Console.WriteLine("\n=== Cosine Similarity ===");
    for (int i = 0; i < embeddings.Length; i++)
    {
        for (int j = i + 1; j < embeddings.Length; j++)
        {
            var sim = CosineSimilarity(embeddings[i], embeddings[j]);
            Console.WriteLine($"  {Path.GetFileName(wavFiles[i])} vs {Path.GetFileName(wavFiles[j])}: {sim:F4}");
        }
    }
}

Console.WriteLine("\nDone!");
return;

// === Helpers ===

static float CosineSimilarity(float[] a, float[] b)
    => TensorPrimitives.CosineSimilarity(a, b);

static void GenerateTestFiles(string dir)
{
    int sr = 16000;
    // 440Hz sine (A4 note — music)
    GenerateTone(Path.Combine(dir, "tone_440hz.wav"), 440, sr, 2);
    // 880Hz sine (A5 note)
    GenerateTone(Path.Combine(dir, "tone_880hz.wav"), 880, sr, 2);
    // White noise
    GenerateNoise(Path.Combine(dir, "noise.wav"), sr, 2);
}

static void GenerateTone(string path, float freq, int sr, int seconds)
{
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * freq * i / sr) * 0.5f;
    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  Generated {seconds}s tone at {freq}Hz → {Path.GetFileName(path)}");
}

static void GenerateNoise(string path, int sr, int seconds)
{
    var rng = new Random(42);
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f;
    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  Generated {seconds}s white noise → {Path.GetFileName(path)}");
}

static void RunSyntheticDemo()
{
    Console.WriteLine("=== Synthetic Demo (no ONNX model) ===");
    Console.WriteLine("This demonstrates the AudioData and AudioIO primitives.\n");

    int sr = 16000;
    // Generate test audio
    var samples = new float[sr * 2];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sr) * 0.5f;

    var audio = new AudioData(samples, sr);
    Console.WriteLine($"  Audio: {audio.Duration.TotalSeconds:F2}s, {audio.SampleRate}Hz, {audio.Channels}ch");

    // Test feature extraction
    var extractor = new MelSpectrogramExtractor(sampleRate: 16000)
    {
        NumMelBins = 80,
        FftSize = 400,
        HopLength = 160
    };

    var features = extractor.Extract(audio);
    Console.WriteLine($"  Mel spectrogram: [{features.GetLength(0)} frames x {features.GetLength(1)} mel bins]");

    // Save and reload
    AudioIO.SaveWav("synthetic_test.wav", audio);
    var reloaded = AudioIO.LoadWav("synthetic_test.wav");
    Console.WriteLine($"  Saved and reloaded: {reloaded.Duration.TotalSeconds:F2}s, {reloaded.Samples.Length} samples");

    File.Delete("synthetic_test.wav");
    Console.WriteLine("\nSynthetic demo complete!");
}

class AudioInput
{
    public float[] Audio { get; set; } = [];
}

class EmbeddingOutput
{
    [Microsoft.ML.Data.ColumnName("Embedding")]
    [Microsoft.ML.Data.VectorType]
    public float[] Embedding { get; set; } = [];
}

