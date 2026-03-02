using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Audio Classification Sample — Audio Spectrogram Transformer (AST)
// ======================================================================
// This sample demonstrates audio classification using ML.NET custom
// transforms with a local ONNX model. Works with any audio classification
// ONNX model (AST, Wav2Vec2, HuBERT, etc.).
//
// Before running:
// 1. Download an AST ONNX model from HuggingFace:
//    huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include "onnx/*" --local-dir models/ast
//    (or export one yourself with optimum-cli)
//
// 2. Place a test WAV file (16kHz mono preferred) in the sample directory.
// ======================================================================

var modelPath = args.Length > 0 ? args[0] : "models/ast/onnx/model.onnx";
var audioPath = args.Length > 1 ? args[1] : "test.wav";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine("Download an AST ONNX model from HuggingFace:");
    Console.WriteLine("  huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include \"onnx/*\" --local-dir models/ast");
    return;
}

if (!File.Exists(audioPath))
{
    Console.WriteLine($"Audio file not found: {audioPath}");
    Console.WriteLine("Generating a synthetic test audio file...");
    GenerateTestWav(audioPath);
}

// Load audio
Console.WriteLine($"Loading audio: {audioPath}");
var audio = AudioIO.LoadWav(audioPath);
Console.WriteLine($"  Duration: {audio.Duration.TotalSeconds:F2}s, Sample Rate: {audio.SampleRate}Hz, Samples: {audio.Samples.Length}");

// Resample to 16kHz if needed
if (audio.SampleRate != 16000)
{
    audio = AudioIO.Resample(audio, 16000);
    Console.WriteLine($"  Resampled to 16kHz ({audio.Samples.Length} samples)");
}

// Define AudioSet labels (top-level subset for demo)
var labels = new[]
{
    "Speech", "Music", "Singing", "Silence", "Animal", "Vehicle", "Water",
    "Wind", "Thunder", "Bell", "Alarm", "Gunshot", "Explosion", "Laughter",
    "Crying", "Cough", "Sneeze", "Clapping", "Typing", "Footsteps"
};

var mlContext = new MLContext();
var options = new OnnxAudioClassificationOptions
{
    ModelPath = modelPath,
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
    {
        NumMelBins = 128, // AST uses 128 mel bins
        FftSize = 400,
        HopLength = 160
    },
    Labels = labels,
    SampleRate = 16000
};

// --- Approach 1: ML.NET Pipeline (Fit / Transform) ---
Console.WriteLine("\n=== ML.NET Pipeline (Fit / Transform) ===");

var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new AudioInput { Audio = audio.Samples }
});

var pipeline = mlContext.Transforms.OnnxAudioClassification(options);
var model = pipeline.Fit(data);
var output = model.Transform(data);

var outputRows = mlContext.Data.CreateEnumerable<ClassificationOutput>(output, reuseRowObject: false).ToList();
foreach (var row in outputRows)
{
    Console.WriteLine($"  Predicted: {row.PredictedLabel} (confidence: {row.Score:P2})");
    Console.WriteLine("  Top 5 classes:");
    var topK = labels
        .Zip(row.Probabilities, (l, p) => (Label: l, Prob: p))
        .OrderByDescending(x => x.Prob)
        .Take(5);
    foreach (var (label, prob) in topK)
        Console.WriteLine($"    {label}: {prob:P2}");
}

// --- Approach 2: Composed Pipeline (individual sub-transforms) ---
Console.WriteLine("\n=== Composed Pipeline (Sub-Transforms) ===");

// Compose individual stages: feature extraction → ONNX scoring → post-processing
// Same pattern as text transforms: tokenizer → model → pooler
var composedPipeline = new AudioFeatureExtractionEstimator(mlContext,
        new AudioFeatureExtractionOptions
        {
            FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000) { NumMelBins = 128, FftSize = 400, HopLength = 160 },
            SampleRate = 16000
        })
    .Append(new OnnxAudioScoringEstimator(mlContext,
        new OnnxAudioScoringOptions { ModelPath = modelPath }))
    .Append(new AudioClassificationPostProcessingEstimator(mlContext,
        new AudioClassificationPostProcessingOptions { Labels = labels }));

var composedModel = composedPipeline.Fit(data);
var composedOutput = composedModel.Transform(data);

var composedRows = mlContext.Data.CreateEnumerable<ClassificationOutput>(composedOutput, reuseRowObject: false).ToList();
foreach (var row in composedRows)
{
    Console.WriteLine($"  Predicted: {row.PredictedLabel} (confidence: {row.Score:P2})");
}

// --- Approach 3: Direct API (convenience) ---
Console.WriteLine("\n=== Direct Classification API ===");

var results = model.Classify(new[] { audio });
foreach (var result in results)
{
    Console.WriteLine($"  Predicted: {result.PredictedLabel} (confidence: {result.Score:P2})");
}

Console.WriteLine("\nDone!");

// --- Helper: Generate a synthetic test WAV ---
static void GenerateTestWav(string path)
{
    // Generate 2 seconds of a 440Hz sine wave (A4 note) at 16kHz
    int sampleRate = 16000;
    int duration = 2;
    var samples = new float[sampleRate * duration];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sampleRate) * 0.5f;

    var audio = new AudioData(samples, sampleRate);
    AudioIO.SaveWav(path, audio);
    Console.WriteLine($"  Generated {duration}s test tone at {path}");
}

class AudioInput
{
    public float[] Audio { get; set; } = [];
}

class ClassificationOutput
{
    [Microsoft.ML.Data.ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";

    [Microsoft.ML.Data.ColumnName("Score")]
    public float Score { get; set; }

    [Microsoft.ML.Data.ColumnName("Probabilities")]
    [Microsoft.ML.Data.VectorType]
    public float[] Probabilities { get; set; } = [];
}

