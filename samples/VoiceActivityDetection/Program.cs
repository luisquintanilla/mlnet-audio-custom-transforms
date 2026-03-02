using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Voice Activity Detection (VAD) Sample — Silero VAD
// ======================================================================
// Detects speech vs. silence segments in audio using a small ONNX model.
// Silero VAD is ~2MB and runs very fast on CPU.
//
// Shows both:
// - Direct API (transformer.DetectSpeech)
// - IVoiceActivityDetector interface (streaming)
// - ML.NET pipeline integration
//
// Before running:
// 1. Download Silero VAD ONNX model:
//    huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad
//    (or download from https://github.com/snakers4/silero-vad/tree/master/src/silero_vad/data)
//
// 2. Place a test WAV file with speech+silence in the sample directory.
// ======================================================================

var modelPath = args.Length > 0 ? args[0] : "models/silero-vad/silero_vad.onnx";
var audioPath = args.Length > 1 ? args[1] : "test_speech.wav";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Silero VAD model not found at: {modelPath}");
    Console.WriteLine("Download from HuggingFace:");
    Console.WriteLine("  huggingface-cli download snakers4/silero-vad --include \"*.onnx\" --local-dir models/silero-vad");
    Console.WriteLine("\nRunning synthetic VAD demo instead...\n");
    RunSyntheticDemo();
    return;
}

if (!File.Exists(audioPath))
{
    Console.WriteLine($"Audio file not found: {audioPath}");
    Console.WriteLine("Generating test audio with speech-like patterns...");
    GenerateTestAudio(audioPath);
}

// Load audio
Console.WriteLine($"Loading audio: {audioPath}");
var audio = AudioIO.LoadWav(audioPath);
Console.WriteLine($"  Duration: {audio.Duration.TotalSeconds:F2}s, Sample Rate: {audio.SampleRate}Hz");

if (audio.SampleRate != 16000)
{
    audio = AudioIO.Resample(audio, 16000);
    Console.WriteLine($"  Resampled to 16kHz");
}

// --- Approach 1: Direct API ---
Console.WriteLine("\n=== Direct VAD API ===");

var mlContext = new MLContext();
var options = new OnnxVadOptions
{
    ModelPath = modelPath,
    Threshold = 0.5f,
    MinSpeechDuration = TimeSpan.FromMilliseconds(250),
    MinSilenceDuration = TimeSpan.FromMilliseconds(100),
    SpeechPad = TimeSpan.FromMilliseconds(30),
    WindowSize = 512,
    SampleRate = 16000
};

var estimator = mlContext.Transforms.OnnxVad(options);
var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
using var transformer = estimator.Fit(emptyData);

var segments = transformer.DetectSpeech(audio);
Console.WriteLine($"  Found {segments.Length} speech segment(s):");
foreach (var seg in segments)
{
    Console.WriteLine($"    [{seg.Start:mm\\:ss\\.fff} → {seg.End:mm\\:ss\\.fff}] " +
                      $"duration={seg.Duration.TotalSeconds:F2}s, confidence={seg.Confidence:P1}");
}

var totalSpeech = segments.Sum(s => s.Duration.TotalSeconds);
Console.WriteLine($"  Total speech: {totalSpeech:F2}s / {audio.Duration.TotalSeconds:F2}s " +
                  $"({totalSpeech / audio.Duration.TotalSeconds:P1})");

// --- Approach 2: IVoiceActivityDetector (streaming) ---
Console.WriteLine("\n=== IVoiceActivityDetector (Streaming) ===");

IVoiceActivityDetector vad = transformer;
using var stream = File.OpenRead(audioPath);

await foreach (var segment in vad.DetectSpeechAsync(stream))
{
    Console.WriteLine($"  Speech: [{segment.Start:mm\\:ss\\.fff} → {segment.End:mm\\:ss\\.fff}] " +
                      $"({segment.Confidence:P1})");
}

Console.WriteLine("\nDone!");
return;

// === Helpers ===

static void GenerateTestAudio(string path)
{
    int sr = 16000;
    int totalSeconds = 5;
    var samples = new float[sr * totalSeconds];

    // Pattern: silence(0-1s) → tone(1-3s) → silence(3-4s) → tone(4-5s)
    for (int i = 0; i < samples.Length; i++)
    {
        float t = (float)i / sr;
        if ((t >= 1.0f && t < 3.0f) || (t >= 4.0f && t < 5.0f))
        {
            // "Speech-like" signal: sum of multiple frequencies
            samples[i] = (MathF.Sin(2 * MathF.PI * 200 * i / sr) * 0.3f +
                          MathF.Sin(2 * MathF.PI * 400 * i / sr) * 0.2f +
                          MathF.Sin(2 * MathF.PI * 800 * i / sr) * 0.1f);
        }
    }

    AudioIO.SaveWav(path, new AudioData(samples, sr));
    Console.WriteLine($"  Generated {totalSeconds}s test audio → {Path.GetFileName(path)}");
}

static void RunSyntheticDemo()
{
    Console.WriteLine("=== Synthetic VAD Demo ===");
    Console.WriteLine("Demonstrates AudioData primitives and VAD concepts.\n");

    int sr = 16000;
    var samples = new float[sr * 4];
    for (int i = 0; i < samples.Length; i++)
    {
        float t = (float)i / sr;
        if (t >= 1.0f && t < 3.0f)
            samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sr) * 0.5f;
    }

    var audio = new AudioData(samples, sr);
    Console.WriteLine($"  Generated audio: {audio.Duration.TotalSeconds:F1}s");
    Console.WriteLine("  Pattern: [silence 0-1s] [tone 1-3s] [silence 3-4s]");
    Console.WriteLine("  (With a real Silero VAD model, this would detect the 1-3s segment as speech)");
    Console.WriteLine("\nSynthetic demo complete!");
}

class AudioInput
{
    public float[] Audio { get; set; } = [];
}
