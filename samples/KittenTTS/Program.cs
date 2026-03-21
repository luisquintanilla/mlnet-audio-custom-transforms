using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ============================================================================
// KittenTTS Text-to-Speech Sample — Local TTS using ONNX + espeak-ng
// ============================================================================
//
// This sample uses OnnxKittenTtsTransformer which provides:
//   1. Text chunking (long text → sentence-boundary chunks)
//   2. Phonemization via espeak-ng subprocess (IPA with stress marks)
//   3. TextCleaner (IPA symbols → token IDs)
//   4. Single ONNX model inference (phoneme IDs + voice embedding → waveform)
//   5. Output as AudioData at 24 kHz
//
// PREREQUISITES:
//   1. Install espeak-ng:
//      Windows: winget install espeak-ng  (or https://github.com/espeak-ng/espeak-ng/releases)
//      Linux:   apt install espeak-ng
//      macOS:   brew install espeak-ng
//
//   2. Download a KittenTTS model from HuggingFace:
//      git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8 models/kittentts
//
//      The model directory should contain:
//        - model.onnx   (the TTS model, ~80 MB for mini)
//        - voices.npz   (voice embeddings for 8 voices)
//
//   3. Run:
//      dotnet run -- "models/kittentts"
//      dotnet run -- "models/kittentts" "Hello, this is KittenTTS!"
//      dotnet run -- "models/kittentts" "Hello!" Luna
//
// Available voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
// Available models:
//   KittenML/kitten-tts-mini-0.8  (80M params, ~80 MB)
//   KittenML/kitten-tts-micro-0.8 (40M params, ~41 MB)
//   KittenML/kitten-tts-nano-0.8  (15M params, ~56 MB)
// ============================================================================

Console.WriteLine("=== KittenTTS Text-to-Speech — ONNX + espeak-ng ===\n");

// --- Resolve model path ---
var modelDir = args.Length > 0 ? args[0] : @"models\kittentts";
// KittenTTS models may be named model.onnx or kitten_tts_*.onnx — auto-detect
var modelPath = Path.Combine(modelDir, "model.onnx");
if (!File.Exists(modelPath))
{
    var onnxFiles = Directory.Exists(modelDir)
        ? Directory.GetFiles(modelDir, "*.onnx")
        : [];
    if (onnxFiles.Length == 1)
        modelPath = onnxFiles[0];
}
var voicesPath = Path.Combine(modelDir, "voices.npz");

if (!File.Exists(modelPath) || !File.Exists(voicesPath))
{
    Console.WriteLine($"Model directory: {Path.GetFullPath(modelDir)}");
    Console.WriteLine($"  *.onnx:      {(File.Exists(modelPath) ? Path.GetFileName(modelPath) : "MISSING")}");
    Console.WriteLine($"  voices.npz:  {(File.Exists(voicesPath) ? "found" : "MISSING")}");
    Console.WriteLine();
    Console.WriteLine("To download the KittenTTS ONNX model:");
    Console.WriteLine("  git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8 models/kittentts");
    Console.WriteLine();
    Console.WriteLine("Also install espeak-ng for phonemization:");
    Console.WriteLine("  Windows: winget install espeak-ng");
    Console.WriteLine("  Linux:   apt install espeak-ng");
    Console.WriteLine("  macOS:   brew install espeak-ng");
    Console.WriteLine();
    Console.WriteLine("Showing API patterns instead...\n");

    ShowApiPatterns();
    return;
}

// --- Setup ---
var mlContext = new MLContext();

var kittenOptions = new OnnxKittenTtsOptions
{
    ModelPath = modelPath,
    VoicesPath = voicesPath,
    DefaultVoice = "Jasper",
    DefaultSpeed = 1.0f,
};

using var transformer = new OnnxKittenTtsTransformer(mlContext, kittenOptions);
Console.WriteLine($"KittenTTS model loaded (single ONNX session)");
Console.WriteLine($"Available voices: {string.Join(", ", transformer.AvailableVoices)}\n");

// --- 1. Basic synthesis ---
Console.WriteLine("--- 1. Direct Synthesis ---\n");

var text = args.Length > 1 ? args[1] : "Hello, this is a text to speech synthesis test using KittenTTS.";
var voice = args.Length > 2 ? args[2] : transformer.AvailableVoices.First();
Console.WriteLine($"  Input: \"{text}\"");
Console.WriteLine($"  Voice: {voice}");

var audio = transformer.Synthesize(text, voice);
Console.WriteLine($"  Output: {audio.Duration.TotalSeconds:F2}s, {audio.SampleRate}Hz, {audio.Samples.Length} samples");

var outputPath = "output.wav";
AudioIO.SaveWav(outputPath, audio);
Console.WriteLine($"  Saved: {outputPath}");

// --- 2. ITextToSpeechClient (official MEAI) ---
Console.WriteLine("\n--- 2. ITextToSpeechClient (MEAI) ---\n");

#pragma warning disable AIEXP001, MEAI001
using var ttsClient = new OnnxTextToSpeechClient(kittenOptions);
var metadata = ttsClient.GetService<TextToSpeechClientMetadata>();
Console.WriteLine($"  Provider: {metadata?.ProviderName}");
Console.WriteLine($"  Model: {metadata?.DefaultModelId}");

var ttsOptions = new TextToSpeechOptions { VoiceId = transformer.AvailableVoices.Skip(1).FirstOrDefault() ?? voice, Speed = 1.0f };
var response = await ttsClient.GetAudioAsync("This is the MEAI client.", ttsOptions);
var audioContent = response.Contents.OfType<DataContent>().FirstOrDefault();
Console.WriteLine($"  Result: {audioContent?.Data.Length ?? 0} WAV bytes, model={response.ModelId}");
#pragma warning restore AIEXP001, MEAI001

// --- 3. Multiple voices ---
Console.WriteLine("\n--- 3. Multiple Voices ---\n");

var voiceNames = transformer.AvailableVoices.Take(4).ToArray();
foreach (var v in voiceNames)
{
    var sample = transformer.Synthesize("Good morning.", v);
    Console.WriteLine($"  {v,-8} → {sample.Duration.TotalSeconds:F2}s, {sample.Samples.Length} samples");
}

// --- 4. ML.NET Pipeline ---
Console.WriteLine("\n--- 4. ML.NET Pipeline ---\n");

var estimator = mlContext.Transforms.KittenTts(kittenOptions);
Console.WriteLine("  Pipeline: Text → espeak-ng → IPA phonemes → Token IDs → ONNX → Audio (24kHz)");
Console.WriteLine("  Can compose with ASR for voice round-trip:");
Console.WriteLine("    var pipeline = mlContext.Transforms");
Console.WriteLine("        .OnnxWhisper(whisperOptions)     // Audio → Text");
Console.WriteLine("        .Append(.KittenTts(ttsOptions)); // Text → Audio");

Console.WriteLine("\n=== Done ===");

// --- Helpers ---

static void ShowApiPatterns()
{
    Console.WriteLine("--- Pattern 1: Direct Synthesis ---");
    Console.WriteLine("""
        var options = new OnnxKittenTtsOptions
        {
            ModelPath = "models/kittentts/model.onnx",
            VoicesPath = "models/kittentts/voices.npz",
        };

        using var transformer = new OnnxKittenTtsTransformer(mlContext, options);
        var audio = transformer.Synthesize("Hello world!", "Jasper", speed: 1.0f);
        AudioIO.SaveWav("output.wav", audio);
    """);

    Console.WriteLine("\n--- Pattern 2: ITextToSpeechClient (official MEAI) ---");
    Console.WriteLine("""
        using var client = new OnnxTextToSpeechClient(kittenOptions);
        var ttsOptions = new TextToSpeechOptions { VoiceId = "Luna", Speed = 1.0f };
        var response = await client.GetAudioAsync("Say something", ttsOptions);
        var audioContent = response.Contents.OfType<DataContent>().First();
        File.WriteAllBytes("output.wav", audioContent.Data!.Value.ToArray());
    """);

    Console.WriteLine("\n--- Pattern 3: ML.NET Pipeline ---");
    Console.WriteLine("""
        var pipeline = mlContext.Transforms.KittenTts(options);
        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);
    """);

    Console.WriteLine("\n--- Pattern 4: All Available Voices ---");
    Console.WriteLine("""
        // KittenTTS ships with 8 built-in voices:
        //   Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
        foreach (var voice in new[] { "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo" })
        {
            var audio = transformer.Synthesize("Hello!", voice);
            AudioIO.SaveWav($"output_{voice}.wav", audio);
        }
    """);
}
