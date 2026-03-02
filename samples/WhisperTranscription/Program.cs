using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.ASR.OnnxGenAI;

// ============================================================================
// Whisper Transcription Sample — Local speech-to-text using ORT GenAI
// ============================================================================
//
// PREREQUISITES:
//   1. Download a Whisper ONNX model in ORT GenAI format:
//      - Use Olive or Hugging Face Optimum to export Whisper to ORT GenAI format
//      - Or download pre-exported from: https://huggingface.co/models?search=whisper+onnx+genai
//
//   2. Model directory should contain:
//      - genai_config.json
//      - encoder_model.onnx (or encoder_model_merged.onnx)
//      - decoder_model_merged.onnx
//      - tokenizer.json, tokenizer_config.json
//
//   3. Add a platform-specific ORT GenAI native package to this project:
//      dotnet add package Microsoft.ML.OnnxRuntimeGenAI         # CPU
//      dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda    # NVIDIA GPU
//      dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML # Windows GPU
//
//   4. Set the model path below or pass as a command-line argument:
//      dotnet run -- "path/to/whisper-model"
//
// ============================================================================

Console.WriteLine("=== Whisper Transcription Sample ===\n");

// --- Model path ---
var modelPath = args.Length > 0 ? args[0] : @"models\whisper-base";

if (!Directory.Exists(modelPath))
{
    Console.WriteLine($"Model directory not found: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("To use this sample:");
    Console.WriteLine("  1. Export a Whisper model to ORT GenAI format (e.g., using Olive)");
    Console.WriteLine("  2. Place the model directory at: " + Path.GetFullPath(modelPath));
    Console.WriteLine("  3. Run: dotnet run -- \"path/to/whisper-model\"");
    Console.WriteLine();
    Console.WriteLine("Showing API patterns instead...\n");

    ShowApiPatterns();
    return;
}

// --- Direct API usage ---
Console.WriteLine("1. Direct API — OnnxSpeechToTextTransformer\n");

var mlContext = new MLContext();

var options = new OnnxSpeechToTextOptions
{
    ModelPath = modelPath,
    Language = "en",
    MaxLength = 256,
    NumMelBins = 80, // 80 for whisper-base/small, 128 for whisper-large-v3
};

using var transformer = new OnnxSpeechToTextTransformer(mlContext, options);

// Load audio (WAV file at 16kHz mono)
var audioPath = args.Length > 1 ? args[1] : "test.wav";
if (File.Exists(audioPath))
{
    var audio = AudioIO.LoadWav(audioPath);

    // Resample to 16kHz if needed
    if (audio.SampleRate != 16000)
        audio = AudioIO.Resample(audio, 16000);

    Console.WriteLine($"  Audio: {audioPath} ({audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz)");

    // Basic transcription
    var results = transformer.Transcribe([audio]);
    Console.WriteLine($"  Transcription: {results[0]}\n");

    // Transcription with timestamps
    var detailed = transformer.TranscribeWithTimestamps([audio]);
    Console.WriteLine($"  Full text: {detailed[0].Text}");
    Console.WriteLine($"  Segments:");
    foreach (var segment in detailed[0].Segments)
    {
        Console.WriteLine($"    [{segment.Start:mm\\:ss\\.ff} → {segment.End:mm\\:ss\\.ff}] {segment.Text}");
    }
    Console.WriteLine();
}
else
{
    // Generate synthetic audio for demo
    Console.WriteLine("  (No test.wav found — generating synthetic audio for demo)");
    var syntheticAudio = GenerateSyntheticAudio();

    var results = transformer.Transcribe([syntheticAudio]);
    Console.WriteLine($"  Transcription: {results[0]}\n");
}

// --- MEAI ISpeechToTextClient usage ---
Console.WriteLine("2. MEAI ISpeechToTextClient — Provider-agnostic API\n");

using var sttClient = new OnnxSpeechToTextClient(options);
Console.WriteLine($"  Provider: {sttClient.Metadata.ProviderName}");
Console.WriteLine($"  Model: {sttClient.Metadata.DefaultModelId}");

if (File.Exists(audioPath))
{
    using var audioStream = File.OpenRead(audioPath);
    var sttResponse = await sttClient.GetTextAsync(audioStream);
    Console.WriteLine($"  Result: {sttResponse.Text}\n");
}
Console.WriteLine();

// --- ML.NET Pipeline usage ---
Console.WriteLine("3. ML.NET Pipeline — Composable transforms\n");

var estimator = mlContext.Transforms.OnnxSpeechToText(options);
Console.WriteLine("  Pipeline created: Audio → Mel Spectrogram → Whisper Encoder → Decoder → Text");
Console.WriteLine("  Can be composed with text transforms:");
Console.WriteLine("    pipeline.Append(mlContext.Transforms.OnnxTextEmbedding(...))");
Console.WriteLine("    pipeline.Append(mlContext.Transforms.OnnxTextClassification(...))");
Console.WriteLine();

Console.WriteLine("=== Done ===");

// --- Helper methods ---

static AudioData GenerateSyntheticAudio()
{
    // Generate 3 seconds of silence + simple tone (as a demo input)
    int sampleRate = 16000;
    int duration = 3;
    var samples = new float[sampleRate * duration];

    for (int i = 0; i < samples.Length; i++)
    {
        float t = (float)i / sampleRate;
        // Simple 440Hz tone
        samples[i] = 0.3f * MathF.Sin(2 * MathF.PI * 440 * t);
    }

    return new AudioData(samples, sampleRate);
}

static void ShowApiPatterns()
{
    Console.WriteLine("--- API Pattern 1: Direct Transcription ---");
    Console.WriteLine(@"
    var options = new OnnxSpeechToTextOptions
    {
        ModelPath = ""models/whisper-base"",
        Language = ""en"",
        MaxLength = 256
    };

    using var transformer = new OnnxSpeechToTextTransformer(mlContext, options);
    var audio = AudioIO.LoadWav(""speech.wav"");
    var results = transformer.Transcribe([audio]);
    Console.WriteLine(results[0]);
    ");

    Console.WriteLine("--- API Pattern 2: MEAI ISpeechToTextClient ---");
    Console.WriteLine(@"
    using var client = new OnnxSpeechToTextClient(options);
    using var stream = File.OpenRead(""speech.wav"");
    var response = await client.GetTextAsync(stream);
    Console.WriteLine(response.Text);
    ");

    Console.WriteLine("--- API Pattern 3: ML.NET Pipeline ---");
    Console.WriteLine(@"
    var pipeline = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
    {
        ModelPath = ""models/whisper-base"",
        Language = ""en""
    });
    var model = pipeline.Fit(trainingData);
    var predictions = model.Transform(testData);
    ");

    Console.WriteLine("--- API Pattern 4: Timestamps ---");
    Console.WriteLine(@"
    var detailed = transformer.TranscribeWithTimestamps([audio]);
    foreach (var segment in detailed[0].Segments)
    {
        Console.WriteLine($""[{segment.Start} → {segment.End}] {segment.Text}"");
    }
    ");
}
