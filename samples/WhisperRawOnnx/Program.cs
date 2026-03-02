using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ============================================================================
// Raw ONNX Whisper Sample — Full-control local speech-to-text
// ============================================================================
//
// This sample uses OnnxWhisperTransformer which manages the FULL encoder-decoder
// pipeline manually — no ORT GenAI dependency needed. It showcases every stage:
//   1. WhisperFeatureExtractor → mel spectrogram (our audio primitive)
//   2. Encoder ONNX session → hidden states
//   3. Decoder loop with WhisperKvCacheManager → token IDs
//   4. TensorPrimitives.IndexOfMax / SoftMax → token sampling
//   5. WhisperTokenizer → text with timestamps (our tokenizer primitive)
//
// PREREQUISITES:
//   1. Export a Whisper model with HuggingFace Optimum:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model openai/whisper-base models/whisper-base/
//
//   2. The exported directory should contain:
//      - encoder_model.onnx
//      - decoder_model_merged.onnx  (handles both prefill and decode-with-past)
//      - config.json, tokenizer.json, etc.
//
//   3. Run:
//      dotnet run -- "models/whisper-base"
//      dotnet run -- "models/whisper-base" "audio.wav"
//
// ============================================================================

Console.WriteLine("=== Raw ONNX Whisper — Full-Control Speech-to-Text ===\n");

// --- Resolve model path ---
var modelDir = args.Length > 0 ? args[0] : @"models\whisper-base";
var encoderPath = Path.Combine(modelDir, "encoder_model.onnx");
var decoderPath = Path.Combine(modelDir, "decoder_model_merged.onnx");

if (!File.Exists(encoderPath) || !File.Exists(decoderPath))
{
    Console.WriteLine($"Model directory: {Path.GetFullPath(modelDir)}");
    Console.WriteLine($"  encoder_model.onnx: {(File.Exists(encoderPath) ? "found" : "MISSING")}");
    Console.WriteLine($"  decoder_model_merged.onnx: {(File.Exists(decoderPath) ? "found" : "MISSING")}");
    Console.WriteLine();
    Console.WriteLine("To export a Whisper model with HuggingFace Optimum:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model openai/whisper-base models/whisper-base/");
    Console.WriteLine();
    Console.WriteLine("Showing API patterns instead...\n");

    ShowApiPatterns();
    return;
}

// --- Setup ---
var mlContext = new MLContext();

var options = new OnnxWhisperOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    Language = "en",
    MaxTokens = 256,
    NumMelBins = 80,   // 80 for whisper-base/small, 128 for whisper-large-v3
    Temperature = 0f,  // 0 = greedy (TensorPrimitives.IndexOfMax), >0 = temperature sampling
};

using var transformer = new OnnxWhisperTransformer(mlContext, options);
Console.WriteLine("Whisper model loaded (raw ONNX — encoder + decoder + KV cache)\n");

// --- Load or generate audio ---
var audioPath = args.Length > 1 ? args[1] : "test.wav";

AudioData audio;
if (File.Exists(audioPath))
{
    audio = AudioIO.LoadWav(audioPath);
    if (audio.SampleRate != 16000)
        audio = AudioIO.Resample(audio, 16000);

    Console.WriteLine($"Audio: {audioPath} ({audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz, {audio.Channels}ch)");
}
else
{
    Console.WriteLine("(No test.wav found — using synthetic 3s tone for demo)");
    audio = GenerateSyntheticAudio();
}

// --- 1. Basic transcription ---
Console.WriteLine("\n--- 1. Direct Transcription ---\n");

var results = transformer.Transcribe([audio]);
Console.WriteLine($"  Text: {results[0]}");

// --- 2. Transcription with timestamps ---
Console.WriteLine("\n--- 2. Transcription with Timestamps ---\n");

var detailed = transformer.TranscribeWithTimestamps([audio]);
Console.WriteLine($"  Full text: {detailed[0].Text}");
Console.WriteLine($"  Language:  {detailed[0].Language}");
Console.WriteLine($"  Tokens:    {detailed[0].TokenIds.Length}");

if (detailed[0].Segments.Length > 0)
{
    Console.WriteLine("  Segments:");
    foreach (var seg in detailed[0].Segments)
        Console.WriteLine($"    [{seg.Start:mm\\:ss\\.ff} → {seg.End:mm\\:ss\\.ff}] {seg.Text}");
}

// --- 3. ML.NET Pipeline ---
Console.WriteLine("\n--- 3. ML.NET Pipeline ---\n");

var estimator = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    Language = "en",
});

Console.WriteLine("  Pipeline: Audio → WhisperFeatureExtractor → Encoder → Decoder (KV Cache) → WhisperTokenizer → Text");
Console.WriteLine("  Can be composed:");
Console.WriteLine("    pipeline.Append(mlContext.Transforms.OnnxTextEmbedding(...))");
Console.WriteLine("    pipeline.Append(mlContext.Transforms.OnnxTextClassification(...))");

// --- 4. Batch transcription ---
Console.WriteLine("\n--- 4. Batch Transcription ---\n");

var batch = new[] { audio, audio };
var batchResults = transformer.Transcribe(batch);
Console.WriteLine($"  Transcribed {batchResults.Length} audio inputs:");
for (int i = 0; i < batchResults.Length; i++)
    Console.WriteLine($"    [{i}]: {batchResults[i]}");

Console.WriteLine("\n=== Done ===");

// --- Helpers ---

static AudioData GenerateSyntheticAudio()
{
    int sampleRate = 16000;
    var samples = new float[sampleRate * 3]; // 3 seconds
    for (int i = 0; i < samples.Length; i++)
        samples[i] = 0.3f * MathF.Sin(2 * MathF.PI * 440 * i / sampleRate);
    return new AudioData(samples, sampleRate);
}

static void ShowApiPatterns()
{
    Console.WriteLine("--- Pattern 1: Direct Transcription ---");
    Console.WriteLine("""
        var options = new OnnxWhisperOptions
        {
            EncoderModelPath = "models/whisper-base/encoder_model.onnx",
            DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
            Language = "en"
        };
    
        using var transformer = new OnnxWhisperTransformer(mlContext, options);
        var audio = AudioIO.LoadWav("speech.wav");
        var results = transformer.Transcribe([audio]);
        Console.WriteLine(results[0]);
    """);

    Console.WriteLine("\n--- Pattern 2: Timestamps ---");
    Console.WriteLine("""
        var detailed = transformer.TranscribeWithTimestamps([audio]);
        foreach (var seg in detailed[0].Segments)
            Console.WriteLine($"[{seg.Start:mm\\:ss} → {seg.End:mm\\:ss}] {seg.Text}");
    """);

    Console.WriteLine("\n--- Pattern 3: ML.NET Pipeline ---");
    Console.WriteLine("""
        var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
        {
            EncoderModelPath = "models/whisper-base/encoder_model.onnx",
            DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
            Language = "en"
        });
        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);
    """);

    Console.WriteLine("\n--- Pattern 4: Temperature Sampling ---");
    Console.WriteLine("""
        // Greedy (default): TensorPrimitives.IndexOfMax on logits
        var options = new OnnxWhisperOptions { ..., Temperature = 0f };
    
        // Temperature sampling: TensorPrimitives.SoftMax + multinomial sample
        var options = new OnnxWhisperOptions { ..., Temperature = 0.6f };
    """);

    Console.WriteLine("\n--- Comparison: Three ASR Approaches ---");
    Console.WriteLine("""
        // 1. Provider-agnostic (any ISpeechToTextClient — Azure, OpenAI, etc.)
        var pipeline = mlContext.Transforms.SpeechToText(sttClient);
    
        // 2. ORT GenAI (easiest local, separate MLNet.ASR.OnnxGenAI package)
        var pipeline = mlContext.Transforms.OnnxSpeechToText(ortGenAiOptions);
    
        // 3. Raw ONNX (full control, uses ALL our primitives, THIS sample)
        var pipeline = mlContext.Transforms.OnnxWhisper(rawOnnxOptions);
    """);
}
