using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Speech-to-Text Sample — Provider-Agnostic ASR
// ======================================================================
// This sample demonstrates the SpeechToTextClientTransformer which wraps
// any ISpeechToTextClient (Azure Speech, OpenAI Whisper API, local, etc.)
// as an ML.NET pipeline step.
//
// The provider-agnostic pattern means you can swap providers without
// changing your ML.NET pipeline code — just change the ISpeechToTextClient.
//
// For local ONNX Whisper inference, two options:
// 1. MLNet.ASR.OnnxGenAI — easiest, uses ORT GenAI (separate package)
// 2. OnnxWhisperTransformer — raw ONNX, full control (in this package)
// ======================================================================

Console.WriteLine("=== Speech-to-Text — Provider-Agnostic ML.NET Transform ===\n");
Console.WriteLine("This sample shows how the SpeechToTextClientTransformer works.");
Console.WriteLine("It wraps any ISpeechToTextClient as an ML.NET pipeline step.\n");

// Demonstrate the pattern (without a real provider)
Console.WriteLine("--- Example: Pipeline Construction ---\n");
Console.WriteLine("""
    // With Azure Speech:
    ISpeechToTextClient sttClient = new AzureSpeechToTextClient(endpoint, key);

    // With OpenAI Whisper API:
    ISpeechToTextClient sttClient = new OpenAIClient(apiKey)
        .GetAudioClient("whisper-1")
        .AsISpeechToTextClient();

    // With local ONNX Whisper (future MLNet.ASR.OnnxGenAI):
    ISpeechToTextClient sttClient = new OnnxSpeechToTextClient("models/whisper-base");

    // Same ML.NET pipeline regardless of provider:
    var pipeline = mlContext.Transforms.SpeechToText(sttClient, new SpeechToTextClientOptions
    {
        SpeechLanguage = "en",
        InputColumnName = "Audio",
        OutputColumnName = "Text"
    });

    var model = pipeline.Fit(data);
    var results = model.Transform(data);
    """);

// Demonstrate AudioData + WAV I/O
Console.WriteLine("\n--- AudioData Primitives Demo ---\n");

int sr = 16000;
var samples = new float[sr * 3];
for (int i = 0; i < samples.Length; i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sr) * 0.5f;

var audio = new AudioData(samples, sr);
Console.WriteLine($"Audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz, {audio.Channels}ch");

// Whisper feature extraction demo
var whisperExtractor = new WhisperFeatureExtractor();
var features = whisperExtractor.Extract(audio);
Console.WriteLine($"Whisper features: [{features.GetLength(0)} frames x {features.GetLength(1)} mel bins]");
Console.WriteLine($"  (Whisper expects 3000 frames x 80 mel bins for 30s input)");

// Show the raw ONNX Whisper approach
Console.WriteLine("\n--- Raw ONNX Whisper (Full Control) ---\n");
Console.WriteLine("""
    // Uses standard HuggingFace optimum-exported ONNX models.
    // No ORT GenAI needed — manages encoder, decoder, KV cache manually.
    // Maximum control + uses ALL our primitives (WhisperFeatureExtractor,
    // WhisperTokenizer, TensorPrimitives).
    //
    // Export model:
    //   optimum-cli export onnx --model openai/whisper-base output_dir/
    //
    // Direct API:
    var transformer = new OnnxWhisperTransformer(mlContext, new OnnxWhisperOptions
    {
        EncoderModelPath = "models/whisper-base/encoder_model.onnx",
        DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
        Language = "en"
    });
    var results = transformer.Transcribe([audioData]);

    // With timestamps:
    var detailed = transformer.TranscribeWithTimestamps([audioData]);
    foreach (var seg in detailed[0].Segments)
        Console.WriteLine($"[{seg.Start:mm\\:ss} → {seg.End:mm\\:ss}] {seg.Text}");

    // ML.NET Pipeline:
    var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions { ... });
    """);

// Show the full ML.NET pipeline chain concept
Console.WriteLine("\n--- Multi-Modal Pipeline (STT → Text Embeddings) ---\n");
Console.WriteLine("""
    // Chain audio-to-text with text processing:
    var pipeline = mlContext.Transforms
        .SpeechToText(sttClient)                           // Audio → Text
        .Append(mlContext.Transforms.OnnxTextEmbedding(...)) // Text → Embedding
        .Append(mlContext.Transforms.OnnxTextClassification(...)); // Text → Classification

    // Or: STT + Sentiment Analysis
    var pipeline = mlContext.Transforms
        .SpeechToText(sttClient)                           // Audio → Text
        .Append(mlContext.Transforms.OnnxTextClassification(new() {
            ModelPath = "sentiment-model.onnx",
            Labels = new[] { "Positive", "Negative", "Neutral" }
        }));
    """);

Console.WriteLine("\nDone! See docs/plan.md for the full roadmap.");
