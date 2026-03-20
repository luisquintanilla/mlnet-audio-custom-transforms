using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ======================================================================
// Speech-to-Text Sample — Provider-Agnostic ASR with MEAI Integration
// ======================================================================
// This sample demonstrates the ISpeechToTextClient abstraction and how
// all three ASR paths plug into the MEAI ecosystem:
//   1. Provider-agnostic (any ISpeechToTextClient — Azure, OpenAI, etc.)
//   2. ORT GenAI Whisper (MLNet.ASR.OnnxGenAI package)
//   3. Raw ONNX Whisper (this package — full control, no ORT GenAI)
//
// All three support:
//   - SpeechToTextClientBuilder middleware (logging, telemetry, config)
//   - AddSpeechToTextClient() DI registration
//   - Rich SpeechToTextResponse (timestamps, model ID, segments)
//   - Streaming with SpeechToTextResponseUpdateKind lifecycle
// ======================================================================

Console.WriteLine("=== Speech-to-Text — MEAI ISpeechToTextClient Integration ===\n");

// ── 1. Three ISpeechToTextClient Implementations ──

Console.WriteLine("--- 1. Three ISpeechToTextClient Implementations ---\n");
Console.WriteLine("""
    // Option A: Provider-agnostic (Azure, OpenAI, any cloud or local provider)
    ISpeechToTextClient sttClient = new AzureSpeechToTextClient(endpoint, key);
    // or: new OpenAIClient(apiKey).GetAudioClient("whisper-1").AsISpeechToTextClient();

    // Option B: Local Whisper via ORT GenAI (easiest local option)
    ISpeechToTextClient sttClient = new OnnxSpeechToTextClient(new OnnxSpeechToTextOptions
    {
        ModelPath = "models/whisper-base",
        Language = "en"
    });

    // Option C: Local Whisper via raw ONNX (full control, no ORT GenAI dep)
    ISpeechToTextClient sttClient = new OnnxWhisperSpeechToTextClient(new OnnxWhisperOptions
    {
        EncoderModelPath = "models/whisper-base/encoder_model.onnx",
        DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
        Language = "en"
    });

    // Same API regardless of provider:
    var response = await sttClient.GetTextAsync(audioStream);
    Console.WriteLine(response.Text);
    Console.WriteLine($"Time: {response.StartTime} → {response.EndTime}");
    Console.WriteLine($"Model: {response.ModelId}");
    """);

// ── 2. SpeechToTextClientBuilder — Middleware Pipeline ──

Console.WriteLine("\n--- 2. MEAI Middleware Pipeline ---\n");
Console.WriteLine("""
    // Wrap any ISpeechToTextClient with logging and telemetry:
    ISpeechToTextClient client = new OnnxWhisperSpeechToTextClient(options)
        .AsBuilder()
        .UseLogging()           // Log invocations and results
        .UseOpenTelemetry()     // OTel tracing spans
        .ConfigureOptions(opts =>
        {
            opts.SpeechLanguage ??= "en";  // Default language
        })
        .Build();

    var response = await client.GetTextAsync(audioStream);
    // → Logged, traced, with default language applied
    """);

// ── 3. DI Registration with AddSpeechToTextClient ──

Console.WriteLine("\n--- 3. Dependency Injection ---\n");
Console.WriteLine("""
    // Register with middleware pipeline in Program.cs:
    builder.Services.AddSpeechToTextClient(
        new OnnxSpeechToTextClient(sttOptions))
        .UseLogging()
        .UseOpenTelemetry();

    // Or with a factory:
    builder.Services.AddSpeechToTextClient(sp =>
        new OnnxWhisperSpeechToTextClient(whisperOptions))
        .UseLogging();

    // Keyed registration for multiple providers:
    builder.Services.AddKeyedSpeechToTextClient("local",
        new OnnxSpeechToTextClient(sttOptions));
    builder.Services.AddKeyedSpeechToTextClient("cloud",
        azureSpeechClient);
    """);

// ── 4. Streaming with Update Kinds ──

Console.WriteLine("\n--- 4. Streaming with SpeechToTextResponseUpdateKind ---\n");
Console.WriteLine("""
    await foreach (var update in client.GetStreamingTextAsync(audioStream))
    {
        switch (update.Kind)
        {
            case var k when k == SpeechToTextResponseUpdateKind.SessionOpen:
                Console.WriteLine("▶ Session started");
                break;
            case var k when k == SpeechToTextResponseUpdateKind.TextUpdated:
                Console.WriteLine($"  [{update.StartTime} → {update.EndTime}] {update.Text}");
                break;
            case var k when k == SpeechToTextResponseUpdateKind.SessionClose:
                Console.WriteLine("■ Session ended");
                break;
        }
    }

    // Or collect into a single response:
    // (If reusing a stream from a previous call, reset position first:
    //  audioStream.Position = 0;)
    var response = await client.GetStreamingTextAsync(audioStream)
        .ToSpeechToTextResponseAsync();
    """);

// ── 5. ML.NET Pipeline Integration ──

Console.WriteLine("\n--- 5. ML.NET Pipeline ---\n");
Console.WriteLine("""
    // Any ISpeechToTextClient as an ML.NET pipeline step:
    var pipeline = mlContext.Transforms.SpeechToText(sttClient);

    // Raw ONNX Whisper via ISpeechToTextClient in ML.NET:
    var pipeline = mlContext.Transforms.OnnxWhisperSpeechToText(whisperOptions);

    // Chain with text processing:
    var pipeline = mlContext.Transforms
        .SpeechToText(sttClient)                              // Audio → Text
        .Append(mlContext.Transforms.OnnxTextEmbedding(...))  // Text → Embedding
        .Append(mlContext.Transforms.OnnxTextClassification(...)); // Text → Label
    """);

// ── 6. AudioData + Feature Extraction Demo ──

Console.WriteLine("\n--- 6. AudioData Primitives Demo ---\n");

int sr = 16000;
var samples = new float[sr * 3];
for (int i = 0; i < samples.Length; i++)
    samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / sr) * 0.5f;

var audio = new AudioData(samples, sr);
Console.WriteLine($"Audio: {audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz, {audio.Channels}ch");

var whisperExtractor = new WhisperFeatureExtractor();
var features = whisperExtractor.Extract(audio);
Console.WriteLine($"Whisper features: [{features.GetLength(0)} frames x {features.GetLength(1)} mel bins]");

// ── 7. GetService for Metadata ──

Console.WriteLine("\n--- 7. Provider Metadata via GetService ---\n");
Console.WriteLine("""
    var metadata = client.GetService<SpeechToTextClientMetadata>();
    Console.WriteLine(metadata?.ProviderName);    // "OnnxGenAI-Whisper" or "OnnxWhisper-RawOnnx"
    Console.WriteLine(metadata?.DefaultModelId);  // "whisper-base"
    """);

Console.WriteLine("\nDone! See docs/meai-integration.md for the full MEAI integration guide.");
