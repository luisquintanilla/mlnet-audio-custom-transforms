# Microsoft.Extensions.AI Integration

## Overview

This project integrates with [Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/ai/ai-extensions) (MEAI) to provide standard interfaces for audio AI tasks. MEAI provides abstractions like `IEmbeddingGenerator`, `ISpeechToTextClient`, etc. that enable provider-agnostic AI code.

## What's in MEAI Today vs What We Prototyped

### Available in MEAI 10.x

| Interface | Purpose | Used Here? |
|-----------|---------|------------|
| `IChatClient` | Text generation | No |
| `IEmbeddingGenerator<TInput, TEmbedding>` | Embeddings | Yes — `IEmbeddingGenerator<AudioData, Embedding<float>>` |
| `ISpeechToTextClient` | Speech-to-text (experimental) | Yes — 3 implementations |
| `IImageGenerator` | Image generation (experimental) | No |

### Prototyped in This Project (Not in MEAI)

| Interface | Purpose | Notes |
|-----------|---------|-------|
| `ITextToSpeechClient` | Text-to-speech | Defined following MEAI patterns |
| `IVoiceActivityDetector` | Voice activity detection | Custom interface |

## Audio Embedding Generator

`OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>`

Uses the generic MEAI embedding interface with `AudioData` as input type. This is the same pattern as text embeddings but with audio input.

```csharp
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = "models/clap/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000),
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true
};

var mlContext = new MLContext();
var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
var transformer = estimator.Fit(emptyData);

IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

var audio = AudioIO.LoadWav("speech.wav");
var embeddings = await generator.GenerateAsync([audio]);
var vector = embeddings[0].Vector; // ReadOnlyMemory<float>
```

Works with DI:

```csharp
builder.Services.AddSingleton<IEmbeddingGenerator<AudioData, Embedding<float>>>(sp =>
{
    var mlContext = new MLContext();
    var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
    var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
    return new OnnxAudioEmbeddingGenerator(transformer);
});
```

## Speech-to-Text Clients

Three `ISpeechToTextClient` implementations — all pluggable into MEAI middleware and DI:

### 1. ORT GenAI Whisper (`MLNet.ASR.OnnxGenAI`)

`OnnxSpeechToTextClient : ISpeechToTextClient` — easiest local Whisper option.

```csharp
using Microsoft.Extensions.AI;
using MLNet.ASR.OnnxGenAI;

ISpeechToTextClient client = new OnnxSpeechToTextClient(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-base",
    Language = "en"
});

using var stream = File.OpenRead("speech.wav");
var response = await client.GetTextAsync(stream);
Console.WriteLine(response.Text);
Console.WriteLine($"Time: {response.StartTime} → {response.EndTime}");
Console.WriteLine($"Model: {response.ModelId}");
```

### 2. Raw ONNX Whisper (`MLNet.AudioInference.Onnx`)

`OnnxWhisperSpeechToTextClient : ISpeechToTextClient` — full control, no ORT GenAI dependency. Uses standard HuggingFace optimum-exported ONNX models with manual encoder-decoder-KV cache management.

```csharp
using Microsoft.Extensions.AI;
using MLNet.AudioInference.Onnx;

ISpeechToTextClient client = new OnnxWhisperSpeechToTextClient(new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-base/encoder_model.onnx",
    DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
    Language = "en"
});

using var stream = File.OpenRead("speech.wav");
var response = await client.GetTextAsync(stream);
Console.WriteLine(response.Text);

// Streaming with per-segment timestamps
stream.Position = 0; // Rewind before reusing the stream
await foreach (var update in client.GetStreamingTextAsync(stream))
{
    switch (update.Kind)
    {
        case var k when k == SpeechToTextResponseUpdateKind.SessionOpen:
            Console.WriteLine("Session started");
            break;
        case var k when k == SpeechToTextResponseUpdateKind.TextUpdated:
            Console.WriteLine($"[{update.StartTime} → {update.EndTime}] {update.Text}");
            break;
        case var k when k == SpeechToTextResponseUpdateKind.SessionClose:
            Console.WriteLine("Session ended");
            break;
    }
}
```

### 3. Provider-Agnostic Wrapper (ML.NET Pipeline)

`SpeechToTextClientTransformer` — wraps **any** `ISpeechToTextClient` as an ML.NET pipeline step.

```csharp
// Works with ANY ISpeechToTextClient — Azure, OpenAI, local
ISpeechToTextClient client = /* any provider */;
var pipeline = mlContext.Transforms.SpeechToText(client);

// Or use the raw ONNX Whisper via ISpeechToTextClient in the ML.NET pipeline:
var pipeline = mlContext.Transforms.OnnxWhisperSpeechToText(whisperOptions);
```

### Choosing the Right Client

| Client | When to Use |
|--------|-------------|
| `OnnxSpeechToTextClient` | Easiest local Whisper (ORT GenAI handles decode) |
| `OnnxWhisperSpeechToTextClient` | Full control, no ORT GenAI dep, manual KV cache |
| Any `ISpeechToTextClient` via `SpeechToText()` | Cloud providers (Azure, OpenAI), custom implementations |

## SpeechToTextClientBuilder — Middleware Pipeline

MEAI provides `SpeechToTextClientBuilder` for composing middleware around any `ISpeechToTextClient`, just like `ChatClientBuilder` for chat:

```csharp
using Microsoft.Extensions.AI;
using MLNet.AudioInference.Onnx;

// Create a client with logging and telemetry middleware
ISpeechToTextClient client = new OnnxWhisperSpeechToTextClient(whisperOptions)
    .AsBuilder()
    .UseLogging()           // Log invocations and results
    .UseOpenTelemetry()     // OTel tracing spans
    .ConfigureOptions(opts =>
    {
        // Set default language for all requests
        opts.SpeechLanguage ??= "en";
    })
    .Build();

var response = await client.GetTextAsync(audioStream);
// → Logged, traced, with default language applied
```

Available middleware (from `Microsoft.Extensions.AI`):

| Middleware | Extension Method | What It Does |
|-----------|-----------------|-------------|
| Logging | `.UseLogging()` | Logs invocations at Info, responses at Debug, sensitive data at Trace |
| OpenTelemetry | `.UseOpenTelemetry()` | Emits spans following OTel GenAI Semantic Conventions |
| ConfigureOptions | `.ConfigureOptions(callback)` | Mutates `SpeechToTextOptions` on each request |
| Custom | `.Use(inner => new MyClient(inner))` | Any custom `DelegatingSpeechToTextClient` |

## Dependency Injection Patterns

### Using `AddSpeechToTextClient()` (Recommended)

MEAI provides builder-based DI registration that supports middleware pipelines:

```csharp
// In Program.cs — registers ISpeechToTextClient with middleware
builder.Services.AddSpeechToTextClient(
    new OnnxSpeechToTextClient(sttOptions))
    .UseLogging()
    .UseOpenTelemetry();

// Or with a factory for deferred construction
builder.Services.AddSpeechToTextClient(sp =>
    new OnnxWhisperSpeechToTextClient(whisperOptions))
    .UseLogging();

// Keyed registration for multiple providers
builder.Services.AddKeyedSpeechToTextClient("local",
    new OnnxSpeechToTextClient(sttOptions));
builder.Services.AddKeyedSpeechToTextClient("cloud",
    azureSpeechClient);
```

### Full Audio AI Service Registration

```csharp
// Register all audio AI services
builder.Services.AddSingleton<IEmbeddingGenerator<AudioData, Embedding<float>>>(sp =>
{
    var mlContext = new MLContext();
    var estimator = mlContext.Transforms.OnnxAudioEmbedding(embeddingOptions);
    var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
    return new OnnxAudioEmbeddingGenerator(transformer);
});

builder.Services.AddSpeechToTextClient(
    new OnnxSpeechToTextClient(sttOptions))
    .UseLogging()
    .UseOpenTelemetry();

builder.Services.AddSingleton<ITextToSpeechClient>(
    new OnnxTextToSpeechClient(ttsOptions));
```

Then inject wherever needed:

```csharp
public class TranscriptionService(ISpeechToTextClient sttClient)
{
    public async Task<string> TranscribeAsync(Stream audio)
    {
        var response = await sttClient.GetTextAsync(audio);
        return response.Text;
    }
}
```

## SpeechToTextOptions & Response Richness

### Options Passed to Providers

The MEAI `SpeechToTextOptions` type carries language, sample rate, and model information. Our clients honor some fields at request time and others at construction time:

```csharp
var options = new SpeechToTextOptions
{
    SpeechLanguage = "es",           // Set at construction time via options class (not per-request)
    TextLanguage = "en",             // Set at construction time via options class (not per-request)
    SpeechSampleRate = 16000,        // ✅ Honored at request time — auto-resamples audio
    ModelId = "whisper-large-v3",    // ✅ Honored at request time — used in response metadata
};

var response = await client.GetTextAsync(audioStream, options);
```

> **Note:** `SpeechLanguage` and `TextLanguage` are configured when creating the client via
> `OnnxWhisperOptions.Language` / `OnnxSpeechToTextOptions.Language` and `Translate`. These fields
> on `SpeechToTextOptions` are not consumed at request time by the current local clients.
> Cloud providers (Azure, OpenAI) may honor them per-request.

### Rich Response Metadata

Our clients populate the full `SpeechToTextResponse` with timestamps and metadata:

```csharp
var response = await client.GetTextAsync(audioStream);

Console.WriteLine(response.Text);                    // Full transcription
Console.WriteLine(response.StartTime);               // First segment start
Console.WriteLine(response.EndTime);                  // Last segment end
Console.WriteLine(response.ModelId);                  // "whisper-base"

// Segment-level timestamps in AdditionalProperties
var segments = response.AdditionalProperties?["segments"];
var language = response.AdditionalProperties?["language"];
```

### GetTextAsync with DataContent

MEAI provides a convenience overload accepting `DataContent` (in-memory audio bytes):

```csharp
var audioContent = new DataContent(audioBytes, "audio/wav");
var response = await client.GetTextAsync(audioContent);
```

## Streaming with Update Kinds

Our `ISpeechToTextClient` implementations use the full MEAI streaming lifecycle:

```csharp
await foreach (var update in client.GetStreamingTextAsync(audioStream))
{
    Console.WriteLine($"Kind: {update.Kind}");

    if (update.Kind == SpeechToTextResponseUpdateKind.SessionOpen)
        Console.WriteLine("▶ Transcription started");

    if (update.Kind == SpeechToTextResponseUpdateKind.TextUpdated)
        Console.WriteLine($"  [{update.StartTime} → {update.EndTime}] {update.Text}");

    if (update.Kind == SpeechToTextResponseUpdateKind.SessionClose)
        Console.WriteLine("■ Transcription complete");
}

// Or collect all updates into a single response:
var response = await client.GetStreamingTextAsync(audioStream)
    .ToSpeechToTextResponseAsync();
Console.WriteLine(response.Text);
```

### Update Kind Lifecycle

| Kind | When Emitted |
|------|-------------|
| `SessionOpen` | Start of transcription session |
| `TextUpdating` | Intermediate result (partial, may change) |
| `TextUpdated` | Final segment result (per Whisper timestamp segment) |
| `SessionClose` | End of transcription session |
| `Error` | Non-blocking error during transcription |

## Metadata via GetService

Both clients support `GetService` for metadata retrieval:

```csharp
var metadata = client.GetService<SpeechToTextClientMetadata>();
Console.WriteLine(metadata?.ProviderName);    // "OnnxGenAI-Whisper" or "OnnxWhisper-RawOnnx"
Console.WriteLine(metadata?.DefaultModelId);  // "whisper-base"
```

## Text-to-Speech Client (Prototype)

`OnnxTextToSpeechClient : ITextToSpeechClient`

> **Note:** MEAI does **not** have `ITextToSpeechClient` as of 10.x. We defined one following MEAI patterns to prototype what it might look like.

### Interface

```csharp
public interface ITextToSpeechClient : IDisposable
{
    TextToSpeechClientMetadata Metadata { get; }

    Task<TextToSpeechResponse> GetAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);
}
```

### Usage

```csharp
using var client = new OnnxTextToSpeechClient(new OnnxSpeechT5Options { ... });

// One-shot
var response = await client.GetAudioAsync("Hello, world!");
AudioIO.SaveWav("output.wav", response.Audio);

// Streaming
await foreach (var update in client.GetStreamingAudioAsync("Hello, world!"))
{
    PlayAudio(update.Audio);
    if (update.IsFinal) break;
}
```

## Voice Activity Detector (Custom Interface)

`IVoiceActivityDetector` — custom interface (no MEAI equivalent exists).

```csharp
public interface IVoiceActivityDetector
{
    IAsyncEnumerable<SpeechSegment> DetectSpeechAsync(
        Stream audioStream,
        VadOptions? options = null,
        CancellationToken cancellationToken = default);
}

public record SpeechSegment(TimeSpan Start, TimeSpan End, float Confidence);
```

## What's Missing in MEAI for Audio

The following capabilities don't exist in MEAI today. We prototyped some of them to explore what audio-first MEAI support could look like:

1. **`ITextToSpeechClient`** — we prototyped it; should be proposed upstream for MEAI.
2. **`IAudioClassifier`** — no MEAI interface for classification tasks (e.g., sound event detection, genre tagging).
3. **`IVoiceActivityDetector`** — very audio-specific; may not fit MEAI's general-purpose scope.
4. **`AudioData` as a first-class type** — MEAI has no audio primitive type; we defined our own.
5. **Streaming audio** — MEAI's streaming patterns work but need audio chunking support for real-time scenarios.

## DataIngestion Integration

The MEAI `IEmbeddingGenerator<AudioData, Embedding<float>>` interface is the bridge between DataIngestion and ML.NET:

```
DataIngestion Layer:  AudioDocumentReader → AudioSegmentChunker → AudioEmbeddingChunkProcessor
                                                                         │
                                                                         ▼
MEAI Layer:                                        IEmbeddingGenerator<AudioData, Embedding<float>>
                                                                         │
                                                                         ▼
ML.NET Layer:                                      OnnxAudioEmbeddingGenerator → OnnxAudioEmbeddingTransformer
```

`AudioEmbeddingChunkProcessor` takes an `IEmbeddingGenerator<AudioData, Embedding<float>>` in its constructor — it doesn't know or care whether the embeddings come from CLAP, Wav2Vec2, or any other model. This proves that `Microsoft.Extensions.DataIngestion` is modality-agnostic: the same Reader → Chunker → Processor pattern works for text, PDF, and audio.

```csharp
// Wire up the 3-layer bridge
var mlContext = new MLContext();
var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var transformer = estimator.Fit(emptyData);

// Layer 2: MEAI — wraps ML.NET transformer
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

// Layer 3: DataIngestion — uses MEAI generator
var processor = new AudioEmbeddingChunkProcessor(generator);
```

See `samples/AudioDataIngestion/` for the full end-to-end pipeline.
