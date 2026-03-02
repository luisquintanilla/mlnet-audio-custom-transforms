# Microsoft.Extensions.AI Integration

## Overview

This project integrates with [Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/ai/ai-extensions) (MEAI) to provide standard interfaces for audio AI tasks. MEAI provides abstractions like `IEmbeddingGenerator`, `ISpeechToTextClient`, etc. that enable provider-agnostic AI code.

## What's in MEAI Today vs What We Prototyped

### Available in MEAI 10.x

| Interface | Purpose | Used Here? |
|-----------|---------|------------|
| `IChatClient` | Text generation | No |
| `IEmbeddingGenerator<TInput, TEmbedding>` | Embeddings | Yes — `IEmbeddingGenerator<AudioData, Embedding<float>>` |
| `ISpeechToTextClient` | Speech-to-text (experimental) | Yes |
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
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(new OnnxAudioEmbeddingOptions
    {
        ModelPath = "models/clap/onnx/model.onnx",
        FeatureExtractor = new MelSpectrogramExtractor(16000),
        Pooling = AudioPoolingStrategy.MeanPooling,
        Normalize = true
    });

var audio = AudioIO.LoadWav("speech.wav");
var embeddings = await generator.GenerateAsync([audio]);
var vector = embeddings[0].Vector; // ReadOnlyMemory<float>
```

Works with DI:

```csharp
builder.Services.AddSingleton<IEmbeddingGenerator<AudioData, Embedding<float>>>(
    new OnnxAudioEmbeddingGenerator(options));
```

## Speech-to-Text Client

`OnnxSpeechToTextClient : ISpeechToTextClient` (in the `MLNet.ASR.OnnxGenAI` package)

Implements the standard MEAI `ISpeechToTextClient` interface for local Whisper inference.

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
```

Because the transform accepts any `ISpeechToTextClient`, you can swap providers without changing pipeline code:

```csharp
// Works with ANY ISpeechToTextClient — Azure, OpenAI, local
ISpeechToTextClient client = /* any provider */;
var pipeline = mlContext.Transforms.SpeechToText(client);
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

### Response Types

```csharp
public class TextToSpeechResponse
{
    public AudioData Audio { get; init; }
    public string? Voice { get; init; }
}

public class TextToSpeechResponseUpdate
{
    public AudioData Audio { get; init; }
    public bool IsFinal { get; init; }
}

public class TextToSpeechOptions
{
    public string? Voice;
    public float Speed = 1.0f;
    public string? Language;
    public float[]? SpeakerEmbedding;
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

## Dependency Injection Patterns

Register all audio AI services in a typical ASP.NET Core app:

```csharp
// In Program.cs
builder.Services.AddSingleton<IEmbeddingGenerator<AudioData, Embedding<float>>>(
    new OnnxAudioEmbeddingGenerator(embeddingOptions));

builder.Services.AddSingleton<ISpeechToTextClient>(
    new OnnxSpeechToTextClient(sttOptions));

builder.Services.AddSingleton<ITextToSpeechClient>(
    new OnnxTextToSpeechClient(ttsOptions));

builder.Services.AddSingleton<IVoiceActivityDetector>(
    new OnnxVadTransformer(mlContext, vadOptions));
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

## Middleware / Decorator Pattern

MEAI supports middleware wrapping (caching, logging, etc.) via the builder pattern. This works with our audio embedding generator out of the box:

```csharp
// Logging decorator for audio embeddings
var generator = new OnnxAudioEmbeddingGenerator(options)
    .AsBuilder()
    .UseLogging()
    .Build();
```

## What's Missing in MEAI for Audio

The following capabilities don't exist in MEAI today. We prototyped some of them to explore what audio-first MEAI support could look like:

1. **`ITextToSpeechClient`** — we prototyped it; should be proposed upstream for MEAI.
2. **`IAudioClassifier`** — no MEAI interface for classification tasks (e.g., sound event detection, genre tagging).
3. **`IVoiceActivityDetector`** — very audio-specific; may not fit MEAI's general-purpose scope.
4. **`AudioData` as a first-class type** — MEAI has no audio primitive type; we defined our own.
5. **Streaming audio** — MEAI's streaming patterns work but need audio chunking support for real-time scenarios.
