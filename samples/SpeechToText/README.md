# Speech-to-Text: Provider Abstraction and Multi-Modal Pipelines

This sample demonstrates the **most important architectural pattern** in the library: provider-agnostic speech-to-text through `ISpeechToTextClient`. No model downloads, no API keys — just pure patterns you can drop into any system.

## What You'll Learn

- **Provider-agnostic architecture** — write pipeline code once, swap providers freely
- **`ISpeechToTextClient`** — the universal contract from `Microsoft.Extensions.AI` for any speech-to-text provider
- **Multi-modal pipeline composition** — chain audio → text → embeddings → classification in a single ML.NET pipeline
- **Whisper feature extraction internals** — what Whisper "sees" when it processes audio (mel spectrograms)

## The Concept: Provider-Agnostic ASR

### Why Provider Abstraction Matters

In production systems, you almost never stay with one speech-to-text provider forever. You might:

- **Develop** with a cheap/fast provider (or a mock) to iterate quickly
- **Test** with a mock client that returns deterministic results for reproducible CI
- **Deploy** with a high-quality cloud provider (Azure Speech, OpenAI Whisper API)
- **Migrate** to local inference (ONNX Whisper) when latency or cost matters

Without abstraction, each switch means rewriting pipeline code, changing data flow, and retesting everything. With abstraction, you change **one line** — the client constructor — and the entire pipeline keeps working.

### `ISpeechToTextClient` — The Universal Contract

[`ISpeechToTextClient`](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.ispeechtotext) is defined in `Microsoft.Extensions.AI` (the same library that gives you `IChatClient` and `IEmbeddingGenerator<T>`). It provides a single, consistent interface that any speech-to-text provider can implement:

```
ISpeechToTextClient
├── Azure Speech Services     → AzureSpeechToTextClient
├── OpenAI Whisper API        → OpenAIClient.GetAudioClient().AsISpeechToTextClient()
├── Local ONNX Whisper        → OnnxSpeechToTextClient (via ORT GenAI)
├── Raw ONNX Whisper          → OnnxWhisperTransformer (manual encoder/decoder)
└── Mock for testing          → new FakeSpeechToTextClient("hello world")
```

### The Pattern: ML.NET Wraps ISpeechToTextClient

The key insight is that `SpeechToTextClientTransformer` doesn't know or care what provider you use. It takes *any* `ISpeechToTextClient` and exposes it as an ML.NET pipeline step:

```
ISpeechToTextClient (any provider)
        ↓
SpeechToTextClientTransformer (ML.NET adapter)
        ↓
ML.NET Pipeline step: IDataView in → IDataView out
```

Your pipeline code is always the same:

```csharp
var pipeline = mlContext.Transforms.SpeechToText(sttClient, new SpeechToTextClientOptions
{
    SpeechLanguage = "en",
    InputColumnName = "Audio",
    OutputColumnName = "Text"
});
```

Swap `sttClient` from Azure → OpenAI → local Whisper → a mock — zero changes to the pipeline.

### Multi-Modal Pipelines: Where Audio Meets Text

The real power emerges when you **compose across modalities**. Speech-to-text produces text. Text can be embedded, classified, or analyzed. ML.NET's `.Append()` chains these seamlessly:

```
Audio (float[]) → STT → Text (string) → Embeddings (float[384]) → Similarity Search
                                       → Classification → Sentiment Label
                                       → TTS → Audio (float[]) — full round-trip
```

This is how real systems work: a call-center pipeline transcribes audio, classifies sentiment, extracts entities, and routes the call — all in one pipeline.

## What This Sample Demonstrates

### 1. Provider-Agnostic `ISpeechToTextClient`

The sample shows how the **same pipeline construction** works with three different providers:

```csharp
// Azure Speech:
ISpeechToTextClient sttClient = new AzureSpeechToTextClient(endpoint, key);

// OpenAI Whisper API:
ISpeechToTextClient sttClient = new OpenAIClient(apiKey)
    .GetAudioClient("whisper-1")
    .AsISpeechToTextClient();

// Local ONNX Whisper:
ISpeechToTextClient sttClient = new OnnxSpeechToTextClient("models/whisper-base");
```

**Why this matters:** Your ML.NET pipeline doesn't change. Not one line. The `SpeechToTextClientTransformer` wraps whichever client you provide and handles the audio-in, text-out contract uniformly.

### 2. Multi-Modal Composition (STT → Text Embeddings → Classification)

The sample demonstrates chaining transforms across modalities:

```csharp
var pipeline = mlContext.Transforms
    .SpeechToText(sttClient)                              // Audio → Text
    .Append(mlContext.Transforms.OnnxTextEmbedding(...))   // Text → Embedding
    .Append(mlContext.Transforms.OnnxTextClassification(...)); // Text → Label
```

**Why this matters:** This is where audio ML and text ML meet. A single pipeline takes raw audio and produces structured output (sentiment labels, topic classifications, embedding vectors for similarity search). No intermediate files, no glue code — just `.Append()`.

### 3. `AudioData` Primitives

The sample creates a synthetic 440 Hz sine wave and demonstrates the `AudioData` type:

```csharp
var audio = new AudioData(samples, sampleRate: 16000);
Console.WriteLine($"{audio.Duration.TotalSeconds:F1}s, {audio.SampleRate}Hz, {audio.Channels}ch");
```

`AudioData` is the universal audio container across the library — mono float PCM normalized to [-1.0, 1.0]. Every transform accepts and produces `AudioData`.

### 4. Whisper Feature Extraction (Mel Spectrograms)

The sample extracts Whisper-compatible features to show what the model "sees":

```csharp
var whisperExtractor = new WhisperFeatureExtractor();
var features = whisperExtractor.Extract(audio);
// → [3000 frames × 80 mel bins]
```

**Why this matters:** Understanding the input representation is critical for debugging and optimization. Whisper expects a **log-mel spectrogram**: 3000 frames (30 seconds at 10ms hop length) × 80 mel frequency bins. Audio shorter than 30 seconds is zero-padded; longer audio is chunked via `ExtractChunked()`.

### 5. Voice Round-Trip Concept (STT → TTS)

The sample illustrates the full audio → text → audio round-trip:

```
Speech recording → ISpeechToTextClient → text → ITextToSpeechClient → synthesized audio
```

**Why this matters:** This demonstrates end-to-end audio processing where the output modality matches the input. Useful for voice translation, accessibility, and voice-cloning pipelines.

## Why No Model Is Required

This sample is **deliberately model-free**. It demonstrates:

- **Pipeline construction patterns** — how to wire up `ISpeechToTextClient` with ML.NET
- **API shape** — what `SpeechToTextClientOptions` controls (language, column names, sample rate)
- **Feature extraction** — real mel spectrogram computation (this part actually runs)
- **Multi-modal composition** — how `.Append()` chains cross modality boundaries

The value is in understanding the **architecture**, not running inference. When you're ready to run a real model, the [WhisperTranscription](../WhisperTranscription/) and [WhisperRawOnnx](../WhisperRawOnnx/) samples provide working inference at two different abstraction levels.

## Code Walkthrough

### `SpeechToTextClientTransformer` — The Adapter

`SpeechToTextClientTransformer` is the bridge between `ISpeechToTextClient` and ML.NET:

```csharp
// Construction: takes any ISpeechToTextClient
var transformer = new SpeechToTextClientTransformer(mlContext, sttClient, options);

// Direct API (outside pipelines):
string[] results = transformer.Transcribe(audioDataList);

// ML.NET pipeline API:
IDataView output = transformer.Transform(inputDataView);
```

It reads `AudioData` from the input column (`"Audio"` by default), calls the `ISpeechToTextClient`, and writes the transcribed text to the output column (`"Text"` by default). The `SpeechToTextClientOptions` controls:

| Property | Default | Purpose |
|---|---|---|
| `InputColumnName` | `"Audio"` | Column containing `AudioData` samples |
| `OutputColumnName` | `"Text"` | Column for transcribed text output |
| `SampleRate` | `16000` | Audio sample rate in Hz |
| `SpeechLanguage` | `null` | Source language (`null` = auto-detect) |
| `TextLanguage` | `null` | Target language for translation |

### Pipeline Composition with `.Append()`

The ML.NET extension methods make pipeline construction fluent:

```csharp
// Single-step: audio → text
var pipeline = mlContext.Transforms.SpeechToText(sttClient);

// Multi-step: audio → text → embeddings
var pipeline = mlContext.Transforms
    .SpeechToText(sttClient)
    .Append(mlContext.Transforms.OnnxTextEmbedding(...));
```

Each `.Append()` connects the output schema of one transform to the input of the next. The `"Text"` column produced by STT flows directly into the text embedding transform — no manual wiring.

### `WhisperFeatureExtractor` — 3000 Frames × 80 Mel Bins

The feature extractor converts raw audio into the log-mel spectrogram Whisper expects:

```
Raw PCM audio (float[], 16kHz)
    → STFT (Short-Time Fourier Transform)
    → Mel filterbank (80 bins)
    → Log scale
    → Pad/truncate to 3000 frames
    → float[3000, 80]
```

- **3000 frames** = 30 seconds of audio at a 10ms hop length (standard Whisper window)
- **80 mel bins** = frequency resolution (use 128 for Whisper v3 via `NumMelBins` option)
- Audio shorter than 30s is **zero-padded** to fill the 3000 frames
- Audio longer than 30s is split via `ExtractChunked()` with optional overlap

### The Three ASR Approaches Compared

This sample sits at the **highest abstraction level**. Here's how the three approaches compare:

| Aspect | `SpeechToText` (this sample) | `WhisperTranscription` | `WhisperRawOnnx` |
|---|---|---|---|
| **Abstraction** | Provider-agnostic | ORT GenAI | Raw ONNX |
| **Model required?** | No (pattern only) | Yes (GenAI format) | Yes (HuggingFace ONNX) |
| **Provider** | Any `ISpeechToTextClient` | Local ONNX only | Local ONNX only |
| **Control level** | Highest-level API | Mid-level API | Full manual control |
| **Manages KV cache?** | Provider handles it | ORT GenAI handles it | You manage it |
| **Best for** | Production pipelines | Easy local inference | Custom decoding, research |
| **Cloud support?** | ✅ Azure, OpenAI, etc. | ❌ Local only | ❌ Local only |

**The key insight:** all three approaches can be consumed through the same `ISpeechToTextClient` interface. `OnnxSpeechToTextClient` wraps ORT GenAI, `OnnxWhisperSpeechToTextClient` wraps raw ONNX — your pipeline code stays identical regardless of which backend you choose.

## Key Takeaways

1. **Provider abstraction is the most important pattern for production systems.** Write your pipeline once against `ISpeechToTextClient`, then swap Azure → OpenAI → local Whisper → a mock without changing pipeline code.

2. **ML.NET pipelines compose across modalities.** Audio → text → vectors → labels in a single `.Append()` chain. No intermediate files, no glue code.

3. **The same `ISpeechToTextClient` interface works with cloud AND local providers.** Azure Speech and a local ONNX Whisper model both implement the same contract.

4. **This sample + WhisperTranscription + WhisperRawOnnx show the same task at three abstraction levels** — from "I don't care how it works" (this sample) to "I control every token" (WhisperRawOnnx).

## Going Further

| Sample | What It Adds |
|---|---|
| [WhisperTranscription](../WhisperTranscription/) | Local inference via ORT GenAI — the easiest way to run Whisper locally with real audio files |
| [WhisperRawOnnx](../WhisperRawOnnx/) | Full manual ONNX control — encoder/decoder, KV cache, token sampling, temperature decoding |
| [TextToSpeech](../TextToSpeech/) | The reverse direction — text → audio via SpeechT5 ONNX, completing the voice round-trip |
| [docs/meai-integration.md](../../docs/meai-integration.md) | Deep dive into the Microsoft.Extensions.AI integration patterns (`ISpeechToTextClient`, `IEmbeddingGenerator<AudioData, Embedding<float>>`) |

## Prerequisites

### Required Software

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

### No Model Required

This sample is **purely demonstrative** — it shows API patterns and architecture without requiring any ONNX model or external service. It generates synthetic audio for feature extraction demonstrations.

This is intentional: the sample teaches you the **patterns** for provider-agnostic speech-to-text so you can plug in any backend (Azure Speech, OpenAI Whisper API, local Whisper model, or a mock for testing).

## Running the Sample

```bash
cd samples/SpeechToText
dotnet run
```

No model downloads or API keys required — the sample demonstrates patterns and runs real feature extraction on synthetic audio.

## Troubleshooting

### "No actual transcription happens"
This is by design. The SpeechToText sample demonstrates **patterns and architecture**, not actual inference. For real transcription:
- **Easiest**: Use the [WhisperTranscription](../WhisperTranscription/) sample with an ORT GenAI Whisper model
- **Most control**: Use the [WhisperRawOnnx](../WhisperRawOnnx/) sample with raw ONNX encoder/decoder models
- **Cloud**: Implement `ISpeechToTextClient` against Azure Speech Services or OpenAI Whisper API

### Which ISpeechToTextClient implementation should I use?

| Approach | Best For | Complexity | Latency |
|----------|----------|------------|---------|
| Cloud provider (Azure, OpenAI) | Production, highest accuracy | Low (API key) | Network-dependent |
| ORT GenAI (`OnnxSpeechToTextClient`) | Local inference, simple API | Medium (model download) | ~2-5s for 30s audio |
| Raw ONNX (`OnnxWhisperSpeechToTextClient`) | Full control, custom sampling | High (manual KV cache) | ~2-5s for 30s audio |
| Mock client | Unit testing | Trivial | Instant |

### Build warnings about missing ONNX models
Some warnings about ONNX model paths are expected in this sample since no model is loaded. The sample runs successfully without any models.
