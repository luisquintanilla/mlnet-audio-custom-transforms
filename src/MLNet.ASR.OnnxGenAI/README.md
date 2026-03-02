# MLNet.ASR.OnnxGenAI

Local Whisper speech-to-text (ASR) for ML.NET using **ONNX Runtime GenAI** — the managed, higher-level abstraction that handles the autoregressive decoder loop natively.

## Why This Package Exists (Dependency Isolation)

This is the **critical design question**: why does this package exist separately from `MLNet.AudioInference.Onnx`, which already contains ASR transformers?

The answer is a **NuGet packaging decision**, not an architectural one.

ONNX Runtime GenAI ships with **platform-specific native binaries** — CPU, CUDA, and DirectML variants, each tens of megabytes. These arrive via runtime NuGet packages like `Microsoft.ML.OnnxRuntimeGenAI.Cuda` or `Microsoft.ML.OnnxRuntimeGenAI.DirectML`. If the ORT GenAI dependency lived inside `MLNet.AudioInference.Onnx`, then **every consumer** of that package — even those only doing audio classification or embedding extraction — would be forced to pull in the ORT GenAI native dependency tree. That's an unacceptable cost for users who never touch speech-to-text.

The isolation is surgical:

- **`MLNet.ASR.OnnxGenAI`** depends on **`MLNet.Audio.Core` directly** — it does _not_ depend on `MLNet.AudioInference.Onnx`
- **`MLNet.AudioInference.Onnx`** continues to carry standard ONNX Runtime (used for classification, embeddings, and raw Whisper), without any ORT GenAI footprint
- Consumers who want ORT GenAI Whisper opt in by referencing this package and its corresponding runtime package — nobody else pays the cost

This is the same pattern used throughout the .NET ecosystem (e.g., `Microsoft.ML.OnnxRuntime` vs `Microsoft.ML.OnnxRuntimeGenAI` are separate NuGet packages for the same reason).

## Key Concepts

### What Is ONNX Runtime GenAI?

Standard ONNX Runtime gives you a session: you feed tensors in, you get tensors out, one forward pass at a time. For encoder-only models (classifiers, embeddings), that's all you need.

But Whisper is an **encoder-decoder** model. Transcription requires an autoregressive loop: run the encoder once, then run the decoder repeatedly — feeding each predicted token back as input — while managing a KV cache that grows with each step. Standard ONNX Runtime makes you write all of that yourself.

**ONNX Runtime GenAI** is a higher-level library that handles this loop natively. You hand it audio features and generation parameters; it runs the encoder, manages the decoder loop and KV cache internally, and gives you back token IDs. It's the same relationship as "raw SQL" vs "an ORM" — you trade visibility for convenience.

The tradeoff: ORT GenAI requires models exported in a **specific format** with a `genai_config.json` manifest (produced by [Olive](https://github.com/microsoft/Olive) or [optimum](https://huggingface.co/docs/optimum/)). You can't point it at an arbitrary ONNX file.

### The Three ASR Approaches

This repository offers three distinct paths to speech-to-text, forming a **convenience-vs-control spectrum**:

| Approach | Package | What It Does | When to Use It |
|---|---|---|---|
| **Provider-Agnostic** | `MLNet.AudioInference.Onnx` | `SpeechToTextClientTransformer` wraps any `ISpeechToTextClient` — Azure, OpenAI, local, etc. | Cloud ASR, swappable providers, no local model needed |
| **ORT GenAI** (this package) | `MLNet.ASR.OnnxGenAI` | `OnnxSpeechToTextTransformer` uses ORT GenAI to run Whisper locally with the decoder loop handled by native code | Simple local deployment — you want Whisper on-device without managing the decoder yourself |
| **Raw ONNX** | `MLNet.AudioInference.Onnx` | `OnnxWhisperTransformer` manages encoder, decoder, KV cache, and token sampling manually using standard ONNX Runtime | Full control — custom sampling strategies, KV cache inspection, or when you can't use the ORT GenAI model format |

**This package is the middle of the spectrum.** It's simpler than raw ONNX (no manual KV cache, no decoder loop code) but less flexible (requires the ORT GenAI model format, can't customize sampling). For most local Whisper deployments, it's the right choice.

## Architecture & Design Decisions

### Why Monolithic (Not Composed into Sub-Transforms)

The shared audio pipeline in `MLNet.AudioInference.Onnx` decomposes inference into lazy stages (Feature → Score → PostProcess) connected via `IDataView`. Whisper **cannot** follow this pattern. The encoder-decoder architecture with autoregressive decoding is fundamentally a single operation from ORT GenAI's perspective — the native engine owns the decoder loop, KV cache lifecycle, and token generation. There's nothing to decompose into independent ML.NET transforms.

This mirrors how `OnnxTextGenerationTransformer` works in the text-inference repo: encoder-decoder and generative models are standalone transformers, not pipelines of sub-transforms.

The transformer uses **eager evaluation**: `Transform()` reads all audio rows, transcribes each one, and returns a new `IDataView` with the results.

### Model Format Requirements

ORT GenAI does not work with arbitrary ONNX files. Models must be exported specifically for GenAI using [Olive](https://github.com/microsoft/Olive) or Hugging Face [optimum](https://huggingface.co/docs/optimum/). The model directory must contain:

- `genai_config.json` — model configuration manifest
- Encoder and decoder ONNX files
- Tokenizer vocabulary files

Pre-exported models are available on Hugging Face (search for "onnxruntime-genai whisper").

### Platform Selection

The hardware backend (CPU, CUDA, DirectML) is controlled by which **NuGet runtime package** you install alongside this package:

- **CPU**: `Microsoft.ML.OnnxRuntimeGenAI` (default, no GPU required)
- **CUDA**: `Microsoft.ML.OnnxRuntimeGenAI.Cuda` (NVIDIA GPUs)
- **DirectML**: `Microsoft.ML.OnnxRuntimeGenAI.DirectML` (Windows GPU-agnostic)

This package references only the **managed** package (`Microsoft.ML.OnnxRuntimeGenAI.Managed`), which contains the C# API surface. The native binaries come from whichever runtime package you choose — a standard .NET native dependency pattern.

### ISpeechToTextClient (MEAI Integration)

`OnnxSpeechToTextClient` implements `Microsoft.Extensions.AI.ISpeechToTextClient`, plugging local Whisper transcription into the MEAI ecosystem. This means it works with:

- Standard .NET dependency injection (`builder.Services.AddSingleton<ISpeechToTextClient>(...)`)
- MEAI middleware pipelines (logging, caching, retry)
- Any code written against the `ISpeechToTextClient` abstraction

You can swap between cloud ASR (Azure, OpenAI) and local Whisper (this package) by changing a DI registration — the consuming code doesn't change.

## API Surface

### `OnnxSpeechToTextOptions`

Configuration record for the transformer. Key properties:

| Property | Type | Default | Purpose |
|---|---|---|---|
| `ModelPath` | `string` | _(required)_ | Path to ORT GenAI model directory (must contain `genai_config.json`) |
| `Language` | `string?` | `null` | Language code (`"en"`, `"es"`, etc.) — `null` for auto-detect |
| `Translate` | `bool` | `false` | Translate to English instead of transcribing in source language |
| `IsMultilingual` | `bool` | `true` | Set `false` for English-only models (`whisper-tiny.en`, `whisper-base.en`) |
| `MaxLength` | `int` | `256` | Maximum tokens to generate |
| `NumMelBins` | `int` | `80` | Mel bins — `80` for Whisper v1/v2, `128` for v3 |
| `SampleRate` | `int` | `16000` | Expected audio sample rate |
| `FeatureExtractor` | `AudioFeatureExtractor?` | `null` | Custom feature extractor — `null` uses default `WhisperFeatureExtractor` |
| `InputColumnName` | `string` | `"Audio"` | ML.NET input column name |
| `OutputColumnName` | `string` | `"Text"` | ML.NET output column name |

### `OnnxSpeechToTextEstimator`

`IEstimator<OnnxSpeechToTextTransformer>` — the ML.NET entry point. `Fit()` creates the transformer (model loading happens here). Accessed via the extension method `mlContext.Transforms.OnnxSpeechToText(options)`.

### `OnnxSpeechToTextTransformer`

The core `ITransformer`. Holds the loaded ORT GenAI `Model`, `Tokenizer`, and feature extractor. Implements `IDisposable` — the native model must be disposed.

Key methods beyond `Transform()`:

- **`Transcribe(IReadOnlyList<AudioData>)`** → `string[]` — direct transcription, bypassing ML.NET `IDataView`
- **`TranscribeWithTimestamps(IReadOnlyList<AudioData>)`** → `TranscriptionResult[]` — structured output with timestamped segments, raw token IDs, and language

Cannot be serialized (`Save()` throws) — the ORT GenAI model must be loaded from disk at runtime.

### `OnnxSpeechToTextClient`

`ISpeechToTextClient` implementation for the MEAI ecosystem. Wraps the transformer and handles WAV loading/resampling. Metadata reports provider `"OnnxGenAI-Whisper"`.

### `TranscriptionResult`

Structured output from `TranscribeWithTimestamps()`:

- `Text` — full transcribed text (cleaned of special tokens)
- `Segments` — array of `TranscriptionSegment` with start/end timestamps
- `TokenIds` — raw decoder token IDs
- `Language` — detected or specified language

### `MLContextExtensions`

Extension method `mlContext.Transforms.OnnxSpeechToText(options)` — idiomatic ML.NET entry point.

## How It Fits in the Architecture

This is a **Layer 2** package — inference-level, sitting alongside (not above or below) `MLNet.AudioInference.Onnx`:

```
Layer 0: MLNet.Audio.Core          ← shared primitives (AudioData, AudioIO, WhisperFeatureExtractor, ...)
Layer 1: (primitives/feature extraction — part of Audio.Core)
Layer 2: MLNet.AudioInference.Onnx ← classification, embeddings, raw Whisper, provider-agnostic ASR
         MLNet.ASR.OnnxGenAI       ← THIS PACKAGE: ORT GenAI Whisper (parallel to AudioInference.Onnx)
Layer 3: Application code
```

The dependency graph is intentionally flat:

```
MLNet.ASR.OnnxGenAI ──→ MLNet.Audio.Core
MLNet.AudioInference.Onnx ──→ MLNet.Audio.Core
```

Both Layer 2 packages depend on `Audio.Core` directly. Neither depends on the other. This is the dependency isolation that justifies this package's existence as a separate assembly.

## Usage

### Direct Transcription (Simplest)

```csharp
var options = new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-base-genai",
    Language = "en"
};

using var transformer = new OnnxSpeechToTextTransformer(new MLContext(), options);

var audio = AudioIO.LoadWav("recording.wav");
string[] results = transformer.Transcribe([audio]);
Console.WriteLine(results[0]);
```

### Transcription with Timestamps

```csharp
TranscriptionResult[] results = transformer.TranscribeWithTimestamps([audio]);

foreach (var segment in results[0].Segments)
{
    Console.WriteLine($"[{segment.Start:F2}s → {segment.End:F2}s] {segment.Text}");
}
```

### ISpeechToTextClient (MEAI)

```csharp
using var client = new OnnxSpeechToTextClient(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-base-genai",
    Language = "en"
});

using var stream = File.OpenRead("recording.wav");
var response = await client.GetTextAsync(stream);
Console.WriteLine(response.Text);
```

Works with DI:

```csharp
builder.Services.AddSingleton<ISpeechToTextClient>(sp =>
    new OnnxSpeechToTextClient(new OnnxSpeechToTextOptions
    {
        ModelPath = "models/whisper-base-genai"
    }));
```

### ML.NET Pipeline

```csharp
var mlContext = new MLContext();

var estimator = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-base-genai",
    Language = "en"
});

var transformer = estimator.Fit(trainingData);
var results = transformer.Transform(audioData);
```

## Dependencies

| Dependency | Version | Role |
|---|---|---|
| `MLNet.Audio.Core` | (project) | Shared audio primitives: `AudioData`, `AudioIO`, `WhisperFeatureExtractor`, `WhisperTokenizer` |
| `Microsoft.ML` | 5.0.0 | `IEstimator<T>`, `ITransformer`, `IDataView`, `MLContext` |
| `Microsoft.ML.OnnxRuntimeGenAI.Managed` | 0.12.1 | ORT GenAI C# API — `Model`, `Generator`, `Tokenizer`, `Tensor`. The native decoder loop engine. |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | `ISpeechToTextClient` interface for MEAI ecosystem integration |

> **Note:** You also need a **runtime** NuGet package for native binaries — `Microsoft.ML.OnnxRuntimeGenAI` (CPU), `Microsoft.ML.OnnxRuntimeGenAI.Cuda`, or `Microsoft.ML.OnnxRuntimeGenAI.DirectML`. The managed package referenced here provides only the C# API surface.
