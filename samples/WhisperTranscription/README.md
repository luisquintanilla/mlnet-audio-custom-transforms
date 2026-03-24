# Whisper Transcription: Local ASR with ORT GenAI

This sample demonstrates **local speech-to-text** using OpenAI's Whisper model via **ONNX Runtime GenAI** (ORT GenAI). It sits at the middle of the convenience-vs-control spectrum: more control than a cloud API, but far simpler than managing raw ONNX encoder/decoder sessions yourself.

## What You'll Learn

- **ORT GenAI abstraction** — how it wraps the autoregressive decoder loop so you don't manage encoder/decoder/KV cache manually
- **Convenience vs. control** — where ORT GenAI sits relative to provider-agnostic cloud APIs (most convenient) and raw ONNX inference (full control)
- **Runtime selection** — how your choice of NuGet package determines CPU, CUDA, or DirectML execution
- **MEAI integration** — using `ISpeechToTextClient` so local Whisper and a cloud provider share the same interface
- **ML.NET composability** — plugging Whisper into an estimator/transformer pipeline alongside other transforms

## The Concept: ORT GenAI for ASR

### What is ONNX Runtime GenAI?

**ONNX Runtime GenAI** is a higher-level API built on top of ONNX Runtime, purpose-built for **generative models** (LLMs, Whisper, vision-language models). For speech-to-text, it manages the full autoregressive decode loop internally:

```
You provide: audio features (mel spectrogram)
ORT GenAI handles: encoder forward pass → decoder token-by-token generation → KV cache management → stopping criteria
You receive: generated token IDs → decoded text
```

With **raw ONNX Runtime**, you would need to:
1. Run the encoder session manually to get encoder hidden states
2. Initialize the decoder with start-of-transcript tokens
3. Loop: run the decoder session, extract the next token, update the KV cache, check for end-of-sequence
4. Manage tensor memory, shapes, and device placement at every step

ORT GenAI collapses all of that into a `Generator` object with a simple `GenerateNextToken()` loop—or, through our library, a single `Transcribe()` call.

### The Convenience–Control Spectrum

This repository provides **three approaches** to the same ASR task (Whisper speech-to-text). Each trades convenience for control:

| Approach | Sample | Package | What You Manage |
|---|---|---|---|
| **Provider-agnostic** | [`SpeechToText`](../SpeechToText/) | `MLNet.ASR` | Nothing — pick a provider, call `GetTextAsync()`. Could be cloud or local. |
| **ORT GenAI** (this) | `WhisperTranscription` | `MLNet.ASR.OnnxGenAI` | Model path, language, max tokens. ORT GenAI handles the decoder loop. |
| **Raw ONNX** | [`WhisperRawOnnx`](../WhisperRawOnnx/) | `MLNet.ASR.OnnxRaw` | Everything: encoder/decoder sessions, KV cache, token sampling, beam search. |

**This sample is the sweet spot for most local ASR use cases.** You get local execution (no cloud dependency, no data leaves your machine) with a simple API, without needing to understand Whisper's internal encoder-decoder architecture.

### Abstraction Layers

```
┌─────────────────────────────────────────────────────────┐
│ MEAI: ISpeechToTextClient                                │
│   OnnxSpeechToTextClient — same interface as Azure/OpenAI│
│   Middleware pipeline: logging, OpenTelemetry, caching   │
├─────────────────────────────────────────────────────────┤
│ ML.NET: ITransformer / IEstimator<T>                     │
│   OnnxSpeechToTextTransformer — composable pipeline      │
│   mlContext.Transforms.OnnxSpeechToText(options)         │
├─────────────────────────────────────────────────────────┤
│ ONNX Runtime GenAI                                       │
│   Handles encoder → decoder loop → KV cache internally   │
│   You provide mel features, receive token IDs            │
├─────────────────────────────────────────────────────────┤
│ Audio Primitives (MLNet.Audio.Core)                      │
│   WhisperFeatureExtractor (mel spectrogram)              │
│   AudioData, AudioIO (WAV I/O, resampling)              │
└─────────────────────────────────────────────────────────┘
```

ORT GenAI sits between you and the raw ONNX models, handling the complex decoder loop. Compare with [WhisperRawOnnx](../WhisperRawOnnx/) where YOU manage the decoder loop, KV cache, and token sampling directly.

### Model Format: ORT GenAI-Specific Export

ORT GenAI requires models in its **own export format**, distinct from a plain ONNX export. The model directory must contain:

```
whisper-base/
├── genai_config.json          ← ORT GenAI configuration (model type, search params)
├── encoder_model.onnx         ← (or encoder_model_merged.onnx)
├── decoder_model_merged.onnx  ← decoder with KV cache merged
├── tokenizer.json             ← BPE tokenizer vocabulary
└── tokenizer_config.json      ← tokenizer settings
```

**How to get this format:**
- **Olive** (Microsoft's model optimization toolkit): `olive convert --model openai/whisper-base --provider onnxruntime-genai`
- **Hugging Face Optimum**: `optimum-cli export onnx --model openai/whisper-base --task automatic-speech-recognition`
- **Pre-exported models**: Search [huggingface.co/models](https://huggingface.co/models?search=whisper+onnx+genai) for `whisper onnx genai`

> **Important:** A standard ONNX export of Whisper (e.g., from `torch.onnx.export`) will **not** work with ORT GenAI. The `genai_config.json` file and the specific graph structure are required.

### Platform Selection: NuGet Runtime Packages

ORT GenAI is a **managed wrapper** around native C++ libraries. Which native library loads at runtime depends entirely on which NuGet package you install:

```xml
<!-- CPU — works everywhere, no GPU required -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" Version="0.12.1" />

<!-- NVIDIA GPU — requires CUDA toolkit installed -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.Cuda" Version="0.12.1" />

<!-- Windows GPU — uses DirectML, works with any DirectX 12 GPU (AMD, Intel, NVIDIA) -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.DirectML" Version="0.12.1" />
```

Your C# code stays **identical** regardless of which package you pick. The runtime package is the only thing that changes. This is why the `.csproj` for this sample intentionally omits the native package — you add the one that matches your hardware.

## What This Sample Demonstrates

### 1. Direct Transcription — `Transcribe()` and `TranscribeWithTimestamps()`

**Why:** The simplest way to go from audio to text locally. No pipeline setup, no DI, no abstractions — just create a transformer and call it.

```csharp
using var transformer = new OnnxSpeechToTextTransformer(mlContext, options);

// Simple: audio in, text out
string[] texts = transformer.Transcribe([audio]);

// Detailed: includes timestamps per segment
TranscriptionResult[] detailed = transformer.TranscribeWithTimestamps([audio]);
foreach (var segment in detailed[0].Segments)
{
    Console.WriteLine($"[{segment.Start:mm\\:ss\\.ff} → {segment.End:mm\\:ss\\.ff}] {segment.Text}");
}
```

`TranscribeWithTimestamps` returns `TranscriptionResult` objects containing the full text, an array of `TranscriptionSegment` (each with `Start`, `End`, and `Text`), the raw token IDs, and the detected language.

**Use cases for timestamps:** subtitle/caption generation, meeting note alignment, podcast chapter markers, audio search indexing.

### 2. MEAI `ISpeechToTextClient` — `OnnxSpeechToTextClient`

**Why:** Plugs local Whisper into the Microsoft.Extensions.AI ecosystem. The same `ISpeechToTextClient` interface works whether the backend is a cloud API (Azure Speech, OpenAI) or this local ORT GenAI model. Swap providers without changing calling code.

```csharp
using var client = new OnnxSpeechToTextClient(options);

// Same interface as any cloud ISpeechToTextClient
using var audioStream = File.OpenRead("speech.wav");
SpeechToTextResponse response = await client.GetTextAsync(audioStream);
Console.WriteLine(response.Text);
```

Because `OnnxSpeechToTextClient` implements `ISpeechToTextClient`, you can:
- Register it in DI: `services.AddSingleton<ISpeechToTextClient>(new OnnxSpeechToTextClient(opts))`
- Use MEAI middleware (logging, caching, rate limiting)
- Swap to a cloud provider for production without changing business logic

### 3. ML.NET Pipeline — `Fit()` / `Transform()`

**Why:** Composable with other ML.NET transforms. Chain speech-to-text with text embedding, classification, or any other transform in a single pipeline.

```csharp
var estimator = mlContext.Transforms.OnnxSpeechToText(options);

// Compose with downstream transforms
var pipeline = estimator
    .Append(mlContext.Transforms.OnnxTextEmbedding(...))
    .Append(mlContext.Transforms.OnnxTextClassification(...));
```

The estimator follows the standard ML.NET pattern: `IEstimator<T>.Fit(data)` returns an `ITransformer` that you call `.Transform(data)` on.

### 4. Timestamps — Word/Segment-Level Timing

**Why:** Many real-world applications need to know *when* something was said, not just *what*. Whisper natively supports timestamp tokens (`<|0.00|>` through `<|30.00|>`) that encode timing information directly in the decoder output.

The `WhisperTokenizer` in `MLNet.Audio.Core` parses these special tokens into structured `TranscriptionSegment` objects with `TimeSpan` start/end values.

## Prerequisites

### 1. Obtain a Whisper Model in ORT GenAI Format

See [Model Format](#model-format-ort-genai-specific-export) above. The model directory must contain `genai_config.json` and the associated ONNX/tokenizer files.

### 2. Install a Platform-Specific Native Runtime

Add **one** of the following to this project (or a project that references it):

```shell
dotnet add package Microsoft.ML.OnnxRuntimeGenAI         # CPU
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda    # NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML # Windows GPU (DirectX 12)
```

### 3. .NET 10 SDK

This sample targets `net10.0`. Ensure you have the .NET 10 SDK installed.

## Running It

### With a Model — Real Transcription

```shell
# Place your model at models/whisper-base (relative to the project), or pass the path:
dotnet run -- "C:\models\whisper-base-genai"

# Optionally provide an audio file as the second argument:
dotnet run -- "C:\models\whisper-base-genai" "recording.wav"
```

The sample will:
1. Load the model and create an `OnnxSpeechToTextTransformer`
2. Load and resample the audio to 16kHz mono (if needed)
3. Run direct transcription and print the text
4. Run timestamped transcription and print each segment with timing
5. Demonstrate the `ISpeechToTextClient` interface
6. Show the ML.NET pipeline pattern

### Without a Model — Pattern Demonstration

If no model directory is found, the sample prints the **API patterns** for all four approaches so you can see the calling conventions without needing a model on disk:

```shell
dotnet run
# Output: API patterns for direct transcription, ISpeechToTextClient, ML.NET pipeline, and timestamps
```

## Code Walkthrough

### Options Configuration

```csharp
var options = new OnnxSpeechToTextOptions
{
    ModelPath = modelPath,       // Path to ORT GenAI model directory
    Language = "en",             // Language code (null = auto-detect)
    MaxLength = 256,             // Max decoder tokens to generate
    NumMelBins = 80,             // 80 for whisper-base/small, 128 for whisper-large-v3
};
```

Key options:
- **`ModelPath`** (required) — directory containing `genai_config.json` and model files
- **`Language`** — ISO language code. Set explicitly for better accuracy; leave `null` for auto-detection
- **`MaxLength`** — caps decoder output length. 256 tokens ≈ 1–2 minutes of speech for most languages
- **`NumMelBins`** — must match the model: 80 for Whisper v1/v2 models (tiny, base, small, medium), 128 for v3 (large-v3)
- **`FeatureExtractor`** — defaults to `WhisperFeatureExtractor`. Override if you need custom mel spectrogram settings
- **`SampleRate`** — defaults to 16000 Hz. Audio is resampled to this rate before feature extraction
- **`Translate`** — set to `true` to translate non-English speech to English (Whisper's built-in translation task)
- **`IsMultilingual`** — set to `false` for English-only models (whisper-tiny.en, whisper-base.en)

### Why `Fit()` is Needed Even for Pre-Trained Models

The `OnnxSpeechToTextEstimator.Fit()` method doesn't train anything — Whisper is already pre-trained. But ML.NET's design pattern requires the estimator → transformer two-step:

```csharp
// Estimator: describes the transform (configuration, schema validation)
IEstimator<OnnxSpeechToTextTransformer> estimator = mlContext.Transforms.OnnxSpeechToText(options);

// Transformer: actually loads the model and can process data
OnnxSpeechToTextTransformer transformer = estimator.Fit(trainingData);

// Now use it
IDataView results = transformer.Transform(testData);
```

`Fit()` exists because ML.NET was designed for **trainable** pipelines (normalization, feature engineering, ML models). Even pre-trained ONNX models go through this pattern for consistency — `Fit()` is where the model gets loaded and the transformer gets instantiated. This means you can compose pre-trained transforms with trainable ones in the same pipeline using `.Append()`.

### `ISpeechToTextClient`: Same Interface, Any Backend

```csharp
using var client = new OnnxSpeechToTextClient(options);

// This is the MEAI interface — identical whether backed by:
// - Local ORT GenAI Whisper (this sample)
// - Azure Cognitive Services Speech
// - OpenAI Whisper API
// - Any other ISpeechToTextClient implementation
using var stream = File.OpenRead("speech.wav");
SpeechToTextResponse response = await client.GetTextAsync(stream);
```

The `OnnxSpeechToTextClient` wraps `OnnxSpeechToTextTransformer` internally. It handles WAV loading and resampling, then delegates to `TranscribeWithTimestamps()`. The `Metadata` property reports `ProviderName: "OnnxGenAI-Whisper"` and uses the model directory name as the `DefaultModelId`.

### Synthetic Audio Generation

When no test WAV file is available, the sample generates a 3-second 440Hz sine wave tone as synthetic input. This won't produce meaningful transcription (Whisper will likely output silence tokens or hallucinated text), but it exercises the full pipeline — feature extraction, encoder, decoder, and token decoding — to verify the setup works end to end.

## Key Takeaways

1. **ORT GenAI is the sweet spot for local ASR.** Simple `Transcribe()` API, no manual decoder management, no KV cache bookkeeping. You provide audio features, ORT GenAI handles the autoregressive generation loop.

2. **Same `ISpeechToTextClient` interface as cloud providers.** Register `OnnxSpeechToTextClient` in DI during development (fast, free, offline), swap to Azure or OpenAI in production — zero code changes in your business logic.

3. **Model format matters.** ORT GenAI requires its own export format with `genai_config.json`. A standard ONNX export won't work. Use Olive or Hugging Face Optimum to produce the right format.

4. **Platform selection is a NuGet concern, not a code concern.** Switch from CPU to CUDA to DirectML by changing a single `<PackageReference>` — your C# stays identical.

5. **This lives in a separate package (`MLNet.ASR.OnnxGenAI`) to isolate native dependencies.** The ORT GenAI native libraries are large (~100–400MB depending on platform). By putting this in its own package, applications that use cloud ASR or raw ONNX don't pay the size cost.

## Troubleshooting

### "Model not found" — API patterns shown instead
This is expected behavior. Without a Whisper model in ORT GenAI format, the sample demonstrates API patterns. To run with real transcription:

```bash
# Option 1: Export with Olive (recommended for ORT GenAI)
pip install olive-ai
olive convert --model openai/whisper-base --provider onnxruntime-genai --output models/whisper-base

# Option 2: Export with Optimum
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base --task automatic-speech-recognition models/whisper-base/
```

### Model format confusion: ORT GenAI vs Raw ONNX
ORT GenAI requires a **`genai_config.json`** file alongside the ONNX models. This is different from the raw ONNX format used by the [WhisperRawOnnx](../WhisperRawOnnx/) sample:

| Format | Config File | Used By | Export Tool |
|--------|-------------|---------|-------------|
| ORT GenAI | `genai_config.json` | This sample (`WhisperTranscription`) | Olive |
| Raw ONNX | `config.json` | [WhisperRawOnnx](../WhisperRawOnnx/) | Optimum |

If you have raw ONNX models but no `genai_config.json`, use the [WhisperRawOnnx](../WhisperRawOnnx/) sample instead.

### ORT GenAI native library not found
The `MLNet.ASR.OnnxGenAI` package depends on `Microsoft.ML.OnnxRuntimeGenAI.Managed` — but you need to add the **native runtime** package for your platform:

```xml
<!-- CPU -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" Version="0.12.1" />

<!-- CUDA (NVIDIA GPU) -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.Cuda" Version="0.12.1" />

<!-- DirectML (Windows GPU) -->
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.DirectML" Version="0.12.1" />
```

### Transcription quality issues
- **Whisper-base** (~74M params) is good for demos but not production. For better accuracy, use `whisper-small` (244M) or `whisper-medium` (769M)
- Audio should be 16kHz mono PCM for best results (the library auto-resamples)
- Whisper works best with 30-second audio chunks — very long audio may need chunking

### When should I use this vs Raw ONNX vs cloud API?

| Approach | Convenience | Control | Dependencies |
|----------|-------------|---------|--------------|
| **Cloud API** | ⭐⭐⭐ | ⭐ | API key only |
| **ORT GenAI** (this sample) | ⭐⭐ | ⭐⭐ | ORT GenAI native lib + model |
| **Raw ONNX** | ⭐ | ⭐⭐⭐ | OnnxRuntime + exported model |

**This sample is the sweet spot** — more control than a cloud API, but the ORT GenAI runtime handles the complex decoder loop and KV cache management for you.

## Going Further

| To... | See... |
|---|---|
| **Full control** over encoder/decoder sessions, KV cache, and token sampling | [`WhisperRawOnnx`](../WhisperRawOnnx/) sample — same Whisper task, raw ONNX Runtime |
| **Provider-agnostic** ASR that abstracts over cloud and local backends | [`SpeechToText`](../SpeechToText/) sample — uses the provider pattern |
| **Understand the 3 ASR approaches** and how they relate architecturally | `docs/architecture.md` — design rationale for the convenience-vs-control spectrum |
| **Audio preprocessing** in detail (mel spectrograms, resampling, feature extraction) | [`MLNet.Audio.Core`](../../src/MLNet.Audio.Core/) — the shared audio primitives library |
| **Other audio tasks** (embeddings, classification, TTS, VAD) | Other samples in [`samples/`](../) — same ML.NET transform patterns, different models |
