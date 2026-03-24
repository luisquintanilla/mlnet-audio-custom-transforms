# Architecture

## Overview

This repository provides ML.NET custom transforms for audio AI tasks — classification, embeddings, voice activity detection (VAD), speech-to-text (ASR), and text-to-speech (TTS) — using local ONNX models. It follows the same `IEstimator<T>`/`ITransformer` patterns as [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms), adapted for audio's unique requirements: raw PCM input, mel spectrogram feature extraction, encoder-decoder architectures with KV cache, and vocoder post-processing.

## Layered Design

### Layer 0: Audio Primitives (`MLNet.Audio.Core`)

Foundation types for audio processing. Zero ML.NET dependency — pure audio DSP and tokenization.

**Dependencies:** NWaves 0.9.6, System.Numerics.Tensors 10.0.3

| Type | Purpose |
|------|---------|
| `AudioData` | Core type: `float[] Samples` (mono, normalized to [-1.0, 1.0]) + `int SampleRate` + `int Channels` + `TimeSpan Duration` |
| `AudioIO` | WAV I/O (8/16/24-bit PCM, 32-bit float), `Resample()` via linear interpolation, `ToMono()` channel mixing |
| `AudioFeatureExtractor` | Abstract base class — audio's equivalent of `Microsoft.ML.Tokenizers.Tokenizer`. Auto-handles resampling and mono conversion before delegating to `ExtractFeatures()` |
| `MelSpectrogramExtractor` | Generic log-mel spectrogram via NWaves `FilterbankExtractor`. Configurable: mel bins, FFT size, hop length, frequency range, window function. Uses SIMD `TensorPrimitives.Log()` for log scaling |
| `WhisperFeatureExtractor` | Whisper-specific mel spectrogram: 16kHz, 400 FFT size, 160 hop length, 80/128 mel bins. Pads/truncates to 3000 frames (30 seconds). `ExtractChunked()` for long audio with overlap |
| `WhisperTokenizer` | Whisper BPE tokenizer with ~1700 special tokens: language codes (99 languages), task tokens (`<\|transcribe\|>`, `<\|translate\|>`), 1501 timestamp tokens (`<\|0.00\|>` through `<\|30.00\|>` in 0.02s steps). Three decode modes: `Decode()` (clean text), `DecodeWithTimestamps()` (inline annotations), `DecodeToSegments()` (structured `TranscriptionSegment` records) |
| `AudioCodecTokenizer` | Abstract base for neural audio codecs (EnCodec, DAC, SpeechTokenizer). `Encode(AudioData) → int[][]` (multi-codebook RVQ), `Decode(int[][]) → AudioData`, `EncodeFlat()` for LM-consumable interleaved tokens |

### Layer 0.5: Tokenizer Extensions (`MLNet.Audio.Tokenizers`)

Extends `Microsoft.ML.Tokenizers` with tokenizer implementations for audio/speech models that aren't supported natively.

**Dependencies:** Microsoft.ML.Tokenizers 2.0.0 (Google.Protobuf transitive)

| Type | Purpose |
|------|---------|
| `SentencePieceCharTokenizer` | Extends `Tokenizer` base class. SentencePiece Char model support (used by SpeechT5). Parses `.model` protobuf, maps characters to vocabulary IDs. Full `Tokenizer` API: `EncodeToIds()`, `EncodeToTokens()`, `Decode()`, `CountTokens()`. Workaround until `Microsoft.ML.Tokenizers` adds native Char model support. |

**Why a separate package?** `MLNet.Audio.Core` stays dependency-free (no ML.Tokenizers dep). This package bridges audio-specific tokenizer needs with the ML.Tokenizers framework. `WhisperTokenizer` and `AudioCodecTokenizer` stay in Core because they don't benefit from extending the `Tokenizer` base class (WhisperTokenizer is a domain-specific decoder; AudioCodecTokenizer operates on a different modality).

### Layer 1: Inference Transforms (`MLNet.AudioInference.Onnx`)

ML.NET `IEstimator<T>`/`ITransformer` implementations for all audio tasks.

**Dependencies:** Microsoft.ML 5.0.0, Microsoft.ML.OnnxRuntime.Managed 1.24.2, Microsoft.ML.Tokenizers 2.0.0, Microsoft.Extensions.AI.Abstractions 10.4.1, System.Numerics.Tensors 10.0.3

**Project reference:** MLNet.Audio.Core

| Subdirectory | Types | Description |
|--------------|-------|-------------|
| Classification/ | `OnnxAudioClassificationEstimator`, `OnnxAudioClassificationTransformer`, `OnnxAudioClassificationOptions`, `AudioClassificationPostProcessingEstimator`, `AudioClassificationPostProcessingTransformer` | Audio classification with softmax + label mapping |
| Embeddings/ | `OnnxAudioEmbeddingEstimator`, `OnnxAudioEmbeddingTransformer`, `OnnxAudioEmbeddingOptions`, `AudioPoolingStrategy`, `AudioEmbeddingPoolingEstimator`, `AudioEmbeddingPoolingTransformer` | Audio embeddings with pooling (MeanPooling, ClsToken, MaxPooling) + L2 normalization |
| VAD/ | `OnnxVadEstimator`, `OnnxVadTransformer`, `OnnxVadOptions`, `IVoiceActivityDetector`, `SpeechSegment`, `VadOptions` | Voice activity detection with frame scoring + segment merging. `OnnxVadTransformer` implements both `ITransformer` and `IVoiceActivityDetector` |
| ASR/ | `SpeechToTextClientEstimator`, `SpeechToTextClientTransformer`, `SpeechToTextClientOptions` | Provider-agnostic ASR wrapping any `ISpeechToTextClient` |
| ASR/ | `OnnxWhisperEstimator`, `OnnxWhisperTransformer`, `OnnxWhisperOptions`, `WhisperKvCacheManager` | Raw ONNX Whisper with manual encoder/decoder/KV cache management |
| TTS/ | `OnnxSpeechT5TtsEstimator`, `OnnxSpeechT5TtsTransformer`, `OnnxSpeechT5Options` | SpeechT5 text-to-speech: encoder → decoder (KV cache) → vocoder |
| TTS/ | `OnnxKittenTtsEstimator`, `OnnxKittenTtsTransformer`, `OnnxKittenTtsOptions` | KittenTTS text-to-speech: espeak-ng phonemization → single ONNX model → 24 kHz audio |
| TTS/ | `OnnxTextToSpeechClient`, `IOnnxTtsSynthesizer` (internal) | Official MEAI `ITextToSpeechClient` — one client for all TTS backends (SpeechT5, KittenTTS) via `IOnnxTtsSynthesizer` |
| MEAI/ | `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` bridge to MEAI |
| Shared/ | `AudioFeatureExtractionEstimator`, `AudioFeatureExtractionTransformer`, `OnnxAudioScoringEstimator`, `OnnxAudioScoringTransformer`, `EnCodecTokenizer` | Reusable sub-transforms for the composed 3-stage pipeline. Feature extraction (Stage 1) and ONNX scoring (Stage 2) are shared across classification and embeddings |

**Entry points** via `MLContextExtensions`:

```csharp
mlContext.Transforms.OnnxAudioClassification(options)
mlContext.Transforms.OnnxAudioEmbedding(options)
mlContext.Transforms.OnnxVad(options)
mlContext.Transforms.OnnxWhisper(options)
mlContext.Transforms.SpeechT5Tts(options)
mlContext.Transforms.KittenTts(options)
mlContext.Transforms.SpeechToText(client, options)
```

### Layer 2: ORT GenAI ASR (`MLNet.ASR.OnnxGenAI`)

Separate package for Whisper speech-to-text via ONNX Runtime GenAI. ORT GenAI handles the autoregressive decoder loop internally — simplest local ASR with minimal code.

**Dependencies:** Microsoft.ML 5.0.0, Microsoft.ML.OnnxRuntimeGenAI.Managed 0.12.1, Microsoft.Extensions.AI.Abstractions 10.4.1

**Project reference:** MLNet.Audio.Core (NOT MLNet.AudioInference.Onnx)

| Type | Description |
|------|-------------|
| `OnnxSpeechToTextEstimator` | `IEstimator<OnnxSpeechToTextTransformer>` |
| `OnnxSpeechToTextTransformer` | Full audio-to-text pipeline: `Transcribe()`, `TranscribeWithTimestamps()`, `Transform()` (ML.NET), `RunGeneration()` (core ORT GenAI logic) |
| `OnnxSpeechToTextOptions` | Config: model path, language, max length, mel bins, sample rate, multilingual/translate flags |
| `OnnxSpeechToTextClient` | `ISpeechToTextClient` implementation: `GetTextAsync()`, `GetStreamingTextAsync()` |
| `MLContextExtensions` | `mlContext.Transforms.OnnxSpeechToText(options)` |

**Why separate?** ORT GenAI has different native dependencies (platform-specific CPU/CUDA/DirectML packages) that would force all `MLNet.AudioInference.Onnx` consumers to take that dependency even if they only need classification or embeddings.

### Layer 3: MEAI Integration (across packages)

[Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) abstractions bridged throughout:

| MEAI Interface | Implementation | Package |
|----------------|---------------|---------|
| `IEmbeddingGenerator<AudioData, Embedding<float>>` | `OnnxAudioEmbeddingGenerator` | MLNet.AudioInference.Onnx |
| `ISpeechToTextClient` | `OnnxSpeechToTextClient` | MLNet.ASR.OnnxGenAI |
| `ISpeechToTextClient` | `OnnxWhisperSpeechToTextClient` | MLNet.AudioInference.Onnx |
| `ITextToSpeechClient` | `OnnxTextToSpeechClient` | MLNet.AudioInference.Onnx |
| `IVoiceActivityDetector` (custom) | `OnnxVadTransformer` | MLNet.AudioInference.Onnx |

`OnnxTextToSpeechClient` implements the official MEAI `ITextToSpeechClient` from 10.4.1. It accepts both `OnnxSpeechT5Options` and `OnnxKittenTtsOptions` — one client for all local TTS backends via the internal `IOnnxTtsSynthesizer` interface. SpeechT5 uses 3 models (encoder + decoder + vocoder, 16 kHz output); KittenTTS uses 1 model + espeak-ng for phonemization (24 kHz output).

### Layer 4: DataIngestion (`MLNet.Audio.DataIngestion`)

[Microsoft.Extensions.DataIngestion](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion.Abstractions) integration — proving DataIngestion is modality-agnostic, not just for text/PDF.

**Dependencies:** Microsoft.Extensions.DataIngestion.Abstractions 10.4.0-preview.1.26160.2, Microsoft.Extensions.AI.Abstractions 10.4.1

**Project reference:** MLNet.Audio.Core

| Type | Base Class | Purpose |
|------|-----------|---------|
| `AudioDocumentReader` | `IngestionDocumentReader` | Reads WAV files → `IngestionDocument`. Stores decoded `AudioData` in `section.Metadata["audio"]` |
| `AudioSegmentChunker` | `IngestionChunker<AudioData>` | Fixed time-window segmentation. Produces `IngestionChunk<AudioData>` with timing metadata |
| `AudioEmbeddingChunkProcessor` | `IngestionChunkProcessor<AudioData>` | Enriches chunks with embeddings via `IEmbeddingGenerator<AudioData, Embedding<float>>`. Stores `float[]` in `chunk.Metadata["embedding"]` |

**DataIngestion pipeline flow:**

```
WAV File → AudioDocumentReader    → IngestionDocument (AudioData in section metadata)
         → AudioSegmentChunker    → IAsyncEnumerable<IngestionChunk<AudioData>> (2s windows)
         → AudioEmbeddingChunkProcessor → IngestionChunk<AudioData> + embedding in metadata
```

**Key design decisions:**

- **`IngestionChunk<AudioData>`** — uses the generic content type for type-safe audio flow (not `IngestionChunk<string>`)
- **Section metadata bridge** — `section.Metadata["audio"]` passes `AudioData` from reader to chunker (IngestionDocument has no generic content property)
- **MEAI connection** — the processor takes `IEmbeddingGenerator<AudioData, Embedding<float>>` connecting Layer 4 → Layer 3 → Layer 1
- **All async streaming** — chunker and processor use `IAsyncEnumerable` for memory-efficient processing of large audio files

### Layer 5: Model Packages + Model Garden (future)

NuGet packages wrapping HuggingFace models with auto-download and caching. Compatible with the ModelPackages SDK pattern for model distribution.

## Three-Stage Pipeline Pattern

Every transform follows a consistent three-stage pattern:

```
Stage 1: Feature Extraction  (audio → numeric features)
Stage 2: ONNX Scoring        (features → model output)
Stage 3: Post-processing      (model output → typed result)
```

For encoder-only transforms, these stages are implemented as **separate, composable sub-transforms** chained via lazy `IDataView`:

- **Stage 1:** `AudioFeatureExtractionTransformer` — audio PCM → mel spectrogram features
- **Stage 2:** `OnnxAudioScoringTransformer` — features → raw ONNX model output
- **Stage 3:** Task-specific post-processor (`AudioClassificationPostProcessingTransformer` or `AudioEmbeddingPoolingTransformer`)

The facade `Fit()` chains these sub-estimators, and the facade `Transform()` chains the resulting sub-transformers. Each sub-transform wraps its output in a lazy `IDataView` — no computation occurs until the cursor is iterated.

### Concrete examples per task

**Classification** (`OnnxAudioClassificationTransformer`):

```
AudioData → MelSpectrogramExtractor (128 mel bins)
          → ONNX encoder (e.g., AST)
          → TensorPrimitives.Softmax() + label index mapping
          → (PredictedLabel, Probabilities, Score)
```

**Embeddings** (`OnnxAudioEmbeddingTransformer`):

```
AudioData → MelSpectrogramExtractor
          → ONNX encoder (e.g., CLAP, Wav2Vec2)
          → AudioPoolingStrategy (MeanPooling/ClsToken/MaxPooling)
          → TensorPrimitives L2 normalization
          → float[] embedding
```

**VAD** (`OnnxVadTransformer`):

```
AudioData → PCM frames (512 samples / 32ms windows)
          → ONNX model (Silero VAD, stateful h/c state)
          → Threshold (0.5) + MinSpeechDuration + MinSilenceDuration + SpeechPad
          → SpeechSegment[] (Start, End, Confidence)
```

**ASR — Whisper** (`OnnxWhisperTransformer`):

```
AudioData → WhisperFeatureExtractor (80/128 mel bins, 3000 frames)
          → ONNX encoder → encoder hidden states [1, 1500, hidden_dim]
          → Decoder loop (WhisperKvCacheManager): greedy argmax or temperature sampling
          → WhisperTokenizer.Decode() / DecodeWithTimestamps() / DecodeToSegments()
          → string or TranscriptionSegment[]
```

**TTS — SpeechT5** (`OnnxSpeechT5TtsTransformer`):

```
string → SentencePiece tokenizer (Microsoft.ML.Tokenizers) → token IDs
       → ONNX encoder → encoder hidden states
       → Decoder loop (KV cache + speaker embedding): autoregressive mel frame generation
       → Stop token threshold check (0.5)
       → ONNX vocoder (postnet + HiFi-GAN) → PCM waveform
       → AudioData (float[] samples, 16kHz)
```

**TTS — KittenTTS** (via `OnnxTextToSpeechClient` + `OnnxKittenTtsOptions`):

```
string → espeak-ng phonemization → phoneme IDs
       → ONNX model (single model, no separate vocoder) → PCM waveform
       → AudioData (float[] samples, 24kHz)
```

KittenTTS is a lightweight alternative to SpeechT5: single model (no encoder/decoder/vocoder split), espeak-ng handles phonemization instead of SentencePiece, and output is 24 kHz. Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo.

## Text vs Audio Architecture Comparison

| Concern | Text World | Audio World |
|---------|-----------|-------------|
| Input type | `string` | `AudioData` (float[] + sample rate) |
| Stage 1 | Tokenization (BPE → token IDs) | Feature Extraction (mel spectrogram) |
| Stage 2 | ONNX Scoring | ONNX Scoring |
| Stage 3 | Post-processing (softmax, pooling) | Post-processing (softmax, pooling, token decoding, vocoder) |
| Model archs | Mostly encoder-only | Encoder-only + encoder-decoder + encoder-decoder-vocoder |
| Key abstraction | `Tokenizer` (`Microsoft.ML.Tokenizers`) | `AudioFeatureExtractor` |
| MEAI interfaces | `IEmbeddingGenerator`, `IChatClient` | `IEmbeddingGenerator`, `ISpeechToTextClient`, `ITextToSpeechClient` |
| KV cache | Not needed (encoder-only) | Required for Whisper ASR and SpeechT5 TTS |
| Output modality | Always text | Text (ASR), audio (TTS), labels (classification), vectors (embeddings), segments (VAD) |

## Encoder-Decoder with KV Cache

Both Whisper (ASR) and SpeechT5 (TTS) use autoregressive decoding with KV cache, managed by `WhisperKvCacheManager`:

### Cache structure per decoder layer

```
past_key_values.{i}.decoder.key    [1, num_heads, seq_len, head_dim]   — grows each step
past_key_values.{i}.decoder.value  [1, num_heads, seq_len, head_dim]   — grows each step
past_key_values.{i}.encoder.key    [1, num_heads, 1500, head_dim]      — fixed after step 1
past_key_values.{i}.encoder.value  [1, num_heads, 1500, head_dim]      — fixed after step 1
```

### Two-phase decode

1. **First step** (`use_cache_branch = false`): Decoder computes full attention over all positions. Cross-attention KV pairs are computed from encoder output and cached.
2. **Subsequent steps** (`use_cache_branch = true`): Decoder only processes the new token. Decoder self-attention KV grows by one position. Cross-attention KV is reused unchanged.

### Output naming convention

The decoder's `present.{i}.{decoder/encoder}.{key/value}` outputs become the next step's `past_key_values.{i}.{decoder/encoder}.{key/value}` inputs. `WhisperKvCacheManager.UpdateFromOutputs()` extracts these tensors and `BuildDecoderInputs()` re-packages them for the next step.

### Model dimension auto-detection

`WhisperKvCacheManager.DetectFromModel(decoderSession)` inspects the ONNX model's input metadata to discover `(numLayers, numHeads, headDim)` from the `past_key_values.0.decoder.key` input shape, eliminating manual configuration.

### Whisper and SpeechT5 as mirror images

```
Whisper (ASR):   audio → mel → encoder → decoder(KV) → text tokens → text
SpeechT5 (TTS): text → tokenize → encoder → decoder(KV) → mel frames → vocoder → audio
```

Both use the same KV cache pattern; the decoder loop differs only in what it produces (token IDs vs. mel spectrogram frames) and its stopping condition (end-of-text token vs. stop probability threshold).

## Three ASR Approaches

The repository provides three approaches to speech-to-text, each optimizing for different trade-offs:

### 1. Provider-agnostic (`SpeechToTextClientTransformer`)

Wraps any `ISpeechToTextClient` implementation (Azure Speech, OpenAI Whisper API, local providers).

```csharp
ISpeechToTextClient client = /* any provider */;
var pipeline = mlContext.Transforms.SpeechToText(client);
```

**Best for:** Cloud APIs, swappable providers, when you don't want to manage local models.

### 2. ORT GenAI (`OnnxSpeechToTextTransformer` in `MLNet.ASR.OnnxGenAI`)

ORT GenAI handles the entire encoder-decoder loop internally.

```csharp
var pipeline = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-base",
    Language = "en"
});
```

**Best for:** Simple local deployment, minimal code, when you don't need fine-grained control over the decode loop.

### 3. Raw ONNX (`OnnxWhisperTransformer` in `MLNet.AudioInference.Onnx`)

Manual encoder/decoder/KV cache with `WhisperFeatureExtractor`, `WhisperTokenizer`, and `WhisperKvCacheManager`.

```csharp
var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-base/encoder_model.onnx",
    DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
    Language = "en"
});
```

**Best for:** Full control, custom sampling strategies, using all audio primitives, and understanding the complete pipeline end-to-end.

### Why three?

| Approach | Runtime dependency | Control level | Code complexity | Decode loop managed by |
|----------|-------------------|---------------|-----------------|----------------------|
| Provider-agnostic | None (pluggable) | Low | Low | Provider |
| ORT GenAI | ORT GenAI native libs | Medium | Low | ORT GenAI |
| Raw ONNX | ORT Managed only | Full | High | Our code (`WhisperKvCacheManager`) |

## Package Dependency Graph

```
┌──────────────────────────────────────────────────┐
│                External NuGet                    │
│  NWaves 0.9.6                                    │
│  System.Numerics.Tensors 10.0.3                  │
│  Microsoft.ML.Tokenizers 2.0.0                   │
└──────────────────────┬───────────────────────────┘
                       │
              ┌────────▼────────┐
              │ MLNet.Audio.Core │  ← Layer 0: Audio primitives
              │  (no ML.NET dep) │
              └──┬──────┬────┬──┘
                 │      │    │
  ┌──────────────▼──┐   │   │
  │ MLNet.Audio     │   │   │
  │ Tokenizers      │   │   │  ← Layer 0.5: ML.Tokenizers extensions
  │                 │   │   │
  │ + ML.Tokenizers │   │   │
  └──────┬──────────┘   │   │
         │              │   │
    ┌────▼───────────▼──┐  ┌▼────────────────────────┐
    │ MLNet.Audio       │  │ MLNet.ASR.OnnxGenAI      │  ← Layer 2
    │ Inference.Onnx    │  │                          │
    │                   │  │ + Microsoft.ML 5.0.0     │
    │ + Microsoft.ML    │  │ + ORT GenAI 0.12.1       │
    │ + ORT 1.24.2      │  │ + MEAI 10.4.1            │
    │ + ML.Tokenizers   │  └──────────────────────────┘
    │ + MEAI 10.4.1     │
    │                   │  ← Layer 1: Inference transforms
    └───────────────────┘
                         │
    ┌────────────────────▼─────────────────────────┐
    │ MLNet.Audio.DataIngestion                    │  ← Layer 4
    │                                              │
    │ + DataIngestion.Abstractions 10.3.0-preview  │
    │ + MEAI.Abstractions 10.4.1                   │
    └──────────────────────────────────────────────┘

  NOTE: MLNet.ASR.OnnxGenAI depends on MLNet.Audio.Core directly,
        NOT on MLNet.AudioInference.Onnx. This keeps the ORT GenAI
        native dependency isolated.

  NOTE: MLNet.Audio.DataIngestion depends on MLNet.Audio.Core and
        MEAI.Abstractions only — it does NOT depend on ML.NET or
        any ONNX runtime. The embedding generator is injected.

  NOTE: MLNet.Audio.Tokenizers extends Microsoft.ML.Tokenizers
        with audio-specific tokenizer implementations (e.g.,
        SentencePiece Char model support for SpeechT5).
```

### Evaluation Strategy

**Encoder-only transforms** (classification, embeddings) use a **composed lazy** pattern. `Fit()` chains 3 sub-estimators, and `Transform()` chains 3 lazy `IDataView` wrappers — computation happens only when a cursor iterates over the output. This matches the text transform architecture where `OnnxTextEmbeddingTransformer` chains `TextTokenizerTransformer` → `OnnxTextModelScorerTransformer` → `EmbeddingPoolingTransformer`.

**Encoder-decoder transforms** (Whisper ASR, SpeechT5 TTS) and **stateful transforms** (Silero VAD) use **eager evaluation** via `LoadFromEnumerable()`. Autoregressive decoding loops with KV cache state don't map cleanly to per-row lazy cursors.

Direct convenience APIs (`Classify()`, `GenerateEmbeddings()`, `Transcribe()`, `Synthesize()`) bypass IDataView entirely — they're always eager.

## ML.NET Pipeline Composition

Transforms compose using standard ML.NET `Append()`:

```csharp
// Single task — audio classification
var pipeline = mlContext.Transforms.OnnxAudioClassification(new OnnxAudioClassificationOptions
{
    ModelPath = "models/ast-audioset.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Labels = ["Speech", "Music", "Silence"]
});

// Single task — raw Whisper ASR
var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-base/encoder_model.onnx",
    DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
    Language = "en"
});

// Multi-modal chain: ASR then TTS (voice conversion)
var pipeline = mlContext.Transforms
    .OnnxWhisper(whisperOptions)           // Audio → Text
    .Append(mlContext.Transforms
        .SpeechT5Tts(ttsOptions));         // Text → Audio

// MEAI embedding generator (outside ML.NET pipeline)
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(embeddingTransformer, modelId: "clap-base");
var embeddings = await generator.GenerateAsync([audio1, audio2]);
```

## Solution Structure

```
mlnet-audio-custom-transforms/
├── MLNet.Audio.slnx                           # Solution file (slnx format)
├── README.md
├── nuget.config
│
├── docs/
│   ├── plan.md                                # Roadmap and implementation plan
│   └── architecture.md                        # This document
│
├── src/
│   ├── Directory.Build.props                  # Shared: author, MIT license, MinVer versioning
│   │
│   ├── MLNet.Audio.Core/                      # Layer 0: Audio primitives
│   │   ├── MLNet.Audio.Core.csproj
│   │   ├── AudioData.cs                       # Core type: float[] samples + sample rate
│   │   ├── AudioIO.cs                         # WAV read/write, resample, mono conversion
│   │   ├── AudioFeatureExtractor.cs           # Abstract base for feature extraction
│   │   ├── MelSpectrogramExtractor.cs         # Generic mel spectrogram via NWaves
│   │   ├── WhisperFeatureExtractor.cs         # Whisper-specific (30s chunks, 80/128 mel)
│   │   └── Tokenizers/
│   │       ├── WhisperTokenizer.cs            # BPE tokenizer + timestamps + language codes
│   │       └── AudioCodecTokenizer.cs         # Abstract base for neural audio codecs
│   │
│   ├── MLNet.Audio.Tokenizers/                # Layer 0.5: ML.Tokenizers extensions
│   │   ├── MLNet.Audio.Tokenizers.csproj
│   │   ├── SentencePieceCharTokenizer.cs      # Extends Tokenizer for SentencePiece Char models
│   │   └── SentencePieceModelParser.cs        # Minimal protobuf parser for .model files
│   │
│   ├── MLNet.AudioInference.Onnx/             # Layer 1: Inference transforms
│   │   ├── MLNet.AudioInference.Onnx.csproj
│   │   ├── MLContextExtensions.cs             # mlContext.Transforms.Onnx*() entry points
│   │   ├── Classification/
│   │   │   ├── OnnxAudioClassificationEstimator.cs
│   │   │   ├── OnnxAudioClassificationTransformer.cs
│   │   │   └── OnnxAudioClassificationOptions.cs
│   │   ├── Embeddings/
│   │   │   ├── OnnxAudioEmbeddingEstimator.cs
│   │   │   ├── OnnxAudioEmbeddingTransformer.cs
│   │   │   └── OnnxAudioEmbeddingOptions.cs   # Includes AudioPoolingStrategy enum
│   │   ├── VAD/
│   │   │   ├── OnnxVadEstimator.cs
│   │   │   ├── OnnxVadTransformer.cs          # Implements ITransformer + IVoiceActivityDetector
│   │   │   ├── OnnxVadOptions.cs
│   │   │   └── IVoiceActivityDetector.cs      # SpeechSegment record, VadOptions
│   │   ├── ASR/
│   │   │   ├── SpeechToTextClientEstimator.cs # Provider-agnostic (wraps ISpeechToTextClient)
│   │   │   ├── SpeechToTextClientTransformer.cs
│   │   │   ├── SpeechToTextClientOptions.cs
│   │   │   ├── OnnxWhisperEstimator.cs        # Raw ONNX Whisper
│   │   │   ├── OnnxWhisperTransformer.cs      # Full encoder/decoder/KV cache pipeline
│   │   │   ├── OnnxWhisperOptions.cs
│   │   │   └── WhisperKvCacheManager.cs       # KV cache state management
│   │   ├── TTS/
│   │   │   ├── OnnxSpeechT5TtsEstimator.cs
│   │   │   ├── OnnxSpeechT5TtsTransformer.cs  # Encoder → decoder(KV) → vocoder pipeline
│   │   │   ├── OnnxSpeechT5Options.cs
│   │   │   ├── IOnnxTtsSynthesizer.cs         # Internal interface — abstracts TTS backends
│   │   │   └── OnnxTextToSpeechClient.cs      # ITextToSpeechClient (official MEAI) — SpeechT5 + KittenTTS
│   │   ├── MEAI/
│   │   │   └── OnnxAudioEmbeddingGenerator.cs # IEmbeddingGenerator<AudioData, Embedding<float>>
│   │   └── Shared/
│   │       └── EnCodecTokenizer.cs            # Meta EnCodec neural audio codec (RVQ)
│   │
│   └── MLNet.ASR.OnnxGenAI/                   # Layer 2: ORT GenAI ASR
│       ├── MLNet.ASR.OnnxGenAI.csproj
│       ├── MLContextExtensions.cs             # mlContext.Transforms.OnnxSpeechToText()
│       ├── OnnxSpeechToTextEstimator.cs
│       ├── OnnxSpeechToTextTransformer.cs     # ORT GenAI handles decoder loop
│       ├── OnnxSpeechToTextOptions.cs
│       └── OnnxSpeechToTextClient.cs          # ISpeechToTextClient implementation
│
│   └── MLNet.Audio.DataIngestion/             # Layer 4: DataIngestion
│       ├── MLNet.Audio.DataIngestion.csproj
│       └── AudioIngestionComponents.cs        # AudioDocumentReader, AudioSegmentChunker,
│                                              # AudioEmbeddingChunkProcessor
│
└── samples/
    ├── AudioClassification/                   # AST model classification
    ├── AudioEmbeddings/                       # CLAP/Wav2Vec2 embeddings + MEAI
    ├── VoiceActivityDetection/                # Silero VAD speech segment detection
    ├── SpeechToText/                          # Provider-agnostic ASR patterns
    ├── WhisperTranscription/                  # ORT GenAI Whisper (simple)
    ├── WhisperRawOnnx/                        # Raw ONNX Whisper (full control)
    ├── TextToSpeech/                          # SpeechT5 local TTS
    ├── KittenTTS/                             # KittenTTS lightweight TTS + espeak-ng
    └── AudioDataIngestion/                    # DataIngestion: Read → Chunk → Embed
```
