# MLNet.AudioInference.Onnx

**Multi-task audio inference transforms for ML.NET using local ONNX models.**

This is **Layer 1** of the audio inference architecture — the core library that provides `IEstimator<T>` / `ITransformer` implementations for five audio tasks: classification, embeddings, voice activity detection, speech-to-text, and text-to-speech. Every task runs locally through ONNX Runtime with no cloud dependencies. Consumed directly by samples and indirectly by the DataIngestion layer via MEAI abstractions.

---

## Why This Package Exists

### Why not just call ONNX Runtime directly?

You absolutely *can* call `InferenceSession.Run()` yourself. But then you're responsible for:

- **Feature extraction** — converting raw PCM audio to the exact mel spectrogram format each model expects (80 bins? 128? Log-mel? Normalized to what range?)
- **Tensor marshalling** — building `DenseTensor<float>` with the right shape (`[1, frames, bins]` vs `[1, bins, frames]` vs `[1, 1, frames, bins]`), managing `OrtValue` lifetimes
- **Post-processing** — softmax, argmax, label mapping, embedding pooling, L2 normalization, autoregressive decode loops with KV cache management
- **Composability** — if you want to chain "load audio → extract features → classify → filter by confidence," you build that plumbing yourself for every pipeline

ML.NET's `IEstimator` / `ITransformer` pattern gives you all of this for free:

| ML.NET Pattern | What it buys you |
|---|---|
| **Pipeline composition** | Chain transforms with `.Append()` — the framework handles column threading between stages |
| **Schema validation** | `GetOutputSchema()` catches column-name mismatches and type errors *before* you run data through the pipeline |
| **Lazy evaluation** | `IDataView` is pull-based — rows are processed on demand, so a 10-million-row dataset doesn't load into memory at once |
| **Fit/Transform** | `Fit()` captures model-specific state (ONNX session, discovered dimensions, tokenizer state); `Transform()` applies it repeatedly without re-initialization |
| **Column annotations** | Metadata flows between stages (e.g., `HiddenDim` from the scorer tells the pooler what embedding dimension to expect) |

This package wraps all the audio-specific complexity behind that standard pattern, so a developer who already knows ML.NET text transforms can work with audio the same way.

---

## Key Concepts

### The Three-Stage Pipeline

Encoder-only audio models (classification, embeddings) decompose into three composable sub-transforms:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
│ Stage 1: Feature     │    │ Stage 2: ONNX         │    │ Stage 3: Post-Processing  │
│ Extraction           │───▶│ Scoring               │───▶│ (softmax/pooling/labels)  │
│                      │    │                        │    │                           │
│ AudioData → mel      │    │ mel features → raw     │    │ raw scores → final output │
│ spectrogram features │    │ model output scores    │    │ (labels, embeddings, etc) │
└─────────────────────┘    └──────────────────────┘    └──────────────────────────┘
   AudioFeature               OnnxAudioScoring            Classification: softmax +
   ExtractionEstimator         Estimator                     argmax + label mapping
                                                           Embeddings: mean/CLS/max
                                                             pooling + L2 normalize
```

Each stage is a full `IEstimator<T>` / `ITransformer` pair. The intermediate data flows through `IDataView` columns — Stage 1 writes a `"Features"` column, Stage 2 reads it and writes `"Scores"`, Stage 3 reads `"Scores"` and writes the final output columns.

### Composed (Lazy) vs. Monolithic (Eager)

This is the most important architectural distinction in the library.

**Composed / Lazy** (encoder-only models: classification, embeddings):

The facade estimator's `Fit()` creates three sub-transformers and chains them. When you call `Transform()`, it returns an `IDataView` that wraps the input through three lazy layers. No data moves until a downstream consumer pulls rows through a cursor. This is the standard ML.NET pattern — it means you can `.Append()` additional transforms (normalization, filtering) and the entire pipeline stays lazy.

```csharp
// Inside OnnxAudioClassificationTransformer.Transform():
var features = _featureTransformer.Transform(input);       // returns lazy IDataView
var scores = _scorerTransformer.Transform(features);        // wraps the above lazily
var results = _postProcessTransformer.Transform(scores);    // wraps again lazily
return results;  // nothing has actually executed yet!
```

Data only flows when you enumerate the cursor (e.g., `mlContext.Data.CreateEnumerable<T>(results)`).

**Monolithic / Eager** (encoder-decoder models: Whisper ASR, SpeechT5 TTS, VAD):

Encoder-decoder models like Whisper and SpeechT5 use **autoregressive generation** — the output of token *N* is the input to token *N+1*. This is fundamentally incompatible with lazy IDataView decomposition because:

1. **The decode loop is stateful**: each step updates a KV cache that the next step reads. You can't represent "run the decoder in a loop until EOS" as a chain of lazy column transforms.
2. **Output length is data-dependent**: Whisper generates a variable number of tokens per audio input. SpeechT5 generates a variable number of mel frames per text input. `IDataView` schemas are fixed — you can't have a column whose type changes per row.
3. **The stages are causally coupled**: the encoder output feeds the decoder's cross-attention at every step, not just once. Breaking this into separate `ITransformer`s would require materializing the entire encoder output and passing it as a column, defeating the purpose.

So these transforms use **eager evaluation** in `Transform()` — they read all input rows, run the full pipeline (feature extraction → encoder → decode loop → post-processing), and return a new `IDataView` loaded from the results via `LoadFromEnumerable()`.

```csharp
// Inside OnnxWhisperTransformer.Transform():
var audioSamples = ReadAudioColumn(input);    // materialize all input
var transcriptions = Transcribe(audioInputs); // run full encoder-decoder pipeline
return _mlContext.Data.LoadFromEnumerable(outputRows); // return materialized results
```

VAD (Silero) is also eager because it is **stateful across frames** — it maintains LSTM hidden state (`h`, `c`) as it slides a window across the audio. Each frame's speech probability depends on the hidden state from the previous frame. This sequential state dependency cannot be decomposed into independent lazy stages.

### Column Annotations for Inter-Stage Metadata

When `OnnxAudioScoringTransformer` loads an ONNX model, it inspects the model's output tensor shape to discover:
- **`HiddenDim`**: the last dimension of the output shape (e.g., 768 for Wav2Vec2-base, 1024 for HuBERT-large)
- **`HasPooledOutput`**: whether the output is `[batch, hidden]` (already pooled) vs `[batch, seq, hidden]` (needs pooling)

These values are attached as **column annotations** on the `"Scores"` output column:

```csharp
// In OnnxAudioScoringTransformer.BuildScoreAnnotations():
metaBuilder.Add<int>("HiddenDim", NumberDataViewType.Int32,
    (ref int value) => value = hiddenDim);
metaBuilder.Add<bool>("HasPooledOutput", BooleanDataViewType.Instance,
    (ref bool value) => value = hasPooledOutput);
```

The downstream `AudioEmbeddingPoolingEstimator` reads these annotations during `Fit()`:

```csharp
// In AudioEmbeddingPoolingEstimator.Fit():
var annotations = col.Value.Annotations;
annotations.GetValue("HiddenDim", ref hiddenDim);
annotations.GetValue("HasPooledOutput", ref hasPooledOutput);
```

This is the ML.NET standard pattern for inter-stage metadata discovery. It means the pooling stage **auto-configures** based on whatever model the scorer loaded — you never manually specify `HiddenDim = 768`. Swap from Wav2Vec2-base (768-dim) to HuBERT-large (1024-dim) and the pooler adapts automatically.

---

## Architecture & Design Decisions

### Why Sub-Transforms Are Public

`AudioFeatureExtractionEstimator`, `OnnxAudioScoringEstimator`, `AudioEmbeddingPoolingEstimator`, and `AudioClassificationPostProcessingEstimator` are all `public`. This is deliberate — it enables power users to compose custom pipelines:

```csharp
// Use the shared scorer with a custom post-processing stage
var scorer = new OnnxAudioScoringEstimator(env, scorerOptions);
var customPostProcess = new MyCustomPostProcessEstimator(env);
var pipeline = scorer.Append(customPostProcess);
```

If these were `internal`, you'd be locked into the facade estimators' fixed 3-stage composition. Making them public follows ML.NET's convention where sub-transforms like `NormalizingEstimator` and `KeyToValueMappingEstimator` are individually usable.

### Why Facade Estimators Exist

While sub-transforms are public, most users shouldn't need to wire them manually. The facade estimators (`OnnxAudioClassificationEstimator`, `OnnxAudioEmbeddingEstimator`) provide:

1. **A single options object** — instead of configuring three separate options classes, you set `OnnxAudioClassificationOptions` once and the facade wires everything
2. **Correct inter-stage column names** — the facade hardcodes `"Features"` and `"Scores"` as intermediate column names, so you don't accidentally misname them
3. **Proper initialization order** — Stage 2 must `Fit()` on the output of Stage 1 (to discover tensor shapes); Stage 3 must `Fit()` on the output of Stage 2 (to discover `HiddenDim`). The facade handles this sequencing.

### SchemaShape.Column Internal Constructor Workaround

`SchemaShape.Column` — ML.NET's compile-time schema descriptor — has an `internal` constructor. Custom `IEstimator` implementations outside `Microsoft.ML` cannot construct columns through the normal API. This library uses reflection to access the 5-parameter constructor:

```csharp
var colCtor = typeof(SchemaShape.Column)
    .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
    .First(c => c.GetParameters().Length == 5);

columns[name] = (SchemaShape.Column)colCtor.Invoke([
    name, vectorKind, dataViewType, isKey, metadataShape
]);
```

This is the same workaround used by the text inference custom transforms project. It's fragile (could break on internal API changes) but is currently the only way to implement `GetOutputSchema()` in a custom estimator. Every estimator in this library uses this pattern.

### WhisperKvCacheManager — KV Cache for Encoder-Decoder Models

Autoregressive decoder models (Whisper, SpeechT5) need a **key-value cache** to avoid recomputing attention over all previous tokens at each step. `WhisperKvCacheManager` manages this:

- **Decoder self-attention KV**: grows each step (new token attends to all previous tokens). Shape per layer: `[1, num_heads, seq_len, head_dim]`
- **Cross-attention KV**: computed once from the encoder output, then reused at every decode step. Shape per layer: `[1, num_heads, encoder_seq_len, head_dim]`

The merged decoder model (`decoder_model_merged.onnx`) uses a `use_cache_branch` boolean input to switch between first-step mode (no cache, compute cross-attention) and subsequent-step mode (use cache, skip cross-attention recomputation).

`WhisperKvCacheManager.DetectFromModel()` auto-discovers `numLayers`, `numHeads`, and `headDim` by inspecting the decoder's `past_key_values.{i}.decoder.key` input metadata. This means the same code works for Whisper tiny (4 layers, 6 heads) through Whisper large-v3 (32 layers, 20 heads) without configuration changes.

The design note in the source acknowledges this pattern is reusable beyond Whisper — SpeechT5 TTS uses the same KV cache pattern inline (identical logic, not yet extracted to a shared manager). A future generalization could be `KvCacheManager<T>` or an `IKvCacheManager` interface.

### ITextToSpeechClient — A Prototype Interface

MEAI (`Microsoft.Extensions.AI`) defines `ISpeechToTextClient` for ASR but does **not yet define** `ITextToSpeechClient` for TTS (as of version 10.x). This library defines its own prototype:

```csharp
public interface ITextToSpeechClient : IDisposable
{
    Task<TextToSpeechResponse> GetAudioAsync(string text, ...);
    IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(string text, ...);
    TextToSpeechClientMetadata Metadata { get; }
}
```

It follows MEAI conventions (async, options pattern, metadata property, streaming via `IAsyncEnumerable`). When MEAI adds an official `ITextToSpeechClient`, this interface should be replaced.

### IVoiceActivityDetector — A Custom Interface

MEAI has no equivalent for voice activity detection (it's not a generative AI task). `IVoiceActivityDetector` is a custom interface:

```csharp
public interface IVoiceActivityDetector : IDisposable
{
    IAsyncEnumerable<SpeechSegment> DetectSpeechAsync(
        Stream audioStream, VadOptions? options = null, CancellationToken cancellationToken = default);
}
```

`OnnxVadTransformer` implements both `ITransformer` (for ML.NET pipelines) and `IVoiceActivityDetector` (for direct use). This dual-interface pattern lets the same object participate in both ML.NET pipeline composition and standalone MEAI-style usage.

### ML.NET Naming Convention

All types follow ML.NET's gerund-form naming convention: `Scoring`, `Pooling`, `PostProcessing`, `Extraction`. This matches the framework's own `NormalizingEstimator`, `TokenizingByCharactersEstimator`, etc. The naming communicates that these are *ongoing processes* (transforms), not one-shot actions.

---

## Supported Tasks

| Task | Entry Point | Estimator | Transformer | Evaluation | Model Architecture |
|---|---|---|---|---|---|
| **Audio Classification** | `OnnxAudioClassification()` | `OnnxAudioClassificationEstimator` | `OnnxAudioClassificationTransformer` | Lazy (3-stage composed) | Wav2Vec2, HuBERT, AST, Audio Spectrogram Transformer |
| **Audio Embeddings** | `OnnxAudioEmbedding()` | `OnnxAudioEmbeddingEstimator` | `OnnxAudioEmbeddingTransformer` | Lazy (3-stage composed) | Wav2Vec2, HuBERT, CLAP, WavLM |
| **Voice Activity Detection** | `OnnxVad()` | `OnnxVadEstimator` | `OnnxVadTransformer` | Eager (stateful LSTM) | Silero VAD v5 |
| **Speech-to-Text (Whisper)** | `OnnxWhisper()` | `OnnxWhisperEstimator` | `OnnxWhisperTransformer` | Eager (autoregressive) | Whisper (tiny → large-v3) |
| **Speech-to-Text (MEAI)** | `SpeechToText()` | `SpeechToTextClientEstimator` | `SpeechToTextClientTransformer` | Eager (provider call) | Any ISpeechToTextClient provider |
| **Text-to-Speech** | `SpeechT5Tts()` | `OnnxSpeechT5TtsEstimator` | `OnnxSpeechT5TtsTransformer` | Eager (autoregressive) | SpeechT5 (encoder-decoder-vocoder) |

---

## API Surface

### Entry Points — `MLContextExtensions`

Six extension methods on `TransformsCatalog` (accessed via `mlContext.Transforms`):

```csharp
mlContext.Transforms.OnnxAudioClassification(options)   // → OnnxAudioClassificationEstimator
mlContext.Transforms.OnnxAudioEmbedding(options)         // → OnnxAudioEmbeddingEstimator
mlContext.Transforms.OnnxVad(options)                     // → OnnxVadEstimator
mlContext.Transforms.OnnxWhisper(options)                 // → OnnxWhisperEstimator
mlContext.Transforms.SpeechToText(client, options?)       // → SpeechToTextClientEstimator
mlContext.Transforms.SpeechT5Tts(options)                 // → OnnxSpeechT5TtsEstimator
```

> **Implementation note:** `GetMLContext()` uses reflection to extract the `MLContext` from `TransformsCatalog.Environment` (a non-public property). This is the same approach used by the text inference custom transforms project. A fallback creates a new `MLContext` if reflection fails.

### Shared/ — Reusable Sub-Transforms

| Type | Role |
|---|---|
| `AudioFeatureExtractionEstimator` / `Transformer` | **Stage 1**: Wraps an `AudioFeatureExtractor` (from Audio.Core) to convert raw PCM audio into mel spectrogram features. Output is a flattened `VBuffer<float>` of shape `[frames × bins]`. |
| `OnnxAudioScoringEstimator` / `Transformer` | **Stage 2**: Loads an ONNX model, runs inference on feature input, produces raw model output scores. Auto-detects tensor names and output shape. Embeds `HiddenDim` and `HasPooledOutput` as column annotations. Owns the `InferenceSession` lifecycle. |
| `EnCodecTokenizer` | Neural audio codec tokenizer (Meta EnCodec). Converts audio ↔ discrete codes via Residual Vector Quantization. Extends `AudioCodecTokenizer` from Audio.Core. Prototypes what could become a first-class audio tokenizer in `Microsoft.ML.Tokenizers`. |

### Classification/

| Type | Role |
|---|---|
| `OnnxAudioClassificationEstimator` | Facade: composes 3 sub-transforms, single `OnnxAudioClassificationOptions` config. |
| `OnnxAudioClassificationTransformer` | Chains sub-transforms lazily. Also exposes `Classify(IReadOnlyList<AudioData>)` for direct use outside ML.NET pipelines. |
| `AudioClassificationPostProcessingEstimator` / `Transformer` | **Stage 3**: Applies `TensorPrimitives.SoftMax()` to raw logits, `TensorPrimitives.IndexOfMax()` for argmax, maps index to label string. Outputs `PredictedLabel`, `Score`, `Probabilities` columns. |
| `OnnxAudioClassificationOptions` | Config: `ModelPath`, `FeatureExtractor`, `Labels[]`, column names, tensor names, `GpuDeviceId`. |
| `AudioClassificationResult` | Result DTO: `PredictedLabel`, `Score`, `Probabilities`, optional `Labels`. |

### Embeddings/

| Type | Role |
|---|---|
| `OnnxAudioEmbeddingEstimator` | Facade: composes 3 sub-transforms, single `OnnxAudioEmbeddingOptions` config. |
| `OnnxAudioEmbeddingTransformer` | Chains sub-transforms lazily. Exposes `GenerateEmbeddings(IReadOnlyList<AudioData>)` and `EmbeddingDimension` property. |
| `AudioEmbeddingPoolingEstimator` / `Transformer` | **Stage 3**: Reduces `[seq, hidden]` → `[hidden]` via mean pooling, CLS token, or max pooling. Optional L2 normalization via `TensorPrimitives.Norm()` / `TensorPrimitives.Divide()`. Auto-discovers `HiddenDim` from scorer annotations. |
| `OnnxAudioEmbeddingOptions` | Config: `ModelPath`, `FeatureExtractor`, `Pooling` strategy, `Normalize`, column names, `GpuDeviceId`. |
| `AudioPoolingStrategy` | Enum: `MeanPooling`, `ClsToken`, `MaxPooling`. |

### VAD/

| Type | Role |
|---|---|
| `OnnxVadEstimator` | Creates `OnnxVadTransformer`. |
| `OnnxVadTransformer` | Implements `ITransformer` + `IVoiceActivityDetector`. Processes audio in fixed-size windows (512 samples = 32ms at 16kHz). Silero VAD is stateful — LSTM hidden states (`h`, `c`) carry across frames. Merges frame-level probabilities into speech segments with configurable thresholding, minimum duration, silence splitting, and padding. |
| `OnnxVadOptions` | Config: `ModelPath`, `Threshold`, `MinSpeechDuration`, `MinSilenceDuration`, `SpeechPad`, `WindowSize`, `SampleRate`. |
| `IVoiceActivityDetector` | Custom MEAI-style interface: `DetectSpeechAsync(Stream, VadOptions?, CancellationToken)` → `IAsyncEnumerable<SpeechSegment>`. |
| `SpeechSegment` | Record: `Start`, `End`, `Confidence`, computed `Duration`. |

### ASR/

| Type | Role |
|---|---|
| `OnnxWhisperEstimator` | Creates `OnnxWhisperTransformer`. |
| `OnnxWhisperTransformer` | Full raw-ONNX Whisper pipeline: (1) `WhisperFeatureExtractor` → mel, (2) encoder session → hidden states, (3) autoregressive decode loop with `WhisperKvCacheManager` → token IDs, (4) `WhisperTokenizer` → text. Uses `TensorPrimitives` for greedy argmax or temperature sampling with softmax. No ORT GenAI dependency. |
| `OnnxWhisperOptions` | Config: `EncoderModelPath`, `DecoderModelPath`, `MaxTokens`, `Language`, `Translate`, `IsMultilingual`, `NumMelBins` (80 for v1-v2, 128 for v3), `Temperature`. Auto-detects `NumDecoderLayers` and `NumAttentionHeads` from model. |
| `WhisperKvCacheManager` | Manages decoder self-attention KV (grows each step) and cross-attention KV (fixed after first step). `DetectFromModel()` auto-discovers dimensions. `BuildDecoderInputs()` / `UpdateFromOutputs()` handle the merged-model `use_cache_branch` protocol. |
| `SpeechToTextClientEstimator` | Wraps any `ISpeechToTextClient` (from MEAI) as an ML.NET estimator. Provider-agnostic — works with Azure, OpenAI, local Whisper, etc. Mirrors the `ChatClientEstimator` pattern from text inference transforms. |
| `SpeechToTextClientTransformer` | Converts `AudioData` → WAV stream → `ISpeechToTextClient.GetTextAsync()` → text column. Eager evaluation. Sync-over-async bridge (same approach as `ChatClientTransformer`). |
| `SpeechToTextClientOptions` | Config: column names, `SampleRate`, `SpeechLanguage`, `TextLanguage`. |

### TTS/

| Type | Role |
|---|---|
| `OnnxSpeechT5TtsEstimator` | Creates `OnnxSpeechT5TtsTransformer`. |
| `OnnxSpeechT5TtsTransformer` | Full SpeechT5 pipeline: (1) SentencePiece tokenizer (`Microsoft.ML.Tokenizers`) → token IDs, (2) encoder → hidden states, (3) autoregressive decoder loop with KV cache → mel frames, (4) vocoder (postnet + HiFi-GAN) → PCM waveform. Loads `.npy` speaker embeddings. Auto-detects decoder dimensions. |
| `OnnxSpeechT5Options` | Config: `EncoderModelPath`, `DecoderModelPath`, `VocoderModelPath`, `TokenizerModelPath` (SentencePiece), `SpeakerEmbeddingPath` (.npy), `MaxMelFrames`, `StopThreshold`, `NumMelBins`. |
| `ITextToSpeechClient` | Prototype MEAI-style interface: `GetAudioAsync()`, `GetStreamingAudioAsync()`, `Metadata`. Follows MEAI conventions (async, options, streaming via `IAsyncEnumerable`). |
| `OnnxTextToSpeechClient` | `ITextToSpeechClient` implementation wrapping `OnnxSpeechT5TtsTransformer`. Single-chunk streaming (full synthesis then yield). |
| `TextToSpeechResponse` / `TextToSpeechResponseUpdate` / `TextToSpeechOptions` / `TextToSpeechClientMetadata` | Supporting types for the TTS client interface. |

### MEAI/

| Type | Role |
|---|---|
| `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` implementation. Bridges `OnnxAudioEmbeddingTransformer` to the MEAI abstraction. Reports `EmbeddingDimension` via `EmbeddingGeneratorMetadata.DefaultModelDimensions`. |

---

## MEAI Integration

This library bridges ML.NET transforms to four [Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.ai) abstractions:

| MEAI Interface | Implementation | Status |
|---|---|---|
| `IEmbeddingGenerator<AudioData, Embedding<float>>` | `OnnxAudioEmbeddingGenerator` | ✅ Official MEAI interface |
| `ISpeechToTextClient` | Consumed by `SpeechToTextClientEstimator` | ✅ Official MEAI interface |
| `ITextToSpeechClient` | `OnnxTextToSpeechClient` (implements), `ITextToSpeechClient` (defines) | ⚠️ **Prototype** — MEAI doesn't define this yet |
| `IVoiceActivityDetector` | `OnnxVadTransformer` | 🔧 **Custom** — no MEAI equivalent exists |

The MEAI pattern enables **provider-agnostic code**. For example, `SpeechToTextClientEstimator` accepts *any* `ISpeechToTextClient` — it doesn't care whether it's backed by a local ONNX Whisper model, Azure Cognitive Services, or OpenAI's API. The same ML.NET pipeline works with any provider.

---

## How It Fits — Layer Architecture

```
┌─────────────────────────────────────────────┐
│  Samples / Applications                     │  ← consume transforms directly
├─────────────────────────────────────────────┤
│  DataIngestion (Layer 2)                    │  ← consumes via MEAI interfaces
├─────────────────────────────────────────────┤
│  MLNet.AudioInference.Onnx (Layer 1)  ◄── YOU ARE HERE
│  IEstimator/ITransformer + MEAI wrappers    │
├─────────────────────────────────────────────┤
│  MLNet.Audio.Core (Layer 0)                 │  ← audio primitives, feature extractors
│  AudioData, AudioFeatureExtractor,          │
│  WhisperFeatureExtractor, WhisperTokenizer, │
│  AudioCodecTokenizer, AudioIO, MelSpectro.. │
└─────────────────────────────────────────────┘
```

**Depends on:** `MLNet.Audio.Core` (audio primitives: `AudioData`, `AudioFeatureExtractor`, `WhisperFeatureExtractor`, `WhisperTokenizer`, `AudioCodecTokenizer`, `AudioIO`)

**Consumed by:**
- Sample projects (direct use of estimators/transformers)
- DataIngestion layer (via `IEmbeddingGenerator`, `ISpeechToTextClient`, `ITextToSpeechClient`, `IVoiceActivityDetector`)

---

## Dependencies

| Package | Version | Why |
|---|---|---|
| `Microsoft.ML` | 5.0.0 | `IEstimator<T>`, `ITransformer`, `IDataView`, `DataViewSchema`, `SchemaShape`, `MLContext`, `VBuffer<T>` — the entire ML.NET pipeline infrastructure |
| `Microsoft.ML.OnnxRuntime.Managed` | 1.24.2 | `InferenceSession`, `OrtValue`, `DenseTensor<T>`, `NamedOnnxValue` — all ONNX model inference. Used by every transform except `SpeechToTextClientTransformer` (which delegates to the MEAI client) |
| `Microsoft.ML.Tokenizers` | 2.0.0 | `SentencePieceTokenizer` — used by `OnnxSpeechT5TtsTransformer` for character-level text tokenization (`spm_char.model`). Only TTS depends on this. |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | `IEmbeddingGenerator<,>`, `ISpeechToTextClient`, `EmbeddingGeneratorMetadata`, `SpeechToTextOptions` — the MEAI abstraction interfaces. `MEAI001` warning is suppressed (experimental API). |
| `System.Numerics.Tensors` | 10.0.3 | `TensorPrimitives` — used throughout for vectorized math: `SoftMax`, `IndexOfMax`, `Norm`, `Divide`, `Add`, `Max`, `MaxMagnitude`. Avoids hand-written loops for these operations. |
| `MLNet.Audio.Core` | (project ref) | Audio primitives: `AudioData`, `AudioFeatureExtractor`, `MelSpectrogramExtractor`, `WhisperFeatureExtractor`, `WhisperTokenizer`, `AudioCodecTokenizer`, `AudioIO` |

---

## Target Framework

- **.NET 10.0** (`net10.0`)
- Nullable reference types enabled
- Implicit usings enabled
