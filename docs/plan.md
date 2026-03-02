# ML.NET Audio Custom Transforms — Master Plan

## Problem Statement

We've built a comprehensive text-based AI pipeline in .NET:
- **Tokenization** → `Microsoft.ML.Tokenizers` (BertTokenizer, TiktokenTokenizer, etc.)
- **Inference** → ML.NET custom transforms with ONNX Runtime (embeddings, classification, NER, QA, reranking, text generation)
- **MEAI integration** → `IEmbeddingGenerator<string, Embedding<float>>`, `IChatClient`
- **Model packaging** → ModelPackages SDK (download, cache, verify from HuggingFace)
- **Model garden** → NuGet packages wrapping pre-trained models under `DotnetAILab.ModelGarden`

Now we need to **do the same thing for audio**. Audio is fundamentally different from text — the input is continuous signal, not discrete tokens — but the architectural patterns (three-stage pipeline, ML.NET IEstimator/ITransformer, MEAI integration, model packaging) should carry over.

## Approach: Follow Existing Patterns, Adapt for Audio

### Text vs Audio — What Changes

| Concern | Text World | Audio World |
|---------|-----------|------------|
| **Input type** | `string` | `float[]` PCM samples + sample rate |
| **Stage 1** | Tokenization (BertTokenizer → token IDs) | Feature Extraction (mel spectrogram, MFCC, log-mel filter banks) |
| **Stage 2** | ONNX Scoring (encoder-only) | ONNX Scoring (encoder-only OR encoder-decoder with autoregressive decoding) |
| **Stage 3** | Post-processing (pooling, softmax, sigmoid, span extraction) | Post-processing (token decoding for ASR, softmax for classification, pooling for embeddings, vocoder for TTS) |
| **Model archs** | Mostly encoder-only (BERT, DeBERTa) + some decoder (Phi) | Encoder-only (Wav2Vec2, AST, HuBERT), Encoder-decoder (Whisper), Autoregressive (TTS models) |
| **MEAI interfaces** | `IEmbeddingGenerator`, `IChatClient` | `ISpeechToTextClient` ✅, `ITextToSpeechClient` ✅ (both experimental in .NET 10) |
| **I/O** | Plain strings | WAV/PCM files, sample rate conversion, channel mixing |
| **Streaming** | Optional | Natural and often required (real-time audio) |

### What Already Exists in the .NET Ecosystem

| Component | Available? | Library/Package |
|-----------|-----------|----------------|
| Audio I/O (WAV read/write) | ✅ | NAudio, or custom (WAV is trivial) |
| Mel spectrogram / MFCC | ✅ | NWaves (pure .NET DSP, NuGet) |
| ONNX Runtime | ✅ | Microsoft.ML.OnnxRuntime |
| ML.NET IEstimator/ITransformer | ✅ | Microsoft.ML |
| ISpeechToTextClient | ✅ | Microsoft.Extensions.AI.Abstractions (experimental) |
| ITextToSpeechClient | ✅ | Microsoft.Extensions.AI.Abstractions (experimental) |
| IEmbeddingGenerator | ✅ | Microsoft.Extensions.AI.Abstractions |
| Whisper ONNX models | ✅ | Available on HuggingFace (encoder + decoder split) |
| Audio classification ONNX | ✅ | AST, Wav2Vec2, HuBERT on HuggingFace |
| TTS ONNX | ✅ | VibeVoice, QwenTTS (ElBruno has .NET wrappers) |
| Silero VAD ONNX | ✅ | ~2MB model, used in ElBruno.Realtime |
| Audio tokenizer (in ML.Tokenizers) | ❌ | Does not exist — audio uses feature extractors, not tokenizers |
| IAudioClassificationClient | ❌ | No MEAI interface for this — custom or propose |
| IAudioEmbeddingGenerator | ❌ | Could use `IEmbeddingGenerator<AudioData, Embedding<float>>` |

### Inspiration from ElBruno's Repos

| Repo | What It Shows | Key Patterns to Adopt |
|------|--------------|----------------------|
| **ElBruno.Realtime** | Full STT+VAD+LLM+TTS pipeline | `ISpeechToTextClient` impl, `ITextToSpeechClient`, `IVoiceActivityDetector`, pluggable providers, DI-ready |
| **ElBruno.QwenTTS** | Local TTS via ONNX (Qwen3-TTS) | Multi-model ONNX pipeline, voice cloning, auto-download from HuggingFace |
| **ElBruno.VibeVoiceTTS** | Local TTS via ONNX (VibeVoice) | GPU acceleration (DirectML/CUDA), voice presets, autoregressive ONNX pipeline with KV-cache |

---

## Audio Task Taxonomy & Transform Map

### Task 1: Speech-to-Text (ASR) — **Priority: HIGH**

**Pipeline:**
```
Audio (float[] @ 16kHz) → Mel Spectrogram (log-mel filter banks, 80/128 channels)
  → ONNX Encoder → encoder hidden states
  → ONNX Decoder (autoregressive) → token IDs
  → Token Decoding → text (with optional timestamps)
```

**Key model:** Whisper (tiny/base/small/medium/large-v3)
- Encoder-decoder architecture
- Input: 30-second chunks, 80/128 mel channels
- Output: token IDs decoded to text
- ONNX export: separate encoder.onnx + decoder.onnx files
- Tokenizer: Whisper uses its own BPE tokenizer (multilingual + special tokens)

**Transforms needed:**
- `AudioFeatureExtractionEstimator` / `AudioFeatureExtractionTransformer` — loads PCM → mel spectrogram
- `OnnxAudioEncoderScorerEstimator` / `OnnxAudioEncoderScorerTransformer` — runs encoder
- `WhisperDecodingTransformer` — autoregressive decoder loop with greedy/beam search
- **Facade:** `OnnxSpeechToTextEstimator` (single-call API)
- **MEAI:** `OnnxSpeechToTextClient : ISpeechToTextClient`

**Challenges:**
- Autoregressive decoding loop is new (text classification/embeddings are single-pass)
- Need mel spectrogram implementation (NWaves or custom)
- Whisper tokenizer needs to be implemented or adapted (it's BPE-based, `Microsoft.ML.Tokenizers` has `BpeTokenizer`)
- 30-second chunking for long audio (sequential or chunked approaches)

### Task 2: Audio Classification — **Priority: HIGH**

**Pipeline:**
```
Audio (float[] @ 16kHz) → Feature Extraction (model-specific)
  → ONNX Encoder → logits [batch x num_classes]
  → Softmax → class probabilities + predicted label
```

**Key models:** AST (Audio Spectrogram Transformer), Wav2Vec2, HuBERT
- Encoder-only architecture (like BERT for text)
- AST: input is mel spectrogram patches
- Wav2Vec2/HuBERT: input is raw waveform

**Transforms needed:**
- `AudioFeatureExtractionTransformer` (reused from ASR) OR `RawWaveformTransformer` (for Wav2Vec2)
- `OnnxAudioModelScorerTransformer` (shared encoder scoring)
- `SoftmaxAudioClassificationTransformer` — softmax + label mapping (reuse text classification pattern)
- **Facade:** `OnnxAudioClassificationEstimator`

**Use cases:** Emotion detection, audio event detection, music genre classification, language identification, speaker verification

### Task 3: Audio Embeddings — **Priority: MEDIUM**

**Pipeline:**
```
Audio (float[] @ 16kHz) → Feature Extraction
  → ONNX Encoder → hidden states [batch x seq x dim]
  → Pooling (mean/CLS) → float[] embedding vector
  → L2 normalization
```

**Key models:** CLAP, Wav2Vec2, HuBERT (as feature extractors)
- Nearly identical to text embeddings pipeline
- Enables audio similarity search, RAG over audio, audio clustering

**Transforms needed:**
- `AudioFeatureExtractionTransformer` (reused)
- `OnnxAudioModelScorerTransformer` (reused)
- `AudioEmbeddingPoolingTransformer` — mean/CLS pooling + L2 norm (reuse text embedding pattern)
- **Facade:** `OnnxAudioEmbeddingEstimator`
- **MEAI:** `OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>`

### Task 4: Voice Activity Detection (VAD) — **Priority: MEDIUM**

**Pipeline:**
```
Audio (float[] @ 16kHz) → Chunked frames (e.g., 30ms windows)
  → ONNX Model (Silero VAD) → speech probability per frame
  → Thresholding → speech/silence segments with timestamps
```

**Key model:** Silero VAD v5 (~2MB ONNX)
- Very small, fast model
- Critical for real-time audio processing, preprocessing for ASR
- Stateful (uses hidden state across frames)

**Transforms needed:**
- `AudioFramingTransformer` — splits audio into frames
- `OnnxVadScorerTransformer` — scores each frame
- `VadSegmentationTransformer` — thresholding + segment merging
- **Facade:** `OnnxVadEstimator`
- **Interface:** `IVoiceActivityDetector` (custom, like ElBruno.Realtime)

### Task 5: Text-to-Speech (TTS) — **Priority: LOWER (most complex)**

**Pipeline (varies by model):**
```
Text → Text Tokenization → token IDs
  → ONNX Language Model (autoregressive) → audio codec tokens
  → ONNX Vocoder / Decoder → float[] audio samples
  → WAV encoding
```

**Key models:** VibeVoice, QwenTTS, VITS, Bark, Kokoro
- Most complex audio task
- Often multi-model (language model + vocoder)
- Autoregressive with KV-cache
- Voice presets / voice cloning capabilities

**Transforms needed:**
- Text tokenization (reuse existing text tokenizers)
- `OnnxTextToAudioTransformer` — runs LM + vocoder pipeline
- `AudioEncodingTransformer` — produces WAV output
- **Facade:** `OnnxTextToSpeechEstimator`
- **MEAI:** `OnnxTextToSpeechClient : ITextToSpeechClient`

**Note:** TTS is architecturally the most different from our existing text pipeline. It may warrant its own package (like `MLNet.TextGeneration.OnnxGenAI` is separate from `MLNet.TextInference.Onnx`).

### Task 6: Speaker Diarization — **Priority: LOW (future)**

**Pipeline:**
```
Audio → VAD → speech segments
  → Audio Embedding per segment
  → Clustering → speaker labels per segment
```

Composes VAD + Audio Embeddings. Can be built on top of Tasks 3 and 4.

---

## Architecture: Layered Design

### Layer 0: Audio Primitives (`MLNet.Audio.Core`)

Foundation types and utilities that everything else depends on.

```csharp
// Core audio data type
public class AudioData
{
    public float[] Samples { get; }     // PCM samples, mono
    public int SampleRate { get; }       // e.g., 16000
    public int Channels { get; }         // 1 for mono
    public TimeSpan Duration => TimeSpan.FromSeconds((double)Samples.Length / SampleRate);
}

// Audio I/O
public static class AudioIO
{
    public static AudioData LoadWav(string path);
    public static AudioData LoadWav(Stream stream);
    public static void SaveWav(string path, AudioData audio);
    public static AudioData Resample(AudioData audio, int targetSampleRate);
    public static AudioData ToMono(AudioData audio);
}

// Feature extraction base
public abstract class AudioFeatureExtractor
{
    public abstract float[,] Extract(AudioData audio);  // [frames x features]
}

// Mel spectrogram extractor
public class MelSpectrogramExtractor : AudioFeatureExtractor
{
    public int FFTSize { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public int NumMelBins { get; init; } = 80;
    public int SampleRate { get; init; } = 16000;
    public override float[,] Extract(AudioData audio);
}
```

**Dependencies:** NWaves (for FFT/mel filter banks) or `System.Numerics.Tensors` for custom impl

### Layer 1: Audio Inference Library (`MLNet.AudioInference.Onnx`)

Shared audio transforms following the three-stage pattern. This is the audio equivalent of `MLNet.TextInference.Onnx`.

```
Stage 1: AudioFeatureExtractionEstimator/Transformer (task-agnostic)
  - Configurable: mel spectrogram params, sample rate, normalization
  - Produces feature tensors ready for ONNX

Stage 2: OnnxAudioModelScorerEstimator/Transformer (task-agnostic)
  - Loads ONNX model, discovers tensor metadata
  - Handles encoder-only and encoder-decoder architectures
  - Configurable execution providers (CPU/CUDA/DirectML)

Stage 3: Task-specific post-processors
  - WhisperDecodingTransformer (ASR)
  - SoftmaxAudioClassificationTransformer
  - AudioEmbeddingPoolingTransformer
  - VadSegmentationTransformer
```

**Facade estimators (single-call convenience API):**
```csharp
// ASR
var estimator = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
{
    EncoderModelPath = "whisper-encoder.onnx",
    DecoderModelPath = "whisper-decoder.onnx",
    TokenizerPath = "tokenizer.json",
    Language = "en",
    Task = SpeechTask.Transcribe
});

// Audio Classification
var estimator = mlContext.Transforms.OnnxAudioClassification(new OnnxAudioClassificationOptions
{
    ModelPath = "ast-audioset.onnx",
    Labels = new[] { "Speech", "Music", "Silence", ... },
    FeatureExtractor = new MelSpectrogramExtractor { NumMelBins = 128 }
});

// Audio Embeddings
var estimator = mlContext.Transforms.OnnxAudioEmbedding(new OnnxAudioEmbeddingOptions
{
    ModelPath = "clap-encoder.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(),
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true
});
```

### Layer 2: MEAI Integration

```csharp
// Speech-to-Text (implements existing MEAI interface)
public class OnnxSpeechToTextClient : ISpeechToTextClient
{
    public Task<SpeechToTextResponse> GetTextAsync(Stream audioStream, SpeechToTextOptions? options, CancellationToken ct);
    public IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(Stream audioStream, SpeechToTextOptions? options, CancellationToken ct);
}

// Text-to-Speech (implements existing MEAI interface)
public class OnnxTextToSpeechClient : ITextToSpeechClient { ... }

// Audio Embeddings (reuses existing MEAI generic interface)
public class OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>
{
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<AudioData> values, ...);
}
```

### Layer 3: Model Packages (reuse ModelPackages SDK)

Same pattern as text: NuGet packages with metadata + tokenizer/config, ONNX downloaded on first use.

```csharp
// Example: Whisper model package
public static class WhisperBaseModel
{
    public static async Task<OnnxSpeechToTextTransformer> CreateSpeechToTextAsync(ModelOptions? options = null);
    public static async Task<OnnxSpeechToTextClient> CreateSpeechToTextClientAsync(ModelOptions? options = null);
    public static async Task<string> EnsureModelAsync(ModelOptions? options = null);
}
```

### Layer 4: Model Garden Entries

```
models/
  asr/
    DotnetAILab.ModelGarden.ASR.WhisperTiny/
    DotnetAILab.ModelGarden.ASR.WhisperBase/
    DotnetAILab.ModelGarden.ASR.WhisperSmall/
  audioclassification/
    DotnetAILab.ModelGarden.AudioClassification.ASTAudioset/
    DotnetAILab.ModelGarden.AudioClassification.EmotionWav2Vec2/
  audioembeddings/
    DotnetAILab.ModelGarden.AudioEmbedding.CLAP/
  tts/
    DotnetAILab.ModelGarden.TTS.VibeVoice/  (future)
    DotnetAILab.ModelGarden.TTS.QwenTTS/    (future)
  vad/
    DotnetAILab.ModelGarden.VAD.SileroV5/
```

---

## CRITICAL: The Encoder-Decoder Problem (Lessons from Text Generation)

### The Problem

Issue [luisquintanilla/mlnet-text-inference-custom-transforms#8](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms/issues/8) documented this exact challenge for text generation:

> **The existing architecture uses a linear `Tokenize → Score → PostProcess` pipeline designed for single-forward-pass encoder models. Text generation requires an autoregressive feedback loop — fundamentally different from the existing pipeline.**

For audio, this problem is **even more pervasive**:
- **Whisper ASR** = encoder-decoder (autoregressive decoding loop)
- **TTS models** = autoregressive (multi-model pipeline: language model + vocoder)
- Only **classification, embeddings, VAD** are encoder-only (single forward pass)

### How Text Generation Solved It

The text repo created **two standalone transformers** that bypass the shared three-stage pipeline entirely:

| Component | What it wraps | How it works |
|-----------|--------------|-------------|
| `ChatClientTransformer` | Any `IChatClient` (MEAI) | Provider-agnostic. Reads all prompts, calls `IChatClient.GetResponseAsync()` per prompt, returns IDataView. |
| `OnnxTextGenerationTransformer` | ORT GenAI `Model`/`Tokenizer`/`Generator` | ONNX-native. Handles tokenization + KV-cache + beam search internally via ORT GenAI's `Generator.GenerateNextToken()` loop. |

**Key patterns from the text generation solution:**
1. **Standalone transformers** — do NOT reuse `TextTokenizerTransformer` or `OnnxTextModelScorerTransformer`
2. **EAGER evaluation** — `Transform(IDataView)` materializes all rows, processes sequentially, returns results via `LoadFromEnumerable()`
3. **`Save()` throws `NotSupportedException`** — native model objects (ORT GenAI `Model`, `IChatClient`) can't be serialized
4. **Dual-face API** — both `Transform(IDataView)` for ML.NET and `Generate(IReadOnlyList<string>)` for direct use
5. **Separate package** — `MLNet.TextGeneration.OnnxGenAI` is a separate project from `MLNet.TextInference.Onnx`

### The Audio Solution: Same Dual Pattern

We apply the exact same strategy. Audio tasks split into two categories:

#### Category A: Encoder-Only (shared three-stage pipeline) ✅

These work exactly like text embeddings/classification/NER/QA:
```
AudioFeatureExtractionTransformer → OnnxAudioModelScorerTransformer → PostProcessor
```

| Task | Post-Processor | Save/Load |
|------|---------------|-----------|
| Audio Classification | `SoftmaxAudioClassificationTransformer` | ✅ ModelPackager zip (model.onnx + feature_extractor_config.json + labels.json) |
| Audio Embeddings | `AudioEmbeddingPoolingTransformer` | ✅ ModelPackager zip (model.onnx + feature_extractor_config.json) |
| VAD | `VadSegmentationTransformer` | ✅ ModelPackager zip (silero_vad.onnx + config.json) |

These get full `ModelPackager.Save()` / `ModelPackager.Load()` support.

#### Category B: Encoder-Decoder / Autoregressive (standalone transformers) ⚠️

These follow the text generation pattern — standalone, bypass shared pipeline:

| Task | Standalone Transformer | MEAI Wrapper | Separate Package? |
|------|----------------------|-------------|-------------------|
| **ASR (Whisper)** | `OnnxSpeechToTextTransformer` | `SpeechToTextClientTransformer` wraps `ISpeechToTextClient` | Yes: `MLNet.ASR.OnnxGenAI` |
| **TTS** | `OnnxTextToSpeechTransformer` | `TextToSpeechClientTransformer` wraps `ITextToSpeechClient` | Yes: `MLNet.TTS.Onnx` |

**The dual pattern for ASR (mirrors text generation exactly):**

```
Text Generation                          Audio ASR
─────────────────                        ─────────────────
ChatClientTransformer                    SpeechToTextClientTransformer
  wraps IChatClient                        wraps ISpeechToTextClient
  provider-agnostic                        provider-agnostic (Azure, Whisper.net, etc.)
  lives in MLNet.TextInference.Onnx        lives in MLNet.AudioInference.Onnx

OnnxTextGenerationTransformer            OnnxSpeechToTextTransformer
  wraps ORT GenAI Model/Tokenizer          wraps ORT GenAI Model (Whisper encoder-decoder)
  handles autoregressive loop              handles autoregressive decoding loop
  lives in MLNet.TextGeneration.OnnxGenAI  lives in MLNet.ASR.OnnxGenAI
```

### ORT GenAI Supports Whisper ✅

Confirmed: `Microsoft.ML.OnnxRuntimeGenAI` (v0.12+) supports Whisper encoder-decoder models. This means we can use the same `Model`/`Tokenizer`/`Generator` pattern:

```csharp
// OnnxSpeechToTextTransformer internal implementation (conceptual)
public class OnnxSpeechToTextTransformer : ITransformer, IDisposable
{
    private readonly Model _model;           // ORT GenAI Whisper model
    private readonly Tokenizer _tokenizer;   // ORT GenAI tokenizer

    public string[] Transcribe(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new List<string>();
        foreach (var audio in audioInputs)
        {
            // 1. Compute mel spectrogram (we handle this — ORT GenAI needs features as input)
            var melFeatures = _featureExtractor.Extract(audio);

            // 2. Set up generator with encoder input
            using var generatorParams = new GeneratorParams(_model);
            generatorParams.SetModelInput("input_features", melFeatures);
            generatorParams.SetSearchOption("max_length", _options.MaxLength);

            // 3. Autoregressive decoding loop (same pattern as text generation)
            using var generator = new Generator(_model, generatorParams);
            using var tokenizerStream = _tokenizer.CreateStream();
            var result = new StringBuilder();
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
                var tokenId = generator.GetSequence(0UL)[^1];
                result.Append(tokenizerStream.Decode(tokenId));
            }
            results.Add(result.ToString());
        }
        return results.ToArray();
    }
}
```

### Save/Load Strategy for Encoder-Decoder Models

**`Save()` throws `NotSupportedException`** — same as text generation. The ORT GenAI `Model` object owns native resources and can't be serialized.

**Model Garden / Model Packages use the factory pattern instead:**
```csharp
// This is exactly what Phi3MiniModel does for text generation:
public static class WhisperBaseModel
{
    public static async Task<OnnxSpeechToTextTransformer> CreateSpeechToTextAsync(ModelOptions? options = null)
    {
        var files = await EnsureModelAsync(options);       // Download encoder.onnx + decoder.onnx
        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
        {
            ModelPath = files.ModelDirectory,               // ORT GenAI reads from directory
            Language = "en",
            Task = SpeechTask.Transcribe
        });
        var dummyData = mlContext.Data.LoadFromEnumerable(new[] { new AudioInput { Audio = Array.Empty<float>() } });
        return estimator.Fit(dummyData);
    }
}
```

This is the **same pattern** as `Phi3MiniModel.CreateTextGeneratorAsync()` in the model garden.

### The Fallback: Raw ONNX InferenceSession

If ORT GenAI's Whisper support is immature or limiting, we have a fallback: manage encoder + decoder `InferenceSession` objects directly. This is more code but more flexible:

```csharp
// Alternative: raw ONNX sessions (no ORT GenAI dependency)
public class OnnxSpeechToTextTransformer : ITransformer, IDisposable
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly WhisperTokenDecoder _tokenDecoder;

    public string[] Transcribe(IReadOnlyList<AudioData> audioInputs)
    {
        foreach (var audio in audioInputs)
        {
            var mel = _featureExtractor.Extract(audio);

            // Run encoder
            var encoderOutput = _encoder.Run(new[] {
                NamedOnnxValue.CreateFromTensor("input_features", mel)
            });
            var hiddenStates = encoderOutput.First().AsTensor<float>();

            // Autoregressive decoder loop (we manage KV-cache manually)
            var tokenIds = new List<int> { _tokenDecoder.StartOfTranscriptId };
            for (int i = 0; i < _options.MaxLength; i++)
            {
                var decoderOutput = _decoder.Run(new[] {
                    NamedOnnxValue.CreateFromTensor("input_ids", tokenIds),
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", hiddenStates),
                    // ... KV-cache tensors ...
                });
                var logits = decoderOutput["logits"].AsTensor<float>();
                var nextToken = ArgMax(logits);  // greedy search
                if (nextToken == _tokenDecoder.EndOfTranscriptId) break;
                tokenIds.Add(nextToken);
            }
            results.Add(_tokenDecoder.Decode(tokenIds));
        }
    }
}
```

**Recommendation:** Start with ORT GenAI (simpler, battle-tested autoregressive loop). Fall back to raw InferenceSession if we hit limitations. Both approaches live in the same `OnnxSpeechToTextTransformer` — the implementation detail is internal, the API surface is identical.

### Updated Package Structure (reflecting the split)

```
MLNet.AudioInference.Onnx                    MLNet.ASR.OnnxGenAI
├── Microsoft.ML                             ├── Microsoft.ML
├── Microsoft.ML.OnnxRuntime.Managed         ├── Microsoft.ML.OnnxRuntimeGenAI
├── MLNet.Audio.Core                         ├── MLNet.Audio.Core
├── Microsoft.Extensions.AI.Abstractions     ├── Microsoft.Extensions.AI.Abstractions
│                                            │
│ Contains:                                  │ Contains:
│ ├── Shared pipeline (feature extract +     │ ├── OnnxSpeechToTextEstimator/Transformer
│ │   ONNX scorer)                           │ │   (wraps ORT GenAI Whisper)
│ ├── SpeechToTextClientTransformer          │ └── MLContextExtensions
│ │   (wraps any ISpeechToTextClient)        │
│ ├── AudioClassification transforms         MLNet.TTS.Onnx (or MLNet.TTS.OnnxGenAI)
│ ├── AudioEmbedding transforms              ├── Microsoft.ML
│ ├── VAD transforms                         ├── Microsoft.ML.OnnxRuntime.Managed (or GenAI)
│ └── MEAI adapters                          ├── MLNet.Audio.Core
│                                            ├── Microsoft.Extensions.AI.Abstractions
Two independent projects.                    │
No cross-dependency.                         │ Contains:
User picks one or both.                      │ ├── OnnxTextToSpeechEstimator/Transformer
                                             │ └── TextToSpeechClientTransformer
```

### Summary: Which Tasks Use Which Pattern

| Task | Pipeline Pattern | Save/Load | Package |
|------|-----------------|-----------|---------|
| Audio Classification | ✅ Shared 3-stage | ✅ ModelPackager.Save/Load | `MLNet.AudioInference.Onnx` |
| Audio Embeddings | ✅ Shared 3-stage | ✅ ModelPackager.Save/Load | `MLNet.AudioInference.Onnx` |
| VAD | ✅ Shared 3-stage | ✅ ModelPackager.Save/Load | `MLNet.AudioInference.Onnx` |
| **ASR (Whisper)** | ❌ Standalone (autoregressive) | ❌ Factory + manifest | `MLNet.ASR.OnnxGenAI` |
| **TTS** | ❌ Standalone (autoregressive) | ❌ Factory + manifest | `MLNet.TTS.Onnx` |

---

## New Abstractions Needed (things that don't exist yet in .NET)

### 1. AudioFeatureExtractor (parallel to Tokenizer)

In text, `Microsoft.ML.Tokenizers.Tokenizer` is the base class for all tokenizers. Audio needs an equivalent:

```csharp
namespace Microsoft.ML.Audio;

public abstract class AudioFeatureExtractor
{
    public int SampleRate { get; }
    public abstract Tensor<float> Extract(ReadOnlySpan<float> samples, int sampleRate);
    public abstract AudioFeatureExtractorConfig Config { get; }
}

public class WhisperFeatureExtractor : AudioFeatureExtractor
{
    // 80/128 mel bins, 30-second chunks, log-mel spectrogram
    // Matches HuggingFace WhisperFeatureExtractor behavior exactly
}

public class Wav2Vec2FeatureExtractor : AudioFeatureExtractor
{
    // Raw waveform normalization (mean/variance)
    // No spectrogram — Wav2Vec2 processes raw audio
}

public class ASTFeatureExtractor : AudioFeatureExtractor
{
    // 128 mel bins, patch-based for Audio Spectrogram Transformer
}
```

**This is arguably the most important new abstraction.** It's the audio equivalent of the tokenizer — without it, every audio model integration is ad-hoc.

### 2. AudioDecoder (for ASR/TTS autoregressive tasks)

Text generation uses `OnnxTextGenerationTransformer` with ORT GenAI. Audio ASR needs something similar:

```csharp
public abstract class AudioTokenDecoder
{
    public abstract string Decode(ReadOnlySpan<int> tokenIds);
    public abstract int[] Encode(string text);  // for TTS
}

public class WhisperTokenDecoder : AudioTokenDecoder
{
    // Wraps the Whisper BPE tokenizer
    // Handles special tokens (<|startoftranscript|>, <|en|>, <|transcribe|>, etc.)
    // Supports timestamp tokens
}
```

### 3. IVoiceActivityDetector (custom interface, not in MEAI)

```csharp
public interface IVoiceActivityDetector
{
    IAsyncEnumerable<SpeechSegment> DetectSpeechAsync(Stream audioStream, VadOptions? options = null);
}

public record SpeechSegment(TimeSpan Start, TimeSpan End, float Confidence);
```

### 4. Proposed MEAI Extensions (future, for broader ecosystem)

These don't exist yet but would be natural extensions:

```csharp
// Audio classification (like HuggingFace pipeline("audio-classification"))
public interface IAudioClassifier
{
    Task<AudioClassificationResult> ClassifyAsync(Stream audioStream, AudioClassificationOptions? options = null);
}

// Audio embedding (already possible via IEmbeddingGenerator<AudioData, Embedding<float>>)
// No new interface needed — just a concrete implementation
```

---

## Proposed Solution Structure

```
mlnet-audio-custom-transforms/
├── MLNet.Audio.slnx
├── nuget.config
├── Directory.Build.props
├── .gitignore
├── README.md
├── docs/
│   ├── architecture.md
│   ├── audio-vs-text-comparison.md
│   ├── encoder-decoder-design.md
│   └── extending.md
├── src/
│   ├── MLNet.Audio.Core/                    # Audio primitives (no ML.NET dependency)
│   │   ├── AudioData.cs
│   │   ├── AudioIO.cs
│   │   ├── AudioFeatureExtractor.cs
│   │   ├── MelSpectrogramExtractor.cs
│   │   ├── WhisperFeatureExtractor.cs
│   │   └── Wav2Vec2FeatureExtractor.cs
│   │
│   ├── MLNet.AudioInference.Onnx/           # Encoder-only transforms + MEAI wrappers
│   │   ├── Shared/                          # Shared 3-stage pipeline (encoder-only tasks)
│   │   │   ├── AudioFeatureExtractionEstimator.cs
│   │   │   ├── AudioFeatureExtractionTransformer.cs
│   │   │   ├── OnnxAudioModelScorerEstimator.cs
│   │   │   └── OnnxAudioModelScorerTransformer.cs
│   │   ├── Classification/                  # Encoder-only: shared pipeline
│   │   │   ├── OnnxAudioClassificationEstimator.cs
│   │   │   ├── OnnxAudioClassificationTransformer.cs
│   │   │   └── OnnxAudioClassificationOptions.cs
│   │   ├── Embeddings/                      # Encoder-only: shared pipeline
│   │   │   ├── OnnxAudioEmbeddingEstimator.cs
│   │   │   ├── OnnxAudioEmbeddingTransformer.cs
│   │   │   └── OnnxAudioEmbeddingOptions.cs
│   │   ├── VAD/                             # Encoder-only: shared pipeline
│   │   │   ├── OnnxVadEstimator.cs
│   │   │   ├── OnnxVadTransformer.cs
│   │   │   └── OnnxVadOptions.cs
│   │   ├── ASR/                             # Provider-agnostic MEAI wrapper (like ChatClientTransformer)
│   │   │   ├── SpeechToTextClientEstimator.cs
│   │   │   └── SpeechToTextClientTransformer.cs
│   │   ├── TTS/                             # Provider-agnostic MEAI wrapper
│   │   │   ├── TextToSpeechClientEstimator.cs
│   │   │   └── TextToSpeechClientTransformer.cs
│   │   ├── MEAI/
│   │   │   └── OnnxAudioEmbeddingGenerator.cs
│   │   ├── MLContextExtensions.cs
│   │   └── ModelPackager.cs                 # Save/Load for encoder-only transforms
│   │
│   ├── MLNet.ASR.OnnxGenAI/                 # ASR via ORT GenAI (like MLNet.TextGeneration.OnnxGenAI)
│   │   ├── OnnxSpeechToTextEstimator.cs     # Loads ORT GenAI Model in Fit()
│   │   ├── OnnxSpeechToTextTransformer.cs   # Wraps ORT GenAI encoder-decoder
│   │   ├── OnnxSpeechToTextOptions.cs
│   │   ├── MEAI/
│   │   │   └── OnnxSpeechToTextClient.cs    # ISpeechToTextClient impl
│   │   └── MLContextExtensions.cs
│   │
│   └── MLNet.TTS.Onnx/                     # TTS (separate package, complex autoregressive)
│       ├── OnnxTextToSpeechEstimator.cs
│       ├── OnnxTextToSpeechTransformer.cs
│       ├── OnnxTextToSpeechOptions.cs
│       ├── VocoderTransformer.cs
│       ├── MEAI/
│       │   └── OnnxTextToSpeechClient.cs    # ITextToSpeechClient impl
│       └── MLContextExtensions.cs
│
└── samples/
    ├── SpeechToTextMeai/                    # Provider-agnostic ASR (wraps ISpeechToTextClient)
    ├── SpeechToTextLocal/                   # Local Whisper via ORT GenAI
    ├── AudioClassification/                 # AST classification sample
    ├── AudioEmbeddings/                     # Audio embedding sample
    └── VoiceActivityDetection/              # VAD sample
```

---

## Implementation Progress

### Phase 1: Foundation ✅ COMPLETE

- [x] Set up solution structure, Directory.Build.props, nuget.config
- [x] Implement `AudioData`, `AudioIO` (WAV read/write, resample, mono conversion)
- [x] Implement `MelSpectrogramExtractor` (using NWaves FilterbankExtractor)
- [x] Implement `WhisperFeatureExtractor` (80/128 mel bins, 30s chunks, padding)
- [x] Implement `AudioFeatureExtractor` abstract base class (audio's equivalent of Tokenizer)

### Phase 2: Classification + Embeddings ✅ COMPLETE

- [x] Implement `OnnxAudioClassificationEstimator/Transformer` (eager, full pipeline)
- [x] Implement `OnnxAudioEmbeddingEstimator/Transformer` (eager, pooling + L2 norm)
- [x] Implement `OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>`
- [x] Create audio classification sample (with synthetic fallback)
- [x] Create audio embeddings sample (with MEAI demo + cosine similarity)

### Phase 3: VAD + Provider-Agnostic ASR ✅ COMPLETE

- [x] Implement `OnnxVadEstimator/Transformer` (stateful Silero VAD: frames→score→merge→segments)
- [x] Implement `IVoiceActivityDetector` interface + streaming `DetectSpeechAsync`
- [x] Implement `SpeechToTextClientEstimator/Transformer` (wraps any ISpeechToTextClient)
- [x] Create VAD sample (with synthetic fallback)
- [x] Create Speech-to-Text sample (pattern documentation + primitives demo)
- [x] Update `MLContextExtensions` with `.OnnxVad()` and `.SpeechToText()` entry points

### Phase 3.5: ORT GenAI Whisper Decoder ✅ COMPLETE

- [x] Create separate `MLNet.ASR.OnnxGenAI` package (mirrors MLNet.TextGeneration.OnnxGenAI pattern)
- [x] Implement `OnnxSpeechToTextTransformer` — full pipeline: audio → mel → encoder → autoregressive decoder → text
  - Uses WhisperFeatureExtractor for mel spectrogram extraction
  - ORT GenAI `Model` + `Generator` for encoder-decoder inference
  - ORT GenAI `Tokenizer` for basic text decoding
  - WhisperTokenizer for structured timestamp output (`DecodeToSegments`)
  - Pinned memory `GCHandle` for passing mel tensor to native ORT GenAI
  - `Generator.SetModelInput("input_features", melTensor)` + `AppendTokens(decoderPrompt)` + decode loop
  - Eager evaluation: `Transform(IDataView)` reads all audio, transcribes, returns text
  - Direct API: `Transcribe(IReadOnlyList<AudioData>)` and `TranscribeWithTimestamps()`
  - `Save()` throws `NotSupportedException` (same pattern as text generation)
- [x] Implement `OnnxSpeechToTextEstimator` (IEstimator wrapper)
- [x] Implement `OnnxSpeechToTextClient : ISpeechToTextClient` (MEAI wrapper)
  - Constructs `SpeechToTextResponse(text)` / `SpeechToTextResponseUpdate(text)` using string constructors
  - Streaming via `GetStreamingTextAsync` (single update for now, future: chunked)
- [x] Implement `MLContextExtensions.OnnxSpeechToText()` entry point
- [x] Create WhisperTranscription sample (direct API + MEAI client + ML.NET pipeline patterns)
- [x] All projects build clean (0 errors)

**Key API discoveries during implementation:**
- ORT GenAI `Tensor(IntPtr data, Int64[] shape, ElementType type)` — takes raw pointer, not managed array
- `ElementType.float32` (not `Float`)
- `Generator.SetModelInput()` is on `Generator`, NOT `GeneratorParams`
- `Generator.AppendTokens(ReadOnlySpan<int>)` for decoder prompt tokens
- MEAI `SpeechToTextResponse.Text` is read-only — use `new SpeechToTextResponse(text)` constructor
- MEAI `SchemaShape.Column` constructor is internal — use reflection (5-arg ctor) like other transforms

### Phase 3.75: Raw ONNX Whisper Decoder ✅ COMPLETE

- [x] Implement `OnnxWhisperTransformer` — raw ONNX encoder-decoder with full KV cache management
  - Uses our `WhisperFeatureExtractor` for mel spectrogram extraction
  - Encoder: `InferenceSession.Run()` on `encoder_model.onnx` → hidden states
  - Decoder: autoregressive loop on `decoder_model_merged.onnx` with `use_cache_branch` toggle
  - `WhisperKvCacheManager`: manages 4 KV tensors per layer (decoder self-attn + cross-attn)
    - Auto-detects numLayers/numHeads/headDim from model metadata
    - Cross-attention KV cached after first step (fixed encoder output)
    - Decoder self-attention KV grows each step
  - `TensorPrimitives.IndexOfMax()` for greedy decode, `TensorPrimitives.SoftMax()` + sampling for temperature
  - Our `WhisperTokenizer` for prompt building + structured decode with timestamps
  - Direct API: `Transcribe()` and `TranscribeWithTimestamps()`
  - NO new package needed — lives in existing `MLNet.AudioInference.Onnx`
  - NO new dependency — uses existing `Microsoft.ML.OnnxRuntime.Managed`
- [x] Implement `OnnxWhisperEstimator` (IEstimator wrapper)
- [x] Add `MLContextExtensions.OnnxWhisper()` entry point
- [x] Updated SpeechToText sample with raw ONNX patterns
- [x] All projects build clean (0 errors)

**Why both ORT GenAI AND raw ONNX approaches exist:**
- ORT GenAI (`MLNet.ASR.OnnxGenAI`): easiest, ORT handles decode loop, but needs extra dependency + specific model format
- Raw ONNX (`OnnxWhisperTransformer`): maximum control, uses ALL our primitives, standard HuggingFace ONNX models, no extra dependencies
- `WhisperKvCacheManager` pattern transfers directly to TTS implementation (same KV cache management)

### Phase 4: SpeechT5 TTS ✅ COMPLETE

- [x] Implement `OnnxSpeechT5TtsTransformer` — full encoder-decoder-vocoder pipeline
  - 3 ONNX sessions: encoder, decoder (merged with KV cache), vocoder (postnet + HiFi-GAN)
  - `Microsoft.ML.Tokenizers` SentencePiece for text tokenization (spm_char.model)
  - Decoder loop with KV cache — same `past_key_values.*` / `present.*` / `use_cache_branch` pattern as Whisper
  - Auto-detects model dimensions (numLayers, numHeads, headDim) from decoder metadata
  - Stop probability head for knowing when to halt mel generation
  - `TensorPrimitives.MaxMagnitude/Divide` for waveform normalization
  - `.npy` file loader for speaker embeddings (x-vector, 512-dim)
  - Direct API: `Synthesize(text)` → `AudioData`, `SynthesizeBatch(texts)`
- [x] Implement `OnnxSpeechT5TtsEstimator` (IEstimator wrapper)
- [x] Define `ITextToSpeechClient` interface (prototype — MEAI doesn't have one yet)
  - `GetAudioAsync(text)` → `TextToSpeechResponse` (contains `AudioData`)
  - `GetStreamingAudioAsync(text)` → `IAsyncEnumerable<TextToSpeechResponseUpdate>`
  - `TextToSpeechOptions` with Voice, Speed, Language, SpeakerEmbedding
- [x] Implement `OnnxTextToSpeechClient : ITextToSpeechClient`
- [x] Add `MLContextExtensions.SpeechT5Tts()` entry point
- [x] Create TextToSpeech sample (direct API + ITextToSpeechClient + ML.NET pipeline + voice round-trip patterns)
- [x] Added `Microsoft.ML.Tokenizers` v2.0.0 dependency (SentencePiece support)
- [x] All projects build clean (0 errors, 10 projects)

**Key primitives showcased:**
- `Microsoft.ML.Tokenizers` — SentencePiece character tokenizer (spm_char.model)
- `TensorPrimitives` — waveform normalization (MaxMagnitude, Divide)
- `AudioData` — output type (PCM samples + sample rate)
- `AudioIO.SaveWav()` — save generated speech to WAV
- KV cache pattern — reused from WhisperKvCacheManager (same `past_key_values.*` convention)

**SpeechT5 mirrors Whisper ASR exactly:**
- Whisper: audio → mel → encoder → decoder(KV) → text tokens → text
- SpeechT5: text → tokenize → encoder → decoder(KV) → mel frames → vocoder → audio

### Phase 5: Model Packaging + Garden — PENDING

- [x] ~~Implement `OnnxSpeechToTextTransformer` via ORT GenAI~~ (moved to Phase 3.5, DONE)
- [x] ~~Implement `OnnxSpeechToTextClient : ISpeechToTextClient`~~ (moved to Phase 3.5, DONE)
- [ ] Create model-manifest.json for audio models
- [ ] Create model packages (Whisper, AST, Silero VAD, CLAP)
- [ ] Add audio model entries to the model garden prototype

### Current Solution Structure

```
MLNet.Audio.slnx
├── src/
│   ├── MLNet.Audio.Core/                    # Audio primitives
│   │   ├── AudioData.cs                     # Core audio type (float[], SampleRate, Channels)
│   │   ├── AudioIO.cs                       # WAV I/O, resample, mono conversion
│   │   ├── AudioFeatureExtractor.cs         # Abstract base (audio's Tokenizer)
│   │   ├── MelSpectrogramExtractor.cs       # NWaves mel spectrogram
│   │   ├── WhisperFeatureExtractor.cs       # Whisper-specific (80/128 mel, 30s chunks)
│   │   └── Tokenizers/
│   │       ├── WhisperTokenizer.cs          # Whisper BPE + special tokens + timestamps
│   │       └── AudioCodecTokenizer.cs       # Abstract neural audio codec (EnCodec/DAC)
│   ├── MLNet.AudioInference.Onnx/           # ML.NET transforms (encoder-only tasks)
│   │   ├── Classification/                  # Audio classification (AST, Wav2Vec2)
│   │   ├── Embeddings/                      # Audio embeddings (CLAP, HuBERT)
│   │   ├── VAD/                             # Voice Activity Detection (Silero VAD)
│   │   ├── ASR/                             # Speech-to-Text
│   │   │   ├── SpeechToTextClient*          # Provider-agnostic (any ISpeechToTextClient)
│   │   │   ├── OnnxWhisper*                 # Raw ONNX Whisper (encoder+decoder+KV cache)
│   │   │   └── WhisperKvCacheManager        # KV cache state mgmt (reusable for TTS)
│   │   ├── MEAI/                            # IEmbeddingGenerator<AudioData, Embedding<float>>
│   │   ├── Shared/                          # EnCodecTokenizer
│   │   └── MLContextExtensions.cs
│   └── MLNet.ASR.OnnxGenAI/                 # Whisper decoder (ORT GenAI)
│       ├── OnnxSpeechToTextOptions.cs
│       ├── OnnxSpeechToTextEstimator.cs
│       ├── OnnxSpeechToTextTransformer.cs   # Full audio→mel→encoder→decoder→text pipeline
│       ├── OnnxSpeechToTextClient.cs        # ISpeechToTextClient (MEAI)
│       └── MLContextExtensions.cs
├── samples/
│   ├── AudioClassification/                 # ✅ Runs
│   ├── AudioEmbeddings/                     # ✅ Runs
│   ├── VoiceActivityDetection/              # ✅ Runs
│   ├── SpeechToText/                        # ✅ Runs (provider-agnostic patterns)
│   └── WhisperTranscription/                # ✅ Builds (needs ORT GenAI native + Whisper model)
└── docs/plan.md
```

### All 4 Samples Run Successfully

- `AudioEmbeddings`: Full synthetic demo with mel spectrogram extraction + WAV I/O
- `SpeechToText`: Pattern documentation + WhisperFeatureExtractor demo (3000×80 output ✅)
- `VoiceActivityDetection`: Synthetic demo showing speech pattern concepts
- `AudioClassification`: Graceful exit with download instructions when model absent

Wrap models as NuGet packages using the ModelPackages SDK.

**Todos:**
28. Create model-manifest.json for Whisper models (tiny, base, small)
29. Create WhisperModel packages (using ModelPackages SDK pattern)
30. Create AST/AudioSet model package
31. Create Silero VAD model package
32. Add audio model entries to the model garden prototype

---

## Key Design Decisions

### 1. NWaves vs Custom Mel Spectrogram
**Recommendation: Start with NWaves, migrate to custom if needed.**
NWaves is a mature, pure-.NET DSP library with mel spectrogram, MFCC, and filter bank support. It avoids reinventing the wheel. If performance becomes an issue, we can build a SIMD-accelerated custom impl using `TensorPrimitives` later.

### 2. Whisper Tokenizer
**Recommendation: Use `Microsoft.ML.Tokenizers.BpeTokenizer` if possible, custom if not.**
Whisper uses a BPE tokenizer. `Microsoft.ML.Tokenizers` has `BpeTokenizer` which may work with Whisper's vocab/merges files. If special token handling (timestamps, language codes) doesn't fit, build a `WhisperTokenizer` wrapper.

### 3. Encoder-Decoder Autoregressive Loop
**Recommendation: Implement in managed C# with ORT InferenceSession.**
Whisper's decoding loop is simpler than LLM text generation (shorter sequences, greedy search is often sufficient). No need for ORT GenAI — direct InferenceSession calls with KV-cache management.

### 4. AudioData vs Raw float[]
**Recommendation: Use `AudioData` wrapper type.**
Audio always needs metadata (sample rate, channels). A wrapper type prevents the "forgot to pass sample rate" bug class. For IDataView integration, define a custom column type.

### 5. Separate TTS Package
**Recommendation: Yes — `MLNet.TTS.Onnx` separate from `MLNet.AudioInference.Onnx`.**
TTS is architecturally different (text→audio vs audio→X). It has different dependencies and model patterns. Mirrors the text world where `MLNet.TextGeneration.OnnxGenAI` is separate from `MLNet.TextInference.Onnx`.

### 6. Feature Extractor Abstraction
**Recommendation: Create `AudioFeatureExtractor` abstract base class.**
This is the audio equivalent of `Tokenizer`. Different models need different feature extraction (mel spectrogram for Whisper/AST, raw waveform for Wav2Vec2, different mel params). An abstract base class with model-specific subclasses is the right pattern.

---

## Dependency Map

```
MLNet.Audio.Core
  ├── NWaves (mel spectrogram, FFT)
  ├── System.Numerics.Tensors (SIMD math)
  └── NAudio (optional, for advanced audio I/O)

MLNet.AudioInference.Onnx
  ├── MLNet.Audio.Core
  ├── Microsoft.ML (IEstimator/ITransformer)
  ├── Microsoft.ML.OnnxRuntime.Managed
  ├── Microsoft.ML.Tokenizers (for Whisper BPE tokenizer)
  ├── Microsoft.Extensions.AI.Abstractions (ISpeechToTextClient, IEmbeddingGenerator)
  └── System.Numerics.Tensors

MLNet.TTS.Onnx
  ├── MLNet.Audio.Core
  ├── Microsoft.ML
  ├── Microsoft.ML.OnnxRuntime.Managed
  ├── Microsoft.ML.Tokenizers
  └── Microsoft.Extensions.AI.Abstractions (ITextToSpeechClient)
```

---

## What This Enables (End-to-End Scenarios)

### Scenario 1: Transcribe a Meeting Recording
```csharp
var transcriber = await WhisperBaseModel.CreateSpeechToTextClientAsync();
var audio = File.OpenRead("meeting.wav");
var result = await transcriber.GetTextAsync(audio);
Console.WriteLine(result.Text);
```

### Scenario 2: Classify Audio Events in a Smart Home
```csharp
var classifier = await ASTAudiosetModel.CreateClassifierAsync();
var audio = AudioIO.LoadWav("doorbell.wav");
var result = classifier.Classify(audio);
Console.WriteLine($"Detected: {result.PredictedLabel} ({result.Confidence:P})");
```

### Scenario 3: Build an Audio Search Index (RAG over Audio)
```csharp
var embedder = await CLAPModel.CreateEmbeddingGeneratorAsync();
var audioFiles = Directory.GetFiles("podcasts/", "*.wav");
foreach (var file in audioFiles)
{
    var audio = AudioIO.LoadWav(file);
    var embedding = await embedder.GenerateAsync(new[] { audio });
    vectorDb.Upsert(file, embedding[0].Vector);
}
```

### Scenario 4: Real-Time Voice Assistant
```csharp
var vad = services.GetRequiredService<IVoiceActivityDetector>();
var stt = services.GetRequiredService<ISpeechToTextClient>();
var llm = services.GetRequiredService<IChatClient>();
var tts = services.GetRequiredService<ITextToSpeechClient>();

await foreach (var segment in vad.DetectSpeechAsync(microphoneStream))
{
    var text = await stt.GetTextAsync(segment.AudioStream);
    var response = await llm.GetResponseAsync(text.Text);
    var audio = await tts.SynthesizeAsync(response.Text);
    PlayAudio(audio);
}
```

### Scenario 5: ML.NET Pipeline Combining Text + Audio
```csharp
var pipeline = mlContext.Transforms
    .OnnxSpeechToText(new OnnxSpeechToTextOptions { ... })    // Audio → Text
    .Append(mlContext.Transforms.OnnxTextEmbedding(new OnnxTextEmbeddingOptions { ... }))  // Text → Embedding
    .Append(mlContext.Transforms.OnnxTextClassification(new OnnxTextClassificationOptions { ... }));  // Text → Classification
```

---

## Open Questions for Future Exploration

1. **Should `AudioFeatureExtractor` live in `Microsoft.ML.Tokenizers` or a new package?** — Conceptually it's the audio equivalent of a tokenizer, but it produces float tensors not int sequences.

2. **Streaming API design** — How should ML.NET IDataView handle streaming audio? Current IDataView is batch-oriented. May need `IAsyncEnumerable<AudioData>` path separate from IDataView.

3. **Multi-modal transforms** — Can we compose audio + text in a single ML.NET pipeline? E.g., transcribe audio then classify the text. This works today by chaining transforms.

4. **Model quantization** — INT8/INT4 Whisper models exist and are much smaller. How to handle quantization variants in model packages?

5. **GPU acceleration strategy** — DirectML (Windows), CUDA (NVIDIA), CoreML (macOS). Same pattern as text transforms — consumer picks their ORT provider package.
