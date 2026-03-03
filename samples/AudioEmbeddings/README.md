# Audio Embeddings: Mapping Sound to Vector Space

This sample demonstrates how to generate **vector embeddings** from audio using ML.NET custom transforms with a local ONNX model. It walks through four different approaches — from the classic ML.NET pipeline to the standardized Microsoft.Extensions.AI (`MEAI`) interface — and shows how to measure audio similarity using cosine distance.

---

## What You'll Learn

| Concept | Why It Matters |
|---|---|
| **Audio embeddings** | Represent audio clips as fixed-length numeric vectors for comparison, search, and clustering |
| **Cosine similarity** | Quantify how "close" two audio clips are in embedding space |
| **Pipeline composition** | Chain ML.NET estimators (`AudioFeatureExtraction → OnnxAudioScoring → AudioEmbeddingPooling`) with auto-discovered metadata |
| **MEAI integration** | Expose your embedding model via `IEmbeddingGenerator<AudioData, Embedding<float>>` for DI, middleware, and provider swapping |

---

## The Concept: Audio Embeddings

### What Is an Embedding?

An **embedding** is a mapping from a complex, variable-length input (like an audio waveform) to a **fixed-length vector in high-dimensional space**. The key property: **similar sounds end up close together**, and dissimilar sounds end up far apart.

For .NET developers used to working with strings or numeric features, think of it this way: instead of comparing two audio clips sample-by-sample (which is both expensive and fragile), you compress each clip into a compact vector (e.g., 512 floats) and compare the vectors instead. Two recordings of the same speaker saying "hello" will produce vectors that are nearly identical — even if the raw waveforms differ due to microphone, background noise, or timing.

### Why This Matters

Once you have embeddings, you can:

- **Similarity search** — find the most similar audio clips in a database ("find sounds like this one")
- **Clustering** — group audio by content without manual labels
- **Recommendation** — suggest similar music, podcasts, or sound effects
- **RAG for audio** — retrieve relevant audio context to ground LLM responses
- **Anomaly detection** — flag audio that doesn't match expected patterns

### How It Works: The Pipeline

```
Raw audio (PCM samples)
    ↓
Mel spectrogram (time-frequency image)
    ↓
Neural network (ONNX model)
    ↓
Raw hidden states: [batch, sequence_length, hidden_dim]
    ↓
Pooling (MeanPooling / ClsToken / MaxPooling)
    ↓
L2 normalization
    ↓
Embedding vector: [hidden_dim] (e.g., 512 floats)
```

Each step transforms the data into a progressively more abstract representation. The mel spectrogram converts raw audio into a time-frequency representation that mirrors human hearing. The neural network — typically a transformer or CNN — learns to extract meaningful patterns from that spectrogram. Pooling collapses the variable-length sequence output into a single fixed-length vector.

### Pooling Strategies

The ONNX model outputs a 3D tensor `[batch, sequence_length, hidden_dim]` — one vector per time frame. To get a single embedding per audio clip, we need to **pool** across the sequence dimension:

| Strategy | How It Works | When to Use |
|---|---|---|
| **`MeanPooling`** | Average all frame vectors element-wise | ✅ **Best default for audio.** Every frame contributes equally, capturing the overall content of the clip. Robust to length variation. |
| **`ClsToken`** | Take only the first output vector (`[0, :]`) | Good when the model was trained with a special `[CLS]` token (common in BERT-style models). Less common for audio encoders. |
| **`MaxPooling`** | Take the element-wise maximum across all frames | Highlights the most prominent features. Can be useful for event detection but loses temporal averaging. |

`MeanPooling` is usually the best choice for audio because audio content is distributed across the entire clip, not concentrated in a single frame.

### Cosine Similarity

Once you have two embedding vectors, you measure their closeness with **cosine similarity** — the cosine of the angle between them:

| Value | Meaning |
|---|---|
| **1.0** | Identical direction → maximally similar |
| **0.0** | Orthogonal → unrelated |
| **-1.0** | Opposite direction → maximally dissimilar |

When embeddings are L2-normalized (as this sample does by default), cosine similarity simplifies to the dot product, which is extremely fast.

### The Model: CLAP

This sample is designed to work with **CLAP** (Contrastive Language-Audio Pretraining) — a model trained to align audio and text in the **same embedding space**. This means you can compare an audio clip's embedding directly against a text description's embedding to find matches. CLAP was trained on large-scale audio-text pairs so it understands a wide variety of sounds.

That said, the pipeline works with **any audio encoder ONNX model** that produces hidden-state outputs (Wav2Vec2, HuBERT, Whisper encoder, etc.).

---

## What This Sample Demonstrates

The sample shows **four approaches** to generate embeddings from the same audio files, each progressively more decoupled:

### Approach 1: ML.NET Pipeline (Fit / Transform)

```csharp
var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var model = estimator.Fit(data);
var output = model.Transform(data);
var rows = mlContext.Data.CreateEnumerable<EmbeddingOutput>(output, reuseRowObject: false);
```

**Why this approach:** This is the standard ML.NET pattern. You create an estimator, fit it to data, transform, and enumerate results via `IDataView`. Familiar to anyone who has used ML.NET for tabular data — the same Fit/Transform contract works for audio.

### Approach 2: Composed Pipeline (`.Append()`)

```csharp
var pipeline = new AudioFeatureExtractionEstimator(mlContext, featureOptions)
    .Append(new OnnxAudioScoringEstimator(mlContext, scoringOptions))
    .Append(new AudioEmbeddingPoolingEstimator(mlContext, poolingOptions));
```

**Why this approach:** This decomposes the monolithic embedding estimator into three explicit stages:

1. **`AudioFeatureExtractionEstimator`** — audio samples → mel spectrogram
2. **`OnnxAudioScoringEstimator`** — mel spectrogram → raw ONNX hidden states
3. **`AudioEmbeddingPoolingEstimator`** — hidden states → pooled, normalized embedding

The key insight is **auto-discovery via column annotations**. When `OnnxAudioScoringTransformer` produces its output column, it attaches metadata annotations including `HiddenDim` (the last dimension of the ONNX output shape) and `HasPooledOutput` (whether the model already pooled its output). When `AudioEmbeddingPoolingEstimator.Fit()` runs, it reads these annotations automatically — you don't need to manually configure the hidden dimension. This is the ML.NET-standard way for pipeline stages to communicate.

### Approach 3: Direct API

```csharp
var embeddings = transformer.GenerateEmbeddings(audioInputs);
```

**Why this approach:** Sometimes you don't need the full `IDataView` ceremony. The `GenerateEmbeddings()` method on the transformer takes a list of `AudioData` objects and returns `float[][]` directly. One line. No schema, no enumeration. Ideal for scripting, prototyping, or when you're embedding audio outside of a larger ML.NET pipeline.

### Approach 4: MEAI `IEmbeddingGenerator`

```csharp
using IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

var embeddings = await generator.GenerateAsync(audioInputs);
```

**Why this approach:** `IEmbeddingGenerator<TInput, TEmbedding>` is the standard MEAI (Microsoft.Extensions.AI) interface for embedding generation. By wrapping the transformer in `OnnxAudioEmbeddingGenerator`, you get:

- **Dependency injection** — register the generator in your DI container
- **Middleware** — add logging, caching, rate-limiting, or telemetry via the MEAI middleware pipeline
- **Provider swapping** — swap the ONNX-based generator for a cloud-based one without changing consuming code
- **Async support** — the interface is natively async

The wrapper is thin: `OnnxAudioEmbeddingGenerator` holds a reference to the fitted transformer and delegates to its `GenerateEmbeddings()` method.

---

## Prerequisites

### ONNX Model

Download a CLAP ONNX model from HuggingFace:

```bash
pip install huggingface-hub
huggingface-cli download lquint/clap-htsat-unfused-onnx --local-dir models/clap
```

Place the model so it's accessible at `models/clap/onnx/model.onnx` relative to the working directory (or pass a custom path as the first argument).

### Test Audio Files

You **don't need to provide WAV files**. If no `.wav` files are found in the working directory, the sample auto-generates three test files:

| File | Content | Purpose |
|---|---|---|
| `tone_440hz.wav` | 2-second 440 Hz sine wave (A4 note) | Musical tone baseline |
| `tone_880hz.wav` | 2-second 880 Hz sine wave (A5 note) | Harmonically related tone (one octave up) |
| `noise.wav` | 2-second white noise (seeded RNG) | Deliberately different from tonal audio |

These are chosen so that the 440 Hz and 880 Hz tones should be **more similar to each other** (both tonal) than either is to noise — which is exactly what you'll see in the cosine similarity matrix.

### SDK

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)

---

## Running It

### With a model

```bash
cd samples/AudioEmbeddings
dotnet run
# Or with a custom model path:
dotnet run -- path/to/model.onnx
# Or with a custom model path and audio directory:
dotnet run -- path/to/model.onnx path/to/wav/files
```

**Expected output** (dimensions depend on your model):

```
=== ML.NET Pipeline (Fit / Transform) ===
  tone_440hz.wav: [512]-dim embedding
    First 5 values: [0.0234, -0.0156, 0.0412, ...]
  tone_880hz.wav: [512]-dim embedding
    ...

=== Composed Pipeline (.Append()) ===
  tone_440hz.wav: [512]-dim embedding
  ...

=== Direct Embedding API ===
  tone_440hz.wav: [512]-dim embedding
  ...

=== MEAI IEmbeddingGenerator<AudioData, Embedding<float>> ===
  Generated 3 embeddings

=== Cosine Similarity ===
  tone_440hz.wav vs tone_880hz.wav: 0.8523
  tone_440hz.wav vs noise.wav: 0.1247
  tone_880hz.wav vs noise.wav: 0.0983
```

The two tones have high similarity; both have low similarity with noise. That's the embedding doing its job.

### Without a model

If the model file isn't found, the sample runs a **synthetic demo** instead — no ONNX runtime needed:

```
Model not found at: models/clap/onnx/model.onnx
Running with synthetic demo instead...

=== Synthetic Demo (no ONNX model) ===
  Audio: 2.00s, 16000Hz, 1ch
  Mel spectrogram: [199 frames x 80 mel bins]
  Saved and reloaded: 2.00s, 32000 samples

Synthetic demo complete!
```

This demonstrates the `AudioData`, `AudioIO`, and `MelSpectrogramExtractor` primitives without requiring a model download.

---

## Code Walkthrough

### Auto-Generated Test Files

```csharp
static void GenerateTone(string path, float freq, int sr, int seconds)
{
    var samples = new float[sr * seconds];
    for (int i = 0; i < samples.Length; i++)
        samples[i] = MathF.Sin(2 * MathF.PI * freq * i / sr) * 0.5f;
    AudioIO.SaveWav(path, new AudioData(samples, sr));
}
```

The test files are generated programmatically: pure sine waves at 440 Hz and 880 Hz, plus white noise (seeded with `Random(42)` for reproducibility). The amplitude is scaled to `0.5f` (tones) or `0.3f` (noise) to avoid clipping. These are saved as 16-bit PCM WAV files via `AudioIO.SaveWav`.

The choice of frequencies is deliberate: 440 Hz and 880 Hz are harmonically related (octave apart), so a well-trained audio encoder should place them closer together in embedding space than either is to random noise.

### Column Annotations: Auto-Discovery of HiddenDim

This is one of the most important patterns in the sample. When you use Approach 2 (composed pipeline), the three stages need to communicate metadata:

1. **`OnnxAudioScoringTransformer`** inspects the ONNX model's output shape during `Fit()`. If the output is `[batch, sequence_length, hidden_dim]`, it stores `HiddenDim` and `HasPooledOutput = false` as **column annotations** (ML.NET metadata) on the output column.

2. **`AudioEmbeddingPoolingEstimator.Fit()`** reads the input column's annotations to auto-populate `HiddenDim` and `IsPrePooled`. This means you can write:

   ```csharp
   new AudioEmbeddingPoolingEstimator(mlContext,
       new AudioEmbeddingPoolingOptions
       {
           Pooling = AudioPoolingStrategy.MeanPooling,
           Normalize = true
           // No HiddenDim — auto-discovered from upstream annotations!
       });
   ```

   The pooler just works, regardless of whether the model outputs 256, 512, or 768-dimensional hidden states.

This is the ML.NET-standard way for pipeline stages to communicate — the same pattern used by built-in transforms like `NormalizeMeanVariance` (which stores mean/variance in column annotations for downstream use).

### TensorPrimitives.CosineSimilarity

```csharp
static float CosineSimilarity(float[] a, float[] b)
    => TensorPrimitives.CosineSimilarity(a, b);
```

`System.Numerics.Tensors.TensorPrimitives` (available in .NET 8+) provides SIMD-accelerated math primitives. This one-liner replaces what would otherwise be a manual 12-line implementation involving dot products, magnitudes, and division. It's hardware-accelerated (AVX2/AVX-512 on x64, NEON on ARM) and handles edge cases like zero-length spans.

### The MEAI Wrapper Pattern

```csharp
using IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);
```

The pattern is straightforward:

1. **Create the transformer once** (via `estimator.Fit()`) — this loads the ONNX model and is expensive
2. **Wrap it** in `OnnxAudioEmbeddingGenerator` — this is cheap, just a reference
3. **Use via the interface** — consuming code only sees `IEmbeddingGenerator<AudioData, Embedding<float>>`

Because `OnnxAudioEmbeddingGenerator` implements `IDisposable`, the `using` declaration ensures both the generator and its underlying transformer are cleaned up. In a real application, you'd register this in your DI container with a singleton or scoped lifetime.

---

## Key Takeaways

1. **Embeddings enable "fuzzy" audio comparison** — you're comparing semantic content, not exact waveform samples. Two different recordings of the same sound will have similar embeddings even if their raw PCM data is completely different.

2. **Column annotations are the ML.NET-standard way for pipeline stages to communicate** — `OnnxAudioScoringTransformer` writes `HiddenDim` metadata; `AudioEmbeddingPoolingEstimator` reads it. No manual configuration, no coupling between stages.

3. **`TensorPrimitives` provides SIMD-accelerated math primitives** — use `TensorPrimitives.CosineSimilarity`, `TensorPrimitives.Dot`, `TensorPrimitives.Normalize`, etc. instead of hand-rolling vector math. They're faster and correct.

4. **MEAI makes the embedding generator pluggable and DI-friendly** — wrapping in `IEmbeddingGenerator<AudioData, Embedding<float>>` means you can swap providers, add middleware (caching, logging, rate-limiting), and inject the generator wherever it's needed — all without changing consuming code.

---

## Going Further

- **[AudioDataIngestion sample](../AudioDataIngestion/)** — uses these same embeddings in a `DataIngestion` pipeline to build a searchable audio index
- **[MEAI Integration Guide](../../docs/meai-integration.md)** — deep dive into the `IEmbeddingGenerator` wrapper, middleware, and DI registration patterns
- **[Architecture Guide](../../docs/architecture.md)** — how the custom transforms, ONNX runtime, and MEAI layers fit together
