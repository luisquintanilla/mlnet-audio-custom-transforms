# Audio Classification: Categorizing Sound with ML.NET

This sample demonstrates how to classify audio into semantic categories (speech, music, silence, animal sounds, etc.) using ML.NET custom transforms backed by an ONNX model. It shows **three different approaches** to the same task — a single-line facade, a composed sub-transform pipeline, and a direct API — so you can pick the abstraction level that fits your scenario.

## What You'll Learn

How to run an Audio Spectrogram Transformer (AST) model through ML.NET's `IEstimator<T>` / `ITransformer` pattern, and why the pipeline decomposes into three composable stages: feature extraction, scoring, and post-processing.

## Where This Fits in the Architecture

Audio classification uses a **3-stage ML.NET pipeline** pattern — the same decomposition used across all encoder-only transforms in this library:

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: AudioFeatureExtractionTransformer               │
│   Raw audio → Mel spectrogram (128 bins, 1024 frames)   │
├─────────────────────────────────────────────────────────┤
│ Stage 2: OnnxAudioScoringTransformer                     │
│   Mel spectrogram → ONNX Runtime inference → logits      │
├─────────────────────────────────────────────────────────┤
│ Stage 3: AudioClassificationPostProcessingTransformer    │
│   Logits → Softmax → Labels + Probabilities              │
└─────────────────────────────────────────────────────────┘
```

All three stages implement `ITransformer` (via `IEstimator<T>.Fit()`), which means they compose via ML.NET's lazy pipeline — computation only happens when you iterate the output `IDataView`. This is the same `Fit()`/`Transform()` pattern used across all ML.NET transforms.

The **single-line facade** (`mlContext.Transforms.OnnxAudioClassification(options)`) wires all three stages internally. The **composed pipeline** approach lets you replace or customize any stage — for example, swapping the feature extractor for a different spectrogram configuration.

This same 3-stage pattern appears in [AudioEmbeddings](../AudioEmbeddings/) (with pooling instead of softmax) — once you understand one, you understand both.

---

## The Concept: Audio Classification

### What It Means to "Classify" Audio

Audio classification assigns a categorical label to a segment of audio. Given a waveform, the system answers the question: _"What kind of sound is this?"_ — producing a label such as **Speech**, **Music**, **Silence**, **Dog Bark**, **Siren**, or any class defined by the model's training set.

Unlike speech recognition (which transcribes _words_), classification identifies the _type_ of sound. A single clip is mapped to one or more labels drawn from a fixed vocabulary.

### Real-World Use Cases

| Domain | Example |
|--------|---------|
| **Content moderation** | Detect gunshots, screams, or profanity in user-uploaded audio |
| **Acoustic event detection** | Smart-home devices recognizing doorbells, glass breaking, smoke alarms |
| **Emotion recognition** | Call-center analytics classifying caller mood (angry, happy, neutral) |
| **Music genre classification** | Tagging tracks as jazz, rock, classical for catalog organization |
| **Environmental monitoring** | Wildlife researchers identifying bird species by call |

### How It Works (High Level)

```
Raw Audio (PCM waveform)
    │
    ▼
┌──────────────────────────┐
│  Mel Spectrogram          │  ← Feature Extraction
│  (time × frequency image) │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Neural Network (AST)     │  ← ONNX Model Scoring
│  → raw logits per class   │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Softmax → Argmax         │  ← Post-Processing
│  → probability distribution│
│  → top predicted label     │
└──────────────────────────┘
```

1. **Feature Extraction** — The raw audio waveform is converted into a **mel spectrogram**: a 2D representation (time × mel-frequency bins) that captures how energy is distributed across frequencies over time. Think of it as a heat-map image of the sound.

2. **Model Scoring** — The spectrogram is fed into a neural network. This sample uses an **Audio Spectrogram Transformer (AST)**, which treats the mel spectrogram as an image and applies Vision Transformer (ViT) techniques — splitting it into patches, projecting them into embeddings, and using self-attention to capture global patterns. The output is a vector of raw logits, one per class.

3. **Post-Processing** — The logits are passed through **softmax** to produce a probability distribution (all values sum to 1.0), and **argmax** selects the class with the highest probability as the predicted label.

### The Model: Audio Spectrogram Transformer (AST)

The AST architecture (Gong et al., 2021) repurposes the Vision Transformer for audio. Instead of image patches, it operates on spectrogram patches — fixed-size windows of the mel spectrogram. This lets it leverage pre-trained image weights and capture long-range temporal dependencies via self-attention, outperforming CNN-based approaches on benchmarks like AudioSet.

The sample defaults to `ast-finetuned-audioset-10-10-0.4593-ONNX` from HuggingFace, fine-tuned on AudioSet's 527-class ontology. You define a subset of labels in your code (see below).

---

## What This Sample Demonstrates

The sample runs the same audio file through three approaches. All produce equivalent results — the difference is in the **abstraction level** and **composability**.

### Approach 1: ML.NET Facade Pipeline

```csharp
var pipeline = mlContext.Transforms.OnnxAudioClassification(options);
var model = pipeline.Fit(data);
var output = model.Transform(data);
```

**What:** A single estimator, configured with a single `OnnxAudioClassificationOptions` object, handles all three stages internally.

**Why it exists:** This is the simplest path. One line creates the estimator, one line fits, one line transforms. Recommended for most scenarios where you just want classification results without customizing the individual stages.

### Approach 2: Composed Pipeline (`.Append()`)

```csharp
var composedPipeline = new AudioFeatureExtractionEstimator(mlContext, featureOptions)
    .Append(new OnnxAudioScoringEstimator(mlContext, scoringOptions))
    .Append(new AudioClassificationPostProcessingEstimator(mlContext, postProcessingOptions));
```

**What:** Chains three independent sub-estimators using ML.NET's `.Append()` composition:

| Stage | Estimator | Input → Output |
|-------|-----------|----------------|
| 1 | `AudioFeatureExtractionEstimator` | Raw audio samples → mel spectrogram features |
| 2 | `OnnxAudioScoringEstimator` | Mel features → raw model logits |
| 3 | `AudioClassificationPostProcessingEstimator` | Logits → `PredictedLabel`, `Score`, `Probabilities` |

**Why it exists:** The sub-transforms are public and independently configurable. You can:
- Swap the feature extractor (e.g., different mel parameters or a completely different feature type)
- Replace the ONNX model without changing extraction or post-processing
- Write custom post-processing (e.g., multi-label thresholding instead of argmax)
- Insert additional transforms between stages (e.g., normalization, augmentation)

This is the same composition pattern used by ML.NET text transforms (tokenizer → model → pooler).

### Approach 3: Direct API

```csharp
var results = model.Classify(new[] { audio });
```

**What:** Calls `Classify()` directly on the transformer, bypassing `IDataView` entirely.

**Why it exists:** When you don't need ML.NET pipeline composition — no `.Append()`, no `IDataView`, no schema — and just want `AudioClassificationResult[]` from audio input. Ideal for one-off inference, scripting, or integration with non-ML.NET code paths.

---

## Prerequisites

### .NET 10 SDK

This sample targets `net10.0`. Install from [dotnet.microsoft.com](https://dotnet.microsoft.com/download/dotnet/10.0).

### ONNX Model

Download an AST model from HuggingFace (requires `huggingface-cli` — install via `pip install huggingface_hub`):

```bash
huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX --include "onnx/*" --local-dir models/ast
```

This places the ONNX model at `models/ast/onnx/model.onnx`.

### Audio File (Optional)

Place a WAV file named `test.wav` in the sample directory. If no file is found, the sample generates a synthetic 2-second 440Hz sine wave (A4 note) for testing.

---

## Running It

```bash
cd samples/AudioClassification
dotnet run
```

Or specify custom paths:

```bash
dotnet run -- path/to/model.onnx path/to/audio.wav
```

### Expected Output

**If the model is present:**

```
Loading audio: test.wav
  Duration: 2.00s, Sample Rate: 16000Hz, Samples: 32000

=== ML.NET Pipeline (Fit / Transform) ===
  Predicted: Music (confidence: 87.32%)
  Top 5 classes:
    Music: 87.32%
    Singing: 5.21%
    Speech: 3.44%
    ...

=== Composed Pipeline (Sub-Transforms) ===
  Predicted: Music (confidence: 87.32%)

=== Direct Classification API ===
  Predicted: Music (confidence: 87.32%)

Done!
```

**If the model is not present**, the sample prints download instructions and exits — no crash, no cryptic error.

---

## Code Walkthrough

### Options Setup

```csharp
var options = new OnnxAudioClassificationOptions
{
    ModelPath = modelPath,
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
    {
        NumMelBins = 128,  // AST expects 128 mel frequency bins
        FftSize = 400,     // 25ms window at 16kHz
        HopLength = 160    // 10ms hop → 100 frames/second
    },
    Labels = labels,
    SampleRate = 16000
};
```

- **`MelSpectrogramExtractor`** — Configures how the raw waveform is converted into a mel spectrogram. The parameters (`NumMelBins`, `FftSize`, `HopLength`) must match what the model was trained with. AST uses 128 mel bins, a 25ms FFT window (400 samples at 16kHz), and a 10ms hop (160 samples).
- **`Labels`** — The class names corresponding to the model's output indices. The sample uses a 20-label subset of AudioSet for demo purposes; the full AudioSet ontology has 527 classes.
- **`SampleRate`** — The expected input sample rate. Audio is resampled to this rate before feature extraction.

### The Fit / Transform Pattern

```csharp
var model = pipeline.Fit(data);
var output = model.Transform(data);
```

Even though the AST model is pre-trained (its weights are frozen), ML.NET requires `Fit()` before `Transform()`. This is by design:

- `Fit()` initializes the transformer — loads the ONNX model, validates the schema, and returns an `ITransformer`.
- `Transform()` runs inference on the data.

The `IEstimator<T>` → `ITransformer` contract is consistent across ML.NET regardless of whether training is involved. Pre-trained model estimators perform initialization in `Fit()` rather than learning.

### How Probabilities Become Labels

The post-processing stage performs:

1. **Softmax** — Converts raw logits into a probability distribution: `P(class_i) = exp(logit_i) / Σ exp(logit_j)`. All values are ≥ 0 and sum to 1.0.
2. **Argmax** — Selects the index with the highest probability.
3. **Label lookup** — Maps the index to a human-readable label from the `Labels` array.

The output schema includes all three pieces of information:

| Column | Type | Description |
|--------|------|-------------|
| `PredictedLabel` | `string` | The top predicted class name |
| `Score` | `float` | The confidence (probability) of the top prediction |
| `Probabilities` | `float[]` | Full probability distribution over all classes |

### The Composed Pipeline and IHostEnvironment Covariance

```csharp
var composedPipeline = new AudioFeatureExtractionEstimator(mlContext, ...)
    .Append(new OnnxAudioScoringEstimator(mlContext, ...))
    .Append(new AudioClassificationPostProcessingEstimator(mlContext, ...));
```

Each sub-estimator's constructor takes `IHostEnvironment` (ML.NET's internal runtime context). You can pass `MLContext` directly because `MLContext` implements `IHostEnvironment`. This is standard ML.NET covariance — every estimator and transformer receives the environment for logging, cancellation, and resource management.

The `.Append()` calls produce an `EstimatorChain<T>` that composes estimators sequentially, fitting and transforming in order. The same composition mechanism powers all ML.NET pipelines (text, image, tabular).

### Input and Output Classes

```csharp
class AudioInput
{
    public float[] Audio { get; set; } = [];
}

class ClassificationOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";

    [ColumnName("Score")]
    public float Score { get; set; }

    [ColumnName("Probabilities")]
    [VectorType]
    public float[] Probabilities { get; set; } = [];
}
```

ML.NET maps between `IDataView` columns and POCO properties using `[ColumnName]` attributes. The `[VectorType]` attribute tells ML.NET that `Probabilities` is a variable-length vector column.

---

## Key Takeaways

1. **The 3-stage pattern is universal.** Audio classification follows the same Feature Extraction → Scoring → Post-Processing decomposition as text transforms (tokenization → model → pooling). Learning this pattern once transfers to every modality.

2. **Sub-transforms are public and composable.** You're not locked into the `OnnxAudioClassification` facade. The three estimators (`AudioFeatureExtractionEstimator`, `OnnxAudioScoringEstimator`, `AudioClassificationPostProcessingEstimator`) are independently usable. Mix, match, replace, or extend any stage.

3. **Classification outputs are rich.** You get `PredictedLabel` (the answer), `Score` (how confident), and `Probabilities[]` (the full distribution). Use the distribution for multi-label scenarios, confidence thresholds, or debugging.

4. **Three abstraction levels for three use cases.** Facade for simplicity, composed pipeline for flexibility, direct API for minimal ceremony. Same model, same results, different ergonomics.

---

## Troubleshooting

### "Model not found" error
The AST ONNX model must be downloaded separately. The `onnx/` subdirectory should contain `model.onnx`:
```bash
huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX --include "onnx/*" --local-dir models/ast
```

### Audio file format issues
- The sample auto-resamples to 16kHz — most formats work
- If you get garbled results, ensure your WAV file is valid PCM (not compressed)
- Very short audio (<0.5s) may not produce meaningful classification results

### Unexpected labels or low confidence
- AST was trained on AudioSet — it recognizes ~527 labels but the sample shows a 20-class subset
- Very quiet audio may classify as "Silence" even if it contains faint sounds
- Mixed audio (speech over music) may split confidence across multiple labels — this is expected behavior, not an error

### GPU vs CPU
The `samples/Directory.Build.props` auto-detects CUDA at build/restore time. If you have a GPU but want CPU-only inference, unset the `CUDA_PATH` environment variable before building:
```bash
# PowerShell
Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
dotnet run -- "models/ast" "test.wav"

# Bash
unset CUDA_PATH
dotnet run -- "models/ast" "test.wav"
```

---

## Going Further

- **[AudioEmbeddings sample](../AudioEmbeddings/)** — Same 3-stage pattern, but post-processing extracts a dense embedding vector instead of a label. Useful for similarity search, clustering, and downstream ML tasks.
- **[Architecture guide](../../docs/architecture.md)** — How the library is structured: core audio primitives, ONNX inference layer, and the ML.NET integration surface.
- **[Transforms guide](../../docs/transforms-guide.md)** — Deep dive into the custom transform pattern: how estimators, transformers, and options classes fit together in ML.NET's extensibility model.
