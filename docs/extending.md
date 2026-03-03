# Extending the Framework

## Overview

This guide explains how to add new audio models and transforms to the framework. Whether you're adding support for a new ONNX model or creating an entirely new audio task, the patterns are consistent and composable.

For background on the layered architecture, see [architecture.md](architecture.md). For a reference of existing transforms, see [transforms-guide.md](transforms-guide.md).

---

## Adding a New Encoder-Only Model

Encoder-only models (classification, embeddings, VAD) are the simplest. Follow the existing patterns in `src/MLNet.AudioInference.Onnx/`.

### Step 1: Create (or Reuse) an AudioFeatureExtractor

If the model needs a different feature extraction than existing extractors, create a new one in `src/MLNet.Audio.Core/`:

```csharp
public class MyModelFeatureExtractor : AudioFeatureExtractor
{
    public override int SampleRate => 16000;

    protected override float[,] ExtractFeatures(float[] samples)
    {
        // Your feature extraction logic here.
        // Return a [frames × features] array.
    }
}
```

The base class `AudioFeatureExtractor` (in `MLNet.Audio.Core`) handles:

- Resampling input audio to the expected sample rate via `AudioIO.Resample()`
- Converting stereo to mono via `AudioIO.ToMono()`
- Calling your `ExtractFeatures()` implementation with clean mono PCM samples

Existing extractors you can reuse:

| Extractor | Location | Notes |
|-----------|----------|-------|
| `MelSpectrogramExtractor` | `MLNet.Audio.Core` | Generic log-mel spectrogram. Configurable bins, FFT size, hop length, frequency range, window function. |
| `WhisperFeatureExtractor` | `MLNet.Audio.Core` | Whisper-specific. 30-second padding/chunking, 80 mel bins, 400 FFT, 160 hop. |

### Step 2: Create an Options Class

Follow the pattern from `OnnxAudioClassificationOptions` (in `Classification/`):

```csharp
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

public class MyModelOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Audio feature extractor to use for preprocessing.</summary>
    public required AudioFeatureExtractor FeatureExtractor { get; set; }

    /// <summary>Name of the input column containing audio samples (float[]). Default: "Audio".</summary>
    public string InputColumnName { get; set; } = "Audio";

    /// <summary>Name of the output column. Default: "MyOutput".</summary>
    public string OutputColumnName { get; set; } = "MyOutput";

    /// <summary>ONNX input tensor name. Null = auto-detect from model.</summary>
    public string? InputTensorName { get; set; }

    /// <summary>ONNX output tensor name. Null = auto-detect from model.</summary>
    public string? OutputTensorName { get; set; }

    /// <summary>Sample rate of the input audio. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>GPU device ID. Null = CPU only.</summary>
    public int? GpuDeviceId { get; set; }
}
```

Key patterns:
- Use `required` for properties that must be set (model path, feature extractor).
- Provide sensible defaults for everything else.
- Include `InputTensorName`/`OutputTensorName` with null = auto-detect.

### Step 3: Create a Transformer

The transformer composes three reusable sub-transforms. Follow `OnnxAudioClassificationTransformer` as a template:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

public sealed class MyModelTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly MyModelOptions _options;
    private readonly AudioFeatureExtractionTransformer _featureTransformer;
    private readonly OnnxAudioScoringTransformer _scorerTransformer;
    private readonly MyPostProcessTransformer _postProcessTransformer;

    public bool IsRowToRowMapper => true;

    internal MyModelTransformer(
        MLContext mlContext,
        MyModelOptions options,
        AudioFeatureExtractionTransformer featureTransformer,
        OnnxAudioScoringTransformer scorerTransformer,
        MyPostProcessTransformer postProcessTransformer)
    {
        _mlContext = mlContext;
        _options = options;
        _featureTransformer = featureTransformer;
        _scorerTransformer = scorerTransformer;
        _postProcessTransformer = postProcessTransformer;
    }

    /// <summary>
    /// ML.NET Transform — composed lazy pattern.
    /// Chains 3 sub-transforms; each is lazy over the IDataView.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var features = _featureTransformer.Transform(input);
        var scores = _scorerTransformer.Transform(features);
        var results = _postProcessTransformer.Transform(scores);
        return results;
    }

    /// <summary>
    /// Direct API — use outside ML.NET pipelines.
    /// </summary>
    public MyResult[] Process(IReadOnlyList<AudioData> audioInputs)
    {
        // Direct APIs bypass IDataView and call sub-transform methods directly.
        // See OnnxAudioClassificationTransformer.Classify() for a production example.
        throw new NotImplementedException("Replace with your direct API logic.");
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        // Chain output schemas through each sub-transform
        var featureSchema = _featureTransformer.GetOutputSchema(inputSchema);
        var scorerSchema = _scorerTransformer.GetOutputSchema(featureSchema);
        return _postProcessTransformer.GetOutputSchema(scorerSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Saving not supported.");

    public void Dispose()
    {
        _featureTransformer?.Dispose();
        _scorerTransformer?.Dispose();
        (_postProcessTransformer as IDisposable)?.Dispose();
    }
}
```

Key implementation notes:
- Use the **composed lazy pattern**: `Transform()` chains 3 sub-transforms (`AudioFeatureExtractionTransformer` → `OnnxAudioScoringTransformer` → post-process), each lazy over the `IDataView`. This is the pattern used by all encoder-only transforms in this framework.
- Encoder-decoder transforms (ASR, TTS) use **eager evaluation** instead, because autoregressive decoding loops don't decompose into lazy sub-transforms.
- Provide a **direct API** method (like `Classify()` or `GenerateEmbeddings()`) for use outside ML.NET pipelines. Direct convenience APIs are always eager.
- The composed transformer **does not** manage `InferenceSession` directly — that's handled by `OnnxAudioScoringTransformer`.

### Step 4: Create an Estimator

The estimator creates the transformer. Follow `OnnxAudioClassificationEstimator`:

```csharp
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

public sealed class MyModelEstimator : IEstimator<MyModelTransformer>
{
    private readonly MLContext _mlContext;
    private readonly MyModelOptions _options;

    public MyModelEstimator(MLContext mlContext, MyModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");

        _mlContext = mlContext;
        _options = options;
    }

    public MyModelTransformer Fit(IDataView input)
    {
        var env = (IHostEnvironment)_mlContext;

        // Stage 1: Feature extraction (audio → mel spectrogram)
        var featureEstimator = new AudioFeatureExtractionEstimator(env,
            new AudioFeatureExtractionOptions { /* configure from _options */ });
        var featureTransformer = featureEstimator.Fit(input);

        // Stage 2: ONNX scoring (features → raw model output)
        var featuredData = featureTransformer.Transform(input);
        var scorerEstimator = new OnnxAudioScoringEstimator(env,
            new OnnxAudioScoringOptions { ModelPath = _options.ModelPath, /* ... */ });
        var scorerTransformer = scorerEstimator.Fit(featuredData);

        // Stage 3: Task-specific post-processing (raw scores → final output)
        var scoredData = scorerTransformer.Transform(featuredData);
        var postProcessEstimator = new MyPostProcessEstimator(env, _options);
        var postProcessTransformer = postProcessEstimator.Fit(scoredData);

        return new MyModelTransformer(
            _mlContext, _options,
            featureTransformer, scorerTransformer, postProcessTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        // SchemaShape.Column has an internal constructor — use reflection
        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,       // or .Vector, .VariableVector
            (DataViewType)TextDataViewType.Instance,     // or NumberDataViewType.Single
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}
```

> **Important:** `SchemaShape.Column`'s 5-parameter constructor is `internal` in ML.NET. All estimators in this framework use the reflection workaround shown above. The 5 parameters are: `name`, `vectorKind`, `itemType`, `isKey`, `metadata`.

### Step 5: Add an MLContext Extension Method

In `src/MLNet.AudioInference.Onnx/MLContextExtensions.cs`, add an entry point:

```csharp
/// <summary>
/// Creates a MyModel estimator using an ONNX model.
/// </summary>
public static MyModelEstimator MyModel(
    this TransformsCatalog catalog,
    MyModelOptions options)
{
    var mlContext = GetMLContext(catalog);
    return new MyModelEstimator(mlContext, options);
}
```

This uses the existing `GetMLContext()` helper which extracts the `MLContext` from the `TransformsCatalog` via reflection.

### Step 6: (Optional) Add an MEAI Interface Wrapper

If the task maps to a `Microsoft.Extensions.AI` interface, create a wrapper class. See [Adding MEAI Support](#adding-meai-support) below.

---

## Adding a New Encoder-Decoder Model

Encoder-decoder models (ASR, TTS) are more complex because they use autoregressive decoding with KV cache. Follow `OnnxWhisperTransformer` (ASR) or `OnnxSpeechT5TtsTransformer` (TTS) as templates.

### Architecture Overview

Encoder-decoder models typically have **two or three** ONNX models:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Encoder** | Processes input features → hidden states | `encoder_model.onnx` |
| **Decoder** | Autoregressive token/frame generation with KV cache | `decoder_model_merged.onnx` |
| **Vocoder** (TTS only) | Converts mel spectrogram → waveform | `vocoder_model.onnx` |

### The KV Cache Pattern

Both Whisper and SpeechT5 use the same KV cache pattern from HuggingFace Optimum exports:

**Inputs:**
- `past_key_values.{layer}.decoder.key` — previous decoder self-attention cache
- `past_key_values.{layer}.decoder.value` — previous decoder self-attention cache
- `past_key_values.{layer}.encoder.key` — cross-attention cache (encoder output)
- `past_key_values.{layer}.encoder.value` — cross-attention cache (encoder output)
- `use_cache_branch` — scalar bool: `false` on first step, `true` after

**Outputs:**
- `present.{layer}.decoder.key` — updated decoder cache
- `present.{layer}.decoder.value` — updated decoder cache
- `present.{layer}.encoder.key` — updated cross-attention cache
- `present.{layer}.encoder.value` — updated cross-attention cache

**Decoding loop:**

1. First step: `use_cache_branch = false`, empty `past_key_values` tensors (seq_len = 0).
2. Run decoder, get output token/frame and `present.*` tensors.
3. Subsequent steps: `use_cache_branch = true`, feed `present` → `past_key_values`.
4. Repeat until EOS token (ASR) or stop probability threshold (TTS).

For ASR, see `WhisperKvCacheManager` — a dedicated class that manages cache tensors across decoder steps. For TTS, `OnnxSpeechT5TtsTransformer` manages cache inline using local variables (same pattern, simpler lifecycle).

### Auto-Detecting Model Dimensions

Rather than hardcoding `numLayers`, `numHeads`, and `headDim`, detect them from the ONNX model metadata. See `WhisperKvCacheManager.DetectFromModel()`:

```csharp
public static (int numLayers, int numHeads, int headDim) DetectFromModel(
    InferenceSession decoderSession)
{
    // Count layers by probing for past_key_values.{i}.decoder.key
    int numLayers = 0;
    while (decoderSession.InputMetadata.ContainsKey($"past_key_values.{numLayers}.decoder.key"))
        numLayers++;

    if (numLayers == 0)
        throw new InvalidOperationException(
            "Could not detect decoder layers. Expected 'past_key_values.0.decoder.key' in model inputs.");

    // Shape is [1, num_heads, seq_len, head_dim]
    var meta = decoderSession.InputMetadata[$"past_key_values.0.decoder.key"];
    var dims = meta.Dimensions;
    int numHeads = dims[1];
    int headDim = dims[3];

    return (numLayers, numHeads, headDim);
}
```

`OnnxSpeechT5TtsTransformer.DetectDecoderDimensions()` uses the same approach. Both transformers also accept override values via options (e.g., `NumDecoderLayers`, `NumAttentionHeads`) for models with non-standard metadata.

### Implementing the Decoder Loop

Here's the general pattern (simplified from the actual implementations):

```csharp
// 1. Run encoder once
using var encoderOutput = RunEncoder(audioFeatures);

// 2. Initialize state
int numLayers = ...; // from DetectFromModel
bool useCache = false;
var decoderKeys = new float[numLayers][];
var decoderValues = new float[numLayers][];
var encoderKeys = new float[numLayers][];
var encoderValues = new float[numLayers][];

// 3. Autoregressive decoding loop
for (int step = 0; step < maxSteps; step++)
{
    // Build inputs with KV cache
    var inputs = new List<NamedOnnxValue>();
    inputs.Add(/* decoder_input_ids or output_sequence */);
    inputs.Add(/* encoder_hidden_states from step 1 */);

    // Add cache tensors
    var cacheTensor = new DenseTensor<bool>(new[] { useCache }, new[] { 1 });
    inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", cacheTensor));

    for (int i = 0; i < numLayers; i++)
    {
        if (useCache && decoderKeys[i] != null)
        {
            // Feed previous present → past_key_values
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_key_values.{i}.decoder.key", BuildTensor(decoderKeys[i])));
            // ... same for .value, .encoder.key, .encoder.value
        }
        else
        {
            // Empty tensors with seq_len = 0
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_key_values.{i}.decoder.key",
                new DenseTensor<float>(new[] { 1, numHeads, 0, headDim })));
            // ... same for others
        }
    }

    // Run decoder
    using var results = decoderSession.Run(inputs);
    var outputDict = results.ToDictionary(r => r.Name, r => r);

    // Extract output token (ASR) or mel frame (TTS)
    ProcessStepOutput(outputDict);

    // Update KV cache from present.* outputs
    for (int i = 0; i < numLayers; i++)
    {
        decoderKeys[i] = ExtractData(outputDict[$"present.{i}.decoder.key"]);
        decoderValues[i] = ExtractData(outputDict[$"present.{i}.decoder.value"]);
        if (!useCache) // encoder cache only changes on first step
        {
            encoderKeys[i] = ExtractData(outputDict[$"present.{i}.encoder.key"]);
            encoderValues[i] = ExtractData(outputDict[$"present.{i}.encoder.value"]);
        }
    }

    useCache = true;

    // Check stop condition
    if (ShouldStop(outputDict))
        break;
}
```

---

## Creating a New AudioFeatureExtractor

For models that need different preprocessing, add a new class in `src/MLNet.Audio.Core/`:

```csharp
using System.Numerics.Tensors;

namespace MLNet.Audio.Core;

/// <summary>
/// Feature extractor for Wav2Vec2-style models that process raw waveform directly.
/// </summary>
public class Wav2Vec2FeatureExtractor : AudioFeatureExtractor
{
    public override int SampleRate => 16000;

    protected override float[,] ExtractFeatures(float[] samples)
    {
        // Wav2Vec2 processes raw waveform — normalize with mean subtraction + variance norm
        float mean = TensorPrimitives.Sum(samples) / samples.Length;
        var centered = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            centered[i] = samples[i] - mean;

        float variance = TensorPrimitives.SumOfSquares(centered) / centered.Length;
        float stdDev = MathF.Sqrt(variance + 1e-7f);
        for (int i = 0; i < centered.Length; i++)
            centered[i] /= stdDev;

        // Return [samples × 1] — raw waveform as single-feature frames
        var result = new float[samples.Length, 1];
        for (int i = 0; i < samples.Length; i++)
            result[i, 0] = centered[i];

        return result;
    }
}
```

The base class `AudioFeatureExtractor.Extract()` handles resampling and mono conversion before calling your `ExtractFeatures()`, so you only need to focus on the actual feature computation.

---

## Creating a Custom Tokenizer

### Using Microsoft.ML.Tokenizers

For models that use SentencePiece tokenization (like SpeechT5), use the `Microsoft.ML.Tokenizers` package. Note that the standard `SentencePieceTokenizer` only supports **BPE** and **Unigram** model types. For **Char** models (used by SpeechT5), use `SentencePieceCharTokenizer` from `MLNet.Audio.Tokenizers`:

```csharp
using Microsoft.ML.Tokenizers;
using MLNet.Audio.Tokenizers;

// For BPE/Unigram models — use the standard path:
using var stream = File.OpenRead("spm.model");
var tokenizer = SentencePieceTokenizer.Create(stream);

// For Char models (e.g., SpeechT5 spm_char.model) — use the Audio.Tokenizers fallback:
var charTokenizer = SentencePieceCharTokenizer.Create("spm_char.model");
var ids = charTokenizer.EncodeToIds(text);
```

`SentencePieceCharTokenizer` extends the `Tokenizer` base class from `Microsoft.ML.Tokenizers`, so it can be used anywhere a `Tokenizer` is expected. See `OnnxSpeechT5TtsTransformer` for a production example — it tries the standard `SentencePieceTokenizer` first and falls back to `SentencePieceCharTokenizer` for Char model types.

### Custom Implementation

For models with non-standard tokenization, implement a custom tokenizer. See `WhisperTokenizer` in `src/MLNet.Audio.Core/Tokenizers/` — it implements:

- **BPE vocabulary** for text tokens
- **Special tokens**: `<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`
- **Timestamp tokens**: `<|0.00|>` through `<|30.00|>` at 20ms resolution
- **Language tokens** for 99+ languages

### Audio Codec Tokenizers

For neural codec models, extend `AudioCodecTokenizer` (abstract base class in `MLNet.Audio.Core/Tokenizers/`). See `EnCodecTokenizer` in `MLNet.AudioInference.Onnx/Shared/` for an example.

---

## The IEstimator/ITransformer Pattern

Every transform in this framework follows the ML.NET contract:

```
IEstimator<ITransformer>.Fit(IDataView) → ITransformer
ITransformer.Transform(IDataView)       → IDataView
```

### Estimator Responsibilities

1. **Validate options** in the constructor (model path exists, required fields set).
2. **`Fit()`** creates and returns a transformer. For encoder-only ONNX models, `Fit()` composes 3 sub-transforms (feature extraction → ONNX scoring → post-processing). For encoder-decoder models, it instantiates the transformer directly.
3. **`GetOutputSchema()`** declares what output columns the transform produces.

### Transformer Responsibilities

1. **Constructor**: accept pre-built sub-transforms (for composed transforms) or create `InferenceSession` (for encoder-decoder transforms).
2. **`Transform()`**: for encoder-only transforms, use the **composed lazy** pattern — chain sub-transforms over the `IDataView`. For encoder-decoder transforms (ASR, TTS), use **eager evaluation** — autoregressive loops don't decompose into lazy sub-transforms.
3. **`GetOutputSchema()`**: declare output schema with `DataViewSchema.Builder`.
4. **`ICanSaveModel.Save()`**: throw `NotSupportedException` (ONNX models are external files).
5. **`IDisposable`**: dispose sub-transforms (composed) or the `InferenceSession` (encoder-decoder).

### The SchemaShape.Column Workaround

`SchemaShape.Column` has an internal 5-parameter constructor that external code can't access directly. All estimators in this framework use this reflection pattern:

```csharp
var colCtor = typeof(SchemaShape.Column)
    .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
    .First(c => c.GetParameters().Length == 5);

// Parameters: (name, vectorKind, itemType, isKey, metadata)
columns[name] = (SchemaShape.Column)colCtor.Invoke([
    name,
    SchemaShape.Column.VectorKind.Scalar,      // .Scalar, .Vector, or .VariableVector
    (DataViewType)TextDataViewType.Instance,    // or NumberDataViewType.Single, etc.
    false,                                       // isKey
    (SchemaShape?)null                           // metadata
]);
```

Common `VectorKind` choices:
- `Scalar` — single value (predicted label, confidence score)
- `Vector` — fixed-length array (probabilities, embeddings)
- `VariableVector` — variable-length array (audio samples)

---

## Adding MEAI Support

If the task maps to a `Microsoft.Extensions.AI` interface, create a wrapper class that owns the transformer.

### Pattern

```csharp
using Microsoft.Extensions.AI;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

public sealed class MyModelMeaiWrapper : IEmbeddingGenerator<AudioData, Embedding<float>>
{
    private readonly MyModelTransformer _transformer;

    public EmbeddingGeneratorMetadata Metadata { get; }

    public MyModelMeaiWrapper(MyModelTransformer transformer, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        _transformer = transformer;

        Metadata = new EmbeddingGeneratorMetadata(
            providerName: "MLNet.AudioInference.Onnx",
            defaultModelId: modelId);
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<AudioData> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var audioList = values.ToList();
        var embeddings = _transformer.GenerateEmbeddings(audioList);

        var result = new GeneratedEmbeddings<Embedding<float>>(
            embeddings.Select(e => new Embedding<float>(e)).ToList());

        return Task.FromResult(result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(MyModelMeaiWrapper))
            return this;
        return null;
    }

    public void Dispose() => _transformer?.Dispose();
}
```

See `OnnxAudioEmbeddingGenerator` for the full production example. The existing MEAI wrappers in this framework are:

| Wrapper | Interface | Location |
|---------|-----------|----------|
| `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` | `MEAI/OnnxAudioEmbeddingGenerator.cs` |
| `OnnxSpeechToTextClient` | `ISpeechToTextClient` | `MLNet.ASR.OnnxGenAI/OnnxSpeechToTextClient.cs` |
| `OnnxTextToSpeechClient` | `ITextToSpeechClient` | `TTS/OnnxTextToSpeechClient.cs` |

---

## Pipeline Composition

Transforms compose via ML.NET's `.Append()`:

```csharp
var pipeline = mlContext.Transforms
    .OnnxWhisper(whisperOptions)           // Audio → Text
    .Append(mlContext.Transforms.SpeechT5Tts(ttsOptions));  // Text → Audio
```

**Limitation:** ML.NET pipelines are strictly linear. For non-linear flows (branching, conditional logic, fan-out), use the direct transformer APIs instead:

```csharp
// Direct API — full control over data flow
var whisper = new OnnxWhisperTransformer(mlContext, whisperOptions);
var transcription = whisper.Transcribe(audioData);

if (transcription.Language == "en")
{
    var tts = new OnnxSpeechT5TtsTransformer(mlContext, ttsOptions);
    var outputAudio = tts.Synthesize(transcription.Text);
}
```

---

## Checklist for Adding a New Model

Use this checklist when adding a new model or task to the framework:

- [ ] **AudioFeatureExtractor** — Create in `MLNet.Audio.Core/` or reuse an existing one
- [ ] **Options class** — Create in the appropriate subfolder of `MLNet.AudioInference.Onnx/`
- [ ] **Transformer** — Implements `ITransformer`, `IDisposable`. Owns the `InferenceSession`.
- [ ] **Estimator** — Implements `IEstimator<T>`. Validates options, creates transformer.
- [ ] **MLContext extension** — Add entry point in `MLContextExtensions.cs`
- [ ] **(Optional) MEAI wrapper** — If task maps to a `Microsoft.Extensions.AI` interface
- [ ] **(Optional) DataIngestion component** — If task can be used in a Read → Chunk → Process pipeline
- [ ] **Sample** — Create a working sample in `samples/`
- [ ] **transforms-guide.md** — Document options, usage, and supported models
- [ ] **architecture.md** — Update if new architectural patterns are introduced

### File Organization Convention

```
src/MLNet.AudioInference.Onnx/
├── YourTask/
│   ├── MyModelOptions.cs
│   ├── MyModelTransformer.cs
│   └── MyModelEstimator.cs
├── MEAI/
│   └── MyModelMeaiWrapper.cs       (if applicable)
└── MLContextExtensions.cs           (add your extension method here)
```

---

## Creating DataIngestion Components

The `MLNet.Audio.DataIngestion` package provides `Microsoft.Extensions.DataIngestion` implementations for audio. The DataIngestion pipeline follows three stages: **Reader → Chunker → Processor**.

### Extending the Reader

To support new audio formats (e.g., MP3, FLAC), extend `AudioDocumentReader` or create a new `IngestionDocumentReader`:

```csharp
public class Mp3DocumentReader : IngestionDocumentReader
{
    public override Task<IngestionDocument> ReadAsync(
        Stream stream, string name, string mediaType, CancellationToken ct = default)
    {
        var audio = DecodeMp3(stream); // Your MP3 decoding logic
        var doc = new IngestionDocument(name);
        var section = new IngestionDocumentSection($"audio:{name}");
        section.Text = $"Audio: {name}, {audio.Duration.TotalSeconds:F2}s";
        section.Metadata["audio"] = audio;
        doc.Sections.Add(section);
        return Task.FromResult(doc);
    }
}
```

**Key pattern:** Store the decoded `AudioData` in `section.Metadata["audio"]` — this is how the reader communicates with the chunker. The `IngestionDocument` API doesn't have a generic content property, so metadata is the bridge.

### Extending the Chunker

To implement different chunking strategies (e.g., VAD-based chunking, silence detection, overlap windows):

```csharp
public class VadBasedChunker : IngestionChunker<AudioData>
{
    private readonly IVoiceActivityDetector _vad;

    public VadBasedChunker(IVoiceActivityDetector vad) => _vad = vad;

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IngestionDocument doc, [EnumeratorCancellation] CancellationToken ct = default)
    {
        // Get audio from reader metadata
        AudioData? audio = null;
        foreach (var section in doc.Sections)
            if (section.Metadata.TryGetValue("audio", out var obj) && obj is AudioData a)
                { audio = a; break; }

        if (audio is null) yield break;

        // Use VAD to find speech segments, yield one chunk per segment
        using var stream = new MemoryStream();
        AudioIO.SaveWav(stream, audio);
        stream.Position = 0;

        await foreach (var segment in _vad.DetectSpeechAsync(stream, cancellationToken: ct))
        {
            var start = (int)(segment.Start.TotalSeconds * audio.SampleRate);
            var end = (int)(segment.End.TotalSeconds * audio.SampleRate);
            var segmentAudio = new AudioData(audio.Samples[start..end], audio.SampleRate);

            var chunk = new IngestionChunk<AudioData>(segmentAudio, doc, $"speech-{segment.Start}");
            chunk.Metadata["startTime"] = segment.Start.TotalSeconds.ToString("F2");
            chunk.Metadata["endTime"] = segment.End.TotalSeconds.ToString("F2");
            chunk.Metadata["confidence"] = segment.Confidence.ToString("F2");
            yield return chunk;
        }
    }
}
```

### Extending the Processor

To add different processing stages (e.g., transcription, classification):

```csharp
public class AudioTranscriptionChunkProcessor : IngestionChunkProcessor<AudioData>
{
    private readonly ISpeechToTextClient _sttClient;

    public AudioTranscriptionChunkProcessor(ISpeechToTextClient sttClient)
        => _sttClient = sttClient;

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<AudioData>> chunks,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var chunk in chunks.WithCancellation(ct))
        {
            using var stream = new MemoryStream();
            AudioIO.SaveWav(stream, chunk.Content);
            stream.Position = 0;

            var response = await _sttClient.GetTextAsync(stream, cancellationToken: ct);
            chunk.Metadata["transcription"] = response.Text;
            yield return chunk;
        }
    }
}
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `IngestionChunk<AudioData>` (not `IngestionChunk<string>`) | Type-safe: chunk content IS the audio segment, not a text description |
| `section.Metadata["audio"]` for reader→chunker bridge | `IngestionDocument` has no generic content — metadata is the only extensible storage |
| `chunk.Metadata["embedding"]` for processor output | Keeps chunks enriched with computed data for downstream consumers |
| Processor takes `IEmbeddingGenerator` (not concrete type) | Provider-agnostic: any MEAI-compatible embedding source works |
| All methods return `IAsyncEnumerable` | Memory-efficient streaming for large audio files |
