# Transforms Guide

A comprehensive reference for every audio transform in the MLNet.Audio project. Each section covers what the transform does, when to use it, its internal architecture, a full options table, code examples (both direct API and ML.NET pipeline), and supported models.

---

## Table of Contents

1. [Audio Classification](#audio-classification)
2. [Audio Embeddings](#audio-embeddings)
3. [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
4. [Speech-to-Text: Provider-Agnostic](#speech-to-text-provider-agnostic)
5. [Speech-to-Text: ORT GenAI (Whisper)](#speech-to-text-ort-genai-whisper)
6. [Speech-to-Text: Raw ONNX (Whisper)](#speech-to-text-raw-onnx-whisper)
7. [Text-to-Speech: SpeechT5](#text-to-speech-speecht5)
8. [Comparison: Three ASR Approaches](#comparison-three-asr-approaches)

---

## Audio Classification

### What It Does

Classifies audio into discrete categories — speech vs. music vs. silence, emotion detection, acoustic event recognition, or any custom label set. The transform outputs a predicted label, a confidence score, and a full probability distribution across all classes.

### When To Use It

- **Content moderation** — detect speech, music, or silence in media files.
- **Emotion recognition** — classify sentiment from voice recordings.
- **Acoustic event detection** — identify environmental sounds (siren, glass breaking, dog bark).
- **Pre-filtering** — route audio to different downstream pipelines based on its type.

### Architecture

```
AudioData → MelSpectrogramExtractor → mel spectrogram (float[,])
         → ONNX encoder model → logits
         → softmax → probabilities
         → argmax + label mapping → PredictedLabel
```

### Key Types

| Type | Role |
|------|------|
| `OnnxAudioClassificationEstimator` | `IEstimator<OnnxAudioClassificationTransformer>` — fits the pipeline |
| `OnnxAudioClassificationTransformer` | `ITransformer, IDisposable` — runs inference |
| `OnnxAudioClassificationOptions` | All configuration properties |
| `AudioClassificationResult` | Structured output from the direct `Classify` API |

### Options

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ModelPath` | `string` | *(required)* | Path to the ONNX classification model file |
| `FeatureExtractor` | `AudioFeatureExtractor` | *(required)* | Feature extraction configuration (e.g., `MelSpectrogramExtractor`) |
| `Labels` | `string[]` | *(required)* | Class labels in the same order as the model's output logits |
| `InputColumnName` | `string` | `"Audio"` | Name of the input column containing `AudioData` |
| `PredictedLabelColumnName` | `string` | `"PredictedLabel"` | Output column for the predicted class label |
| `ProbabilitiesColumnName` | `string` | `"Probabilities"` | Output column for the full probability distribution (`float[]`) |
| `ScoreColumnName` | `string` | `"Score"` | Output column for the top-class confidence score |
| `InputTensorName` | `string?` | `null` | ONNX input tensor name; auto-detected from model if `null` |
| `OutputTensorName` | `string?` | `null` | ONNX output tensor name; auto-detected from model if `null` |
| `SampleRate` | `int` | `16000` | Expected sample rate of input audio |
| `GpuDeviceId` | `int?` | `null` | CUDA device ID for GPU inference; `null` = CPU, `0` = first GPU |

### Code Examples

**Direct API — classify a list of audio files:**

```csharp
var mlContext = new MLContext();

var options = new OnnxAudioClassificationOptions
{
    ModelPath = "models/ast/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Labels = new[] { "Speech", "Music", "Silence" }
};

var estimator = mlContext.Transforms.OnnxAudioClassification(options);
var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
var transformer = estimator.Fit(emptyData);

// Direct API: classify without building a full pipeline
var audioFiles = new[]
{
    AudioIO.LoadWav("samples/speech.wav"),
    AudioIO.LoadWav("samples/music.wav")
};

AudioClassificationResult[] results = transformer.Classify(audioFiles);

foreach (var result in results)
{
    Console.WriteLine($"Label: {result.PredictedLabel}, Score: {result.Score:P1}");
    for (int i = 0; i < result.Labels.Length; i++)
        Console.WriteLine($"  {result.Labels[i]}: {result.Probabilities[i]:P1}");
}
```

**ML.NET pipeline — batch processing with IDataView:**

```csharp
var pipeline = mlContext.Transforms.OnnxAudioClassification(
    new OnnxAudioClassificationOptions
    {
        ModelPath = "models/ast/onnx/model.onnx",
        FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
        Labels = new[] { "Speech", "Music", "Silence" },
        GpuDeviceId = 0 // use GPU
    });

var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new AudioInput { Audio = AudioIO.LoadWav("audio1.wav") },
    new AudioInput { Audio = AudioIO.LoadWav("audio2.wav") }
});

var model = pipeline.Fit(data);
var results = model.Transform(data);

// Read results from the IDataView
var predictedLabels = results.GetColumn<string>("PredictedLabel").ToArray();
var scores = results.GetColumn<float>("Score").ToArray();
```

### Supported Models

| Model | Description | Typical `NumMelBins` |
|-------|-------------|---------------------|
| **AST** (Audio Spectrogram Transformer) | General-purpose audio classification; pre-trained on AudioSet | 128 |
| **Wav2Vec2 for classification** | Fine-tuned Wav2Vec2 with a classification head | 80 |

Export from HuggingFace with `optimum-cli export onnx` or use a pre-exported ONNX model.

---

## Audio Embeddings

### What It Does

Generates fixed-size vector embeddings from audio for similarity search, RAG (Retrieval-Augmented Generation), clustering, or any downstream task that needs a numeric representation of audio content. Embeddings are pooled from the encoder hidden states and optionally L2-normalized.

### When To Use It

- **Similarity search** — find audio clips that sound alike.
- **RAG for audio** — embed audio into a vector database alongside text embeddings.
- **Clustering** — group audio by speaker, topic, or acoustic properties.
- **Transfer learning** — use embeddings as features in another ML model.

### Architecture

```
AudioData → MelSpectrogramExtractor → mel spectrogram (float[,])
         → ONNX encoder model → hidden states (float[sequence_len, hidden_dim])
         → pooling (mean / max / CLS token) → single vector (float[hidden_dim])
         → L2 normalization (optional) → Embedding
```

### Key Types

| Type | Role |
|------|------|
| `OnnxAudioEmbeddingEstimator` | `IEstimator<OnnxAudioEmbeddingTransformer>` — fits the pipeline |
| `OnnxAudioEmbeddingTransformer` | `ITransformer, IDisposable` — runs inference, exposes `GenerateEmbeddings` |
| `OnnxAudioEmbeddingOptions` | All configuration properties |
| `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` — MEAI integration |
| `AudioPoolingStrategy` | Enum: `MeanPooling`, `MaxPooling`, `ClsToken` |

### Options

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ModelPath` | `string` | *(required)* | Path to the ONNX encoder model |
| `FeatureExtractor` | `AudioFeatureExtractor` | *(required)* | Feature extraction configuration |
| `InputColumnName` | `string` | `"Audio"` | Input column containing `AudioData` |
| `OutputColumnName` | `string` | `"Embedding"` | Output column for the embedding vector (`float[]`) |
| `Pooling` | `AudioPoolingStrategy` | `MeanPooling` | How to reduce sequence-level hidden states to a single vector |
| `Normalize` | `bool` | `true` | Apply L2 normalization to the output embedding |
| `InputTensorName` | `string?` | `null` | ONNX input tensor name; auto-detected if `null` |
| `OutputTensorName` | `string?` | `null` | ONNX output tensor name; auto-detected if `null` |
| `SampleRate` | `int` | `16000` | Expected sample rate |
| `GpuDeviceId` | `int?` | `null` | CUDA device ID; `null` = CPU |

**`AudioPoolingStrategy` enum:**

| Value | Description |
|-------|-------------|
| `MeanPooling` | Average all hidden states across the sequence dimension |
| `MaxPooling` | Take the element-wise maximum across the sequence dimension |
| `ClsToken` | Use the first token's hidden state (CLS token) as the embedding |

### Code Examples

**Direct API — generate embeddings for similarity comparison:**

```csharp
var mlContext = new MLContext();

var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = "models/ast/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true
};

var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));

Console.WriteLine($"Embedding dimension: {transformer.EmbeddingDimension}");

var audios = new[]
{
    AudioIO.LoadWav("clip_a.wav"),
    AudioIO.LoadWav("clip_b.wav")
};

float[][] embeddings = transformer.GenerateEmbeddings(audios);

// Cosine similarity (already L2-normalized, so dot product = cosine similarity)
float similarity = TensorPrimitives.Dot(embeddings[0], embeddings[1]);
Console.WriteLine($"Similarity: {similarity:F4}");
```

**ML.NET pipeline:**

```csharp
var pipeline = mlContext.Transforms.OnnxAudioEmbedding(
    new OnnxAudioEmbeddingOptions
    {
        ModelPath = "models/ast/onnx/model.onnx",
        FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
        Pooling = AudioPoolingStrategy.ClsToken
    });

var model = pipeline.Fit(data);
var results = model.Transform(data);
var embeddings = results.GetColumn<float[]>("Embedding").ToArray();
```

**MEAI integration — `IEmbeddingGenerator<AudioData, Embedding<float>>`:**

```csharp
var transformer = estimator.Fit(emptyData);
var generator = new OnnxAudioEmbeddingGenerator(transformer, modelId: "ast-embed");

// Standard MEAI interface
GeneratedEmbeddings<Embedding<float>> embeddings = await generator.GenerateAsync(
    new[] { AudioIO.LoadWav("clip.wav") }
);

float[] vector = embeddings[0].Vector.ToArray();
Console.WriteLine($"Dimension: {vector.Length}");
```

### Supported Models

Any ONNX encoder model that takes a mel spectrogram as input and outputs hidden states works. Common choices:

| Model | Hidden Dim | Description |
|-------|-----------|-------------|
| **AST** (Audio Spectrogram Transformer) | 768 | Pre-trained on AudioSet |
| **Wav2Vec2** | 768 | Self-supervised speech representations |
| **HuBERT** | 768 / 1024 | Hidden-unit BERT for speech |

---

## Voice Activity Detection (VAD)

### What It Does

Detects speech segments in audio — returns a list of `(start, end, confidence)` tuples indicating where speech occurs. Uses the Silero VAD model, which processes PCM audio in small frames (512 samples) and outputs a speech probability per frame. The transform merges consecutive high-probability frames into speech segments with configurable thresholds.

### When To Use It

- **Pre-processing for ASR** — strip silence before transcription to reduce compute and improve accuracy.
- **Speaker diarization** — identify when *someone* is speaking (though not *who*).
- **Audio editing** — automatically trim silence or detect pauses.
- **Real-time applications** — the `IVoiceActivityDetector` interface supports streaming.

### Architecture

```
AudioData → PCM frames (512 samples each)
         → ONNX Silero VAD model → speech probability per frame
         → threshold (default 0.5) → binary speech/silence per frame
         → merge consecutive speech frames → apply min duration filters
         → pad boundaries → SpeechSegment[]
```

### Key Types

| Type | Role |
|------|------|
| `OnnxVadEstimator` | `IEstimator<OnnxVadTransformer>` — fits the pipeline |
| `OnnxVadTransformer` | `ITransformer, IVoiceActivityDetector, IDisposable` — runs inference |
| `OnnxVadOptions` | All configuration properties for the ML.NET transform |
| `IVoiceActivityDetector` | Standalone streaming interface |
| `VadOptions` | Options for the `IVoiceActivityDetector` interface |
| `SpeechSegment` | `record SpeechSegment(TimeSpan Start, TimeSpan End, float Confidence)` with computed `Duration` |

### Options (`OnnxVadOptions` — ML.NET pipeline)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ModelPath` | `string` | *(required)* | Path to Silero VAD ONNX model |
| `InputColumnName` | `string` | `"Audio"` | Input column containing `AudioData` |
| `OutputColumnName` | `string` | `"SpeechSegments"` | Output column for detected segments |
| `Threshold` | `float` | `0.5f` | Speech probability threshold (0.0–1.0) |
| `MinSpeechDuration` | `TimeSpan` | `250ms` | Minimum duration to count as speech |
| `MinSilenceDuration` | `TimeSpan` | `100ms` | Minimum silence duration to split segments |
| `SpeechPad` | `TimeSpan` | `30ms` | Padding added to start/end of each segment |
| `WindowSize` | `int` | `512` | Number of PCM samples per frame |
| `SampleRate` | `int` | `16000` | Expected sample rate |
| `GpuDeviceId` | `int?` | `null` | CUDA device ID; `null` = CPU |

### Options (`VadOptions` — standalone interface)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `Threshold` | `float` | `0.5f` | Speech probability threshold |
| `MinSpeechDuration` | `TimeSpan` | `250ms` | Minimum speech duration |
| `MinSilenceDuration` | `TimeSpan` | `100ms` | Minimum silence gap to split |
| `SpeechPad` | `TimeSpan` | `30ms` | Padding around segments |
| `SampleRate` | `int` | `16000` | Expected sample rate |

### Code Examples

**Direct API — detect speech in a single file:**

```csharp
var mlContext = new MLContext();

var options = new OnnxVadOptions
{
    ModelPath = "models/silero-vad/silero_vad.onnx",
    Threshold = 0.5f,
    MinSpeechDuration = TimeSpan.FromMilliseconds(250),
    MinSilenceDuration = TimeSpan.FromMilliseconds(100),
    SpeechPad = TimeSpan.FromMilliseconds(30)
};

var estimator = mlContext.Transforms.OnnxVad(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));

var audio = AudioIO.LoadWav("meeting.wav");
IReadOnlyList<SpeechSegment> segments = transformer.DetectSpeech(audio);

foreach (var seg in segments)
{
    Console.WriteLine($"[{seg.Start:mm\\:ss\\.ff} → {seg.End:mm\\:ss\\.ff}] " +
                      $"confidence={seg.Confidence:F2}, duration={seg.Duration.TotalSeconds:F1}s");
}
```

**Streaming API — `IVoiceActivityDetector` interface:**

```csharp
IVoiceActivityDetector detector = transformer; // OnnxVadTransformer implements IVoiceActivityDetector

await using var stream = File.OpenRead("meeting.wav");

await foreach (SpeechSegment segment in detector.DetectSpeechAsync(stream, new VadOptions
{
    Threshold = 0.6f,
    MinSpeechDuration = TimeSpan.FromMilliseconds(500)
}))
{
    Console.WriteLine($"Speech: {segment.Start} → {segment.End} ({segment.Confidence:P0})");
}
```

**ML.NET pipeline:**

```csharp
var pipeline = mlContext.Transforms.OnnxVad(new OnnxVadOptions
{
    ModelPath = "models/silero-vad/silero_vad.onnx",
    Threshold = 0.5f
});

var model = pipeline.Fit(data);
var results = model.Transform(data);
```

### Supported Models

| Model | Description |
|-------|-------------|
| **Silero VAD** | Lightweight, fast, accurate. The recommended model for this transform. Download from [Silero Models](https://github.com/snakers4/silero-vad). |

---

## Speech-to-Text: Provider-Agnostic

### What It Does

Wraps any `ISpeechToTextClient` implementation (Azure Speech, OpenAI Whisper API, a local ONNX model, or your own custom client) as a standard ML.NET pipeline step. This lets you swap ASR providers without changing your pipeline code.

### When To Use It

- **Cloud APIs** — Azure Speech Services, OpenAI Whisper API, Google Cloud Speech.
- **Provider portability** — swap providers by changing one line; the pipeline stays the same.
- **Hybrid pipelines** — combine cloud ASR with local VAD or embedding steps.

### Architecture

```
AudioData → ISpeechToTextClient.GetTextAsync() → text string
```

The transform delegates entirely to the `ISpeechToTextClient`. It does not do feature extraction, tokenization, or decoding itself.

### Key Types

| Type | Role |
|------|------|
| `SpeechToTextClientEstimator` | `IEstimator<SpeechToTextClientTransformer>` — fits the pipeline |
| `SpeechToTextClientTransformer` | `ITransformer, IDisposable` — runs inference via the client |
| `SpeechToTextClientOptions` | Configuration for column names and language settings |

### Options (`SpeechToTextClientOptions`)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `InputColumnName` | `string` | `"Audio"` | Input column containing `AudioData` |
| `OutputColumnName` | `string` | `"Text"` | Output column for transcribed text |
| `SampleRate` | `int` | `16000` | Expected sample rate of input audio |
| `SpeechLanguage` | `string?` | `null` | Speech language hint (e.g., `"en"`, `"es"`); `null` = auto-detect |
| `TextLanguage` | `string?` | `null` | Target text language for translation; `null` = same as speech |

### Code Examples

**Swapping providers — same pipeline, different client:**

```csharp
var mlContext = new MLContext();

// Option 1: Azure Speech
ISpeechToTextClient client = new AzureSpeechToTextClient(endpoint, key);

// Option 2: OpenAI Whisper API
ISpeechToTextClient client = new OpenAIClient(apiKey)
    .GetAudioClient("whisper-1")
    .AsISpeechToTextClient();

// Option 3: Local ONNX (via the ORT GenAI package)
ISpeechToTextClient client = new OnnxSpeechToTextClient(
    new OnnxSpeechToTextOptions { ModelPath = "models/whisper-tiny" });

// Same pipeline regardless of provider
var pipeline = mlContext.Transforms.SpeechToText(client);
var model = pipeline.Fit(data);
var results = model.Transform(data);
var texts = results.GetColumn<string>("Text").ToArray();
```

**With explicit options:**

```csharp
var pipeline = mlContext.Transforms.SpeechToText(client, new SpeechToTextClientOptions
{
    InputColumnName = "Audio",
    OutputColumnName = "Transcription",
    SpeechLanguage = "en",
    SampleRate = 16000
});
```

### Supported Providers

Any implementation of `ISpeechToTextClient` works. Known implementations:

| Provider | Package / Class |
|----------|----------------|
| Azure Speech Services | `AzureSpeechToTextClient` |
| OpenAI Whisper API | `OpenAIClient.GetAudioClient().AsISpeechToTextClient()` |
| Local ONNX (ORT GenAI) | `OnnxSpeechToTextClient` (from `MLNet.ASR.OnnxGenAI`) |

---

## Speech-to-Text: ORT GenAI (Whisper)

### What It Does

Runs Whisper ASR locally using **ONNX Runtime GenAI**. This is the simplest local approach — ORT GenAI handles the entire encoder-decoder loop, KV cache management, and beam search internally. You just provide a model directory and get text back.

### When To Use It

- **Simple local deployment** — no cloud dependency, minimal configuration.
- **ORT GenAI model format** — you already have a model exported for ORT GenAI.
- **Timestamps needed** — supports word/segment-level timestamps via `TranscribeWithTimestamps`.
- **MEAI integration** — implements `ISpeechToTextClient` for provider-agnostic use.

### Architecture

```
AudioData → WhisperFeatureExtractor → mel spectrogram (float[frames, num_mel_bins])
         → ORT GenAI model (encoder + decoder in one package)
         → text tokens → decoded text
```

ORT GenAI manages the autoregressive decoder loop, KV cache, and beam search. You don't need to handle these yourself.

### Key Types

| Type | Role |
|------|------|
| `OnnxSpeechToTextEstimator` | `IEstimator<OnnxSpeechToTextTransformer>` — fits the pipeline |
| `OnnxSpeechToTextTransformer` | `ITransformer, IDisposable` — runs inference; exposes `Transcribe` and `TranscribeWithTimestamps` |
| `OnnxSpeechToTextOptions` | All configuration properties |
| `OnnxSpeechToTextClient` | `ISpeechToTextClient` — MEAI integration |
| `TranscriptionResult` | Structured output with text, segments, token IDs, language |

> **Package:** `MLNet.ASR.OnnxGenAI` — this is a separate NuGet package from the main `MLNet.AudioInference.Onnx`.

### Options (`OnnxSpeechToTextOptions`)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ModelPath` | `string` | *(required)* | Path to the ORT GenAI model directory (contains `model.onnx`, `config.json`, etc.) |
| `FeatureExtractor` | `AudioFeatureExtractor?` | `null` | Custom feature extractor; `null` = use built-in `WhisperFeatureExtractor` |
| `InputColumnName` | `string` | `"Audio"` | Input column containing `AudioData` |
| `OutputColumnName` | `string` | `"Text"` | Output column for transcribed text |
| `MaxLength` | `int` | `256` | Maximum number of tokens to generate |
| `SampleRate` | `int` | `16000` | Expected sample rate |
| `Language` | `string?` | `null` | Language code (e.g., `"en"`, `"fr"`); `null` = auto-detect |
| `Translate` | `bool` | `false` | If `true`, translate to English |
| `IsMultilingual` | `bool` | `true` | Whether the model supports multiple languages |
| `NumMelBins` | `int` | `80` | Number of mel frequency bins (80 for most Whisper models, 128 for large-v3) |

### Code Examples

**Direct API — transcribe with timestamps:**

```csharp
var mlContext = new MLContext();

var options = new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-tiny",
    Language = "en",
    MaxLength = 256
};

var estimator = mlContext.Transforms.OnnxSpeechToText(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));

// Simple transcription
var audios = new[] { AudioIO.LoadWav("interview.wav") };
string[] texts = transformer.Transcribe(audios);
Console.WriteLine(texts[0]);

// Transcription with timestamps
TranscriptionResult[] results = transformer.TranscribeWithTimestamps(audios);
foreach (var segment in results[0].Segments)
{
    Console.WriteLine($"[{segment.Start:mm\\:ss} → {segment.End:mm\\:ss}] {segment.Text}");
}
// Output: [00:00 → 00:05] Hello world
//         [00:05 → 00:12] This is a test recording
```

**ML.NET pipeline:**

```csharp
var pipeline = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-tiny",
    Language = "en"
});

var model = pipeline.Fit(data);
var results = model.Transform(data);
var transcriptions = results.GetColumn<string>("Text").ToArray();
```

**MEAI integration — `ISpeechToTextClient`:**

```csharp
ISpeechToTextClient client = new OnnxSpeechToTextClient(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-tiny"
});

// Use with the provider-agnostic pipeline
var pipeline = mlContext.Transforms.SpeechToText(client);

// Or use directly
await using var stream = File.OpenRead("audio.wav");
var response = await client.GetTextAsync(stream);
Console.WriteLine(response.Text);
```

### Supported Models

Any Whisper model exported to ORT GenAI format:

| Model | Parameters | Speed | Quality |
|-------|-----------|-------|---------|
| `whisper-tiny` | 39M | Fastest | Good for simple audio |
| `whisper-base` | 74M | Fast | Better accuracy |
| `whisper-small` | 244M | Medium | Good balance |
| `whisper-medium` | 769M | Slow | High accuracy |
| `whisper-large-v3` | 1.5B | Slowest | Best accuracy (set `NumMelBins = 128`) |

---

## Speech-to-Text: Raw ONNX (Whisper)

### What It Does

Full-control Whisper ASR with manual encoder/decoder management and KV cache handling. This approach uses separate encoder and decoder ONNX models, giving you complete control over the autoregressive generation loop, temperature-based sampling, and KV cache lifecycle.

### When To Use It

- **Full control** — you need to customize the decoding strategy, manage KV cache, or modify generation behavior.
- **HuggingFace Optimum export** — your model is exported as separate `encoder_model.onnx` and `decoder_model_merged.onnx` files.
- **Learning / research** — understand how autoregressive ASR works at the ONNX level.
- **Custom decoding** — implement beam search, constrained decoding, or other non-standard strategies.

### Architecture

```
AudioData → WhisperFeatureExtractor → mel spectrogram (float[frames, num_mel_bins])
         → ONNX encoder model → encoder hidden states
         → ONNX decoder (autoregressive loop):
             token IDs + encoder states + KV cache → next token logits
             → temperature scaling → softmax (TensorPrimitives.SoftMax) → sampling
             → append token → update KV cache (WhisperKvCacheManager)
             → repeat until <|endoftext|> or MaxTokens
         → WhisperTokenizer → text + timestamps
```

### Key Types

| Type | Role |
|------|------|
| `OnnxWhisperEstimator` | `IEstimator<OnnxWhisperTransformer>` — fits the pipeline |
| `OnnxWhisperTransformer` | `ITransformer, IDisposable` — runs inference; exposes `Transcribe` and `TranscribeWithTimestamps` |
| `OnnxWhisperOptions` | All configuration properties |
| `WhisperKvCacheManager` | *(internal)* Manages key-value cache tensors for autoregressive decoding |
| `WhisperTranscriptionResult` | Structured output with text, segments, token IDs, language |
| `TranscriptionSegment` | `record TranscriptionSegment(string Text, TimeSpan Start, TimeSpan End)` |

### Options (`OnnxWhisperOptions`)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EncoderModelPath` | `string` | *(required)* | Path to the Whisper encoder ONNX model |
| `DecoderModelPath` | `string` | *(required)* | Path to the Whisper decoder ONNX model (merged with KV cache) |
| `InputColumnName` | `string` | `"Audio"` | Input column containing `AudioData` |
| `OutputColumnName` | `string` | `"Text"` | Output column for transcribed text |
| `MaxTokens` | `int` | `256` | Maximum number of tokens to generate per audio |
| `SampleRate` | `int` | `16000` | Expected sample rate |
| `Language` | `string?` | `null` | Language code; `null` = auto-detect |
| `Translate` | `bool` | `false` | If `true`, translate to English |
| `IsMultilingual` | `bool` | `true` | Whether the model supports multiple languages |
| `NumMelBins` | `int` | `80` | Number of mel frequency bins |
| `NumDecoderLayers` | `int` | `0` | Number of decoder transformer layers; `0` = auto-detect from model |
| `NumAttentionHeads` | `int` | `0` | Number of attention heads; `0` = auto-detect from model |
| `Temperature` | `float` | `0f` | Decoding temperature; `0` = greedy (argmax), `>0` = sampling with `TensorPrimitives.SoftMax` |

### Code Examples

**Direct API — transcribe with full control:**

```csharp
var mlContext = new MLContext();

var options = new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-tiny/encoder_model.onnx",
    DecoderModelPath = "models/whisper-tiny/decoder_model_merged.onnx",
    Language = "en",
    MaxTokens = 256,
    Temperature = 0f // greedy decoding
};

var estimator = mlContext.Transforms.OnnxWhisper(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));

// Simple transcription
var audios = new[] { AudioIO.LoadWav("meeting.wav") };
string[] texts = transformer.Transcribe(audios);
Console.WriteLine(texts[0]);

// Transcription with timestamps
WhisperTranscriptionResult[] results = transformer.TranscribeWithTimestamps(audios);
foreach (var segment in results[0].Segments)
{
    Console.WriteLine($"[{segment.Start:mm\\:ss} → {segment.End:mm\\:ss}] {segment.Text}");
}
// Output: [00:00 → 00:05] Hello world
//         [00:05 → 00:12] This is a test recording
```

**With temperature-based sampling:**

```csharp
var options = new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-small/encoder_model.onnx",
    DecoderModelPath = "models/whisper-small/decoder_model_merged.onnx",
    Temperature = 0.2f, // low temperature for near-greedy sampling
    MaxTokens = 448
};
```

**ML.NET pipeline:**

```csharp
var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-tiny/encoder_model.onnx",
    DecoderModelPath = "models/whisper-tiny/decoder_model_merged.onnx"
});

var model = pipeline.Fit(data);
var results = model.Transform(data);
var transcriptions = results.GetColumn<string>("Text").ToArray();
```

### Supported Models

Any Whisper model exported to ONNX with Optimum (`optimum-cli export onnx`):

| Model | Parameters | `NumDecoderLayers` | `NumAttentionHeads` |
|-------|-----------|-------------------|---------------------|
| `whisper-tiny` | 39M | 4 | 6 |
| `whisper-base` | 74M | 6 | 8 |
| `whisper-small` | 244M | 12 | 12 |
| `whisper-medium` | 769M | 24 | 16 |
| `whisper-large-v3` | 1.5B | 32 | 20 (set `NumMelBins = 128`) |

> Set `NumDecoderLayers` and `NumAttentionHeads` to `0` for auto-detection from the model.

---

## Text-to-Speech: SpeechT5

### What It Does

Generates speech audio from text using Microsoft's **SpeechT5** model. The pipeline tokenizes text with SentencePiece, encodes it, autoregressively generates mel spectrogram frames, and converts them to a waveform using a HiFi-GAN vocoder. Supports speaker customization via x-vector embeddings.

### When To Use It

- **Local TTS** — generate speech without a cloud API.
- **Speaker customization** — use different speaker embeddings (512-dim x-vectors) for different voices.
- **Batch synthesis** — generate audio for multiple texts in one call.
- **MEAI integration** — use the `ITextToSpeechClient` interface for provider-agnostic TTS.

### Architecture

```
text → SentencePiece tokenizer (spm_char.model) → token IDs
     → ONNX encoder (343 MB) → encoder hidden states
     → ONNX decoder (244 MB, autoregressive with KV cache):
         mel frames + encoder states + speaker embedding + KV cache
         → next mel frame + stop probability
         → repeat until stop probability > StopThreshold or MaxMelFrames
     → ONNX vocoder / HiFi-GAN (55 MB) → PCM waveform → AudioData
```

### Key Types

| Type | Role |
|------|------|
| `OnnxSpeechT5TtsEstimator` | `IEstimator<OnnxSpeechT5TtsTransformer>` — fits the pipeline |
| `OnnxSpeechT5TtsTransformer` | `ITransformer, IDisposable` — runs inference; exposes `Synthesize` and `SynthesizeBatch` |
| `OnnxSpeechT5Options` | All configuration properties |
| `OnnxTextToSpeechClient` | `ITextToSpeechClient` — our prototype MEAI-style client |
| `ITextToSpeechClient` | Interface: `GetAudioAsync`, `GetStreamingAudioAsync` |
| `TextToSpeechResponse` | Response containing `AudioData`, `Voice`, `Duration` |
| `TextToSpeechResponseUpdate` | Streaming chunk with `AudioData` and `IsFinal` flag |
| `TextToSpeechOptions` | `Voice`, `Speed`, `Language`, `SpeakerEmbedding` |

### Options (`OnnxSpeechT5Options`)

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EncoderModelPath` | `string` | *(required)* | Path to SpeechT5 encoder ONNX model (~343 MB) |
| `DecoderModelPath` | `string` | *(required)* | Path to SpeechT5 decoder ONNX model (~244 MB) |
| `VocoderModelPath` | `string` | *(required)* | Path to HiFi-GAN vocoder ONNX model (~55 MB) |
| `TokenizerModelPath` | `string?` | `null` | Path to SentencePiece `.model` file; `null` = look for `spm_char.model` alongside encoder |
| `SpeakerEmbeddingPath` | `string?` | `null` | Path to `.npy` file with 512-dim x-vector speaker embedding |
| `InputColumnName` | `string` | `"Text"` | Input column containing text to synthesize |
| `OutputColumnName` | `string` | `"Audio"` | Output column for generated `AudioData` |
| `MaxMelFrames` | `int` | `1000` | Maximum mel frames to generate (controls max audio length) |
| `StopThreshold` | `float` | `0.5f` | Stop probability threshold for decoder termination |
| `NumMelBins` | `int` | `80` | Number of mel frequency bins |
| `SampleRate` | `int` | `16000` | Output audio sample rate |
| `NumDecoderLayers` | `int` | `0` | Number of decoder layers; `0` = auto-detect |
| `NumAttentionHeads` | `int` | `0` | Number of attention heads; `0` = auto-detect |

### Required Model Files

| File | Size | Description |
|------|------|-------------|
| `encoder_model.onnx` | ~343 MB | Text encoder |
| `decoder_model_merged.onnx` | ~244 MB | Autoregressive mel decoder with KV cache |
| `decoder_postnet_and_vocoder.onnx` | ~55 MB | HiFi-GAN vocoder (mel → waveform) |
| `spm_char.model` | ~10 KB | SentencePiece character-level tokenizer |
| `speaker.npy` | ~2 KB | Default speaker embedding (512-dim x-vector) |

### Code Examples

**Direct API — synthesize speech:**

```csharp
var mlContext = new MLContext();

var options = new OnnxSpeechT5Options
{
    EncoderModelPath = "models/speecht5/encoder_model.onnx",
    DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
    VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx",
    TokenizerModelPath = "models/speecht5/spm_char.model",
    SpeakerEmbeddingPath = "models/speecht5/speaker.npy"
};

var estimator = mlContext.Transforms.SpeechT5Tts(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<TextInput>()));

// Synthesize a single utterance
AudioData audio = transformer.Synthesize("Hello, this is a test of speech synthesis.");
AudioIO.SaveWav("output.wav", audio);
Console.WriteLine($"Generated {audio.Duration.TotalSeconds:F1}s of audio at {audio.SampleRate}Hz");

// Batch synthesis
AudioData[] audios = transformer.SynthesizeBatch(new[]
{
    "First sentence.",
    "Second sentence.",
    "Third sentence."
});
```

**With custom speaker embedding:**

```csharp
// Load a different speaker's x-vector (512-dim float array)
float[] customSpeaker = LoadNpy("speakers/female_voice.npy");

AudioData audio = transformer.Synthesize("Hello world", speakerEmbedding: customSpeaker);
```

**ML.NET pipeline:**

```csharp
var pipeline = mlContext.Transforms.SpeechT5Tts(new OnnxSpeechT5Options
{
    EncoderModelPath = "models/speecht5/encoder_model.onnx",
    DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
    VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx"
});

var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new TextInput { Text = "Hello world" },
    new TextInput { Text = "Goodbye world" }
});

var model = pipeline.Fit(data);
var results = model.Transform(data);
```

**`ITextToSpeechClient` — our prototype interface:**

```csharp
ITextToSpeechClient client = new OnnxTextToSpeechClient(new OnnxSpeechT5Options
{
    EncoderModelPath = "models/speecht5/encoder_model.onnx",
    DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
    VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx"
});

// Simple generation
TextToSpeechResponse response = await client.GetAudioAsync("Hello, world!");
AudioIO.SaveWav("output.wav", response.Audio);
Console.WriteLine($"Duration: {response.Duration.TotalSeconds:F1}s");

// With options
TextToSpeechResponse response = await client.GetAudioAsync("Hello!", new TextToSpeechOptions
{
    Voice = "default",
    Speed = 1.0f,
    Language = "en",
    SpeakerEmbedding = customSpeakerVector
});

// Streaming generation
await foreach (var update in client.GetStreamingAudioAsync("A longer piece of text..."))
{
    ProcessAudioChunk(update.Audio);
    if (update.IsFinal)
        Console.WriteLine("Done!");
}
```

### Supported Models

| Model | Description |
|-------|-------------|
| **SpeechT5** (`microsoft/speecht5_tts`) | Microsoft's unified speech model. Export from HuggingFace with Optimum. |

---

## Comparison: Three ASR Approaches

| Aspect | Provider-Agnostic | ORT GenAI | Raw ONNX |
|--------|-------------------|-----------|----------|
| **Entry point** | `mlContext.Transforms.SpeechToText(client)` | `mlContext.Transforms.OnnxSpeechToText(options)` | `mlContext.Transforms.OnnxWhisper(options)` |
| **Package** | `MLNet.AudioInference.Onnx` | `MLNet.ASR.OnnxGenAI` | `MLNet.AudioInference.Onnx` |
| **Models** | Any cloud or local provider | ORT GenAI format | HuggingFace Optimum ONNX export |
| **Complexity** | Lowest | Low | Highest |
| **Control** | None (black box) | Medium | Full (KV cache, temperature, decoding) |
| **Dependencies** | Just MEAI abstractions | `OnnxRuntimeGenAI` | `OnnxRuntime` + our primitives |
| **KV Cache** | N/A (provider handles) | Managed by ORT GenAI internally | Manual (`WhisperKvCacheManager`) |
| **Timestamps** | Provider-dependent | Yes (`TranscribeWithTimestamps`) | Yes (`TranscribeWithTimestamps` + `WhisperTokenizer`) |
| **Feature extraction** | Provider handles | Ours (`WhisperFeatureExtractor`) or built-in | Ours (`WhisperFeatureExtractor`) |
| **Temperature control** | N/A | Via ORT GenAI config | Direct (`Temperature` option, `TensorPrimitives.SoftMax`) |
| **MEAI client** | ✅ The client itself | ✅ `OnnxSpeechToTextClient : ISpeechToTextClient` | ❌ Not directly |
| **Best for** | Cloud APIs, provider swapping | Simple local deployment | Full control, research, learning |

### Decision Flowchart

```
Need ASR in your app?
  ├─ Using a cloud API (Azure, OpenAI)? → Provider-Agnostic
  ├─ Want simple local inference? → ORT GenAI
  └─ Need full control over decoding? → Raw ONNX
```

### Migration Between Approaches

All three produce the same output column (`"Text"` by default), so downstream pipeline steps work unchanged:

```csharp
// Step 1: start with cloud
var asrStep = mlContext.Transforms.SpeechToText(azureClient);

// Step 2: move to local (just change one line)
var asrStep = mlContext.Transforms.OnnxSpeechToText(new OnnxSpeechToTextOptions
{
    ModelPath = "models/whisper-tiny"
});

// Rest of pipeline unchanged
var fullPipeline = asrStep
    .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"));
```
