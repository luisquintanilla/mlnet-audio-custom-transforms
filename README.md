# ML.NET Audio Custom Transforms

Multi-task audio inference transforms for ML.NET using local ONNX models. Brings the same patterns from [text-based ML.NET transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) to the audio domain — classification, embeddings, speech-to-text, text-to-speech, and voice activity detection.

> **New to audio ML?** Start with the [Audio Processing Primer](docs/audio-primer.md) — no prior audio knowledge required.

## Packages

| Package | Description | Key Dependencies |
|---------|-------------|-----------------|
| `MLNet.Audio.Core` | Audio primitives: `AudioData`, WAV I/O, mel spectrogram, `WhisperTokenizer` | NWaves, System.Numerics.Tensors |
| `MLNet.Audio.Tokenizers` | Text tokenizer extensions for audio models: `SentencePieceCharTokenizer` | Microsoft.ML.Tokenizers |
| `MLNet.AudioInference.Onnx` | ML.NET transforms: classification, embeddings, VAD, raw ONNX ASR/TTS | Microsoft.ML, OnnxRuntime, ML.Tokenizers, MEAI |
| `MLNet.ASR.OnnxGenAI` | Local Whisper speech-to-text via ORT GenAI | Microsoft.ML, OnnxRuntimeGenAI, MEAI |
| `MLNet.Audio.DataIngestion` | DataIngestion components: audio document reader, chunker, embedding processor | DataIngestion.Abstractions, MEAI, Audio.Core |

## Supported Audio Tasks

| Task | Status | Key Types | MEAI Interface |
|------|--------|-----------|---------------|
| Audio Classification | ✅ | `OnnxAudioClassificationTransformer` | — |
| Audio Embeddings | ✅ | `OnnxAudioEmbeddingTransformer`, `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` |
| Voice Activity Detection | ✅ | `OnnxVadTransformer` | `IVoiceActivityDetector` (custom) |
| Speech-to-Text (Provider) | ✅ | `SpeechToTextClientTransformer` | `ISpeechToTextClient` (any provider) |
| Speech-to-Text (ORT GenAI) | ✅ | `OnnxSpeechToTextTransformer`, `OnnxSpeechToTextClient` | `ISpeechToTextClient` |
| Speech-to-Text (Raw ONNX) | ✅ | `OnnxWhisperTransformer`, `OnnxWhisperSpeechToTextClient` | `ISpeechToTextClient` |
| Text-to-Speech (SpeechT5) | ✅ | `OnnxSpeechT5TtsTransformer`, `OnnxTextToSpeechClient` | `ITextToSpeechClient` |
| Text-to-Speech (KittenTTS) | ✅ | `OnnxTextToSpeechClient` via `OnnxKittenTtsOptions` | `ITextToSpeechClient` |

## Quick Start

```csharp
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

var mlContext = new MLContext();
var audio = AudioIO.LoadWav("audio.wav");

// --- Audio Classification (AST) — full Fit/Transform pattern ---
var options = new OnnxAudioClassificationOptions
{
    ModelPath = "models/ast/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Labels = new[] { "Speech", "Music", "Silence" }
};

// Load data
var data = mlContext.Data.LoadFromEnumerable(new[] { new AudioInput { Audio = audio.Samples } });

// Create pipeline, fit, and transform
var pipeline = mlContext.Transforms.OnnxAudioClassification(options);
var model = pipeline.Fit(data);
var output = model.Transform(data);

// Enumerate results
var results = mlContext.Data.CreateEnumerable<ClassificationOutput>(output, reuseRowObject: false);
```

The other tasks follow the same Fit/Transform pattern. Pipeline creation for each:

```csharp
// --- Audio Embeddings (CLAP) ---
var pipeline = mlContext.Transforms.OnnxAudioEmbedding(new OnnxAudioEmbeddingOptions
{
    ModelPath = "models/clap/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000),
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true
});

// --- Voice Activity Detection (Silero) ---
var pipeline = mlContext.Transforms.OnnxVad(new OnnxVadOptions
{
    ModelPath = "models/silero-vad/silero_vad.onnx",
    Threshold = 0.5f
});

// --- Speech-to-Text (provider-agnostic) ---
ISpeechToTextClient sttClient = /* Azure, OpenAI, local, etc. */;
var pipeline = mlContext.Transforms.SpeechToText(sttClient);

// --- Speech-to-Text (Raw ONNX Whisper via ISpeechToTextClient) ---
ISpeechToTextClient whisperClient = new OnnxWhisperSpeechToTextClient(whisperOptions);
var pipeline = mlContext.Transforms.SpeechToText(whisperClient);
// or: mlContext.Transforms.OnnxWhisperSpeechToText(whisperOptions);

// --- Speech-to-Text (Raw ONNX Whisper — direct transformer) ---
var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = "models/whisper-base/encoder_model.onnx",
    DecoderModelPath = "models/whisper-base/decoder_model_merged.onnx",
    Language = "en"
});

// --- Text-to-Speech (SpeechT5) ---
var pipeline = mlContext.Transforms.SpeechT5Tts(new OnnxSpeechT5Options
{
    EncoderModelPath = "models/speecht5/encoder_model.onnx",
    DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
    VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx",
});
```

## MEAI Integration

Integrates with [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai):

```csharp
// Audio Embeddings via MEAI
var estimator = mlContext.Transforms.OnnxAudioEmbedding(embeddingOptions);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);
var embeddings = await generator.GenerateAsync([audio]);

// Speech-to-Text via MEAI (ORT GenAI)
ISpeechToTextClient sttClient = new OnnxSpeechToTextClient(sttOptions);
var response = await sttClient.GetTextAsync(audioStream);

// Speech-to-Text via MEAI (Raw ONNX — no ORT GenAI dep)
ISpeechToTextClient rawClient = new OnnxWhisperSpeechToTextClient(whisperOptions);
var response = await rawClient.GetTextAsync(audioStream);
Console.WriteLine($"{response.Text} [{response.StartTime} → {response.EndTime}]");

// Speech-to-Text with middleware pipeline
var client = new OnnxSpeechToTextClient(sttOptions)
    .AsBuilder()
    .UseLogging()
    .UseOpenTelemetry()
    .Build();

// Text-to-Speech via official MEAI ITextToSpeechClient
ITextToSpeechClient ttsClient = new OnnxTextToSpeechClient(ttsOptions);
var response = await ttsClient.GetAudioAsync("Hello, world!");
var audioContent = response.Contents.OfType<DataContent>().First();
File.WriteAllBytes("output.wav", audioContent.Data.ToArray());
```

See [MEAI Integration Guide](docs/meai-integration.md) for DI patterns and middleware.

## DataIngestion Integration

Integrates with [Microsoft.Extensions.DataIngestion](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion.Abstractions) to prove that DataIngestion is modality-agnostic — not just for text/PDF:

```csharp
using MLNet.Audio.DataIngestion;

// Layer 3: DataIngestion — Read audio files into documents
var reader = new AudioDocumentReader(targetSampleRate: 16000);
var doc = await reader.ReadAsync(stream, "audio.wav", "audio/wav");

// Layer 3: DataIngestion — Chunk into fixed time-windows
var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2));
var chunks = chunker.ProcessAsync(doc);

// Layer 3: DataIngestion — Enrich with embeddings via MEAI → ML.NET
var processor = new AudioEmbeddingChunkProcessor(generator);
await foreach (var chunk in processor.ProcessAsync(chunks))
{
    var embedding = (float[])chunk.Metadata["embedding"];
    // Use for similarity search, clustering, RAG, etc.
}
```

See [Architecture Guide](docs/architecture.md) for the full layered design.

## Samples

| Sample | Task | Description |
|--------|------|-------------|
| [`AudioClassification`](samples/AudioClassification) | Classification | Classify audio using AST (Audio Spectrogram Transformer) |
| [`AudioEmbeddings`](samples/AudioEmbeddings) | Embeddings | Generate vector embeddings + cosine similarity |
| [`VoiceActivityDetection`](samples/VoiceActivityDetection) | VAD | Detect speech segments using Silero VAD |
| [`SpeechToText`](samples/SpeechToText) | STT | Provider-agnostic ASR patterns + multi-modal pipeline |
| [`WhisperTranscription`](samples/WhisperTranscription) | STT | Local Whisper via ORT GenAI |
| [`WhisperRawOnnx`](samples/WhisperRawOnnx) | STT | Full-control Whisper with manual KV cache |
| [`TextToSpeech`](samples/TextToSpeech) | TTS | SpeechT5 encoder-decoder-vocoder synthesis |
| [`KittenTTS`](samples/KittenTTS) | TTS | Lightweight KittenTTS with espeak-ng phonemization |
| [`AudioDataIngestion`](samples/AudioDataIngestion) | DataIngestion | End-to-end Read → Chunk → Embed → Similarity Search |

All samples run without models — they show API patterns and download instructions as graceful fallback.

## Architecture

```
Layer 1 (ML.NET):         Audio (PCM) → Feature Extraction → ONNX Scoring → Post-processing → Result
Layer 2 (MEAI):           IEmbeddingGenerator<AudioData, Embedding<float>> / ISpeechToTextClient / ITextToSpeechClient
Layer 3 (DataIngestion):  AudioDocumentReader → AudioSegmentChunker → AudioEmbeddingChunkProcessor
```

Three-stage pipeline pattern mirroring the text transform architecture. See [Architecture Guide](docs/architecture.md).

- **Encoder-only** (classification, embeddings, VAD): single-pass, mel → ONNX → result. Uses a **composed** 3-stage pattern with lazy `IDataView` wrappers (feature extraction → scoring → post-processing as separate `ITransformer` stages).
- **Encoder-decoder** (Whisper ASR): mel → encoder → decoder loop with KV cache → text. Stays **monolithic** — the autoregressive decode loop cannot be split across lazy stages.
- **Encoder-decoder-vocoder** (SpeechT5 TTS): tokens → encoder → decoder loop with KV cache → mel → vocoder → audio. Also **monolithic** due to the sequential decode loop.

## .NET Primitives Used

| Primitive | Where Used |
|-----------|-----------|
| `System.Numerics.Tensors` / `TensorPrimitives` | Softmax, normalization, mel features, argmax, temperature sampling |
| `Microsoft.ML.Tokenizers` (SentencePiece) | SpeechT5 text tokenization (with `SentencePieceCharTokenizer` fallback from Audio.Tokenizers for Char models) |
| `Microsoft.Extensions.AI` | `IEmbeddingGenerator`, `ISpeechToTextClient`, `ITextToSpeechClient` |
| `Microsoft.Extensions.DataIngestion` | `IngestionDocumentReader`, `IngestionChunker<AudioData>`, `IngestionChunkProcessor<AudioData>` |
| Custom `WhisperTokenizer` | Whisper BPE + timestamps + language codes |
| `AudioFeatureExtractor` (abstract) | Audio's equivalent of `Tokenizer` |
| `AudioData` | Core audio type: float[] samples + sample rate + channels |

## Documentation

📖 **[Full Documentation](docs/README.md)** — Architecture, transforms guide, audio primer, MEAI integration, models guide, extending the framework, and samples walkthrough.

## Prerequisites

- .NET 10 SDK
- ONNX models from HuggingFace (see [Models Guide](docs/models-guide.md))

## Getting Started

### Codespaces / DevContainer
Open in GitHub Codespaces for a pre-configured environment with .NET 10, Python (for model downloads), and C# Dev Kit.

### NuGet Packages
Published to [GitHub Packages](https://github.com/luisquintanilla?tab=packages&repo_name=mlnet-audio-custom-transforms):
- `MLNet.Audio.Core`
- `MLNet.Audio.Tokenizers`
- `MLNet.AudioInference.Onnx`
- `MLNet.ASR.OnnxGenAI`
- `MLNet.Audio.DataIngestion`

Add the GitHub Packages source to your `nuget.config`:
```xml
<add key="github" value="https://nuget.pkg.github.com/luisquintanilla/index.json" />
```

## Related Projects

| Project | Description |
|---------|-------------|
| [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) | Text-based ML.NET transforms (same architectural patterns) |
| [model-packages-prototype](https://github.com/luisquintanilla/model-packages-prototype) | ModelPackages SDK for NuGet-wrapped AI models |
| [dotnet-model-garden-prototype](https://github.com/luisquintanilla/dotnet-model-garden-prototype) | Model garden with pre-packaged AI models |
| [dotnet-tokenizers-guide](https://github.com/luisquintanilla/dotnet-tokenizers-guide) | Microsoft.ML.Tokenizers guide |
| [dotnet-tensors-guide](https://github.com/luisquintanilla/dotnet-tensors-guide) | System.Numerics.Tensors guide |
