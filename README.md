# ML.NET Audio Custom Transforms

Multi-task audio inference transforms for ML.NET using local ONNX models. Brings the same patterns from [text-based ML.NET transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) to the audio domain — classification, embeddings, speech-to-text, text-to-speech, and voice activity detection.

> **New to audio ML?** Start with the [Audio Processing Primer](docs/audio-primer.md) — no prior audio knowledge required.

## Packages

| Package | Description | Key Dependencies |
|---------|-------------|-----------------|
| `MLNet.Audio.Core` | Audio primitives: `AudioData`, WAV I/O, mel spectrogram, `WhisperTokenizer` | NWaves, System.Numerics.Tensors |
| `MLNet.AudioInference.Onnx` | ML.NET transforms: classification, embeddings, VAD, raw ONNX ASR/TTS | Microsoft.ML, OnnxRuntime, ML.Tokenizers, MEAI |
| `MLNet.ASR.OnnxGenAI` | Local Whisper speech-to-text via ORT GenAI | Microsoft.ML, OnnxRuntimeGenAI, MEAI |

## Supported Audio Tasks

| Task | Status | Key Types | MEAI Interface |
|------|--------|-----------|---------------|
| Audio Classification | ✅ | `OnnxAudioClassificationTransformer` | — |
| Audio Embeddings | ✅ | `OnnxAudioEmbeddingTransformer`, `OnnxAudioEmbeddingGenerator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` |
| Voice Activity Detection | ✅ | `OnnxVadTransformer` | `IVoiceActivityDetector` (custom) |
| Speech-to-Text (Provider) | ✅ | `SpeechToTextClientTransformer` | `ISpeechToTextClient` (any provider) |
| Speech-to-Text (ORT GenAI) | ✅ | `OnnxSpeechToTextTransformer` | `ISpeechToTextClient` |
| Speech-to-Text (Raw ONNX) | ✅ | `OnnxWhisperTransformer`, `WhisperKvCacheManager` | — |
| Text-to-Speech (SpeechT5) | ✅ | `OnnxSpeechT5TtsTransformer`, `OnnxTextToSpeechClient` | `ITextToSpeechClient` (prototype) |

## Quick Start

```csharp
using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

var mlContext = new MLContext();

// --- Audio Classification (AST) ---
var pipeline = mlContext.Transforms.OnnxAudioClassification(new OnnxAudioClassificationOptions
{
    ModelPath = "models/ast/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Labels = new[] { "Speech", "Music", "Silence" }
});

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

// --- Speech-to-Text (Raw ONNX Whisper) ---
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
IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(embeddingOptions);
var embeddings = await generator.GenerateAsync([audio]);

// Speech-to-Text via MEAI
ISpeechToTextClient sttClient = new OnnxSpeechToTextClient(sttOptions);
var response = await sttClient.GetTextAsync(audioStream);

// Text-to-Speech via our prototype ITextToSpeechClient
ITextToSpeechClient ttsClient = new OnnxTextToSpeechClient(ttsOptions);
var response = await ttsClient.GetAudioAsync("Hello, world!");
AudioIO.SaveWav("output.wav", response.Audio);
```

See [MEAI Integration Guide](docs/meai-integration.md) for DI patterns and middleware.

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

All samples run without models — they show API patterns and download instructions as graceful fallback.

## Architecture

```
Audio (PCM) → Feature Extraction → ONNX Scoring → Post-processing → Result
```

Three-stage pipeline pattern mirroring the text transform architecture. See [Architecture Guide](docs/architecture.md).

- **Encoder-only** (classification, embeddings, VAD): single-pass, mel → ONNX → result
- **Encoder-decoder** (Whisper ASR): mel → encoder → decoder loop with KV cache → text
- **Encoder-decoder-vocoder** (SpeechT5 TTS): tokens → encoder → decoder loop with KV cache → mel → vocoder → audio

## .NET Primitives Used

| Primitive | Where Used |
|-----------|-----------|
| `System.Numerics.Tensors` / `TensorPrimitives` | Softmax, normalization, mel features, argmax, temperature sampling |
| `Microsoft.ML.Tokenizers` (SentencePiece) | SpeechT5 text tokenization |
| Custom `WhisperTokenizer` | Whisper BPE + timestamps + language codes |
| `AudioFeatureExtractor` (abstract) | Audio's equivalent of `Tokenizer` |
| `AudioData` | Core audio type: float[] samples + sample rate + channels |

## Documentation

📖 **[Full Documentation](docs/README.md)** — Architecture, transforms guide, audio primer, MEAI integration, models guide, extending the framework, and samples walkthrough.

## Prerequisites

- .NET 10 SDK
- ONNX models from HuggingFace (see [Models Guide](docs/models-guide.md))

## Related Projects

| Project | Description |
|---------|-------------|
| [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) | Text-based ML.NET transforms (same architectural patterns) |
| [model-packages-prototype](https://github.com/luisquintanilla/model-packages-prototype) | ModelPackages SDK for NuGet-wrapped AI models |
| [dotnet-model-garden-prototype](https://github.com/luisquintanilla/dotnet-model-garden-prototype) | Model garden with pre-packaged AI models |
| [dotnet-tokenizers-guide](https://github.com/luisquintanilla/dotnet-tokenizers-guide) | Microsoft.ML.Tokenizers guide |
| [dotnet-tensors-guide](https://github.com/luisquintanilla/dotnet-tensors-guide) | System.Numerics.Tensors guide |
