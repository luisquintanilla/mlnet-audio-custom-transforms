# KittenTTS: Lightweight Text-to-Speech with espeak-ng Phonemization

## What You'll Learn

- **KittenTTS architecture** — a single-pass ONNX model using espeak-ng for phonemization
- **Voice embeddings from NPZ** — how KittenTTS selects voices from a compressed NumPy archive
- **ITextToSpeechClient** — using the official MEAI interface with KittenTTS as a backend
- **SpeechT5 vs KittenTTS** — contrasting a 3-model pipeline with a single-model approach

## The Concept: Lightweight Neural TTS

### How KittenTTS Works

KittenTTS takes a fundamentally different approach from SpeechT5:

| Aspect | SpeechT5 | KittenTTS |
|--------|----------|-----------|
| ONNX models | 3 (encoder + decoder + vocoder) | 1 (single-pass) |
| Tokenization | SentencePiece (character-level) | espeak-ng IPA phonemes → token IDs |
| Decoding | Autoregressive with KV cache | Single forward pass |
| Voice mechanism | External speaker embedding (float[]) | NPZ voice archive with named voices |
| Output sample rate | 16 kHz | 24 kHz |
| Model sizes | ~642 MB total | 41–80 MB |

### Pipeline Stages

```
"Hello world!"
    ↓ Text chunking (max 400 chars, sentence boundaries)
    ↓ Phonemization via espeak-ng (IPA with stress marks)
    ↓ TextCleaner (IPA symbols → integer token IDs)
    ↓ Single ONNX model (token IDs + voice embedding + speed → waveform)
    ↓ Trim trailing samples, concatenate chunks
    → AudioData at 24 kHz
```

### Available Voices

KittenTTS ships with **8 built-in voices**: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo.

Voice embeddings are stored in `voices.npz` (a NumPy compressed ZIP archive). The transformer reads this file natively using a built-in NPZ/NPY parser — no Python required.

## Prerequisites

### Required Software

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- **espeak-ng** — required for phonemization:
  - Windows: `winget install espeak-ng`
  - Linux: `apt install espeak-ng`
  - macOS: `brew install espeak-ng`

### Model Files

Download a KittenTTS model from HuggingFace:

```bash
cd samples/KittenTTS
git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8 models/kittentts
```

Available model variants:

| Model | Params | Size | HuggingFace |
|-------|--------|------|-------------|
| Mini | 80M | ~80 MB | `KittenML/kitten-tts-mini-0.8` |
| Micro | 40M | ~41 MB | `KittenML/kitten-tts-micro-0.8` |
| Nano | 15M | ~56 MB | `KittenML/kitten-tts-nano-0.8` |

The model directory should contain:

| File | Purpose |
|------|---------|
| `model.onnx` | Single-pass TTS model |
| `voices.npz` | Voice embeddings (8 voices) |

## Running It

### With model files (full synthesis)

```bash
cd samples/KittenTTS

# Default text and voice (Jasper)
dotnet run -- "models/kittentts"

# Custom text
dotnet run -- "models/kittentts" "Hello, this is KittenTTS!"

# Custom text and voice
dotnet run -- "models/kittentts" "Hello!" Luna
```

Output: `output.wav` — a 24kHz mono PCM WAV file.

### Without model files (pattern demonstration)

```bash
cd samples/KittenTTS
dotnet run
```

Shows download instructions and API pattern examples.

## What This Sample Demonstrates

### 1. Direct Synthesis

```csharp
using var transformer = new OnnxKittenTtsTransformer(mlContext, options);
var audio = transformer.Synthesize("Hello world!", "Jasper", speed: 1.0f);
AudioIO.SaveWav("output.wav", audio);
```

### 2. ITextToSpeechClient (official MEAI)

```csharp
using var client = new OnnxTextToSpeechClient(kittenOptions);
var ttsOptions = new TextToSpeechOptions { VoiceId = "Luna", Speed = 1.0f };
var response = await client.GetAudioAsync("Say something", ttsOptions);
var audioContent = response.Contents.OfType<DataContent>().First();
File.WriteAllBytes("output.wav", audioContent.Data.ToArray());
```

Note: `OnnxTextToSpeechClient` accepts both `OnnxSpeechT5Options` and `OnnxKittenTtsOptions` — one client serves all TTS backends via the internal `IOnnxTtsSynthesizer` interface.

### 3. Multiple Voices

```csharp
foreach (var voice in new[] { "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo" })
{
    var audio = transformer.Synthesize("Hello!", voice);
    AudioIO.SaveWav($"output_{voice}.wav", audio);
}
```

### 4. ML.NET Pipeline

```csharp
var pipeline = mlContext.Transforms.KittenTts(options);
var model = pipeline.Fit(data);
var predictions = model.Transform(data);
```

## Going Further

| Sample / Doc | What It Shows |
|---|---|
| [`samples/TextToSpeech`](../TextToSpeech/) | SpeechT5 TTS — 3-model encoder-decoder-vocoder pipeline |
| [`samples/SpeechToText`](../SpeechToText/) | ASR via ISpeechToTextClient — round-trip partner for TTS |
| [`docs/architecture.md`](../../docs/architecture.md) | Full system architecture and TTS pipeline details |
| [`docs/meai-integration.md`](../../docs/meai-integration.md) | MEAI interface mapping including ITextToSpeechClient |
