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
| Tokenization | SentencePiece (character-level) | KittenTtsTokenizer (ML.Tokenizers) — 176-symbol IPA vocabulary |
| Text processing | Built-in encoder | MEDI pipeline: TtsSentenceChunker → EspeakPhonemizationProcessor |
| Decoding | Autoregressive with KV cache | Single forward pass |
| Voice mechanism | External speaker embedding (float[]) | NPZ voice archive with named voices |
| Output sample rate | 16 kHz | 24 kHz |
| Model sizes | ~642 MB total | 41–80 MB |

### Pipeline Stages

```
"Hello world!"
    ↓ TtsSentenceChunker (MEDI IngestionChunker<string>)
    │   Splits at sentence boundaries [.!?], token-aware sizing
    ↓ EspeakPhonemizationProcessor (MEDI IngestionChunkProcessor<string>)
    │   English text → IPA phonemes via espeak-ng subprocess
    ↓ KittenTtsTokenizer (Microsoft.ML.Tokenizers.Tokenizer)
    │   IPA characters → integer token IDs (176-symbol vocabulary)
    ↓ Single ONNX model
    │   Token IDs + voice embedding + speed → raw PCM waveform
    → AudioData at 24 kHz
```

### Available Voices

KittenTTS ships with **8 built-in voices**: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo.

Voice embeddings are stored in `voices.npz` (a NumPy compressed ZIP archive). The transformer reads this file natively using a built-in NPZ/NPY parser — no Python required.

### Understanding Phonemes and IPA

If you've ever noticed that "read" (present) and "read" (past) are spelled identically but sound different, you've discovered why TTS models can't just feed raw text to a neural network. English spelling is famously inconsistent:

| Word pair | Same spelling? | Same sound? |
|-----------|---------------|-------------|
| read (present) / read (past) | ✅ | ❌ (/ɹiːd/ vs /ɹɛd/) |
| lead (verb) / lead (noun) | ✅ | ❌ (/liːd/ vs /lɛd/) |
| through / though / thought | Similar | ❌ (all different vowels) |

**Phonemes** are the actual sounds of speech — the building blocks that distinguish one word from another. The **International Phonetic Alphabet (IPA)** provides a universal notation where each symbol maps to exactly one sound, regardless of language or spelling.

For example, "Hello world" becomes:

```
"Hello world" → /həˈloʊ wɜːld/

h  → voiceless glottal fricative (the breathy "h")
ə  → schwa (unstressed vowel, like "a" in "about")
ˈ  → primary stress marker (next syllable is emphasized)
l  → lateral approximant (tongue touches roof of mouth)
oʊ → diphthong (the "oh" sound glides from o to ʊ)
w  → labial-velar approximant (the "w" sound)
ɜː → open-mid central vowel (the "ur" in "world")
l  → lateral approximant again
d  → voiced alveolar plosive (the "d" sound)
```

**espeak-ng** is the industry-standard open-source phonemizer. It handles all the ambiguities of English spelling — stress placement, silent letters, irregular pronunciations — and produces clean IPA output.

KittenTTS uses **176 IPA symbols** in its vocabulary: 1 pad symbol, 11 punctuation marks, 52 ASCII letters, and 112 IPA extension characters. Each IPA character maps to exactly one integer token ID via `KittenTtsTokenizer`.

> **How is this different from SpeechT5?** SpeechT5 uses SentencePiece character-level tokenization — it feeds raw letters to the model and lets the encoder learn pronunciation implicitly. KittenTTS makes pronunciation *explicit* via phonemization, which gives the model cleaner input at the cost of requiring espeak-ng as an external dependency.

### How the Abstraction Layers Compose

KittenTTS is built from four layers, each using standard .NET abstractions:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Microsoft.Extensions.DataIngestion (MEDI)      │
│   TtsSentenceChunker → EspeakPhonemizationProcessor     │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Microsoft.ML.Tokenizers                        │
│   KittenTtsTokenizer (Tokenizer base class)             │
├─────────────────────────────────────────────────────────┤
│ Layer 1: ML.NET Custom Transforms                       │
│   OnnxKittenTtsTransformer (ITransformer)               │
├─────────────────────────────────────────────────────────┤
│ Layer 0: ONNX Runtime                                   │
│   Single-pass neural network inference                  │
└─────────────────────────────────────────────────────────┘
```

This layered design isn't accidental — each layer aligns with a real .NET ecosystem abstraction:

- **Composability** — `TtsSentenceChunker` and `EspeakPhonemizationProcessor` implement MEDI's `IngestionChunker<string>` and `IngestionChunkProcessor<string>`. They can be used independently or plugged into any pipeline that works with MEDI abstractions.
- **Ecosystem alignment** — the same `IngestionChunker<string>` abstraction that chunks documents for text RAG also chunks text for TTS. If you've used MEDI for search indexing, the chunking model is already familiar.
- **Swappability** — the transformer constructor accepts any `Microsoft.ML.Tokenizers.Tokenizer` via `options.Tokenizer`. Want a custom IPA tokenizer with a different symbol set? Provide it — the pipeline doesn't care.
- **Token-aware chunking** — instead of splitting at a fixed character count, `TtsSentenceChunker` accepts a `Func<string, int> measureLength` parameter. KittenTTS passes `tokenizer.CountTokens`, so chunks are measured in *tokens*, not characters. This matters because IPA symbols like `oʊ` are two Unicode characters but one token.

Here's how the transformer wires these layers together:

```csharp
// In OnnxKittenTtsTransformer constructor:
_tokenizer = options.Tokenizer ?? KittenTtsTokenizer.Create();
_phonemizer = new EspeakPhonemizationProcessor(options.EspeakPath);
_chunker = new TtsSentenceChunker(
    maxLengthPerChunk: options.MaxTokensPerChunk ?? options.MaxChunkLength,
    measureLength: s => _tokenizer.CountTokens(s));
```

## About KittenTTS

### What It Is

KittenTTS is a **lightweight neural text-to-speech model** from [KittenML](https://github.com/KittenML/KittenTTS). It generates natural-sounding speech from text using a single-pass neural network, in contrast to multi-stage models like SpeechT5 that require separate encoder, decoder, and vocoder stages.

### How It Differs from Traditional TTS

Traditional TTS (like SpeechT5) works in stages:
1. Encode text → hidden representations
2. Decode hidden states → mel spectrogram (one frame at a time, autoregressively)
3. Vocoder converts mel → PCM waveform

KittenTTS collapses all of this into **one model** that takes phoneme token IDs and outputs raw audio directly. The trade-off: it's simpler and faster, but has a fixed maximum input length (handled via token-aware chunking using Microsoft.Extensions.DataIngestion's `IngestionChunker` abstraction).

### What Problems It Solves

- **Accessibility** — generate spoken versions of text content for visually impaired users
- **Content creation** — narrate articles, generate voiceovers, create audio previews
- **Prototyping** — quickly test TTS in your application without cloud API costs or latency
- **Education** — learn how neural TTS works with a simple, inspectable pipeline
- **Edge deployment** — small model sizes (41–80 MB) suitable for devices with limited resources

### Quality Expectations

- Natural-sounding English speech with good prosody and intonation
- 8 distinct voices with different characteristics (pitch, timbre, speaking style)
- 24 kHz output — higher fidelity than 16 kHz models like SpeechT5
- Best with short to medium text (1–3 sentences). Very long text is chunked automatically
- Occasional artifacts at chunk boundaries for very long passages

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

## Expected Output

When run with a model, you'll see something like:

```
=== KittenTTS Text-to-Speech Sample ===

Model: models/kittentts
ONNX model: models/kittentts/model.onnx
Voices: models/kittentts/voices.npz

=== 1. Direct Synthesis ===
  Synthesizing with voice: Jasper, speed: 1.0x
  Generated 3.52s of audio at 24000Hz (84480 samples)
  Saved to output.wav

=== 2. ITextToSpeechClient (MEAI) ===
  Provider: OnnxKittenTts, Model: kittentts
  Audio: 132 KB WAV

=== 3. Available Voices ===
  Found 8 voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo

=== 4. Voice Showcase ===
  Bella: 2.31s | Jasper: 2.45s | Luna: 2.28s | Bruno: 2.52s

Done! Check output.wav and output_*.wav for results.
```

Without a model, the sample prints download instructions and API pattern examples — no crash, no exception.

## Limitations

- **English only** — KittenTTS is trained on English speech data. Other languages will produce garbled output or errors from espeak-ng phonemization.
- **Fixed voice set** — unlike SpeechT5 which supports custom speaker embeddings for voice cloning, KittenTTS only supports its 8 built-in voices.
- **Requires espeak-ng** — the external `espeak-ng` tool must be installed and on PATH for phonemization. This is an extra setup step compared to SpeechT5 which handles tokenization internally.
- **Chunk boundary artifacts** — very long text is split into token-based chunks (default 400 tokens). Audio quality may degrade slightly at chunk boundaries.
- **Model variant differences** — voice names and ONNX filenames differ between model sizes (mini/micro/nano). The transformer auto-detects and falls back gracefully, but the mini model's voice names (Bella, Jasper, etc.) are the canonical ones.
- **No streaming** — the ONNX model generates the full waveform in one pass. There's no incremental audio output during generation.

## Troubleshooting

### "espeak-ng not found" or phonemization errors

espeak-ng must be installed and accessible on your system PATH:

```bash
# Windows
winget install espeak-ng
# Restart your terminal after installation

# Linux / Codespaces
sudo apt install -y espeak-ng

# macOS
brew install espeak-ng
```

If installed but not found, set the path explicitly:
```csharp
var options = new OnnxKittenTtsOptions
{
    ModelPath = "models/kittentts/model.onnx",
    EspeakPath = @"C:\Program Files\eSpeak NG\espeak-ng.exe"  // Windows
};
```

### Voice not found warnings

If you see "Voice 'Bella' not found, falling back to first available voice", your model variant uses different voice names. The **nano** model uses `expr-voice-2-f`, `expr-voice-3-m`, etc. instead of Bella/Jasper. The transformer automatically falls back — this is expected behavior, not an error.

### Model path not found when using `dotnet run --project`

When running from the repo root (`dotnet run --project samples/KittenTTS`), relative paths resolve from your current directory, not the project directory. The sample handles this with an `AppContext.BaseDirectory` fallback. If you still have issues, use an absolute path:

```bash
dotnet run --project samples/KittenTTS -- "C:\full\path\to\models\kittentts"
```

### Output sounds garbled or silent

- Verify the model downloaded completely (`model.onnx` should be 41–80 MB depending on variant)
- Ensure `voices.npz` is present alongside the ONNX model
- Check that the audio player supports 24 kHz mono WAV files

## Going Further

| Sample / Doc | What It Shows |
|---|---|
| [`samples/TextToSpeech`](../TextToSpeech/) | SpeechT5 TTS — 3-model encoder-decoder-vocoder pipeline |
| [`samples/SpeechToText`](../SpeechToText/) | ASR via ISpeechToTextClient — round-trip partner for TTS |
| [`src/MLNet.Audio.DataIngestion`](../../src/MLNet.Audio.DataIngestion/) | TtsSentenceChunker, EspeakPhonemizationProcessor — the MEDI components used by KittenTTS |
| [`src/MLNet.Audio.Tokenizers`](../../src/MLNet.Audio.Tokenizers/) | KittenTtsTokenizer — the ML.Tokenizers IPA tokenizer |
| [`docs/architecture.md`](../../docs/architecture.md) | Full system architecture and TTS pipeline details |
| [`docs/meai-integration.md`](../../docs/meai-integration.md) | MEAI interface mapping including ITextToSpeechClient |
