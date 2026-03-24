# Text-to-Speech: SpeechT5 Synthesis and the Encoder-Decoder-Vocoder Chain

## What You'll Learn

- **TTS synthesis** — how text becomes natural-sounding speech waveforms
- **Encoder-decoder-vocoder architecture** — why TTS needs *three* models, not two
- **Speaker embeddings** — how a fixed vector controls voice identity without retraining
- **`ITextToSpeechClient`** — the official MEAI interface for TTS, backed by local ONNX models
- **Whisper ↔ SpeechT5 symmetry** — how ASR and TTS are mirror-image pipelines sharing the same KV cache pattern

## Where This Fits in the Architecture

SpeechT5 is the most complex transform in the library — a **3-model autoregressive pipeline**:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 2: MEAI                                            │
│   ITextToSpeechClient (same interface for SpeechT5 AND  │
│   KittenTTS — swap backends without changing your code)  │
├─────────────────────────────────────────────────────────┤
│ Layer 1: ML.NET Transform                                │
│   OnnxSpeechT5TtsTransformer (ITransformer)              │
│   ├─ SentencePieceCharTokenizer (ML.Tokenizers)         │
│   ├─ ONNX Encoder → hidden states                        │
│   ├─ ONNX Decoder (autoregressive + KV cache) → mel     │
│   └─ ONNX Vocoder (HiFi-GAN) → PCM waveform            │
├─────────────────────────────────────────────────────────┤
│ Layer 0: Audio Primitives                                 │
│   AudioData, AudioIO (WAV I/O)                           │
└─────────────────────────────────────────────────────────┘
```

The `ITextToSpeechClient` abstraction means you can switch from SpeechT5 to [KittenTTS](../KittenTTS/) (or a cloud API) with a single line change. The [KittenTTS sample](../KittenTTS/) shows the lightweight alternative.

## The Concept: Text-to-Speech Synthesis

### What TTS Is

Text-to-speech (TTS) converts written text into natural-sounding speech waveforms — the inverse of speech recognition (ASR/STT). A modern neural TTS system doesn't just concatenate pre-recorded phoneme clips; it *generates* audio from scratch, producing fluid, natural speech with appropriate prosody, rhythm, and intonation.

### The SpeechT5 Architecture — Three Models, Not Two

Unlike simpler encoder-decoder models, SpeechT5 TTS requires a **three-stage pipeline**: encoder → decoder → vocoder. Each stage has a distinct role:

#### 1. Encoder: Text → Hidden Representations

The encoder takes a sequence of token IDs (produced by a SentencePiece tokenizer) and transforms them into **hidden state vectors** — dense representations that capture the meaning, structure, and phonetic content of the input text. Think of this as the model "understanding" what it needs to say.

```
"Hello world" → SentencePiece → [token_ids] → Encoder ONNX → [hidden_states]
```

#### 2. Decoder: Hidden Representations → Mel Spectrogram Frames

The decoder takes the encoder's hidden states plus a **speaker embedding** (voice identity vector) and autoregressively generates **mel spectrogram frames** — a frequency-domain representation of the sound. Each step produces one frame of 80 mel bins, and the output of each step feeds back as input to the next.

This is where the KV cache comes in: just like Whisper's decoder caches key/value tensors to avoid recomputing attention over already-generated tokens, SpeechT5's decoder caches attention state over already-generated mel frames. The decoder also outputs a **stop probability** at each step — when this exceeds a threshold (default 0.5), generation stops. This is different from ASR, where an end-of-text token signals completion.

The mel spectrogram is a "sound blueprint" — it describes *what frequencies should be present at each moment* but isn't audio you can play directly. It lives in the frequency domain.

```
[hidden_states] + [speaker_embedding] → Decoder loop (KV cache) → [mel_frames]
```

#### 3. Vocoder (HiFi-GAN): Mel Spectrogram → PCM Waveform

The vocoder converts the mel spectrogram from the frequency domain into a **time-domain PCM waveform** — the actual audio signal. SpeechT5 uses a HiFi-GAN vocoder combined with a postnet refinement stage, both packaged in a single ONNX model (`decoder_postnet_and_vocoder.onnx`).

**Why a separate model?** The decoder generates a compact, abstract representation (80 values per frame). The vocoder must expand this into thousands of audio samples per frame at high fidelity. These are fundamentally different tasks requiring different architectures. It's like the decoder drawing an architectural blueprint and the vocoder building the actual house from those plans.

```
[mel_frames] → Vocoder ONNX (postnet + HiFi-GAN) → [PCM samples] → AudioData
```

### Speaker Embeddings: Voice Identity as a Vector

A speaker embedding is a **fixed-length vector** (typically 512 dimensions) that encodes a speaker's voice characteristics — pitch, timbre, speaking rate, accent. The SpeechT5 decoder conditions its output on this vector, so:

- **Same text + same embedding** = identical audio
- **Same text + different embedding** = same words, different voice

This enables **voice selection** (pick from pre-extracted embeddings) and **voice cloning** (extract an embedding from a reference audio sample using a model like speechbrain/ECAPA-TDNN). The default `speaker.npy` file bundled with the model provides one pre-extracted voice.

### Tokenization: SentencePiece

SpeechT5 uses a **SentencePiece** character-level tokenizer (`spm_char.model`) to convert input text into token IDs. This project loads it via **`SentencePieceCharTokenizer`** from `MLNet.Audio.Tokenizers` (which extends the `Tokenizer` base class from `Microsoft.ML.Tokenizers`). The standard `SentencePieceTokenizer` doesn't support the Char model type, so the library provides this specialized implementation. The tokenizer handles the full Unicode range and doesn't require any external Python dependencies.

### Stop Condition: Probability Threshold vs. End Token

TTS and ASR handle stopping differently:

| | **ASR (Whisper)** | **TTS (SpeechT5)** |
|---|---|---|
| **Stops when** | Decoder emits `<end-of-text>` token | Stop probability exceeds threshold (0.5) |
| **Why** | Discrete tokens have a natural end symbol | Mel frames are continuous — no "end frame" token |
| **Safety limit** | Max token count | `MaxMelFrames` (default: 1000 ≈ 15 seconds) |

### Whisper and SpeechT5: Mirror Images

These two models are architectural inverses — the same encoder-decoder-with-KV-cache pattern running in opposite directions:

```
Whisper (ASR):   audio → mel extraction → encoder → decoder(KV) → text tokens   → text
SpeechT5 (TTS): text  → tokenize       → encoder → decoder(KV) → mel frames    → vocoder → audio
```

Both use autoregressive decoding with KV cache. Both have an encoder that transforms input into hidden states and a decoder that generates output one step at a time. The key differences:

- **Input/output modality is flipped**: Whisper consumes audio and produces text; SpeechT5 consumes text and produces audio.
- **SpeechT5 needs an extra stage** (the vocoder) because its decoder outputs mel spectrograms, not a directly usable format. Whisper's decoder outputs token IDs that map directly to text.
- **SpeechT5 uses speaker embeddings** — Whisper has no equivalent because text transcription doesn't need to encode a "voice identity."

## What This Sample Demonstrates

The sample shows **six different approaches** to TTS, each with a distinct purpose:

### 1. Direct Synthesis — `transformer.Synthesize(text)`

The simplest approach: create a transformer, call `Synthesize()`, get `AudioData` back.

```csharp
using var transformer = new OnnxSpeechT5TtsTransformer(mlContext, options);
var audio = transformer.Synthesize("Hello world!");
AudioIO.SaveWav("output.wav", audio);
```

**Why:** Minimal boilerplate. Best for quick experiments and scripts.

### 2. ITextToSpeechClient (official MEAI) — `client.GetAudioAsync(text)`

The official `ITextToSpeechClient` from `Microsoft.Extensions.AI.Abstractions` 10.4.1. `OnnxTextToSpeechClient` implements this interface using SpeechT5 (or KittenTTS) as the backend.

```csharp
using var client = new OnnxTextToSpeechClient(options);
var response = await client.GetAudioAsync("Say something");
var audioContent = response.Contents.OfType<DataContent>().First();
File.WriteAllBytes("output.wav", audioContent.Data.ToArray());
```

**Why:** Standard MEAI interface — same pattern used for `ISpeechToTextClient` and `IEmbeddingGenerator<,>`. Supports middleware, DI, and provider swapping.

### 3. ML.NET Pipeline — `Fit` / `Transform`

Standard ML.NET estimator/transformer pattern, composable with other transforms.

```csharp
var estimator = mlContext.Transforms.SpeechT5Tts(options);
var model = estimator.Fit(data);
var predictions = model.Transform(data);
```

**Why:** Composability. You can chain TTS with other ML.NET transforms in a single pipeline — including ASR for voice round-trips.

### 4. Custom Speaker Embedding — `.npy` file loading

Load a different speaker embedding to change the output voice.

```csharp
var speakerEmbedding = LoadSpeakerEmbedding("reference.npy");
var audio = transformer.Synthesize("Clone this voice", speakerEmbedding);
```

**Why:** Voice cloning and voice selection. Extract x-vectors from reference audio samples (using speechbrain/ECAPA-TDNN or similar), save as `.npy`, and use them to synthesize speech in any voice — without retraining the model.

### 5. Voice Round-Trip — STT → TTS Composition

Compose Whisper ASR and SpeechT5 TTS into a single ML.NET pipeline: transcribe audio to text, then synthesize it back as speech (potentially in a different voice).

```csharp
var pipeline = mlContext.Transforms
    .OnnxWhisper(whisperOptions)         // Audio → Text
    .Append(.SpeechT5Tts(ttsOptions));   // Text → Audio (different voice!)
```

**Why:** Demonstrates bidirectional audio ML — the full audio-to-audio loop. Practical use cases include voice conversion, accent normalization, and accessibility tools.

### 6. Batch Synthesis — Multiple Texts

Process multiple texts in one call.

```csharp
var texts = new[] { "Good morning.", "How are you today?" };
var batch = transformer.SynthesizeBatch(texts);
```

**Why:** Throughput. Avoids per-call overhead when synthesizing multiple utterances.

## Prerequisites

### Required Software

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

### Model Files

Download the SpeechT5 ONNX model from HuggingFace:

```bash
cd samples/TextToSpeech
git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5
```

The model directory should contain:

| File | Size | Purpose |
|------|------|---------|
| `encoder_model.onnx` | ~343 MB | Text encoder — token IDs → hidden states |
| `decoder_model_merged.onnx` | ~244 MB | Autoregressive decoder — hidden states → mel frames |
| `decoder_postnet_and_vocoder.onnx` | ~55 MB | Postnet refinement + HiFi-GAN vocoder — mel → PCM |
| `spm_char.model` | Small | SentencePiece tokenizer model |
| `speaker.npy` | Small | Default speaker embedding (512-dim x-vector) |

## Running It

### With model files (full synthesis)

```bash
cd samples/TextToSpeech

# Default text
dotnet run -- "models/speecht5"

# Custom text
dotnet run -- "models/speecht5" "Hello, I am a speech synthesis model."
```

Output: `output.wav` — a 16kHz mono PCM WAV file you can play with any audio player.

### Without model files (pattern demonstration)

```bash
cd samples/TextToSpeech
dotnet run
```

When models aren't found, the sample prints download instructions and demonstrates all five API patterns with code examples — no crash, no exception.

## Code Walkthrough

### OnnxSpeechT5Options — Configuration

The options class configures all three ONNX models, the tokenizer, and generation parameters:

```csharp
var options = new OnnxSpeechT5Options
{
    // Three ONNX models — one for each pipeline stage
    EncoderModelPath = "models/speecht5/encoder_model.onnx",
    DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
    VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx",

    // Optional — auto-detected from model directory if null
    TokenizerModelPath = null,       // defaults to spm_char.model
    SpeakerEmbeddingPath = null,     // defaults to speaker.npy

    // Generation parameters
    MaxMelFrames = 500,   // safety limit (~7.5 seconds at 16kHz)
    StopThreshold = 0.5f, // stop probability threshold
    NumMelBins = 80,      // mel spectrogram frequency bins
    SampleRate = 16000,   // output audio sample rate
};
```

Key design note: `NumDecoderLayers` and `NumAttentionHeads` are **auto-detected** from the ONNX model at load time, so you don't need to specify them manually.

### SentencePiece Tokenization

The transformer loads a SentencePiece model (`spm_char.model`) via **`SentencePieceCharTokenizer`** from `MLNet.Audio.Tokenizers` to convert input text into token IDs (the standard `SentencePieceTokenizer` from `Microsoft.ML.Tokenizers` does not support the Char model type):

```
"Hello!" → SentencePiece → [H, e, l, l, o, !] → [token_id_1, token_id_2, ...]
```

SpeechT5 uses a character-level tokenizer, so each character (including punctuation and spaces) gets its own token. This is different from Whisper, which uses a BPE tokenizer with subword units.

### The Decoder Loop — Autoregressive Mel Generation

The decoder generates mel spectrogram frames one at a time in a loop:

1. **Initialize**: Start with a zero mel frame as input, encoder hidden states, and empty KV cache
2. **Step**: Feed current mel frame + speaker embedding + KV cache → decoder produces next mel frame + updated KV cache + stop probability
3. **Accumulate**: Append the new mel frame to the output sequence
4. **Check stop**: If stop probability > `StopThreshold` (0.5) or frame count ≥ `MaxMelFrames`, stop
5. **Repeat** from step 2 with the new mel frame as input

This is the same autoregressive pattern Whisper uses for text token generation, but producing 80-dimensional mel vectors instead of discrete token IDs.

### Speaker Embedding — The `.npy` File

The `speaker.npy` file is a NumPy array containing a 512-dimensional float32 vector — the default speaker's voice "fingerprint." The transformer loads it using an internal `.npy` parser (`LoadNpyFloat32`) that reads the NumPy binary format directly, with no Python dependency.

To use a different voice:
1. Extract an x-vector from reference audio using a speaker verification model (e.g., speechbrain/ECAPA-TDNN)
2. Save it as a `.npy` float32 array
3. Pass it to `Synthesize(text, speakerEmbedding)` or set `TextToSpeechOptions.SpeakerEmbedding`

### ITextToSpeechClient — Official MEAI Interface

The `ITextToSpeechClient` interface from `Microsoft.Extensions.AI.Abstractions` 10.4.1:

```csharp
public interface ITextToSpeechClient : IDisposable
{
    Task<TextToSpeechResponse> GetAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    object? GetService(Type serviceType, object? serviceKey = null);
}
```

`OnnxTextToSpeechClient` implements this interface using SpeechT5 as the backend (or KittenTTS via the `OnnxKittenTtsOptions` constructor overload). Audio is returned as `DataContent` with WAV bytes in `TextToSpeechResponse.Contents`. Provider metadata is available via `GetService<TextToSpeechClientMetadata>()`.

`TextToSpeechOptions` supports voice selection (`VoiceId`), speed, language, and custom speaker embeddings (via `AdditionalProperties["speakerEmbedding"]`).

### Voice Round-Trip Pipeline

The ML.NET pipeline API enables composing ASR and TTS into a single transform chain:

```csharp
var pipeline = mlContext.Transforms
    .OnnxWhisper(whisperOptions)         // Audio → Text (Whisper ASR)
    .Append(.SpeechT5Tts(ttsOptions));   // Text → Audio (SpeechT5 TTS)
```

This creates an audio-to-audio pipeline: input audio is transcribed by Whisper, then the transcription is synthesized into new audio by SpeechT5 — potentially in a different voice via speaker embedding. Use cases include voice conversion, re-voicing content for accessibility, and audio translation (when combined with a text translation step).

## Key Takeaways

1. **TTS needs THREE models** (encoder + decoder + vocoder) — more complex than ASR, which only needs encoder + decoder. The vocoder is necessary because mel spectrograms are a frequency-domain representation that must be converted to time-domain audio.

2. **Speaker embeddings enable voice selection without retraining.** A single trained model can produce speech in any voice by swapping the 512-dimensional embedding vector. This is fundamentally different from fine-tuning.

3. **`ITextToSpeechClient` is the official MEAI interface.** Available in `Microsoft.Extensions.AI.Abstractions` 10.4.1 (marked `[Experimental]`). `OnnxTextToSpeechClient` accepts both `OnnxSpeechT5Options` and `OnnxKittenTtsOptions` — one client for all local TTS backends via the internal `IOnnxTtsSynthesizer` interface.

4. **Whisper and SpeechT5 are mirrors.** Same KV cache autoregressive pattern, opposite direction. Understanding one helps you understand the other. The key insight: both models generate output *one step at a time*, feeding each output back as input to the next step.

## Troubleshooting

### "Model not found" errors
SpeechT5 requires **three** separate ONNX files. Verify all are present:
```bash
ls models/speecht5/
# Should contain:
#   encoder_model.onnx           (~343 MB)
#   decoder_model_merged.onnx    (~244 MB)
#   decoder_postnet_and_vocoder.onnx (~55 MB)
#   spm_char.model               (SentencePiece tokenizer)
#   speaker.npy                  (default speaker embedding)
```
Download with:
```bash
git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5
```

### Output sounds garbled or robotic
- Verify all three ONNX files downloaded completely (check file sizes above)
- Ensure `spm_char.model` is present — if the tokenizer file is missing, SpeechT5 will throw a file-not-found exception when loading
- Very long text may accumulate decoder errors. Try shorter sentences first
- The stop threshold (0.5) may cause early cutoff for some voices — try lowering to 0.3

### Output is silent (0-length audio)
- The autoregressive decoder may not have converged. This can happen with:
  - Empty or whitespace-only input text
  - Characters not in the SentencePiece vocabulary
  - Corrupted model files (re-download)

### Comparing with KittenTTS
If you're choosing between SpeechT5 and KittenTTS:

| Criterion | SpeechT5 | KittenTTS |
|-----------|----------|-----------|
| Quality | Good, natural prosody | Good, natural prosody |
| Speed | Slower (autoregressive decoder loop) | Faster (single forward pass) |
| Model size | ~642 MB (3 files) | 41-80 MB (1 file) |
| Voices | Custom via speaker embedding | 8 built-in (Bella, Jasper, etc.) |
| Voice cloning | ✅ Any x-vector embedding | ❌ Fixed voice set |
| External deps | None | Requires espeak-ng |
| Output quality | 16 kHz | 24 kHz |

**Recommendation**: Use KittenTTS for quick prototyping and lightweight deployment. Use SpeechT5 when you need voice cloning or don't want external dependencies.

## Going Further

| Sample / Doc | What It Shows | Relationship to TTS |
|---|---|---|
| [`samples/WhisperRawOnnx`](../WhisperRawOnnx/) | Raw ONNX Whisper ASR with manual KV cache | **Reverse direction**: audio → text using the same KV cache pattern |
| [`samples/KittenTTS`](../KittenTTS/) | Lightweight TTS with espeak-ng phonemization | **Alternative TTS backend**: single-model approach vs SpeechT5's 3-model pipeline |
| [`samples/SpeechToText`](../SpeechToText/) | Provider-agnostic ASR via `ISpeechToTextClient` | **Round-trip partner**: pairs with TTS for voice conversion pipelines |
| [`docs/architecture.md`](../../docs/architecture.md) | Full system architecture, TTS pipeline section | Deep dive into encoder-decoder-vocoder data flow and KV cache management |
| [`docs/meai-integration.md`](../../docs/meai-integration.md) | MEAI interface mapping and DI patterns | Design rationale for `ITextToSpeechClient` and integration with ASP.NET |
