# Audio Processing Fundamentals for .NET Developers

A guide for C#/.NET developers who know ML.NET and text-based ML but are new to audio processing. This document bridges the gap between text ML concepts you already understand and audio ML concepts used throughout this project.

---

## Why Audio Needs Special Processing

If you've built ML.NET pipelines for text — tokenization, embeddings, transformers — you already know the pattern: **raw data can't go straight into a model**. Audio is the same, just with a different kind of raw data.

- **Computers store audio as PCM (Pulse Code Modulation)** — a long sequence of amplitude samples measured over time. A WAV file is essentially a giant `float[]`.
- **Unlike text (discrete tokens), audio is a continuous signal.** There are no natural word boundaries or discrete units — just a waveform that varies smoothly over time.
- **ML models can't process raw PCM efficiently.** A 30-second audio clip at 16kHz is 480,000 float values. That's too large and too unstructured for a model to learn from directly.
- **Audio needs feature extraction**, just like text needs tokenization. The analogy is direct:

| Text ML | Audio ML |
|---------|----------|
| Raw text string | Raw PCM samples (`float[]`) |
| Tokenizer → token IDs | Feature extractor → mel spectrogram |
| Feed tokens to model | Feed spectrogram to model |

In this project, `AudioFeatureExtractor` is audio's equivalent of `Microsoft.ML.Tokenizers.Tokenizer`.

---

## PCM Audio Basics

PCM (Pulse Code Modulation) is how digital audio works at the lowest level. Think of it as "sampling the air pressure at regular intervals."

### Samples

Each sample is a single amplitude measurement — one number representing how far the speaker cone is displaced at that instant. In this project, samples are `float` values normalized to the range **-1.0 to 1.0**:

```
 1.0 ─┐          ╱╲
      │        ╱    ╲
 0.0 ─┤──────╱────────╲──────╱──  ← silence
      │                ╲  ╱
-1.0 ─┘                 ╲╱
       sample 0  1  2  3  4  5 ...
```

### Sample Rate

How many samples are captured per second, measured in Hertz (Hz).

- **16,000 Hz (16kHz)** — the standard for speech ML models (Whisper, Silero VAD, SpeechT5). Captures frequencies up to 8kHz, which covers the full range of human speech.
- **24,000 Hz (24kHz)** — used by KittenTTS for higher-fidelity speech synthesis. Captures frequencies up to 12kHz, providing clearer consonants and more natural-sounding output than 16kHz.
- **44,100 Hz (44.1kHz)** — CD quality. Common for music files.
- **48,000 Hz (48kHz)** — common for video/broadcast audio.

**For ML, 16kHz is almost always what you want.** Higher sample rates waste compute without improving model accuracy for speech tasks.

### Channels

- **Mono (1 channel):** One stream of samples. What ML models expect.
- **Stereo (2 channels):** Two interleaved streams (left/right). Must be mixed down to mono before ML processing.

### Bit Depth

The precision of each sample value:

- **16-bit integer:** Standard WAV file format. Values from -32,768 to +32,767, normalized to -1.0 to 1.0 on load.
- **32-bit float:** Higher precision, used internally by this project. Values directly in the -1.0 to 1.0 range.

### Duration Formula

```
Duration (seconds) = Number of Samples / Sample Rate
```

For example:
- 160,000 samples ÷ 16,000 Hz = **10 seconds**
- 480,000 samples ÷ 16,000 Hz = **30 seconds** (one Whisper chunk)
- 960,000 samples ÷ 16,000 Hz = **60 seconds**

### How This Maps to `AudioData`

`AudioData` is the core audio type in this project — it bundles samples, sample rate, and channel count together:

```csharp
// Load from a WAV file — automatically converts to mono, normalizes to float
AudioData audio = AudioIO.LoadWav("speech.wav");

// Or construct directly from raw samples
float[] samples = new float[16000 * 10]; // 10 seconds of silence at 16kHz
var audio = new AudioData(samples, sampleRate: 16000, channels: 1);

// Properties
int rate = audio.SampleRate;      // 16000
int channels = audio.Channels;    // 1
TimeSpan length = audio.Duration; // 00:00:10
float[] pcm = audio.Samples;     // the raw float[] array
```

---

## From Time Domain to Frequency Domain

This is the single most important concept in audio ML. If you understand this section, everything else falls into place.

### Time Domain: What a WAV File Stores

A WAV file stores amplitude over time — for each moment, "how loud is the sound?" This is the **time domain** representation.

```
Amplitude
    │   ╱╲      ╱╲
    │  ╱  ╲    ╱  ╲    ╱╲
 0 ─┤╱──────╲╱──────╲╱────╲──
    │                        ╲
    └────────────────────────── Time
```

The problem: two very different sounds (a whistle and a guitar chord) can have similar amplitude patterns, but very different **frequencies**. A model looking only at amplitude over time misses the information that matters.

### Frequency Domain: What the Ear Hears

Your ear doesn't track amplitude over time — it separates sound into frequencies (like a prism splits light into colors). A "middle C" on a piano is 262 Hz; an "A above middle C" is 440 Hz. Speech is a complex mix of many frequencies changing over time.

The **frequency domain** answers a different question: "what frequencies are present, and how strong is each one?"

### FFT (Fast Fourier Transform)

The FFT is the math that converts time-domain samples into frequency-domain information. Given N samples, it produces N/2 frequency bins, each telling you how much energy is at that frequency.

You don't need to understand the math — just know that FFT is the bridge from "amplitude over time" to "frequencies present."

### STFT (Short-Time Fourier Transform)

Audio frequencies change over time (that's what makes speech intelligible). A single FFT over the entire audio would blend everything together. Instead, we apply FFT to short **overlapping windows** of audio — this is the STFT.

Three parameters control the STFT:

**Window size (FFT size):** How many samples per analysis window.
- 400 samples at 16kHz = 25ms window
- Larger windows → better frequency resolution, worse time resolution
- 400 is the standard for speech models

**Hop length:** How far to advance between consecutive windows.
- 160 samples at 16kHz = 10ms hop
- This means windows overlap (400 - 160 = 240 samples of overlap)
- Smaller hops → more frames → finer time resolution

**Result:** A **spectrogram** — a 2D grid where one axis is time (frames) and the other is frequency (bins), with values representing intensity.

```
Frequency ▲
 8000 Hz  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░
 4000 Hz  │ ░░▓▓▓▓░░░░░░▓▓▓▓░░░░░░▓▓▓▓  ← vowel formants
 2000 Hz  │ ▓▓████▓▓░░▓▓████▓▓░░▓▓████▓▓
 1000 Hz  │ ████████▓▓████████▓▓████████
  500 Hz  │ ██████████████████████████████ ← fundamental frequency
    0 Hz  │ ██████████████████████████████
          └──────────────────────────────► Time (frames)
            Frame 0     Frame 50    Frame 100
```

With a 16kHz sample rate, 400-sample window, and 160-sample hop:
- Each **frame** covers 25ms of audio
- Frames are spaced 10ms apart
- You get **100 frames per second** of audio

---

## Mel Spectrograms: What Models Actually See

A raw spectrogram has a problem: it treats all frequencies equally. But human hearing doesn't work that way.

### The Mel Scale

Human hearing is **logarithmic** with respect to pitch:
- The difference between 100 Hz and 200 Hz sounds like a huge jump (one octave).
- The difference between 8000 Hz and 8100 Hz is barely noticeable.

The **mel scale** is a perceptual frequency scale that matches how humans actually hear. Low frequencies get more resolution; high frequencies get compressed.

### Mel Filter Bank

A mel filter bank is a set of overlapping triangular filters spaced according to the mel scale. It takes the FFT frequency bins and groups them into a smaller number of **mel-spaced bands**:

```
FFT bins:    |0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|...|199|
              \_____/ \______/ \_________/ \______________/
Mel bin:        0        1         2              ...  79

(Low frequencies get narrow, closely-spaced filters;
 high frequencies get wide, spread-out filters)
```

### Number of Mel Bins

How many frequency bands in the output. This varies by model:

- **80 mel bins** — Whisper (original), SpeechT5
- **128 mel bins** — Whisper (large-v3), AST (Audio Spectrogram Transformer)

### Log Scaling

After computing mel energies, we take the **logarithm** — this compresses the dynamic range (loud vs. quiet) to match human loudness perception. The result is a **log-mel spectrogram**.

### Result Shape

The final output is a 2D array: **[frames × mel_bins]**

For example, 30 seconds of audio at 16kHz with 160-sample hop and 80 mel bins:
- 30 seconds × 100 frames/second = **3000 frames**
- Each frame has **80** mel energy values
- Result: `float[3000, 80]`

### How This Maps to `MelSpectrogramExtractor`

```csharp
// Create an extractor matching your model's requirements
var extractor = new MelSpectrogramExtractor(sampleRate: 16000)
{
    NumMelBins = 80,      // 80 for Whisper, 128 for AST
    FftSize = 400,        // 25ms window at 16kHz
    HopLength = 160,      // 10ms hop → 100 frames/sec
    LogScale = true,      // log-mel (almost always true)
    LowFrequency = 0f,    // start at 0 Hz
};

AudioData audio = AudioIO.LoadWav("speech.wav");
float[,] features = extractor.Extract(audio); // [frames, 80]

Console.WriteLine($"Frames: {features.GetLength(0)}");   // e.g., 3000
Console.WriteLine($"Mel bins: {features.GetLength(1)}");  // 80
```

Note that `Extract()` automatically handles resampling and mono conversion — you don't need to preprocess the audio yourself.

---

## Key Vocabulary Quick Reference

| Term | What It Is | Analogy to Text |
|------|-----------|-----------------|
| PCM Sample | A single amplitude value (`float` in -1.0 to 1.0) | A single character |
| Sample Rate | Samples per second (16kHz for speech ML) | N/A (text is discrete) |
| Feature Extraction | Audio → mel spectrogram | Tokenization (text → token IDs) |
| Mel Spectrogram | Frequency features over time (`float[,]`) | Token embeddings |
| FFT Size | Analysis window width (400 samples = 25ms) | N-gram window |
| Hop Length | Step between windows (160 samples = 10ms) | Stride |
| Mel Bins | Number of frequency bands (80 or 128) | Vocabulary size (sort of) |
| Frame | One time-step of features (one row of the spectrogram) | One token position |
| Whisper 30s chunk | 30s × 100 frames/sec = 3000 frames | Max sequence length (512/2048) |
| `AudioData` | Samples + sample rate + channels | A `string` (raw input) |
| `AudioFeatureExtractor` | Base class for audio → features | `Tokenizer` base class |
| `MelSpectrogramExtractor` | Log-mel spectrogram extraction | A specific tokenizer (e.g., BPE) |

---

## How Different Models Process Audio

Each model in this project processes audio differently. Here's the pipeline for each:

### Whisper (Speech Recognition / ASR)

```
WAV → AudioData → WhisperFeatureExtractor → [3000 × 80] mel spectrogram
    → encoder → decoder → text tokens → transcribed text
```

- Fixed **30-second chunks**, padded or truncated to exactly 3000 frames
- Uses **80 mel bins** (or 128 for large-v3)
- Encoder-decoder architecture: encoder processes the spectrogram, decoder generates text tokens autoregressively
- Long audio is split into 30s chunks via `ExtractChunked()`

```csharp
var extractor = new WhisperFeatureExtractor(numMelBins: 80);
float[,] features = extractor.Extract(audio); // always [3000, 80]
```

### AST (Audio Spectrogram Transformer / Classification)

```
WAV → AudioData → MelSpectrogramExtractor → [frames × 128] mel spectrogram
    → patch embeddings → transformer → class logits → "dog barking"
```

- **Variable length** input (no fixed chunk size)
- Uses **128 mel bins**
- Treats the spectrogram like an image — splits it into patches, then processes with a Vision Transformer (ViT)
- Outputs class probabilities over a label set

### CLAP (Audio Embeddings)

```
WAV → AudioData → MelSpectrogramExtractor → [frames × mel_bins] mel spectrogram
    → audio encoder → pooling → L2 normalize → 512-dim float[] vector
```

- **Variable length** input
- Produces a fixed-size **embedding vector** (typically 512 dimensions)
- Useful for audio similarity, search, and retrieval
- The text-ML equivalent: sentence embeddings (like `text-embedding-ada-002`)

### Silero VAD (Voice Activity Detection)

```
WAV → AudioData → raw PCM in 512-sample windows → LSTM → speech probability
```

- **No mel spectrogram!** Silero VAD works directly on raw PCM samples.
- Processes audio in tiny **512-sample windows** (~32ms at 16kHz)
- Stateful: maintains LSTM hidden state (`h`, `c`) across frames
- Outputs a **speech probability** (0.0 to 1.0) per window
- Used to find when someone is talking vs. silence

### SpeechT5 (Text-to-Speech / TTS)

```
text → SentencePiece tokenizer → token IDs → encoder → decoder loop
    → [frames × 80] mel spectrogram → vocoder (HiFi-GAN) → PCM → AudioData
```

- **Reverse of Whisper** — text in, audio out
- Uses `Microsoft.ML.Tokenizers.SentencePieceTokenizer` for text input
- Decoder generates mel frames autoregressively (like GPT generating tokens)
- Vocoder converts mel spectrogram back into a PCM waveform
- Outputs `AudioData` with the synthesized speech

### KittenTTS (Lightweight Text-to-Speech)

```
text → text chunking (max 400 chars, sentence boundaries)
    → espeak-ng (external tool: text → IPA phonemes)
    → TextCleaner (IPA symbols → integer token IDs)
    → single ONNX model (token IDs + voice embedding + speed → waveform)
    → trim + concatenate → AudioData at 24 kHz
```

- **Fundamentally different from SpeechT5** — no encoder/decoder/vocoder split, no autoregressive loop
- Uses **espeak-ng** (an external phonemization tool) instead of SentencePiece tokenization
- **Single forward pass** through one ONNX model — faster and simpler than SpeechT5's 3-model pipeline
- Voice selection via **named voices** stored in a `voices.npz` archive (8 built-in: Bella, Jasper, Luna, Bruno, etc.)
- **24 kHz output** (higher than SpeechT5's 16 kHz)
- Models range from 41 MB (Micro) to 80 MB (Mini) — much smaller than SpeechT5's 643 MB total

**How it compares to SpeechT5:**

| | SpeechT5 | KittenTTS |
|---|---|---|
| Models | 3 ONNX files (643 MB) | 1 ONNX file (41–80 MB) |
| Decoding | Autoregressive (token by token) | Single forward pass |
| Text processing | SentencePiece tokenizer | espeak-ng IPA phonemization |
| Voice mechanism | Speaker embedding (`.npy` x-vector) | Named voices from `.npz` archive |
| Sample rate | 16 kHz | 24 kHz |

```csharp
var options = new OnnxKittenTtsOptions { ModelPath = "models/kittentts/model.onnx" };
var estimator = mlContext.Transforms.KittenTts(options);
```

---

## Audio Processing in .NET: The Stack

Here's how the pieces fit together in this project, from lowest to highest level:

### Core Types (`MLNet.Audio.Core`)

| Class | Role | Text Equivalent |
|-------|------|-----------------|
| `AudioData` | Core type: `float[] Samples` + `SampleRate` + `Channels` | `string` |
| `AudioIO` | Load/save WAV, resample, mono conversion | File I/O + text normalization |
| `AudioFeatureExtractor` | Abstract base for feature extraction | `Tokenizer` (abstract base) |
| `MelSpectrogramExtractor` | NWaves-powered log-mel spectrogram | A specific tokenizer implementation |
| `WhisperFeatureExtractor` | Whisper-specific (30s padding, configurable mel bins) | `WhisperTokenizer` (model-specific) |

### Inference Layer (`MLNet.AudioInference.Onnx`)

| Class | Role |
|-------|------|
| `OnnxWhisperTransformer` | Whisper ASR (speech → text) |
| `OnnxAudioClassificationTransformer` | Audio classification (speech → label) |
| `OnnxAudioEmbeddingTransformer` | Audio embeddings (speech → vector) |
| `OnnxVadTransformer` | Voice activity detection (speech → segments) |
| `OnnxSpeechT5TtsTransformer` | Text-to-speech (text → audio) |
| `OnnxKittenTtsTransformer` | Lightweight text-to-speech (text → audio, single-pass) |

### Key Dependencies

- **NWaves** — FFT and mel filter bank computation (the heavy math behind `MelSpectrogramExtractor`)
- **ONNX Runtime** — runs the neural network models
- **System.Numerics.Tensors / TensorPrimitives** — SIMD-accelerated math used throughout for operations like log, softmax, normalization, and vector math
- **Microsoft.ML.Tokenizers** — SentencePiece tokenizer for SpeechT5 TTS

---

## Common Gotchas for Text-ML Developers

### 1. Audio Files Are BIG

Text is tiny. Audio is not.

| Duration | Samples (16kHz mono) | Memory (float32) |
|----------|---------------------|-------------------|
| 1 second | 16,000 | ~64 KB |
| 10 seconds | 160,000 | ~640 KB |
| 1 minute | 960,000 | ~3.8 MB |
| 10 minutes | 9,600,000 | ~38 MB |
| 1 hour | 57,600,000 | ~230 MB |

Plan for memory accordingly, especially when processing batches.

### 2. Sample Rate Matters — A Lot

Feeding 44.1kHz audio to a model trained on 16kHz will produce **garbage results**. The model sees frequencies shifted and stretched — it's like feeding Spanish text to an English-only model.

```csharp
// ❌ WRONG: audio might be 44.1kHz, model expects 16kHz
var audio = AudioIO.LoadWav("recording.wav");
model.Transcribe(audio); // garbage output

// ✅ RIGHT: resample first
var audio = AudioIO.LoadWav("recording.wav");
var audio16k = AudioIO.Resample(audio, 16000);
model.Transcribe(audio16k); // correct output
```

> **Good news:** `AudioFeatureExtractor.Extract()` resamples automatically. But if you're passing raw samples to a model directly (like Silero VAD), you must resample yourself.

### 3. Always Convert to Mono

ML models don't handle stereo. Two channels of interleaved audio will be misinterpreted as one channel at double the length.

```csharp
// AudioIO.LoadWav() already converts to mono, but if you have raw stereo data:
var mono = AudioIO.ToMono(stereoAudio);
```

### 4. Feature Extraction Is NOT Optional

Raw PCM → model is like feeding raw bytes → BERT. It technically runs but produces meaningless output. The model expects a mel spectrogram (or similar features), not a raw waveform.

The one exception: **Silero VAD**, which is specifically designed to process raw PCM.

### 5. Whisper Is Special

Whisper expects **exactly 3000 frames** of input — that's exactly 30 seconds of audio at 10ms hop length. Shorter audio is zero-padded; longer audio must be chunked.

```csharp
var extractor = new WhisperFeatureExtractor(numMelBins: 80);

// Short audio (5 seconds) → padded to 3000 frames with zeros
float[,] features = extractor.Extract(shortAudio);
// features.GetLength(0) == 3000 (padded)

// Long audio (2 minutes) → split into 30s chunks
List<float[,]> chunks = extractor.ExtractChunked(longAudio);
// chunks.Count == 4 (four 30-second chunks)
```

### 6. The Processing Pipeline Order

Always follow this order:

```
Load WAV → Resample to 16kHz → Convert to mono → Extract features → Feed to model
             AudioIO.Resample()  AudioIO.ToMono()  extractor.Extract()
```

`AudioFeatureExtractor.Extract()` handles steps 2-4 internally, so in practice:

```csharp
var audio = AudioIO.LoadWav("any-format.wav");  // any sample rate, any channels
var features = extractor.Extract(audio);         // resamples + mono + mel spectrogram
// Ready for model input
```
