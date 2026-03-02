# MLNet.Audio.Core — Audio Primitives for .NET

**The foundational audio types and feature extraction library for .NET — no ML.NET dependency required.**

MLNet.Audio.Core is Layer 0 of the audio transforms architecture. It provides the primitive types (`AudioData`), I/O operations (`AudioIO`), and feature extractors (`MelSpectrogramExtractor`, `WhisperFeatureExtractor`) that every other package in the stack builds on. Because it has zero ML.NET coupling, you can use it anywhere — console apps, web APIs, Avalonia UIs, or inside ML.NET pipelines.

---

## Why This Package Exists

### The problem: .NET has no audio primitive type

Text ML in .NET has `string`. Image ML has `MLImage`. Audio ML has... `float[]`?

A raw `float[]` tells you nothing. Is it 16 kHz or 44.1 kHz? Mono or stereo? How long is it? Every function that touches audio needs these answers, and without a shared type, every function re-asks them via extra parameters, or worse, silently assumes.

**`AudioData` solves this.** It bundles PCM samples with their sample rate and channel count into a single type that flows through pipelines without ambiguity. `Duration` is computed from `Samples.Length / SampleRate` — not stored — because the samples *are* the source of truth. Storing duration separately would create an invariant that could drift.

### The problem: models don't understand waveforms

If you've worked with text ML, you know that models don't consume raw strings — they consume *token IDs*. A tokenizer converts human-readable text into the numeric vocabulary a model was trained on.

Audio works the same way, but the "tokenization" step is feature extraction. Raw PCM waveforms are enormous (16,000 samples per second) and carry information in a format that's hostile to neural networks. Models need *spectral features* — compact representations of what frequencies are present at each moment in time.

**Feature extraction is audio's tokenization step.** This package provides it as a first-class concept via `AudioFeatureExtractor`, making the pipeline explicit:

| Modality | Raw input | Transform | Model input |
|----------|-----------|-----------|-------------|
| Text | `string` | `Tokenizer.Encode()` | `int[]` token IDs |
| Audio | `AudioData` | `AudioFeatureExtractor.Extract()` | `float[,]` mel spectrogram |

### Why separate from ML.NET?

ML.NET's `ITransformer` / `IDataView` pipeline is powerful but opinionated — it assumes you're inside an ML.NET context. Audio primitives need to work *everywhere*:

- **In an ML.NET pipeline** — as the building block for custom transforms
- **In a standalone ONNX inference app** — load WAV, extract features, run model, done
- **In a preprocessing tool** — batch-convert audio without touching ML.NET
- **In unit tests** — test feature extraction without bootstrapping `MLContext`

Zero ML.NET dependency means zero friction for any of these scenarios.

---

## Key Concepts

If you're a .NET developer comfortable with text ML (tokenizers, embeddings, transformers) but new to audio ML, this section maps what you already know to the audio domain.

### PCM Audio and Sample Rates

Audio is stored as **PCM (Pulse Code Modulation)** — a sequence of amplitude measurements taken at regular intervals. Each measurement is a **sample**, and the number of samples per second is the **sample rate**.

```
Sample rate: 16,000 Hz = 16,000 measurements per second
1 second of audio = 16,000 float values in [-1.0, 1.0]
30 seconds of audio = 480,000 float values
```

The sample rate matters because **models are trained at a specific rate**. Whisper expects 16 kHz. If you feed it 44.1 kHz audio (CD quality), every frequency will appear shifted — the model hears gibberish. This is why `AudioFeatureExtractor` automatically resamples: you hand it any audio, and it handles the conversion.

**Channels** are separate audio streams — stereo has 2 (left + right), mono has 1. ML models almost universally expect mono. Multi-channel audio is mixed down by averaging.

### Mel Spectrograms: What Models Actually See

A raw waveform tells you amplitude over time. But speech, music, and environmental sounds are distinguished by *which frequencies* are present at *which moments*. A **spectrogram** decomposes the waveform into a time-frequency representation:

```
Waveform (1D):     [amplitude₀, amplitude₁, amplitude₂, ...]
                              ↓ STFT (Short-Time Fourier Transform)
Spectrogram (2D):  rows = frequency bins, columns = time frames
```

A **mel spectrogram** goes one step further. Human hearing is *logarithmic* — we perceive the difference between 100 Hz and 200 Hz as the same "distance" as 1000 Hz and 2000 Hz. The **mel scale** warps the frequency axis to match human perception, compressing high frequencies where our ears are less sensitive. This is the same idea behind mel-frequency cepstral coefficients (MFCCs), which audio engineers have used for decades.

**Why log-mel and not raw spectrogram?** Two reasons:

1. **Perceptual alignment** — mel spacing matches how we hear, so models learn more efficiently
2. **Dynamic range compression** — log scaling tames the huge amplitude range, making features better-behaved for neural networks

The result is a 2D array `[frames × mel_bins]` — essentially a grayscale image where the x-axis is time, the y-axis is (mel-scaled) frequency, and brightness is energy. This is why many audio models use architectures borrowed from computer vision.

### The Analogy: Tokenization ↔ Feature Extraction

| Text pipeline | Audio pipeline |
|---|---|
| Raw text: `"Hello world"` | Raw audio: `AudioData` (480,000 samples) |
| Tokenizer splits into subwords | STFT splits into time frames |
| Token IDs: `[15496, 995]` | Mel spectrogram: `float[3000, 80]` |
| Embedding layer maps IDs → vectors | First model layer processes spectrogram |
| Sequence of vectors → transformer | Sequence of vectors → transformer |

The parallel is deep. Just as different text models use different tokenizers (BPE, SentencePiece, WordPiece), different audio models use different feature extraction settings (80 vs 128 mel bins, different FFT sizes, different window functions). And just as you'd never hardcode a tokenizer into a model runner, feature extraction is a separate, configurable step.

---

## Architecture & Design Decisions

### Abstract `AudioFeatureExtractor` base class

The base class owns the *boring but critical* preprocessing that every extractor needs:

```csharp
public float[,] Extract(AudioData audio)
{
    var resampled = audio.SampleRate != SampleRate
        ? AudioIO.Resample(audio, SampleRate)
        : audio;

    var mono = resampled.Channels > 1
        ? AudioIO.ToMono(resampled)
        : resampled;

    return ExtractFeatures(mono.Samples);
}

protected abstract float[,] ExtractFeatures(float[] samples);
```

Subclasses only implement `ExtractFeatures(float[] samples)` — they receive **guaranteed mono audio at the correct sample rate**. This eliminates an entire class of bugs (wrong sample rate, stereo fed to mono-expecting code) by construction, not convention.

This is the [Template Method pattern](https://en.wikipedia.org/wiki/Template_method_pattern): the base class defines the algorithm skeleton (resample → mono → extract), and subclasses fill in the variable step.

### Why NWaves for mel computation?

Computing a mel spectrogram from scratch requires: STFT (windowing, FFT), mel filter bank construction, and matrix multiplication. [NWaves](https://github.com/ar1st0crat/NWaves) is a mature, .NET-native DSP library that provides all of this through `FilterbankExtractor`. It handles the hairy signal processing (correct FFT normalization, proper mel-scale triangular filters, overlap-add) so this package doesn't have to reimplement textbook DSP.

### Why `System.Numerics.Tensors` and `TensorPrimitives`?

The mel spectrogram's log-scaling step applies `max(x, floor)` then `log(x)` to every element of every frame. `TensorPrimitives.Max` and `TensorPrimitives.Log` are SIMD-accelerated — they operate on multiple float values per CPU instruction via AVX2/SSE. On a 3000-frame, 80-bin spectrogram (240,000 values), this is measurably faster than a scalar loop:

```csharp
// SIMD: clamp to floor then log — processes ~8 floats per instruction
TensorPrimitives.Max(row, 1e-10f, row);
TensorPrimitives.Log(row, row);
```

### Why `WhisperTokenizer` is custom (not `Microsoft.ML.Tokenizers`)

Whisper's tokenizer is a GPT-2 BPE tokenizer *plus* ~1,700 special tokens with deep semantic meaning:

- **1,501 timestamp tokens**: `<|0.00|>` through `<|30.00|>` in 0.02s steps — these aren't mere markers, they encode *precise time positions* convertible to `TimeSpan`
- **99 language codes**: `<|en|>`, `<|zh|>`, `<|de|>`, etc. — each maps to a specific language
- **Task tokens**: `<|transcribe|>`, `<|translate|>` — control the model's behavior
- **Control tokens**: `<|nospeech|>`, `<|notimestamps|>`, `<|startoftranscript|>`

`Microsoft.ML.Tokenizers.BpeTokenizer` handles standard BPE encode/decode well, but has no concept of:

1. **Special token registries with semantic meaning** — timestamp token IDs need to convert to/from `TimeSpan`
2. **Domain-specific decode modes** — `Decode()` strips timestamps, `DecodeWithTimestamps()` preserves them as `[0.00s]` annotations, `DecodeToSegments()` returns structured `TranscriptionSegment` records
3. **Token ID ranges with programmatic meaning** — "is this token a timestamp?" requires range checks, not dictionary lookups

A custom tokenizer was the pragmatic choice. The design notes in the source code sketch what a future `Microsoft.ML.Tokenizers` extension point could look like.

### Why `AudioCodecTokenizer` is abstract with a generic design note

Neural audio codecs (EnCodec, DAC, SpeechTokenizer, Mimi) represent a fundamentally different approach to audio "tokenization." Instead of extracting spectral features for a downstream model, they *compress the waveform itself into discrete codes* — enabling audio to be treated as a token sequence by language models.

The key architectural insight is **Residual Vector Quantization (RVQ)**: the encoder produces not one stream of codes but `N` parallel streams (codebooks), where each successive codebook captures finer reconstruction detail. This is why `Encode()` returns `int[][]` (codebooks × frames) rather than `int[]`.

`AudioCodecTokenizer` is abstract because the actual encode/decode requires running neural network inference (ONNX models), which lives in higher layers. This base class defines the *contract* — codebook count, vocabulary size, frame layout, interleaved flattening for LM consumption — without depending on any inference runtime.

The source code includes a design note proposing that `Microsoft.ML.Tokenizers.Tokenizer` could become generic:

```
Tokenizer<TInput, TToken>
  ├── TextTokenizer : Tokenizer<string, EncodedToken>     // what exists today
  └── AudioCodecTokenizer : Tokenizer<AudioData, AudioCodeToken>  // what this prototypes
```

This would unify text and audio tokenization under a single abstraction.

---

## API Surface

### `AudioData` — The fundamental audio type

```csharp
public class AudioData
{
    public float[] Samples { get; }      // PCM samples, mono, normalized to [-1.0, 1.0]
    public int SampleRate { get; }       // Hz (e.g., 16000)
    public int Channels { get; }         // Always 1 after loading (stereo is mixed down)
    public TimeSpan Duration { get; }    // Computed: Samples.Length / SampleRate

    public static AudioData Empty(int sampleRate = 16000);
}
```

**Design rationale:** `Duration` is a computed property, not a stored field. The samples array *is* the ground truth for duration — storing it separately would create an invariant that could silently break if samples are modified. Constructor validates with `ArgumentOutOfRangeException.ThrowIfNegativeOrZero` for both `sampleRate` and `channels`.

### `AudioIO` — WAV I/O and basic transforms

```csharp
public static class AudioIO
{
    // Load WAV from file path or stream (always returns mono)
    static AudioData LoadWav(string path);
    static AudioData LoadWav(Stream stream);

    // Save as 16-bit PCM WAV to file path or stream
    static void SaveWav(string path, AudioData audio);
    static void SaveWav(Stream stream, AudioData audio);

    // Resample to target rate (linear interpolation)
    static AudioData Resample(AudioData audio, int targetSampleRate);

    // Mix multi-channel to mono by averaging
    static AudioData ToMono(AudioData audio);
}
```

**WAV format support:** Reads 8-bit PCM, 16-bit PCM, 24-bit PCM, and 32-bit IEEE float WAV files. Writes 16-bit PCM. Chunk-based parsing correctly skips unknown chunks (e.g., metadata, LIST) and handles the `fmt` → `data` chunk ordering requirement.

**Resampling tradeoff:** `Resample` uses linear interpolation — simple, fast, and adequate for the typical use case (44.1 kHz → 16 kHz downsampling for speech models). It won't match the quality of a polyphase sinc resampler for music production, but for ML feature extraction, the mel spectrogram's frequency binning absorbs minor interpolation artifacts. If the source and target rates match, it returns the original `AudioData` unchanged (no allocation).

**`ToMono`:** Averages across channels — the standard approach. Returns the original if already mono.

### `AudioFeatureExtractor` — Abstract base for all feature extractors

```csharp
public abstract class AudioFeatureExtractor
{
    public abstract int SampleRate { get; }                        // Expected input rate
    public float[,] Extract(AudioData audio);                      // Resample → mono → extract
    protected abstract float[,] ExtractFeatures(float[] samples);  // Subclass implements this
}
```

**Why abstract:** Different models need different features. Whisper needs 80-bin log-mel at 16 kHz. AST needs 128-bin. A future MFCC extractor might need cepstral coefficients. The base class guarantees correct preprocessing; subclasses define the spectral transform.

**Return type `float[,]`:** A 2D array of `[frames × features]`. This maps directly to ONNX model input tensors, which expect dense rectangular arrays.

### `MelSpectrogramExtractor` — Configurable mel spectrogram extraction

```csharp
public class MelSpectrogramExtractor : AudioFeatureExtractor
{
    public int NumMelBins { get; init; }          // Default: 80 (Whisper), or 128 (AST)
    public int FftSize { get; init; }             // Default: 400 (25ms window at 16kHz)
    public int HopLength { get; init; }           // Default: 160 (10ms hop at 16kHz)
    public float LowFrequency { get; init; }      // Default: 0 Hz
    public float? HighFrequency { get; init; }     // Default: SampleRate / 2 (Nyquist)
    public bool LogScale { get; init; }            // Default: true (log-mel)
    public WindowType Window { get; init; }        // Default: Hann
}
```

Uses NWaves `FilterbankExtractor` internally for FFT and mel filter bank computation. All parameters are exposed as `init`-only properties for clean configuration:

```csharp
var extractor = new MelSpectrogramExtractor(sampleRate: 16000)
{
    NumMelBins = 128,
    FftSize = 512,
    HopLength = 256
};
```

When `LogScale` is `true`, each frame is processed with SIMD-accelerated `TensorPrimitives.Max` (floor clamp at 1e-10) and `TensorPrimitives.Log`.

### `WhisperFeatureExtractor` — Whisper-specific mel extraction with padding and chunking

```csharp
public class WhisperFeatureExtractor : MelSpectrogramExtractor
{
    public int MaxFrames { get; init; }          // Default: 3000 (30s at 10ms hop)
    public bool PadToMaxFrames { get; init; }    // Default: true

    public new float[,] Extract(AudioData audio);                           // Padded/truncated
    public List<float[,]> ExtractChunked(AudioData audio, int overlapFrames = 0);  // Long audio
}
```

**Why it's special:** Whisper's architecture requires *exactly* 3000 frames of input (30 seconds at 10ms hop). Shorter audio is zero-padded; longer audio is truncated. This is baked into the model's positional encoding — you can't just feed it arbitrary-length spectrograms.

**`ExtractChunked`** handles audio longer than 30 seconds by splitting it into overlapping chunks, extracting features for each, and returning a list. This is the pattern used for transcribing long recordings: process each chunk independently, then stitch the results together (handled at a higher layer).

Constructor locks down the parameters Whisper requires: 16 kHz sample rate, 400-sample FFT, 160-sample hop, log-scaled.

### `WhisperTokenizer` — BPE + timestamps + language codes

```csharp
public class WhisperTokenizer : IDisposable
{
    // Special token IDs
    public int EndOfTextId { get; }
    public int StartOfTranscriptId { get; }
    public int TranscribeId { get; }
    public int TranslateId { get; }
    public int NoSpeechId { get; }
    public int NoTimestampsId { get; }
    public int TimestampBeginId { get; }
    public IReadOnlyDictionary<string, int> LanguageTokens { get; }
    public bool IsMultilingual { get; }

    // Three decode modes
    string Decode(IReadOnlyList<int> tokenIds);                              // Clean text only
    string DecodeWithTimestamps(IReadOnlyList<int> tokenIds);                // "[0.00s] Hello [1.52s]"
    IReadOnlyList<TranscriptionSegment> DecodeToSegments(IReadOnlyList<int> tokenIds);  // Structured

    // Prompt construction
    int[] GetStartOfTranscriptSequence(string? language = null, bool translate = false);

    // Timestamp utilities
    bool IsTimestamp(int tokenId);
    TimeSpan? TokenToTimestamp(int tokenId);
    int TimestampToToken(TimeSpan time);
}

public record TranscriptionSegment(string Text, TimeSpan Start, TimeSpan End);
```

**Three decode modes** serve different downstream needs:

| Mode | Output | Use case |
|------|--------|----------|
| `Decode()` | `"Hello world"` | Clean transcription text for display or search |
| `DecodeWithTimestamps()` | `"[0.00s] Hello world [1.52s]"` | Subtitle generation, debug inspection |
| `DecodeToSegments()` | `[TranscriptionSegment("Hello world", 0:00, 0:01.52)]` | Programmatic access to timed segments |

**Timestamp token math:** Token ID → time is `(tokenId - TimestampBeginId) * 0.02` seconds. Time → token ID is `Round(seconds / 0.02) + TimestampBeginId`. The 0.02s resolution gives 1,501 timestamps covering 0.00–30.00 seconds — matching Whisper's 30-second processing window.

**Multilingual layout:** The multilingual tokenizer has a base vocabulary of 50,258 tokens (vs 50,257 for English-only), followed by 99 language tokens, task tokens, control tokens, and 1,501 timestamp tokens. `GetStartOfTranscriptSequence()` builds the initial decoder prompt: `[<|startoftranscript|>, <|en|>, <|transcribe|>]`.

### `AudioCodecTokenizer` — Abstract base for neural audio codecs

```csharp
public abstract class AudioCodecTokenizer : IDisposable
{
    public abstract int NumCodebooks { get; }     // e.g., 8 for EnCodec at 6kbps
    public abstract int CodebookSize { get; }     // Typically 1024
    public abstract int SampleRate { get; }
    public abstract int HopLength { get; }        // Samples per codec frame

    public abstract int[][] Encode(AudioData audio);    // → [codebooks][frames]
    public abstract AudioData Decode(int[][] codes);    // → reconstructed audio
    public virtual int[] EncodeFlat(AudioData audio);   // Interleaved for LM input
    public int VocabularySize { get; }                  // NumCodebooks * CodebookSize
    public int GetFrameCount(TimeSpan duration);
}

public record AudioCodeToken(int[] CodebookValues, TimeSpan Timestamp, int FrameIndex);
```

**RVQ multi-codebook design:** `Encode()` returns `int[][]` where `codes[c][f]` is the code from codebook `c` at frame `f`. Codebook 0 captures coarse structure (intelligibility); higher codebooks add detail (quality). This is fundamentally different from text tokenization, which produces a single sequence.

**`EncodeFlat()`** bridges the gap to language models, which expect a single token sequence. It interleaves codebooks and offsets each by `codebook_index * CodebookSize` so tokens are globally unique:

```
Frame 0: [cb0_code + 0*1024, cb1_code + 1*1024, ..., cb7_code + 7*1024]
Frame 1: [cb0_code + 0*1024, cb1_code + 1*1024, ..., cb7_code + 7*1024]
...
```

---

## How It Fits in the Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Your Application / ML.NET Pipeline                     │
├─────────────────────────────────────────────────────────┤
│  MLNet.Audio.Transforms   (ML.NET ITransformer layer)   │
│  MLNet.Audio.Whisper      (Whisper-specific pipeline)   │
├─────────────────────────────────────────────────────────┤
│  MLNet.Audio.Core  ← YOU ARE HERE (Layer 0)             │
│    AudioData, AudioIO, AudioFeatureExtractor,           │
│    MelSpectrogramExtractor, WhisperFeatureExtractor,    │
│    WhisperTokenizer, AudioCodecTokenizer                │
├─────────────────────────────────────────────────────────┤
│  NWaves (DSP)    System.Numerics.Tensors (SIMD)         │
└─────────────────────────────────────────────────────────┘
```

**Layer 0 means:** Every other package in the stack takes a dependency on `MLNet.Audio.Core`. Nothing in `MLNet.Audio.Core` depends on any other package in the stack. This is a deliberate architectural boundary — audio primitives are stable, rarely-changing types that higher layers build on.

---

## Usage

### Loading and inspecting audio

```csharp
using MLNet.Audio.Core;

// Load a WAV file — always returns mono AudioData
var audio = AudioIO.LoadWav("speech.wav");
Console.WriteLine($"Duration: {audio.Duration.TotalSeconds:F1}s");
Console.WriteLine($"Sample rate: {audio.SampleRate} Hz");
Console.WriteLine($"Samples: {audio.Samples.Length:N0}");
```

### Extracting mel spectrogram features

```csharp
// Generic mel spectrogram (configurable for any model)
var melExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
{
    NumMelBins = 128,
    FftSize = 512,
    HopLength = 256
};
float[,] features = melExtractor.Extract(audio);
// features shape: [frames, 128]
```

### Extracting Whisper features

```csharp
// Whisper-specific: 80 mel bins, padded to 30s (3000 frames)
var whisperExtractor = new WhisperFeatureExtractor(numMelBins: 80);
float[,] whisperFeatures = whisperExtractor.Extract(audio);
// whisperFeatures shape: [3000, 80] — always, regardless of input length

// For audio longer than 30s, use chunked extraction
var chunks = whisperExtractor.ExtractChunked(longAudio, overlapFrames: 150);
// chunks: List<float[3000, 80]> — one per 30s window
```

### Resampling and saving

```csharp
// Resample 44.1kHz audio to 16kHz for speech models
var resampled = AudioIO.Resample(audio, targetSampleRate: 16000);

// Save processed audio
AudioIO.SaveWav("output.wav", resampled);

// Stream-based I/O for web scenarios
using var memoryStream = new MemoryStream();
AudioIO.SaveWav(memoryStream, resampled);
```

### Decoding Whisper output tokens

```csharp
var tokenizer = new WhisperTokenizer(isMultilingual: true);

int[] modelOutput = /* ... from ONNX inference ... */;

// Clean transcription text
string text = tokenizer.Decode(modelOutput);

// Text with timestamp annotations
string timestamped = tokenizer.DecodeWithTimestamps(modelOutput);
// "[0.00s] Hello world [1.52s]"

// Structured segments for programmatic use
var segments = tokenizer.DecodeToSegments(modelOutput);
foreach (var seg in segments)
    Console.WriteLine($"[{seg.Start} → {seg.End}] {seg.Text}");
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [NWaves](https://github.com/ar1st0crat/NWaves) | 0.9.6 | Mel filter bank construction, FFT, windowing — the DSP heavy lifting via `FilterbankExtractor` |
| [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors) | 10.0.3 | SIMD-accelerated math via `TensorPrimitives` — used for vectorized log-mel computation (`Max`, `Log`) |

**Notable absence:** No ML.NET packages. No ONNX Runtime. No inference dependencies. This is by design — `MLNet.Audio.Core` is pure data types and CPU-bound computation.

**Target framework:** .NET 10.0
