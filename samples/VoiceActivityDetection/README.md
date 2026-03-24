# Voice Activity Detection: Finding Speech in Audio Streams

This sample demonstrates **Voice Activity Detection (VAD)** — detecting *when* speech occurs in an audio stream using the [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model and the `MLNet.AudioInference.Onnx` library.

## What You'll Learn

- How **windowed/streaming detection** differs from whole-clip classification
- How a **stateful ONNX model** (Silero VAD) processes audio frame-by-frame, carrying hidden state between windows
- The **detection paradigm**: input is a continuous audio stream, output is a set of temporal segments (`SpeechSegment(Start, End, Confidence)`)
- Three approaches to VAD: direct API, streaming via `IVoiceActivityDetector`, and composable ML.NET pipelines

## The Concept: Voice Activity Detection

### What VAD Is

Voice Activity Detection answers a deceptively simple question: **which parts of an audio stream contain speech, and which parts are silence or background noise?**

The output is not a single label like "this clip contains speech." Instead, VAD produces a list of **time segments** — precise start and end timestamps marking where speech occurs within the audio:

```
Input:  ──silence──[speech 1.0s→3.0s]──silence──[speech 4.0s→5.0s]──
Output: SpeechSegment(Start=1.0s, End=3.0s, Confidence=0.92)
        SpeechSegment(Start=4.0s, End=5.0s, Confidence=0.88)
```

### VAD vs. Audio Classification

This distinction is critical for .NET developers coming from other ML tasks:

| Aspect | Audio Classification | Voice Activity Detection |
|---|---|---|
| **Question** | "What is this clip?" | "Where is speech in this stream?" |
| **Input** | Fixed-length clip | Continuous/variable-length stream |
| **Output** | Single label + confidence | List of time segments |
| **Processing** | Stateless (each clip independent) | Stateful (model carries context between frames) |
| **Example** | "This is music" (95%) | Speech at 1.0s→3.0s (92%), 4.0s→5.0s (88%) |

Classification labels a *whole clip*; VAD finds *time segments within* a clip.

### Real-World Use Cases

- **Pre-processing for ASR (Automatic Speech Recognition):** Only send speech segments to a transcription model like Whisper — skip silence to save compute and improve accuracy.
- **Meeting recording analysis:** Identify who spoke when, calculate talk-time ratios, find periods of cross-talk or silence.
- **Real-time communication:** Mute detection, echo cancellation triggers, bandwidth optimization (don't transmit silence).
- **Podcast/video editing:** Automatically trim dead air, detect pauses for chapter markers, identify segment boundaries.

### Why VAD Is a Critical Preprocessing Step

VAD is rarely the end goal — it's the **first stage** in larger audio processing pipelines:

| Use Case | How VAD Helps |
|----------|---------------|
| **ASR Preprocessing** | Skip silence, reduce Whisper processing time by 30-60% |
| **Meeting Analysis** | Identify who spoke when (combined with speaker diarization) |
| **Real-time Communication** | Mute detection, bandwidth optimization, echo cancellation |
| **Podcast/Video Editing** | Auto-trim silence, find content boundaries |
| **Security/Surveillance** | Trigger recording only when speech is detected |
| **Accessibility** | Alert deaf users when someone starts speaking |

In this library, you could compose VAD with ASR:

```csharp
// Step 1: Detect speech segments
var segments = vadTransformer.DetectSpeech(audio);

// Step 2: Transcribe only the speech segments (skip silence)
foreach (var segment in segments)
{
    var speechAudio = audio.Slice(segment.Start, segment.End);
    var text = whisperTransformer.Transcribe(speechAudio);
}
```

### How It Works: Frame-by-Frame Scoring

VAD processes audio through a **sliding window** pipeline:

```
Audio stream
  │
  ▼
┌─────────────────────────────────┐
│ 1. Frame extraction             │  Split into 512-sample windows
│    (512 samples = 32ms at 16kHz)│  (overlapping or contiguous)
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 2. ONNX model scoring          │  Each frame → speech probability (0.0–1.0)
│    (Silero VAD with h/c state)  │  Hidden state carries forward between frames
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 3. Thresholding                 │  probability ≥ 0.5 → "speech"
│                                 │  probability < 0.5 → "silence"
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 4. Segment merging              │  Adjacent speech frames → single segment
│    + post-processing            │  Apply MinSpeechDuration, MinSilenceDuration,
│                                 │  SpeechPad to clean up boundaries
└─────────────┬───────────────────┘
              │
              ▼
  SpeechSegment[] (Start, End, Confidence)
```

### The Model: Silero VAD

[Silero VAD](https://github.com/snakers4/silero-vad) is purpose-built for voice activity detection:

- **Tiny:** ~2MB ONNX model — runs fast on CPU without GPU acceleration.
- **Stateful LSTM:** Unlike classification or embedding models that process each input independently, Silero VAD is an LSTM that maintains **hidden state (`h`) and cell state (`c`)** across frames. This gives it temporal context — it "remembers" what it heard in previous frames when scoring the current one.
- **Real-time capable:** 32ms frames at 16kHz means the model can score audio faster than real-time on modern CPUs.
- **Input format:** Expects mono 16kHz PCM audio in 512-sample windows.

The stateful nature is visible in the ONNX session's inputs and outputs:

```
Inputs:  input[1, 512], sr[1], h[2, 1, 64], c[2, 1, 64]
Outputs: output[1, 1],        hn[2, 1, 64], cn[2, 1, 64]
                               ▲                ▲
                               └── updated hidden/cell state
                                   fed back as input for next frame
```

Each frame receives the previous frame's hidden state and produces updated state for the next frame — an explicit recurrence loop managed in application code.

## What This Sample Demonstrates

The sample shows three approaches to VAD, each suited to different scenarios:

### 1. Direct API — `transformer.DetectSpeech(audioData)`

```csharp
var segments = transformer.DetectSpeech(audio);
```

**Why:** Simplest possible API. Load audio, get segments. Best when you have the complete audio in memory and just want results.

### 2. IVoiceActivityDetector Streaming — `DetectSpeechAsync` with `IAsyncEnumerable`

```csharp
IVoiceActivityDetector vad = transformer;
await foreach (var segment in vad.DetectSpeechAsync(stream))
{
    Console.WriteLine($"Speech: [{segment.Start} → {segment.End}]");
}
```

**Why:** Streaming-compatible. Works with live audio from a microphone, network stream, or file stream. Segments are yielded as they're detected — you don't need to wait for the entire audio to finish. The `IVoiceActivityDetector` interface is custom to this library (no Microsoft.Extensions.AI equivalent exists for VAD).

### 3. ML.NET Pipeline — `Fit` / `Transform`

```csharp
var estimator = mlContext.Transforms.OnnxVad(options);
var transformer = estimator.Fit(emptyData);
```

**Why:** Composable with other ML.NET transforms. Chain VAD with feature extraction, classification, or other audio processing in a single pipeline. Follows the standard ML.NET estimator/transformer pattern.

## Prerequisites

### Silero VAD Model

Download the Silero VAD ONNX model (~2MB):

```bash
# Using huggingface-cli
huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad
```

Or download `silero_vad.onnx` manually from [the Silero VAD GitHub repo](https://github.com/snakers4/silero-vad/tree/master/src/silero_vad/data).

The model should be placed at `models/silero-vad/silero_vad.onnx` relative to the sample directory (or pass a custom path as the first command-line argument).

### .NET 10 SDK

This sample targets `net10.0`. Ensure you have the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) installed.

## Running It

```bash
cd samples/VoiceActivityDetection

# With model + audio file:
dotnet run -- models/silero-vad/silero_vad.onnx my_audio.wav

# With model only (generates test audio automatically):
dotnet run -- models/silero-vad/silero_vad.onnx

# Without model (runs synthetic demo):
dotnet run
```

### With Model

When the Silero VAD model is available, the sample:

1. Loads the WAV file (or generates a test file with speech-like tone patterns).
2. Resamples to 16kHz mono if needed.
3. Runs VAD using all three approaches.
4. Prints detected speech segments with timestamps, durations, and confidence scores.

Expected output:

```
=== Direct VAD API ===
  Found 2 speech segment(s):
    [00:01.000 → 00:03.000] duration=2.00s, confidence=92.0%
    [00:04.000 → 00:05.000] duration=1.00s, confidence=88.0%
  Total speech: 3.00s / 5.00s (60.0%)

=== IVoiceActivityDetector (Streaming) ===
  Speech: [00:01.000 → 00:03.000] (92.0%)
  Speech: [00:04.000 → 00:05.000] (88.0%)
```

### Without Model (Synthetic Demo)

When no model is found, the sample generates a 4-second synthetic audio signal with a deliberate `[silence → tone → silence]` pattern and explains what VAD *would* detect:

```
=== Synthetic VAD Demo ===
  Generated audio: 4.0s
  Pattern: [silence 0-1s] [tone 1-3s] [silence 3-4s]
  (With a real Silero VAD model, this would detect the 1-3s segment as speech)
```

## Code Walkthrough

### Synthetic Audio Generation

The sample creates test audio with a known pattern so you can verify VAD results:

```csharp
// Pattern: silence(0-1s) → tone(1-3s) → silence(3-4s) → tone(4-5s)
for (int i = 0; i < samples.Length; i++)
{
    float t = (float)i / sr;
    if ((t >= 1.0f && t < 3.0f) || (t >= 4.0f && t < 5.0f))
    {
        // "Speech-like" signal: sum of multiple frequencies
        samples[i] = (MathF.Sin(2 * MathF.PI * 200 * i / sr) * 0.3f +
                      MathF.Sin(2 * MathF.PI * 400 * i / sr) * 0.2f +
                      MathF.Sin(2 * MathF.PI * 800 * i / sr) * 0.1f);
    }
}
```

The multi-frequency tone (200Hz + 400Hz + 800Hz) simulates the harmonic structure of human speech, which occupies a similar frequency range. This gives the VAD model a signal that is more speech-like than a simple sine wave.

### VAD Options: Tuning Detection Quality

```csharp
var options = new OnnxVadOptions
{
    ModelPath = modelPath,
    Threshold = 0.5f,                                    // Speech probability cutoff
    MinSpeechDuration = TimeSpan.FromMilliseconds(250),  // Ignore segments < 250ms
    MinSilenceDuration = TimeSpan.FromMilliseconds(100), // Merge speech across < 100ms gaps
    SpeechPad = TimeSpan.FromMilliseconds(30),           // Pad segment edges by 30ms
    WindowSize = 512,                                    // 512 samples = 32ms at 16kHz
    SampleRate = 16000                                   // Required for Silero VAD
};
```

**Why each parameter matters:**

| Parameter | Purpose | Effect of Increasing |
|---|---|---|
| `Threshold` | Speech probability cutoff (0–1) | Fewer false positives, may miss quiet speech |
| `MinSpeechDuration` | Discard segments shorter than this | Filters out brief clicks/pops |
| `MinSilenceDuration` | Merge speech across silence gaps shorter than this | Keeps "um..." pauses within a single segment |
| `SpeechPad` | Extend segment boundaries by this amount | Avoids clipping the start/end of words |
| `WindowSize` | Samples per frame (512 = 32ms at 16kHz) | Larger = coarser time resolution, faster processing |

### SpeechSegment: Temporal Output

The output type highlights the difference between detection and classification:

```csharp
public record SpeechSegment(TimeSpan Start, TimeSpan End, float Confidence)
{
    public TimeSpan Duration => End - Start;
}
```

- **Start/End:** *Where* in the audio stream speech occurs — this is the temporal dimension that classification lacks.
- **Confidence:** Average speech probability across frames in this segment.
- **Duration:** Computed property for convenience.

Compare with classification output (a single label + score for the entire clip) or embedding output (a feature vector). VAD returns *multiple* results per input, each localized in time.

## Key Takeaways

1. **VAD is a detection task (temporal segments), not a classification task (single label).** The output is "speech at 1.0s–3.0s" rather than "this clip contains speech." This temporal precision is what makes VAD useful as a preprocessing step.

2. **Windowed/streaming processing differs fundamentally from batch.** Silero VAD carries LSTM hidden state (`h`, `c`) between frames. Each frame's prediction is influenced by all prior frames — the model accumulates context over time. This is unlike classification or embedding models, which process each input independently.

3. **VAD is a critical preprocessing step for ASR pipelines.** Running Whisper on an entire meeting recording is wasteful and error-prone. VAD identifies the speech segments first, then only those segments are sent for transcription — saving compute and improving accuracy.

4. **The `IVoiceActivityDetector` interface is custom to this library.** Unlike embeddings (`IEmbeddingGenerator`), speech-to-text (`ISpeechToTextClient`), or text-to-speech (`ITextToSpeechClient`), there is no Microsoft.Extensions.AI abstraction for voice activity detection. The `IVoiceActivityDetector` interface provides a streaming-first API with `IAsyncEnumerable<SpeechSegment>`.

## Troubleshooting

### "Model not found" — synthetic demo runs instead

This is expected behavior. Without a Silero VAD model, the sample generates synthetic audio and demonstrates what VAD would detect. To run with real detection:

```bash
huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad
```

The model is tiny (~2 MB) — one of the smallest ONNX models you'll use.

### No speech segments detected

- **Threshold too high**: Default is 0.5. Try lowering to 0.3 for quieter speech: `options.Threshold = 0.3f`
- **MinSpeechDuration too long**: Default is 250ms. Very short utterances ("yes", "no") may be filtered out. Try `options.MinSpeechDuration = TimeSpan.FromMilliseconds(100)`
- **Audio is too quiet**: Normalize audio to [-1.0, 1.0] range before processing
- **Wrong sample rate**: Silero VAD expects 16kHz. The transformer auto-resamples, but extremely low sample rate audio (<8kHz) may lose speech information

### Too many speech segments (over-segmentation)

- **MinSilenceDuration too short**: Default is 100ms. Increase to 300-500ms to merge segments separated by brief pauses: `options.MinSilenceDuration = TimeSpan.FromMilliseconds(300)`
- **SpeechPad too small**: Default is 30ms. Increase padding to merge nearby segments: `options.SpeechPad = TimeSpan.FromMilliseconds(100)`
- **Threshold too low**: Raise from 0.5 to 0.6-0.7 to be more selective about what counts as speech

### Understanding the LSTM state

Silero VAD is a **stateful model** — it carries hidden state (`h`, `c` tensors) between frames. This means:

- Processing order matters — don't shuffle audio frames
- The model "remembers" recent context, improving detection at speech/silence boundaries
- Reset state between unrelated audio files (the transformer handles this automatically)
- This is why VAD uses 512-sample windows (32ms at 16kHz) — small enough for real-time, large enough for the LSTM to be effective

## Going Further

- **[SpeechToText](../SpeechToText/)** — Automatic speech recognition that conceptually builds on VAD: detect speech segments, then transcribe them.
- **[WhisperRawOnnx](../WhisperRawOnnx/)** — Raw ONNX Whisper inference with full control over the transcription pipeline.
- **[AudioClassification](../AudioClassification/)** — Contrast with VAD: classification labels an entire clip rather than finding temporal segments.
- **[Architecture documentation](../../docs/architecture.md)** — The 5-layer design of the library, including how VAD fits into the inference transforms layer.
