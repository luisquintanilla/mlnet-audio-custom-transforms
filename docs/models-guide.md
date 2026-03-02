# Models Guide

A practical guide to downloading, configuring, and using ONNX models with every audio transform in this project. Follow the instructions here to get any model running immediately.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Download Methods](#model-download-methods)
3. [Speech-to-Text Models (Whisper)](#speech-to-text-models-whisper)
4. [Audio Classification Models](#audio-classification-models)
5. [Audio Embedding Models](#audio-embedding-models)
6. [Voice Activity Detection Models](#voice-activity-detection-models)
7. [Text-to-Speech Models](#text-to-speech-models)
8. [Model Directory Layout](#model-directory-layout)
9. [GPU Acceleration](#gpu-acceleration)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

---

## Overview

All transforms use ONNX models downloaded from HuggingFace. Each transform expects a specific model format — some use a single `.onnx` file, others need a directory with multiple files (encoder, decoder, vocoder, tokenizer). This guide covers every supported model, how to download it, what files you need, and the exact C# options to configure.

**Key principle:** Models are never bundled in this repository. You download them separately and point the options to the local path.

---

## Model Download Methods

### HuggingFace CLI (recommended for most models)

Best for downloading models that already have ONNX exports in their repository (AST, CLAP, Silero VAD).

```bash
pip install huggingface-hub
huggingface-cli download <repo-id> --include "onnx/*" --local-dir models/<name>
```

The `--include` flag avoids downloading PyTorch weights you don't need. The `--local-dir` flag places files in a predictable location.

### Optimum CLI (for Whisper ONNX export)

Required for Whisper models, which don't ship pre-exported ONNX files. Optimum exports the encoder and decoder as separate ONNX files with proper KV cache inputs/outputs.

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base models/whisper-base/
```

This produces `encoder_model.onnx` and `decoder_model_merged.onnx` (merged = KV cache inputs/outputs baked in), plus config files and tokenizer assets.

### Git Clone (for complete repos)

Use when you need every file in the repo, including tokenizer configs, speaker embeddings, or SentencePiece models.

```bash
git lfs install
git clone https://huggingface.co/<repo-id> models/<name>
```

> **Note:** Git LFS is required — model weights are stored as LFS objects. Without it, you'll get pointer files instead of actual weights.

### Manual Download

For any model, you can browse `https://huggingface.co/<repo-id>/tree/main` and download individual `.onnx` files directly. This works when you only need one file and don't want to install any CLI tools.

---

## Speech-to-Text Models (Whisper)

OpenAI's Whisper is a family of encoder-decoder models for automatic speech recognition. This project supports two approaches — each requires a different model format.

### Approach 1: ORT GenAI (`MLNet.ASR.OnnxGenAI`)

Uses ONNX Runtime GenAI format (exported via Olive or the ORT GenAI model builder). The model is a single directory containing config files and ONNX weights. ORT GenAI manages the autoregressive decoder loop internally.

**Download:**

ORT GenAI models need a specific export format. The simplest approach is to use a pre-built model from the ONNX Runtime GenAI model zoo, or export with the ORT GenAI model builder:

```bash
# Pre-built (check onnxruntime-genai releases for latest)
huggingface-cli download microsoft/whisper-base-onnx-genai --local-dir models/whisper-base
```

**Required files:** A directory containing `genai_config.json`, ONNX model files, and tokenizer assets. Point `ModelPath` to the directory, not a specific file.

**Usage:**

```csharp
using MLNet.ASR.OnnxGenAI;

var options = new OnnxSpeechToTextOptions
{
    ModelPath = @"models\whisper-base",   // directory path
    Language = "en",
    MaxLength = 256,
    NumMelBins = 80,
    IsMultilingual = true
};

using var transformer = new OnnxSpeechToTextTransformer(mlContext, options);
var results = transformer.Transcribe([audio]);
Console.WriteLine(results[0]);
```

### Approach 2: Raw ONNX (`MLNet.AudioInference.Onnx`)

Uses standard HuggingFace Optimum-exported ONNX: separate encoder and decoder files. This approach gives you full control over the decoder loop — you manage the KV cache, sampling strategy, and token decoding through `WhisperKvCacheManager` and `WhisperTokenizer`.

**Download:**

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base models/whisper-base/
```

**Required files:**

| File | Purpose |
|------|---------|
| `encoder_model.onnx` | Audio encoder — converts mel spectrogram to hidden states `[1, 1500, hidden_dim]` |
| `decoder_model_merged.onnx` | Autoregressive text decoder with KV cache inputs/outputs for two-phase decoding |

**Usage:**

```csharp
using MLNet.AudioInference.Onnx;

var options = new OnnxWhisperOptions
{
    EncoderModelPath = @"models\whisper-base\encoder_model.onnx",
    DecoderModelPath = @"models\whisper-base\decoder_model_merged.onnx",
    Language = "en",
    NumMelBins = 80,
    MaxTokens = 256,
    Temperature = 0f   // 0 = greedy decoding, >0 = temperature sampling
};

using var transformer = new OnnxWhisperTransformer(mlContext, options);

// Plain text
var results = transformer.Transcribe([audio]);

// With timestamps
var detailed = transformer.TranscribeWithTimestamps([audio]);
foreach (var seg in detailed[0].Segments)
    Console.WriteLine($"[{seg.Start:mm\\:ss} → {seg.End:mm\\:ss}] {seg.Text}");
```

### Whisper Model Comparison

All Whisper variants use the same architecture — they differ in depth and width. Larger models are more accurate but slower and use more memory.

| Model | Params | ONNX Size (approx.) | Speed | Accuracy | Mel Bins | Export Command |
|-------|--------|---------------------|-------|----------|----------|----------------|
| `whisper-tiny` | 39M | ~150 MB | Fastest | Good | 80 | `optimum-cli export onnx --model openai/whisper-tiny models/whisper-tiny/` |
| `whisper-base` | 74M | ~290 MB | Fast | Better | 80 | `optimum-cli export onnx --model openai/whisper-base models/whisper-base/` |
| `whisper-small` | 244M | ~960 MB | Medium | Great | 80 | `optimum-cli export onnx --model openai/whisper-small models/whisper-small/` |
| `whisper-medium` | 769M | ~3 GB | Slow | Excellent | 80 | `optimum-cli export onnx --model openai/whisper-medium models/whisper-medium/` |
| `whisper-large-v3` | 1.5B | ~6 GB | Slowest | Best | 128 | `optimum-cli export onnx --model openai/whisper-large-v3 models/whisper-large-v3/` |

**Recommendation:** Start with `whisper-base` for development and testing. It's fast enough for interactive use and accurate enough for most English content. Move to `whisper-small` or larger for production if accuracy matters.

### Architecture Details

- **Input:** 30-second audio chunks (longer audio is chunked automatically by `WhisperFeatureExtractor.ExtractChunked()`)
- **Feature extraction:** Log-mel spectrogram, 16kHz sample rate, 400 FFT size, 160 hop length, padded/truncated to 3000 frames
- **Mel bins:** 80 for tiny/base/small/medium, 128 for large-v3 — set `NumMelBins` accordingly
- **Encoder output:** Hidden states `[1, 1500, hidden_dim]`
- **Decoder:** Autoregressive with KV cache — see [architecture.md](architecture.md) for the two-phase decode pattern
- **Output:** Token IDs decoded via `WhisperTokenizer` (BPE with 99+ language codes and 1501 timestamp tokens)
- **Languages:** 99+ languages supported by multilingual models

**Model-specific dimensions:**

| Model | Encoder Layers | Decoder Layers | Attention Heads | Hidden Dim | Head Dim |
|-------|---------------|----------------|-----------------|------------|----------|
| whisper-tiny | 4 | 4 | 6 | 384 | 64 |
| whisper-base | 6 | 6 | 8 | 512 | 64 |
| whisper-small | 12 | 12 | 8 | 768 | 96 |
| whisper-medium | 24 | 24 | 16 | 1024 | 64 |
| whisper-large-v3 | 32 | 32 | 20 | 1280 | 64 |

> **Note:** `NumDecoderLayers` and `NumAttentionHeads` in `OnnxWhisperOptions` default to 0, which triggers auto-detection from the ONNX model metadata via `WhisperKvCacheManager.DetectFromModel()`. You only need to set these manually if auto-detection fails.

---

## Audio Classification Models

### AST (Audio Spectrogram Transformer)

The Audio Spectrogram Transformer, fine-tuned on AudioSet. Classifies audio into 527 categories covering speech, music, and environmental sounds (e.g., "Speech", "Dog", "Siren", "Piano", "Laughter").

**Download:**

```bash
huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include "onnx/*" --local-dir models/ast
```

**Required files:**

| File | Size | Purpose |
|------|------|---------|
| `models/ast/onnx/model.onnx` | ~340 MB | Single encoder model |

**Architecture:**

- **Input:** Mel spectrogram with 128 mel bins, variable length
- **Output:** 527 AudioSet class logits → softmax → probabilities
- **Feature extraction:** `MelSpectrogramExtractor` with `NumMelBins = 128`

**Usage:**

```csharp
using MLNet.AudioInference.Onnx;

// Load AudioSet labels (527 classes)
var labels = File.ReadAllLines("models/ast/labels.txt");

var options = new OnnxAudioClassificationOptions
{
    ModelPath = @"models\ast\onnx\model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000) { NumMelBins = 128 },
    Labels = labels,
    SampleRate = 16000
};

var estimator = mlContext.Transforms.OnnxAudioClassification(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
var results = transformer.Classify([audio]);

Console.WriteLine($"Predicted: {results[0].PredictedLabel} ({results[0].Score:P1})");
```

> **Labels:** The AST AudioSet model expects 527 labels in the exact order matching the model's output logits. Download the label file from the HuggingFace model repo, or define them manually.

### Other Classification Models

Any ONNX encoder model that takes a mel spectrogram input and outputs class logits works with `OnnxAudioClassificationTransformer`. Configure `NumMelBins`, `FftSize`, and `HopLength` on the `MelSpectrogramExtractor` to match the model's expected input, and provide the correct `Labels` array.

---

## Audio Embedding Models

### CLAP (Contrastive Language-Audio Pretraining)

CLAP learns a shared embedding space for audio and text — audio clips and text descriptions that match are close together in the 512-dimensional space. This enables text-to-audio search, audio similarity, and zero-shot classification.

**Download:**

```bash
huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap
```

**Required files:**

| File | Size | Purpose |
|------|------|---------|
| `models/clap/onnx/model.onnx` | ~600 MB | Audio encoder (HTSAT backbone) |

**Architecture:**

- **Input:** Mel spectrogram with 80 mel bins (default `MelSpectrogramExtractor` settings work)
- **Output:** 512-dimensional embedding vector
- **Pooling:** `MeanPooling` (default) works well; `ClsToken` also supported
- **Normalization:** L2-normalized when `Normalize = true` (recommended for cosine similarity)

**Usage:**

```csharp
using MLNet.AudioInference.Onnx;

var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = @"models\clap\onnx\model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000),
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true
};

var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
var embeddings = transformer.GenerateEmbeddings([audio1, audio2]);

// Cosine similarity (embeddings are already L2-normalized)
float similarity = TensorPrimitives.CosineSimilarity(
    embeddings[0].AsSpan(), embeddings[1].AsSpan());
Console.WriteLine($"Similarity: {similarity:F3}");
```

**MEAI integration:**

```csharp
using MLNet.AudioInference.Onnx;

IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer, modelId: "clap-htsat-unfused");

var result = await generator.GenerateAsync([audio]);
```

### Other Embedding Models

Any ONNX encoder that maps mel spectrogram input to a fixed-size vector works with `OnnxAudioEmbeddingTransformer`. Common alternatives include Wav2Vec2 and HuBERT (both produce per-frame embeddings that get pooled). Set the `Pooling` strategy and `Normalize` flag based on your downstream use case:

| Pooling Strategy | When to Use |
|-----------------|-------------|
| `MeanPooling` | General purpose — averages across all time frames. Best default. |
| `ClsToken` | When the model has a dedicated `[CLS]` token in position 0 (BERT-style). |
| `MaxPooling` | When you want to capture the most activated features across time. |

---

## Voice Activity Detection Models

### Silero VAD v5

A tiny, fast model that detects speech vs. non-speech in audio. Outputs a speech probability for each frame, which the transform merges into contiguous `SpeechSegment` regions.

**Download:**

```bash
huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad
```

**Required files:**

| File | Size | Purpose |
|------|------|---------|
| `models/silero-vad/silero_vad.onnx` | ~2 MB | Stateful VAD model |

**Architecture:**

- **Input:** Raw PCM samples in 512-sample windows (32ms at 16kHz) — **no mel spectrogram needed**
- **Output:** Speech probability per frame (0.0 to 1.0)
- **State:** Carries internal `h` and `c` hidden states between frames (the transform manages this automatically)
- **Post-processing:** Threshold + minimum duration filtering + padding → `SpeechSegment[]`

**Usage:**

```csharp
using MLNet.AudioInference.Onnx;

var options = new OnnxVadOptions
{
    ModelPath = @"models\silero-vad\silero_vad.onnx",
    Threshold = 0.5f,
    MinSpeechDuration = TimeSpan.FromMilliseconds(250),
    MinSilenceDuration = TimeSpan.FromMilliseconds(100),
    SpeechPad = TimeSpan.FromMilliseconds(30),
    WindowSize = 512,
    SampleRate = 16000
};

var estimator = mlContext.Transforms.OnnxVad(options);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
var segments = transformer.DetectSpeech(audio);

foreach (var seg in segments)
    Console.WriteLine($"Speech: {seg.Start:mm\\:ss\\.ff} → {seg.End:mm\\:ss\\.ff} (confidence: {seg.Confidence:F2})");
```

**IVoiceActivityDetector interface:**

`OnnxVadTransformer` implements `IVoiceActivityDetector`, so you can also use the `DetectSpeechAsync()` method and pass the transformer anywhere that interface is expected.

**Tuning parameters:**

| Parameter | Effect of Increasing | Typical Range |
|-----------|---------------------|---------------|
| `Threshold` | Fewer false positives (misses quiet speech) | 0.3 – 0.7 |
| `MinSpeechDuration` | Ignores very short speech bursts | 100 – 500 ms |
| `MinSilenceDuration` | Merges nearby segments (bridges short pauses) | 50 – 300 ms |
| `SpeechPad` | Adds padding around detected segments | 0 – 100 ms |

---

## Text-to-Speech Models

### SpeechT5 (Microsoft)

Microsoft's SpeechT5 is an encoder-decoder model for text-to-speech. The pipeline is: text → SentencePiece tokenizer → encoder → decoder (autoregressive mel frame generation with KV cache) → postnet + HiFi-GAN vocoder → PCM audio.

**Download:**

```bash
git lfs install
git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5
```

**Required files:**

| File | Size | Purpose |
|------|------|---------|
| `encoder_model.onnx` | ~343 MB | Text encoder — converts token IDs to hidden states |
| `decoder_model_merged.onnx` | ~244 MB | Mel decoder with KV cache — autoregressive mel frame generation |
| `decoder_postnet_and_vocoder.onnx` | ~55 MB | Postnet refinement + HiFi-GAN vocoder — converts mel to PCM waveform |
| `spm_char.model` | ~3 KB | SentencePiece character-level tokenizer |
| `speaker.npy` | ~2 KB | Default 512-dimensional x-vector speaker embedding |
| **Total** | **~643 MB** | |

**Architecture:**

- **Tokenizer:** SentencePiece character model (loaded via `Microsoft.ML.Tokenizers`)
- **Encoder:** 6 layers, 12 attention heads, 64 head dim, 768 hidden dim
- **Decoder:** Autoregressive — generates one mel frame per step, uses KV cache (same `WhisperKvCacheManager` pattern as Whisper)
- **Output:** 80 mel bins per frame, stop probability head determines when to halt
- **Vocoder:** Postnet + HiFi-GAN converts mel spectrogram to 16kHz PCM audio
- **Speaker embedding:** 512-dim x-vector controls voice characteristics

**Usage:**

```csharp
using MLNet.AudioInference.Onnx;

var options = new OnnxSpeechT5Options
{
    EncoderModelPath = @"models\speecht5\encoder_model.onnx",
    DecoderModelPath = @"models\speecht5\decoder_model_merged.onnx",
    VocoderModelPath = @"models\speecht5\decoder_postnet_and_vocoder.onnx",
    TokenizerModelPath = @"models\speecht5\spm_char.model",        // optional — auto-detected if in same dir
    SpeakerEmbeddingPath = @"models\speecht5\speaker.npy",         // optional — auto-detected if in same dir
    MaxMelFrames = 500,
    StopThreshold = 0.5f,
    NumMelBins = 80,
    SampleRate = 16000
};

using var transformer = new OnnxSpeechT5TtsTransformer(mlContext, options);
var audio = transformer.Synthesize("Hello, world! This is a test of text to speech.");
AudioIO.SaveWav("output.wav", audio);
```

**ITextToSpeechClient interface:**

```csharp
using var client = new OnnxTextToSpeechClient(options);
var response = await client.GetAudioAsync("Say something");
AudioIO.SaveWav("output.wav", response.Audio);
```

> **Note:** `ITextToSpeechClient` is a prototype MEAI-style interface. When Microsoft.Extensions.AI adds an official TTS interface, this will be updated to match.

**Limitations:**

- English-only (SpeechT5 is trained on LibriTTS)
- Best for short utterances (1–2 sentences). Quality degrades on very long text.
- Single speaker by default — provide a different `.npy` x-vector for voice cloning

---

## Model Directory Layout

Recommended directory structure for each sample. Each sample keeps its models in a local `models/` subdirectory:

```
samples/
├── AudioClassification/
│   └── models/
│       └── ast/
│           └── onnx/
│               └── model.onnx
│
├── AudioEmbeddings/
│   └── models/
│       └── clap/
│           └── onnx/
│               └── model.onnx
│
├── VoiceActivityDetection/
│   └── models/
│       └── silero-vad/
│           └── silero_vad.onnx
│
├── WhisperTranscription/         (ORT GenAI)
│   └── models/
│       └── whisper-base/
│           ├── genai_config.json
│           └── *.onnx
│
├── WhisperRawOnnx/               (Raw ONNX)
│   └── models/
│       └── whisper-base/
│           ├── encoder_model.onnx
│           └── decoder_model_merged.onnx
│
├── TextToSpeech/
│   └── models/
│       └── speecht5/
│           ├── encoder_model.onnx
│           ├── decoder_model_merged.onnx
│           ├── decoder_postnet_and_vocoder.onnx
│           ├── spm_char.model
│           └── speaker.npy
│
└── SpeechToText/                 (Provider-agnostic — uses any ISpeechToTextClient)
    └── (no local models needed)
```

> **Tip:** Models directories are `.gitignore`'d. Never commit model weights to the repository.

---

## GPU Acceleration

ONNX Runtime supports multiple execution providers. By default, all transforms run on CPU. To enable GPU acceleration, install the appropriate NuGet package and set `GpuDeviceId` in the options.

### Execution Providers

| Provider | Platform | NuGet Package | Notes |
|----------|----------|---------------|-------|
| CPU | All | `Microsoft.ML.OnnxRuntime.Managed` (already included) | Default. Works everywhere. |
| CUDA | NVIDIA GPUs | `Microsoft.ML.OnnxRuntime.Gpu` | Requires CUDA toolkit + cuDNN |
| DirectML | Windows (AMD/Intel/NVIDIA) | `Microsoft.ML.OnnxRuntime.DirectML` | Broadest Windows GPU support |

### Enabling GPU

Set `GpuDeviceId` to the GPU device index (typically `0`):

```csharp
// Audio classification on GPU
var options = new OnnxAudioClassificationOptions
{
    ModelPath = @"models\ast\onnx\model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000) { NumMelBins = 128 },
    Labels = labels,
    GpuDeviceId = 0   // Use first GPU
};

// Audio embeddings on GPU
var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = @"models\clap\onnx\model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000),
    GpuDeviceId = 0
};

// VAD on GPU
var options = new OnnxVadOptions
{
    ModelPath = @"models\silero-vad\silero_vad.onnx",
    GpuDeviceId = 0
};
```

> **Note:** `OnnxWhisperOptions` and `OnnxSpeechT5Options` do not currently expose `GpuDeviceId` — GPU support for encoder-decoder transforms is planned.

### When GPU Helps

- **Large models** (whisper-medium, whisper-large-v3): significant speedup from GPU parallelism
- **Batch processing**: multiple audio files processed together
- **Real-time pipelines**: when CPU can't keep up with incoming audio

GPU provides minimal benefit for Silero VAD (already sub-millisecond on CPU) or single-file classification with small models.

---

## Performance Considerations

### Inference Speed (approximate, CPU)

| Task | Model | Input | Time | Notes |
|------|-------|-------|------|-------|
| Speech-to-Text | whisper-tiny | 30s audio | ~1–2s | Fastest Whisper option |
| Speech-to-Text | whisper-base | 30s audio | ~2–5s | Good accuracy/speed balance |
| Speech-to-Text | whisper-small | 30s audio | ~5–15s | Production-quality accuracy |
| Speech-to-Text | whisper-medium | 30s audio | ~15–40s | High accuracy, slow |
| Audio Classification | AST | Single clip | ~100ms | Fast enough for real-time |
| Audio Embeddings | CLAP | Single clip | ~200ms | Depends on audio length |
| Voice Activity Detection | Silero VAD | Per frame | <1ms | Real-time capable |
| Text-to-Speech | SpeechT5 | Short sentence | ~1–3s | Depends on output length |

### Memory Usage

| Model | RAM (approximate) |
|-------|-------------------|
| Silero VAD | ~10 MB |
| AST (AudioSet) | ~400 MB |
| CLAP (HTSAT) | ~700 MB |
| whisper-tiny | ~200 MB |
| whisper-base | ~350 MB |
| whisper-small | ~1.2 GB |
| whisper-medium | ~3.5 GB |
| whisper-large-v3 | ~7 GB |
| SpeechT5 (all 3 files) | ~800 MB |

### Optimization Tips

1. **Load once, reuse:** Model loading is the biggest one-time cost. Create the transformer once and reuse it for all inference calls. Never create a new transformer per audio file.

2. **Choose the right model size:** whisper-base handles most English content well. Only scale up if you need multilingual support or maximum accuracy.

3. **Use VAD as a pre-filter:** Run Silero VAD first to find speech segments, then only transcribe those segments with Whisper. This avoids wasting compute on silence or background noise.

4. **Chunk long audio:** `WhisperFeatureExtractor.ExtractChunked()` splits long audio into 30-second chunks with overlap. Each chunk is processed independently — this is how Whisper handles audio longer than 30 seconds.

5. **Greedy decoding:** Set `Temperature = 0f` in `OnnxWhisperOptions` for deterministic, faster decoding. Temperature sampling (`Temperature > 0`) adds diversity but slows down inference.

---

## Troubleshooting

### Common Issues

**"Could not find ONNX model file"**
- Check the path is correct and the file exists. Use absolute paths or paths relative to the working directory.
- For Whisper (raw ONNX), you need both `encoder_model.onnx` and `decoder_model_merged.onnx`.
- For SpeechT5, you need all three: encoder, decoder, and vocoder.

**"Model input shape mismatch"**
- Verify `NumMelBins` matches the model. AST expects 128; Whisper tiny/base/small/medium expect 80; Whisper large-v3 expects 128.
- Make sure the `FeatureExtractor` sample rate matches the audio sample rate (16kHz for all models in this project).

**"Git LFS pointer files instead of model weights"**
- Install Git LFS (`git lfs install`) before cloning. If you already cloned, run `git lfs pull` to fetch the actual weights.

**"Optimum export fails"**
- Ensure you have `optimum[onnxruntime]` installed: `pip install optimum[onnxruntime]`.
- Some model versions may need a specific `transformers` version. Check the model card on HuggingFace for compatibility notes.

**"Out of memory"**
- Large models (whisper-medium, whisper-large-v3) require significant RAM. Consider using a smaller model variant.
- On GPU, check that your GPU has enough VRAM for the model size.

**"KV cache dimension mismatch"**
- If `WhisperKvCacheManager` auto-detection fails, set `NumDecoderLayers` and `NumAttentionHeads` explicitly in `OnnxWhisperOptions` or `OnnxSpeechT5Options` using the values from the table in [Architecture Details](#architecture-details).

---

## Quick Reference

| Transform | Model | Download Command | Model Path |
|-----------|-------|-----------------|------------|
| Classification | AST AudioSet | `huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include "onnx/*" --local-dir models/ast` | `models/ast/onnx/model.onnx` |
| Embeddings | CLAP HTSAT | `huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap` | `models/clap/onnx/model.onnx` |
| VAD | Silero VAD | `huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad` | `models/silero-vad/silero_vad.onnx` |
| STT (Raw ONNX) | Whisper | `optimum-cli export onnx --model openai/whisper-base models/whisper-base/` | `encoder_model.onnx` + `decoder_model_merged.onnx` |
| TTS | SpeechT5 | `git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5` | `encoder_model.onnx` + `decoder_model_merged.onnx` + `decoder_postnet_and_vocoder.onnx` |
