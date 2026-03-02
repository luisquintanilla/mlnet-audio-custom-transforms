# Samples Guide

## Overview

The project includes 8 runnable samples demonstrating different audio AI tasks. Each sample gracefully handles missing models — showing API patterns and download instructions instead of crashing.

## Prerequisites

- .NET 10 SDK
- Models downloaded from HuggingFace (see [Models Guide](models-guide.md))

## AudioClassification

**Location:** `samples/AudioClassification/`
**Task:** Classify audio into categories using AST (Audio Spectrogram Transformer)
**Model needed:** `onnx-community/ast-finetuned-audioset-10-10-0.4593`

### Setup

```bash
cd samples/AudioClassification
huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include "onnx/*" --local-dir models/ast
dotnet run
```

### What it demonstrates

- Loading ONNX model for audio classification
- Using `MelSpectrogramExtractor` with 128 mel bins, 400-sample FFT, 160-sample hop
- ML.NET pipeline: `mlContext.Transforms.OnnxAudioClassification(options)`
- Supports AST, Wav2Vec2, and HuBERT model architectures
- Softmax post-processing with label mapping and top-K probabilities
- Graceful fallback when model is absent

### Without model

Prints download instructions and exits cleanly — no synthetic demo.

## AudioEmbeddings

**Location:** `samples/AudioEmbeddings/`
**Task:** Generate vector embeddings from audio using CLAP
**Model needed:** `laion/clap-htsat-unfused`

### Setup

```bash
cd samples/AudioEmbeddings
huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap
dotnet run
```

### What it demonstrates

- Audio embedding generation via direct API
- `OnnxAudioEmbeddingGenerator : IEmbeddingGenerator<AudioData, Embedding<float>>` (MEAI)
- Mean pooling with normalization
- Cosine similarity between audio clips
- Synthetic demo with mel spectrogram extraction (80 mel bins, 400-sample FFT, 160-sample hop)
- Shows `AudioData` and `AudioIO` primitives

### Without model

Runs a synthetic demo: creates a 2-second 440 Hz tone, demonstrates `MelSpectrogramExtractor` outputting `[N frames × 80 mel bins]`, and shows WAV save/reload via `AudioIO`.

## VoiceActivityDetection

**Location:** `samples/VoiceActivityDetection/`
**Task:** Detect speech segments in audio using Silero VAD
**Model needed:** `snakers4/silero-vad`

### Setup

```bash
cd samples/VoiceActivityDetection
huggingface-cli download snakers4/silero-vad --include "*.onnx" --local-dir models/silero-vad
dotnet run
```

### What it demonstrates

- VAD transform: `mlContext.Transforms.OnnxVad(options)`
- `IVoiceActivityDetector` streaming interface
- Speech segment detection with timestamps, confidence, and duration
- Synthetic demo generating a tone pattern (silence → speech → silence)
- Configurable threshold, min speech/silence duration, and padding

### Without model

Runs a synthetic demo: generates 4-second audio with `[silence 0–1 s] [440 Hz tone 1–3 s] [silence 3–4 s]` and explains what a real VAD model would detect.

## SpeechToText

**Location:** `samples/SpeechToText/`
**Task:** Provider-agnostic speech-to-text pipeline
**Model needed:** None (demonstrates patterns without actual cloud/local providers)

### Setup

```bash
cd samples/SpeechToText
dotnet run
```

### What it demonstrates

- Provider-agnostic pattern: same pipeline works with Azure, OpenAI, or local
- `SpeechToTextClientTransformer` wrapping any `ISpeechToTextClient`
- Multi-modal pipeline composition (STT → Text Embeddings → Classification)
- `AudioData` primitives demo (create 3 s audio, extract Whisper features: `[N frames × 80 mel bins]`)
- Whisper expects 3000 frames × 80 mel bins for a full 30 s input
- Voice round-trip concept (STT → TTS)

### Key takeaway

The ML.NET pipeline is provider-agnostic — swap the `ISpeechToTextClient` implementation without changing the pipeline.

## WhisperTranscription

**Location:** `samples/WhisperTranscription/`
**Task:** Local Whisper ASR via ORT GenAI
**Model needed:** ORT GenAI format Whisper model (exported via Olive)

### Setup

```bash
cd samples/WhisperTranscription
# Export via Olive or download pre-exported ORT GenAI model
# Place at: models/whisper-base/
# Required files: genai_config.json, encoder_model.onnx, decoder_model_merged.onnx,
#                 tokenizer.json, tokenizer_config.json
dotnet run
```

### What it demonstrates

- `OnnxSpeechToTextTransformer` direct API: `Transcribe()`, `TranscribeWithTimestamps()`
- `OnnxSpeechToTextClient : ISpeechToTextClient` (MEAI interface)
- ML.NET pipeline: `mlContext.Transforms.OnnxSpeechToText(options)`
- Timestamp output format
- Generates synthetic 3 s audio (440 Hz tone) when no WAV file is provided

### Without model

Shows API patterns with code examples and download instructions, then exits.

## WhisperRawOnnx

**Location:** `samples/WhisperRawOnnx/`
**Task:** Full-control Whisper ASR with manual encoder/decoder/KV cache
**Model needed:** HuggingFace Optimum-exported Whisper ONNX

### Setup

```bash
cd samples/WhisperRawOnnx
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base models/whisper-base/
dotnet run
```

### What it demonstrates

- Raw ONNX Whisper with full control over every stage
- `OnnxWhisperTransformer` direct API
- `WhisperFeatureExtractor` → mel spectrogram (80 mel bins)
- `WhisperKvCacheManager` for manual KV cache handling
- Timestamp support via `WhisperTokenizer`
- Temperature sampling vs greedy decoding
- All 3 ASR approaches compared side by side
- ML.NET pipeline composition

### Without model

Shows 5 patterns: direct transcription, timestamps, ML.NET pipeline, temperature sampling, and a 3-approach comparison.

## TextToSpeech

**Location:** `samples/TextToSpeech/`
**Task:** Text-to-speech using SpeechT5 (encoder-decoder-vocoder)
**Model needed:** `NeuML/txtai-speecht5-onnx`

### Setup

```bash
cd samples/TextToSpeech
git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5
dotnet run
```

### What it demonstrates

- `OnnxSpeechT5TtsTransformer` direct synthesis: `Synthesize("Hello!")`
- `OnnxTextToSpeechClient : ITextToSpeechClient` (MEAI-style prototype)
- ML.NET pipeline: `mlContext.Transforms.SpeechT5Tts(options)`
- SentencePiece tokenization → Encoder → Decoder (KV cache) → Vocoder → PCM
- Custom speaker embedding from `.npy` file
- Voice round-trip: STT → TTS pipeline composition
- Batch synthesis

### Without model

Shows 5 patterns: direct synthesis, `ITextToSpeechClient`, ML.NET pipeline, custom speaker embedding, and voice round-trip.

## AudioDataIngestion

**Location:** `samples/AudioDataIngestion/`
**Task:** End-to-end DataIngestion pipeline: Read → Chunk → Embed → Similarity Search
**Model needed:** `laion/clap-htsat-unfused`

### Setup

```bash
cd samples/AudioDataIngestion
huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap
dotnet run
```

### What it demonstrates

- **Microsoft.Extensions.DataIngestion** integration for audio (proves DataIngestion is modality-agnostic)
- `AudioDocumentReader` — loads WAV files into `IngestionDocument` with audio metadata
- `AudioSegmentChunker` — fixed time-window segmentation into `IngestionChunk<AudioData>`
- `AudioEmbeddingChunkProcessor` — enriches chunks with embeddings via `IEmbeddingGenerator<AudioData, Embedding<float>>`
- The 3-layer bridge: DataIngestion → MEAI → ML.NET
- Cosine similarity search across embedded audio segments
- Generates 4 synthetic audio types: 440Hz tone, 880Hz tone, white noise, chirp

### Without model

Runs a synthetic demo: creates a 4-second test WAV, demonstrates `AudioDocumentReader` loading it into an `IngestionDocument`, and `AudioSegmentChunker` splitting it into 1-second `IngestionChunk<AudioData>` segments. Skips embedding generation (requires model).

### Key takeaway

The same `IngestionDocumentReader → IngestionChunker<T> → IngestionChunkProcessor<T>` pattern that works for text/PDF documents works for audio. `IngestionChunk<AudioData>` keeps the pipeline type-safe with audio content.

## Running All Samples

```bash
cd C:\Dev\mlnet-audio-custom-transforms

# Build everything
dotnet build

# Run each sample
dotnet run --project samples/AudioClassification
dotnet run --project samples/AudioEmbeddings
dotnet run --project samples/VoiceActivityDetection
dotnet run --project samples/SpeechToText
dotnet run --project samples/WhisperTranscription
dotnet run --project samples/WhisperRawOnnx
dotnet run --project samples/TextToSpeech
dotnet run --project samples/AudioDataIngestion
```

All samples exit with code 0 even without models — they show API patterns as documentation.

## Sample Architecture

All samples follow the same structure:

1. Check if model files exist
2. If yes: run actual inference
3. If no: print download instructions and/or show API patterns as code examples
4. Always exit cleanly

Each sample exposes up to three API levels:

| Level | Example |
|-------|---------|
| **Direct API** | `new OnnxSpeechToTextTransformer(mlContext, options)` |
| **MEAI Interface** | `new OnnxSpeechToTextClient(options)` — implements standard MEAI contracts |
| **ML.NET Pipeline** | `mlContext.Transforms.OnnxSpeechToText(options)` — composable with other transforms |

This ensures samples serve as both runnable demos AND API documentation.
