# Documentation

## ML.NET Audio Custom Transforms

Welcome to the documentation for ML.NET Audio Custom Transforms — a comprehensive framework for audio AI tasks in .NET using local ONNX models.

### For Newcomers
If you're a .NET developer new to audio processing and audio ML, start here:

1. **[Audio Processing Primer](audio-primer.md)** — Learn the fundamentals: PCM audio, mel spectrograms, FFT, and how audio ML differs from text ML. No prior audio knowledge required.

2. **[Architecture Overview](architecture.md)** — Understand the layered design, the three-stage pipeline pattern, and how this project relates to the text-based ML.NET transform ecosystem.

3. **[Samples Guide](samples-guide.md)** — Get hands-on quickly. Every sample runs even without models, showing API patterns as documentation.

### For Developers
Ready to use the transforms in your projects:

4. **[Transforms Guide](transforms-guide.md)** — Complete reference for every transform: Audio Classification, Embeddings, VAD, Speech-to-Text (3 approaches), Text-to-Speech. Includes all options, code examples, and supported models.

5. **[Models Guide](models-guide.md)** — Which models to use, how to download them, architecture details, performance tradeoffs, and directory layout.

6. **[MEAI Integration](meai-integration.md)** — How to use these transforms with Microsoft.Extensions.AI interfaces (IEmbeddingGenerator, ISpeechToTextClient, ITextToSpeechClient), dependency injection, and middleware patterns.

### For Contributors
Want to extend the framework:

7. **[Extending the Framework](extending.md)** — Step-by-step guide to adding new models, creating feature extractors, implementing MEAI interfaces, and the KV cache pattern.

8. **[Master Plan](plan.md)** — The full implementation roadmap, design decisions, and future vision.

### Quick Links

| Task | Transform | MEAI Interface | Sample |
|------|-----------|---------------|--------|
| Audio Classification | `OnnxAudioClassification` | — | `samples/AudioClassification` |
| Audio Embeddings | `OnnxAudioEmbedding` | `IEmbeddingGenerator<AudioData, Embedding<float>>` | `samples/AudioEmbeddings` |
| Voice Activity Detection | `OnnxVad` | `IVoiceActivityDetector` | `samples/VoiceActivityDetection` |
| Speech-to-Text (Provider) | `SpeechToText` | `ISpeechToTextClient` | `samples/SpeechToText` |
| Speech-to-Text (ORT GenAI) | `OnnxSpeechToText` | `ISpeechToTextClient` | `samples/WhisperTranscription` |
| Speech-to-Text (Raw ONNX) | `OnnxWhisper` | `ISpeechToTextClient` | `samples/WhisperRawOnnx` |
| Text-to-Speech (SpeechT5) | `SpeechT5Tts` | `ITextToSpeechClient` | `samples/TextToSpeech` |
| Text-to-Speech (KittenTTS) | `KittenTts` | `ITextToSpeechClient` | `samples/KittenTTS` |
| Audio DataIngestion | `AudioDocumentReader` / `AudioSegmentChunker` / `AudioEmbeddingChunkProcessor` | `IEmbeddingGenerator` | `samples/AudioDataIngestion` |

### Project Status

✅ 7 audio tasks implemented
✅ 5 source packages (Audio.Core, Audio.Tokenizers, AudioInference.Onnx, ASR.OnnxGenAI, Audio.DataIngestion)
✅ 9 runnable samples
✅ MEAI integration (IEmbeddingGenerator, ISpeechToTextClient, ITextToSpeechClient)
✅ Microsoft.Extensions.DataIngestion integration (Reader → Chunker → Processor for audio)
✅ Microsoft.ML.Tokenizers (SentencePiece for SpeechT5 via Audio.Tokenizers, custom WhisperTokenizer)
✅ System.Numerics.Tensors / TensorPrimitives throughout
📋 Model packaging and model garden integration (planned)
