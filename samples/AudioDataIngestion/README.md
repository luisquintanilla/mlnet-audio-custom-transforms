# Audio Data Ingestion: RAG Pipelines for Audio Content

This sample demonstrates that **Microsoft.Extensions.DataIngestion is modality-agnostic** — the same Reader → Chunker → Processor pipeline designed for text documents works seamlessly with audio. It builds a complete audio RAG (Retrieval-Augmented Generation) pipeline: read WAV files, chunk into segments, generate embeddings, and search by similarity.

## What You'll Learn

- How `Microsoft.Extensions.DataIngestion` works for **audio**, not just text
- The **3-layer bridge** architecture: DataIngestion → MEAI → ML.NET
- How to perform **similarity search** over embedded audio chunks
- Streaming audio processing with `IAsyncEnumerable` for memory efficiency

## The Concept: DataIngestion for Audio

### What Is DataIngestion?

`Microsoft.Extensions.DataIngestion` provides a standard pipeline for processing documents into vector-store-ready chunks. The pipeline has three stages:

```
Reader → Chunker → Processor
```

- **`IngestionDocumentReader`** — reads raw content (a file, a stream) into an `IngestionDocument`
- **`IngestionChunker<T>`** — splits an `IngestionDocument` into a stream of `IngestionChunk<T>` pieces
- **`IngestionChunkProcessor<T>`** — enriches each chunk (typically by computing embeddings)

This was designed for text workflows — reading Markdown, PDF, or HTML, chunking into paragraphs, and embedding with a text model. But the abstractions are **completely generic**:

| Abstraction | Text Pipeline | Audio Pipeline |
|---|---|---|
| `IngestionDocumentReader` | Reads Markdown/PDF/HTML | Reads WAV files |
| `IngestionChunker<T>` | Splits into text paragraphs (`T = string`) | Splits into time segments (`T = AudioData`) |
| `IngestionChunkProcessor<T>` | Embeds text chunks | Embeds audio chunks |
| Chunk content type `T` | `string` | `AudioData` |

The key insight: **`T` is generic**. When `T = AudioData`, each `IngestionChunk<AudioData>` carries the actual audio segment as its content — type-safe, no serialization hacks, and directly consumable by the embedding generator.

### Why This Matters for RAG

Audio files — podcasts, meetings, music libraries, voice recordings — need to be searchable too. The RAG pattern for audio is:

1. **Split** audio into time-based segments
2. **Generate embeddings** for each segment
3. **Store** embeddings in a vector database
4. **Query** by similarity (find segments that sound like X)

DataIngestion gives you this pipeline out of the box. You just need audio-specific implementations of each stage.

### The 3-Layer Bridge Architecture

This sample connects three abstraction layers, each decoupled from the others:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: DataIngestion                                     │
│  AudioDocumentReader → AudioSegmentChunker                  │
│  → AudioEmbeddingChunkProcessor                             │
│                          │                                  │
│                          │ uses IEmbeddingGenerator          │
│                          ▼                                  │
│  Layer 2: MEAI (Microsoft.Extensions.AI)                    │
│  IEmbeddingGenerator<AudioData, Embedding<float>>           │
│  implemented by OnnxAudioEmbeddingGenerator                 │
│                          │                                  │
│                          │ wraps ML.NET transformer          │
│                          ▼                                  │
│  Layer 1: ML.NET                                            │
│  AudioFeatureExtraction → OnnxScoring → EmbeddingPooling    │
│  = float[] embedding per audio segment                      │
└─────────────────────────────────────────────────────────────┘
```

- **Layer 1 (ML.NET):** The `OnnxAudioEmbeddingTransformer` runs the actual ONNX model — extracting mel spectrograms, scoring through CLAP, and pooling into a fixed-length `float[]` embedding.
- **Layer 2 (MEAI):** `OnnxAudioEmbeddingGenerator` implements `IEmbeddingGenerator<AudioData, Embedding<float>>`, wrapping the ML.NET transformer behind the standard MEAI abstraction. Any consumer that speaks MEAI can use it.
- **Layer 3 (DataIngestion):** `AudioEmbeddingChunkProcessor` takes an `IEmbeddingGenerator` — it doesn't know or care that ML.NET is underneath. It just calls `GenerateAsync()` on each chunk's `AudioData` content and stores the resulting embedding in `chunk.Metadata["embedding"]`.

This separation means you can swap any layer independently. Replace CLAP with a different audio model? Only Layer 1 changes. Switch from ML.NET to a cloud embedding API? Only Layer 2 changes. DataIngestion code stays the same.

## What This Sample Demonstrates

The sample runs an end-to-end audio DataIngestion pipeline with four stages:

### 1. Synthetic Audio Generation

Four deliberately different WAV files are generated programmatically:

| File | Content | Purpose |
|---|---|---|
| `music_440hz.wav` | 4 s pure tone at 440 Hz (concert A) | Baseline tonal signal |
| `music_880hz.wav` | 4 s pure tone at 880 Hz (octave above) | Similar but higher — tests similarity |
| `noise.wav` | 4 s white noise (seeded RNG) | Maximally different from tones |
| `chirp.wav` | 4 s frequency sweep 200→2000 Hz | Blend of tonal and broadband characteristics |

These are chosen to create **interesting similarity relationships**: the two tones should be more similar to each other than to noise, and the chirp should fall somewhere in between.

### 2. Read — `AudioDocumentReader`

```csharp
var reader = new AudioDocumentReader(targetSampleRate: 16000);
using var stream = File.OpenRead(wavFile);
var doc = await reader.ReadAsync(stream, "music_440hz.wav", "audio/wav");
```

The reader decodes a WAV stream into an `IngestionDocument`. The audio samples (as `AudioData`) are stored in `section.Metadata["audio"]` for downstream chunkers. A text summary (filename, duration, sample rate, sample count) is stored in `section.Text` for inspection.

### 3. Chunk — `AudioSegmentChunker`

```csharp
var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2));
var chunks = chunker.ProcessAsync(doc);
```

The chunker splits the audio into fixed-duration segments (2 seconds each). A 4-second file produces 2 chunks. Each `IngestionChunk<AudioData>` carries:
- **`Content`** — the `AudioData` segment itself (samples + sample rate)
- **`Metadata["startTime"]`** / **`Metadata["endTime"]`** — timing info (e.g., `"0.00"` → `"2.00"`)
- **`Metadata["segmentIndex"]`** — zero-based segment index
- **`Metadata["sourceFile"]`** — original filename

### 4. Embed — `AudioEmbeddingChunkProcessor`

```csharp
var processor = new AudioEmbeddingChunkProcessor(generator);
var enrichedChunks = processor.ProcessAsync(chunks);
```

The processor takes the `IAsyncEnumerable<IngestionChunk<AudioData>>` stream from the chunker and enriches each chunk by:
1. Extracting the `AudioData` from `chunk.Content`
2. Calling `IEmbeddingGenerator.GenerateAsync()` → which invokes MEAI → ML.NET → CLAP
3. Storing the resulting `float[]` in `chunk.Metadata["embedding"]`

The enriched chunks flow out as another `IAsyncEnumerable` — fully streaming, no buffering of the entire collection.

### 5. Search — Cosine Similarity

```csharp
var similarity = TensorPrimitives.CosineSimilarity(query.Embedding, candidate.Embedding);
```

After all chunks are embedded, the sample uses the first chunk as a query and ranks all other chunks by cosine similarity using `System.Numerics.Tensors.TensorPrimitives`. This demonstrates the retrieval step of audio RAG: given a query audio segment, find the most acoustically similar segments across the entire collection.

## Prerequisites

**CLAP ONNX model** — download from HuggingFace:

```bash
pip install huggingface-hub
huggingface-cli download laion/clap-htsat-unfused --include "onnx/*" --local-dir models/clap
```

The model file should end up at `models/clap/onnx/model.onnx` (relative to where you run the sample).

**.NET 10 SDK** — this sample targets `net10.0`.

## Running It

### With CLAP model (full pipeline)

```bash
cd samples/AudioDataIngestion
dotnet run
# or specify model path:
dotnet run -- path/to/model.onnx
```

This runs the complete Read → Chunk → Embed → Search pipeline:

```
=== Generating Synthetic Audio Files ===

  music_440hz.wav: 4s tone at 440Hz
  music_880hz.wav: 4s tone at 880Hz
  noise.wav: 4s white noise
  chirp.wav: 4s chirp 200→2000Hz

=== Setting Up Embedding Pipeline (ML.NET → MEAI) ===

  ML.NET model loaded → MEAI IEmbeddingGenerator ready

=== DataIngestion Pipeline: Read → Chunk → Embed ===

  Processing: music_440hz.wav
    Read: Audio file: music_440hz.wav, Duration: 4.00s, SampleRate: 16000Hz, Samples: 64000
    Chunk [0.00s → 2.00s]: [512]-dim embedding
    Chunk [2.00s → 4.00s]: [512]-dim embedding
  ...

=== Audio Similarity Search ===

  Query: music_440hz.wav [0.00s → 2.00s]
  Results (sorted by similarity):
    0.9998  music_440hz.wav [2.00s → 4.00s]
    0.9876  music_880hz.wav [0.00s → 2.00s]
    ...
```

### Without model (synthetic demo)

If no model is found, the sample automatically falls back to a demonstration of the Reader and Chunker stages:

```bash
dotnet run
```

```
Model not found at: models/clap/onnx/model.onnx
Running with synthetic demo (no model) instead...

=== Synthetic Demo (DataIngestion without model) ===

  AudioDocumentReader: Audio file: test.wav, Duration: 4.00s, SampleRate: 16000Hz, Samples: 64000
  AudioSegmentChunker: chunk [0.00s → 1.00s], 1.00s of audio
  AudioSegmentChunker: chunk [1.00s → 2.00s], 1.00s of audio
  AudioSegmentChunker: chunk [2.00s → 3.00s], 1.00s of audio
  AudioSegmentChunker: chunk [3.00s → 4.00s], 1.00s of audio

  Total chunks: 4

  (AudioEmbeddingChunkProcessor skipped — requires ONNX model)
```

This is useful for exploring the DataIngestion pipeline structure without downloading a model.

## Code Walkthrough

### Creating the 3-Layer Bridge

The setup wires together all three layers:

```csharp
// Layer 1: ML.NET — build the ONNX audio embedding estimator and fit it
var mlContext = new MLContext();
var options = new OnnxAudioEmbeddingOptions
{
    ModelPath = modelPath,
    FeatureExtractor = new MelSpectrogramExtractor(sampleRate: 16000)
    {
        NumMelBins = 64, FftSize = 512, HopLength = 160
    },
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true,
    SampleRate = 16000
};
var estimator = mlContext.Transforms.OnnxAudioEmbedding(options);
var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>());
using var transformer = estimator.Fit(emptyData);

// Layer 2: MEAI — wrap the ML.NET transformer as IEmbeddingGenerator
using IEmbeddingGenerator<AudioData, Embedding<float>> generator =
    new OnnxAudioEmbeddingGenerator(transformer);

// Layer 3: DataIngestion — create pipeline components
var reader = new AudioDocumentReader(targetSampleRate: 16000);
var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2));
var processor = new AudioEmbeddingChunkProcessor(generator);
```

Note how the `AudioEmbeddingChunkProcessor` only sees `IEmbeddingGenerator<AudioData, Embedding<float>>` — it is completely decoupled from ML.NET and ONNX.

### The Async Streaming Pattern

The pipeline uses `IAsyncEnumerable` throughout, enabling streaming without buffering entire collections:

```csharp
// Chunker produces a stream of chunks
IAsyncEnumerable<IngestionChunk<AudioData>> chunks = chunker.ProcessAsync(doc);

// Processor consumes and enriches the stream, producing another stream
IAsyncEnumerable<IngestionChunk<AudioData>> enrichedChunks = processor.ProcessAsync(chunks);

// Consume one chunk at a time
await foreach (var chunk in enrichedChunks)
{
    // Each chunk arrives fully enriched with its embedding
    var embedding = (float[])chunk.Metadata["embedding"];
}
```

This is critical for large audio collections — you process one chunk at a time rather than loading all embeddings into memory. The chunker yields a segment, the processor immediately embeds it, and the consumer stores or indexes it.

### Enriched Metadata Flow

The `chunk.Metadata` dictionary carries data through the pipeline:

```csharp
// Set by AudioSegmentChunker:
chunk.Metadata["startTime"]     // "0.00" — segment start in seconds
chunk.Metadata["endTime"]       // "2.00" — segment end in seconds
chunk.Metadata["segmentIndex"]  // "0" — zero-based index
chunk.Metadata["sourceFile"]    // "music_440hz.wav" — original filename

// Added by AudioEmbeddingChunkProcessor:
chunk.Metadata["embedding"]     // float[] — the CLAP embedding vector
```

### Cosine Similarity Search

After collecting all embedded chunks, the sample performs brute-force similarity search using `TensorPrimitives.CosineSimilarity` from `System.Numerics.Tensors`:

```csharp
var results = allChunks
    .Skip(1)
    .Select(c => new
    {
        c.File, c.Segment,
        Similarity = TensorPrimitives.CosineSimilarity(query.Embedding, c.Embedding)
    })
    .OrderByDescending(r => r.Similarity)
    .ToList();
```

This is the retrieval phase of audio RAG. In production, you'd store embeddings in a vector database (e.g., Qdrant, Pinecone, Azure AI Search) and query it, but the similarity math is the same.

### Synthetic Audio Patterns

The four test signals are deliberately chosen to create meaningful similarity structure:

- **440 Hz tone** — pure sine wave at concert A pitch. Predictable, narrow-band spectrum.
- **880 Hz tone** — one octave higher. Harmonically related to 440 Hz, so embedding models often see them as similar.
- **White noise** — flat spectrum with random phase. Maximally different from any tonal signal.
- **Chirp (200→2000 Hz)** — a frequency sweep. Contains tonal energy at many frequencies, so it has partial similarity to both tones but is more broadband than either.

Expected similarity ordering: `440 Hz ↔ 880 Hz > 440 Hz ↔ chirp > 440 Hz ↔ noise`.

## Key Takeaways

1. **DataIngestion is modality-agnostic.** The same `Reader → Chunker → Processor` pattern works for text _and_ audio. The generic `IngestionChunk<T>` type parameter makes this possible — `T = string` for text, `T = AudioData` for audio.

2. **The 3-layer bridge shows how abstractions compose.** DataIngestion (Layer 3) calls MEAI (Layer 2), which calls ML.NET (Layer 1). Each layer only knows about the interface above it. You can swap implementations at any layer without touching the others.

3. **`IAsyncEnumerable` streaming enables memory-efficient processing.** Large audio collections (thousands of files, hours of audio) can be processed chunk-by-chunk without loading everything into memory. Each chunk flows through the pipeline and can be indexed incrementally.

4. **This enables audio RAG.** Index audio libraries by embedding, search by acoustic similarity, retrieve relevant segments. The same vector-store infrastructure used for text RAG works for audio — because the embeddings are just `float[]` vectors regardless of modality.

## Going Further

| Resource | What It Covers |
|---|---|
| [`samples/AudioEmbeddings/`](../AudioEmbeddings/) | Embedding generation in detail — the Layer 1 + Layer 2 pipeline without DataIngestion |
| [`src/MLNet.Audio.DataIngestion/`](../../src/MLNet.Audio.DataIngestion/) | The library source: `AudioDocumentReader`, `AudioSegmentChunker`, `AudioEmbeddingChunkProcessor` |
| [`docs/meai-integration.md`](../../docs/meai-integration.md) | MEAI integration guide, including the DataIngestion section |
| [`docs/extending.md`](../../docs/extending.md) | How to create custom DataIngestion components for new modalities |
