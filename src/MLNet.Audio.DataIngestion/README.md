# MLNet.Audio.DataIngestion

Audio data ingestion components for RAG and vector-store pipelines — proving that `Microsoft.Extensions.DataIngestion` is modality-agnostic.

This package implements the DataIngestion `Reader → Chunker → Processor` pipeline for audio. Same abstractions that handle text/PDF documents, now handling WAV audio files — chunked by time windows, enriched with embeddings, ready for vector database storage and similarity search.

> **Layer 4** of the [architecture](../../docs/architecture.md). Depends only on `Audio.Core` and `MEAI.Abstractions` — does *not* depend on ML.NET or ONNX Runtime. The embedding implementation is injected.

---

## Why This Package Exists

### The DataIngestion Pattern

[Microsoft.Extensions.DataIngestion](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion.Abstractions) defines a pipeline pattern for ingesting documents into vector stores:

```
IngestionDocumentReader → IngestionChunker<T> → IngestionChunkProcessor<T>
        ↓                        ↓                         ↓
   Read raw files         Split into chunks         Enrich chunks
   (WAV, PDF, ...)       (segments, pages)      (embeddings, metadata)
```

This pattern was designed with text and PDF in mind — the first-party implementation is [`Microsoft.Extensions.DataIngestion.Markdig`](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion.Markdig), which reads Markdown files, chunks them into paragraphs, and enriches them with text embeddings.

### But the Abstractions Are Generic

The key insight: **nothing in the DataIngestion abstractions is text-specific**. The generic type parameter `T` in `IngestionChunker<T>` and `IngestionChunkProcessor<T>` means you choose what content type flows through the pipeline. For Markdig, `T = string`. For audio, `T = AudioData`.

This package **proves** the pattern works for audio:

| Text Pipeline | Audio Pipeline |
|---------------|----------------|
| `IngestionDocumentReader` reads Markdown | `AudioDocumentReader` reads WAV files |
| `IngestionChunker<string>` splits into paragraphs | `AudioSegmentChunker` (which is `IngestionChunker<AudioData>`) splits into time windows |
| `IngestionChunkProcessor<string>` adds text embeddings | `AudioEmbeddingChunkProcessor` (which is `IngestionChunkProcessor<AudioData>`) adds audio embeddings |
| `IngestionChunk<string>` — chunk content is a text paragraph | `IngestionChunk<AudioData>` — chunk content is an audio segment |

### Why This Matters for RAG

Retrieval-Augmented Generation (RAG) isn't limited to text documents. Audio files — podcasts, meetings, lectures, music — can be:

1. **Chunked** into segments (by time window, silence detection, speaker turns)
2. **Embedded** into vector space (using audio embedding models like CLAP)
3. **Stored** in vector databases (Qdrant, Pinecone, Azure AI Search, etc.)
4. **Retrieved** by similarity search ("find audio clips similar to this one")

The DataIngestion pipeline is the standard .NET way to build this kind of ingestion flow. By implementing it for audio, we get the same DI-friendly, composable, streaming pipeline that text gets — no custom plumbing required.

---

## Key Concepts

### The IngestionChunk\<T\> Type Parameter

The most important design decision in this package is choosing `T = AudioData` for `IngestionChunk<T>`.

When you see `IngestionChunk<string>` in text pipelines, the chunk's `Content` property is a string — a paragraph or sentence of text. When you see `IngestionChunk<AudioData>` here, the chunk's `Content` property is an `AudioData` instance — the actual PCM samples of an audio segment.

This is **type-safe audio flow**. The chunk content IS the audio segment itself, not a text description of it, not a file path to it. Downstream processors receive strongly-typed `AudioData` and can operate on it directly — compute embeddings, run VAD, transcribe, etc.

```
IngestionChunk<string>     → Content is "The quick brown fox..."  (text paragraph)
IngestionChunk<AudioData>  → Content is AudioData { Samples, SampleRate, Duration }  (audio segment)
```

### The Reader → Chunker → Processor Pipeline

Each stage has a clear responsibility:

1. **Reader** (`IngestionDocumentReader`): Takes a raw `Stream` and produces an `IngestionDocument`. This is format-specific — the reader knows how to decode WAV files, handle different bit depths, resample to a target sample rate. The output is a modality-neutral document structure.

2. **Chunker** (`IngestionChunker<T>`): Takes an `IngestionDocument` and produces a stream of `IngestionChunk<T>`. This is where the content gets split into manageable pieces. For audio, that means fixed time-window segments (e.g., 2-second clips). Each chunk carries metadata about its position in the source.

3. **Processor** (`IngestionChunkProcessor<T>`): Takes a stream of chunks and enriches them. This is where embeddings get computed, transcriptions get added, or any other per-chunk processing happens. Processors chain — you can have multiple processors in sequence.

### Connection to RAG

After the pipeline runs, each `IngestionChunk<AudioData>` has:

- **Content**: The audio segment (`AudioData` with PCM samples)
- **Metadata["embedding"]**: A `float[]` vector from the embedding model
- **Metadata["startTime"]** / **Metadata["endTime"]**: Where in the source file this chunk came from
- **Metadata["sourceFile"]**: Which file it originated from

These embedded chunks can be stored in any vector database. To search, embed a query audio clip with the same embedding model and find the nearest neighbors. The timing metadata tells you exactly where in the original file the match is.

---

## Architecture & Design Decisions

### Section Metadata Bridge

`IngestionDocument` is the handoff point between the Reader and the Chunker. But `IngestionDocument` has no generic content property — it doesn't know about `AudioData`. So how does the reader pass decoded audio to the chunker?

The answer: **section metadata**. `IngestionDocumentSection` has a `Metadata` dictionary (`IDictionary<string, object>`), and that's the only extensible storage point the API provides. `AudioDocumentReader` stores the decoded `AudioData` in:

```csharp
section.Metadata["audio"] = audio;  // AudioData instance
```

The chunker retrieves it:

```csharp
if (section.Metadata.TryGetValue("audio", out var obj) && obj is AudioData a)
{
    audio = a;
}
```

This is a pragmatic bridge. The section's `Text` property is still populated with a human-readable summary (file name, duration, sample rate) for logging/debugging, but the actual audio data lives in metadata. Any custom chunker that knows the `"audio"` key convention can consume it.

### Why AudioData as Chunk Content

`IngestionChunk<T>` lets you choose `T`. We chose `AudioData` because:

- **Type safety**: Processors receive `AudioData` directly, not untyped objects
- **Self-describing**: Each chunk knows its sample rate, duration, and channel count
- **Composable**: The same `AudioData` type is used throughout the entire audio transforms ecosystem — ML.NET transforms, MEAI generators, and now DataIngestion all speak the same type
- **Rich operations**: You can resample, mix, analyze, or save any chunk as a standalone WAV file

The alternative — using `IngestionChunk<string>` and storing audio in metadata — would lose type safety and make the pipeline opaque.

### IAsyncEnumerable Streaming

Both the chunker and processor use `IAsyncEnumerable<IngestionChunk<AudioData>>` as their return type:

```csharp
public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
    IngestionDocument doc, CancellationToken cancellationToken = default)

public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
    IAsyncEnumerable<IngestionChunk<AudioData>> chunks, CancellationToken cancellationToken = default)
```

This matters for audio because files can be large. A 1-hour podcast at 16kHz mono is ~115 MB of PCM data. Streaming means:

- Chunks are produced and consumed one at a time — no need to hold all chunks in memory
- Backpressure is natural — the consumer controls the pace
- Cancellation is cooperative via `CancellationToken`
- Processors can be chained without intermediate collections

### The 3-Layer Bridge

`AudioEmbeddingChunkProcessor` demonstrates a clean layered dependency:

```
AudioEmbeddingChunkProcessor          ← Layer 4: DataIngestion
    ↓ depends on
IEmbeddingGenerator<AudioData, Embedding<float>>   ← Layer 3: MEAI Abstractions
    ↓ implemented by
OnnxAudioEmbeddingGenerator           ← Layer 1: ML.NET + ONNX Runtime
```

The processor takes an `IEmbeddingGenerator<AudioData, Embedding<float>>` in its constructor. It doesn't know or care whether the embedding comes from:

- `OnnxAudioEmbeddingGenerator` (CLAP model via ML.NET + ONNX Runtime)
- A remote API (Azure AI, OpenAI)
- A mock for testing

This is dependency inversion in action. The DataIngestion layer depends on abstractions (MEAI), not on concrete ML infrastructure. You can swap embedding providers without touching any ingestion code.

### Dependency Injection Friendly

All three components accept their dependencies through constructors:

```csharp
new AudioDocumentReader(targetSampleRate: 16000)
new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2))
new AudioEmbeddingChunkProcessor(generator)  // IEmbeddingGenerator<AudioData, Embedding<float>>
```

This makes them natural to register in a DI container:

```csharp
services.AddSingleton(new AudioDocumentReader(targetSampleRate: 16000));
services.AddSingleton(new AudioSegmentChunker(TimeSpan.FromSeconds(2)));
services.AddSingleton<IngestionChunkProcessor<AudioData>>(sp =>
    new AudioEmbeddingChunkProcessor(sp.GetRequiredService<IEmbeddingGenerator<AudioData, Embedding<float>>>()));
```

---

## API Surface

### AudioDocumentReader

**Extends**: `IngestionDocumentReader`

Reads WAV audio files into `IngestionDocument` objects.

```csharp
public class AudioDocumentReader : IngestionDocumentReader
{
    public AudioDocumentReader(int targetSampleRate = 16000);

    public override Task<IngestionDocument> ReadAsync(
        Stream stream, string name, string mediaType, CancellationToken cancellationToken = default);
}
```

**Behavior**:

- Decodes WAV via `AudioIO.LoadWav(stream)` — supports 8/16/24-bit PCM and 32-bit float formats, stereo is mixed to mono
- Resamples to `targetSampleRate` if the source sample rate differs (default: 16000 Hz, standard for speech models)
- Creates an `IngestionDocument` with one `IngestionDocumentSection`:
  - `section.Text` = human-readable summary: `"Audio file: clip.wav, Duration: 3.50s, SampleRate: 16000Hz, Samples: 56000"`
  - `section.Metadata["audio"]` = the decoded `AudioData` instance (used by downstream chunkers)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targetSampleRate` | `int` | `16000` | Resample all audio to this rate. 16kHz is standard for most speech/audio ML models. |

### AudioSegmentChunker

**Extends**: `IngestionChunker<AudioData>`

Chunks audio documents into fixed time-window segments.

```csharp
public class AudioSegmentChunker : IngestionChunker<AudioData>
{
    public AudioSegmentChunker(TimeSpan? segmentDuration = null);

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IngestionDocument doc, CancellationToken cancellationToken = default);
}
```

**Behavior**:

- Retrieves `AudioData` from `section.Metadata["audio"]` (stored by `AudioDocumentReader`)
- Splits into fixed-duration segments (default: 2 seconds). The last segment may be shorter.
- Each chunk's `Content` is a standalone `AudioData` segment with the same sample rate as the source
- Timing metadata is stored per chunk:
  - `chunk.Metadata["startTime"]` — segment start time in seconds (e.g., `"4.00"`)
  - `chunk.Metadata["endTime"]` — segment end time in seconds (e.g., `"6.00"`)
  - `chunk.Metadata["segmentIndex"]` — zero-based index of this segment
  - `chunk.Metadata["sourceFile"]` — the `IngestionDocument.Identifier` (original file name)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segmentDuration` | `TimeSpan?` | `2 seconds` | Duration of each chunk. Shorter = more chunks, finer granularity for search. Longer = fewer chunks, more context per chunk. |

**Example**: A 10-second audio file with 2-second segments produces 5 chunks:

| Chunk | startTime | endTime | segmentIndex |
|-------|-----------|---------|--------------|
| 0 | 0.00 | 2.00 | 0 |
| 1 | 2.00 | 4.00 | 1 |
| 2 | 4.00 | 6.00 | 2 |
| 3 | 6.00 | 8.00 | 3 |
| 4 | 8.00 | 10.00 | 4 |

### AudioEmbeddingChunkProcessor

**Extends**: `IngestionChunkProcessor<AudioData>`

Enriches audio chunks with vector embeddings from any `IEmbeddingGenerator<AudioData, Embedding<float>>`.

```csharp
public class AudioEmbeddingChunkProcessor : IngestionChunkProcessor<AudioData>
{
    public AudioEmbeddingChunkProcessor(
        IEmbeddingGenerator<AudioData, Embedding<float>> generator);

    public override async IAsyncEnumerable<IngestionChunk<AudioData>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<AudioData>> chunks, CancellationToken cancellationToken = default);
}
```

**Behavior**:

- Iterates over the input chunk stream
- For each chunk, calls `_generator.GenerateAsync([chunk.Content])` to compute the embedding vector
- Stores the result in `chunk.Metadata["embedding"]` as `float[]`
- Yields the enriched chunk downstream

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator` | `IEmbeddingGenerator<AudioData, Embedding<float>>` | Any MEAI-compatible audio embedding generator. In practice, this is `OnnxAudioEmbeddingGenerator` from `MLNet.AudioInference.Onnx`, but the processor doesn't know or care. |

---

## How It Fits — Layered Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  Layer 4: DataIngestion                                           │
│  AudioDocumentReader → AudioSegmentChunker                        │
│                         → AudioEmbeddingChunkProcessor            │
│                                                                   │
│  Depends on: Audio.Core, MEAI.Abstractions,                       │
│              DataIngestion.Abstractions                            │
│  Does NOT depend on: ML.NET, ONNX Runtime                         │
├───────────────────────────────────────────────────────────────────┤
│  Layer 3: MEAI Abstractions                                       │
│  IEmbeddingGenerator<AudioData, Embedding<float>>                 │
│  (interface only — implementation injected)                       │
├───────────────────────────────────────────────────────────────────┤
│  Layer 2: ML.NET + ONNX (MLNet.AudioInference.Onnx)              │
│  OnnxAudioEmbeddingGenerator — implements the MEAI interface      │
│  OnnxAudioEmbeddingTransformer — the actual ML.NET transform      │
├───────────────────────────────────────────────────────────────────┤
│  Layer 1: Audio.Core                                              │
│  AudioData, AudioIO, MelSpectrogramExtractor                      │
└───────────────────────────────────────────────────────────────────┘
```

This package (Layer 4) only takes dependencies on:

- **Audio.Core** (Layer 1) — for the `AudioData` type and `AudioIO.LoadWav`
- **MEAI Abstractions** (Layer 3 interfaces) — for `IEmbeddingGenerator<AudioData, Embedding<float>>`
- **DataIngestion.Abstractions** — for the `IngestionDocumentReader`, `IngestionChunker<T>`, `IngestionChunkProcessor<T>` base classes

It does **not** reference ML.NET, ONNX Runtime, or any concrete model infrastructure. The embedding generator is injected at runtime, keeping this package lightweight and testable.

---

## Usage

### Full Read → Chunk → Embed → Search Pipeline

```csharp
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using MLNet.Audio.Core;
using MLNet.Audio.DataIngestion;

// --- Setup: create the pipeline components ---

// Reader: decodes WAV files, resamples to 16kHz
var reader = new AudioDocumentReader(targetSampleRate: 16000);

// Chunker: 2-second fixed time-window segments
var chunker = new AudioSegmentChunker(segmentDuration: TimeSpan.FromSeconds(2));

// Processor: enriches chunks with embeddings
// The generator is injected — could be OnnxAudioEmbeddingGenerator, a mock, etc.
IEmbeddingGenerator<AudioData, Embedding<float>> generator = /* your embedding generator */;
var processor = new AudioEmbeddingChunkProcessor(generator);

// --- Pipeline: Read → Chunk → Embed ---

using var stream = File.OpenRead("podcast-episode.wav");
var doc = await reader.ReadAsync(stream, "podcast-episode.wav", "audio/wav");

// Chunk the document into segments (IAsyncEnumerable)
var chunks = chunker.ProcessAsync(doc);

// Enrich each chunk with an embedding vector (IAsyncEnumerable)
var embeddedChunks = processor.ProcessAsync(chunks);

// --- Consume: store in vector DB or search ---

var storedChunks = new List<(AudioData Audio, float[] Embedding, double StartTime, double EndTime)>();

await foreach (var chunk in embeddedChunks)
{
    var embedding = (float[])chunk.Metadata["embedding"];
    var startTime = double.Parse((string)chunk.Metadata["startTime"]);
    var endTime = double.Parse((string)chunk.Metadata["endTime"]);

    storedChunks.Add((chunk.Content, embedding, startTime, endTime));

    // In production: store in Qdrant, Pinecone, Azure AI Search, etc.
    // await vectorDb.UpsertAsync(chunk.Content, embedding, metadata);
}

// --- Search: find similar audio segments ---

var queryAudio = AudioIO.LoadWav("query-clip.wav");
var queryEmbedding = await generator.GenerateAsync([queryAudio]);
var queryVector = queryEmbedding[0].Vector.ToArray();

// Cosine similarity search (simplified — use your vector DB in production)
var results = storedChunks
    .Select(c => (Chunk: c, Similarity: CosineSimilarity(queryVector, c.Embedding)))
    .OrderByDescending(r => r.Similarity)
    .Take(5);

foreach (var (chunk, similarity) in results)
{
    Console.WriteLine($"  [{chunk.StartTime:F2}s - {chunk.EndTime:F2}s] similarity: {similarity:F4}");
}
```

### With OnnxAudioEmbeddingGenerator

```csharp
using Microsoft.ML;
using MLNet.AudioInference.Onnx;

// Create the ML.NET-based embedding generator (Layer 1 → Layer 3)
var mlContext = new MLContext();
var embeddingOptions = new OnnxAudioEmbeddingOptions
{
    ModelPath = "models/clap/onnx/model.onnx",
    FeatureExtractor = new MelSpectrogramExtractor(16000),
    Pooling = AudioPoolingStrategy.MeanPooling,
    Normalize = true
};
var estimator = mlContext.Transforms.OnnxAudioEmbedding(embeddingOptions);
var transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<AudioInput>()));
IEmbeddingGenerator<AudioData, Embedding<float>> generator = new OnnxAudioEmbeddingGenerator(transformer);

// Now use with DataIngestion (Layer 4) — the processor doesn't know about ML.NET
var processor = new AudioEmbeddingChunkProcessor(generator);
```

---

## Dependencies

| Package | Version | What It Provides |
|---------|---------|------------------|
| [`Microsoft.Extensions.DataIngestion.Abstractions`](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion.Abstractions) | 10.3.0-preview.1 | `IngestionDocumentReader`, `IngestionChunker<T>`, `IngestionChunkProcessor<T>`, `IngestionDocument`, `IngestionDocumentSection`, `IngestionChunk<T>` |
| [`Microsoft.Extensions.AI.Abstractions`](https://www.nuget.org/packages/Microsoft.Extensions.AI.Abstractions) | 10.3.0 | `IEmbeddingGenerator<TInput, TEmbedding>`, `Embedding<T>` |
| [`MLNet.Audio.Core`](../MLNet.Audio.Core) | (project ref) | `AudioData` (PCM audio type), `AudioIO` (WAV decode/encode/resample) |

**Not referenced** (injected at runtime): `MLNet.AudioInference.Onnx`, `Microsoft.ML`, `Microsoft.ML.OnnxRuntime`.

---

## API Discovery Notes

The `Microsoft.Extensions.DataIngestion.Abstractions` API has some signatures that differ from what you might expect. These are the actual signatures we discovered and implemented against:

### IngestionDocumentReader

```csharp
// ReadAsync takes a Stream, not a file path
public abstract Task<IngestionDocument> ReadAsync(
    Stream stream, string name, string mediaType, CancellationToken cancellationToken = default);
```

- `stream` — the raw file stream (caller opens/closes it)
- `name` — a display name or identifier for the document (e.g., `"audio.wav"`)
- `mediaType` — MIME type (e.g., `"audio/wav"`, `"text/markdown"`)

### IngestionChunker\<T\> and IngestionChunkProcessor\<T\>

```csharp
// The chunker's method is called ProcessAsync (not ChunkAsync)
public abstract IAsyncEnumerable<IngestionChunk<T>> ProcessAsync(
    IngestionDocument doc, CancellationToken cancellationToken = default);

// The processor also uses ProcessAsync, taking the chunk stream
public abstract IAsyncEnumerable<IngestionChunk<T>> ProcessAsync(
    IAsyncEnumerable<IngestionChunk<T>> chunks, CancellationToken cancellationToken = default);
```

Both the chunker and the processor use `ProcessAsync` as the method name. The chunker takes an `IngestionDocument`, the processor takes `IAsyncEnumerable<IngestionChunk<T>>`.

### Constructors

```csharp
// IngestionDocument takes an identifier string
var doc = new IngestionDocument("podcast-episode.wav");

// IngestionChunk<T> takes content, document, and optional section ID
var chunk = new IngestionChunk<AudioData>(segmentAudio, doc, "segment-0");

// IngestionDocumentSection takes an identifier string
var section = new IngestionDocumentSection("audio:podcast-episode.wav");
```

### Metadata Storage

Both `IngestionDocumentSection` and `IngestionChunk<T>` have a `Metadata` property of type `IDictionary<string, object>`. This is the extensibility point for passing modality-specific data between pipeline stages.

---

## The Key Takeaway

**DataIngestion is not just for text — and this package proves it.**

The same `Reader → Chunker → Processor` pipeline that ingests Markdown documents into a vector store works for audio files. The generic `T` in `IngestionChunk<T>` is the escape hatch from text-only thinking: set `T = AudioData` and you have type-safe, streaming, DI-friendly audio ingestion that plugs into the same RAG infrastructure as text.

If you can embed it, you can ingest it. If you can ingest it, you can search it.
