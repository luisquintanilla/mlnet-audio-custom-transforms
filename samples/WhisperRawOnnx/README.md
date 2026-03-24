# Raw ONNX Whisper: Autoregressive Decoding and KV Cache Deep Dive

This is the **deep-dive** sample in the repo. It exposes every stage of the Whisper speech-to-text pipeline — from mel spectrogram extraction through the encoder, autoregressive decoder loop with manual KV cache management, token sampling, and tokenizer decoding. If you want to truly understand how encoder-decoder speech recognition works under the hood, start here.

---

## What You'll Learn

| Concept | Why It Matters |
|---------|---------------|
| **Encoder-decoder architecture** | The foundational pattern behind Whisper, T5, BART, and most sequence-to-sequence models |
| **KV cache mechanics** | The optimization that makes autoregressive generation practical (O(n) vs O(n²)) |
| **Token sampling strategies** | Greedy (argmax) vs temperature sampling — how models choose the next token |
| **Whisper's special tokens** | Language codes, task tokens (`<\|transcribe\|>`), and timestamp tokens (`<\|0.00\|>` – `<\|30.00\|>`) |
| **ML.NET integration** | How raw ONNX inference composes with ML.NET's `Fit`/`Transform` pipeline |

---

## The Concept: Autoregressive Encoder-Decoder Models

### What Is an Encoder-Decoder?

An encoder-decoder model has two distinct halves:

1. **Encoder** — Processes the **entire input** in a single forward pass and produces a sequence of hidden states (contextual representations). For Whisper, the input is a mel spectrogram and the output is a matrix of shape `[1, 1500, hidden_dim]`.

2. **Decoder** — Generates output tokens **one at a time**. Each step is conditioned on:
   - The encoder's output (via cross-attention — "what did the audio say?")
   - All previously generated tokens (via self-attention — "what have I said so far?")

This is fundamentally different from **encoder-only** models (like BERT or AST) that process the entire input in parallel and produce a fixed output. Encoder-decoder models produce **variable-length sequential output** where each token depends on every token before it.

### Why "Autoregressive"?

The term means each output feeds back as input to the next step:

```
Step 1:  [<|startoftranscript|>, <|en|>, <|transcribe|>]  →  decoder  →  "Hello"
Step 2:  [..., "Hello"]                                     →  decoder  →  " world"
Step 3:  [..., "Hello", " world"]                            →  decoder  →  <|endoftext|>
```

The decoder **cannot** generate token N+1 without first having generated tokens 1 through N. This sequential dependency is what makes caching critical.

### The Whisper Pipeline

```
Audio (PCM float[])
  │
  ▼
WhisperFeatureExtractor          ← Audio primitive (Layer 0)
  │  FFT (400 samples / 25ms window, 160 hop / 10ms)
  │  → log-mel spectrogram [3000 frames × 80 bins]
  │  Padded/truncated to exactly 30 seconds
  ▼
Encoder ONNX Session             ← Single forward pass
  │  Input:  [1, 80, 3000]
  │  Output: [1, 1500, hidden_dim]  (encoder hidden states)
  ▼
Decoder Loop with KV Cache       ← Token-by-token autoregressive loop
  │  Each step:
  │    input_ids + encoder_hidden_states + past KV cache
  │    → logits → sample token → update cache
  │  Repeats until <|endoftext|> or MaxTokens reached
  ▼
WhisperTokenizer                 ← Tokenizer primitive (Layer 0)
  │  Token IDs → text
  │  Timestamp tokens → segments with TimeSpan boundaries
  ▼
Transcription text / segments
```

### Abstraction Layers

```
┌─────────────────────────────────────────────────────────┐
│ MEAI: ISpeechToTextClient (optional wrapper)             │
│   OnnxWhisperSpeechToTextClient — provider-agnostic API  │
├─────────────────────────────────────────────────────────┤
│ ML.NET: ITransformer / IEstimator<T>                     │
│   OnnxWhisperTransformer — composable pipeline           │
│   Exposes Transcribe() and TranscribeWithTimestamps()    │
├─────────────────────────────────────────────────────────┤
│ System.Numerics.Tensors: TensorPrimitives               │
│   SoftMax (temperature sampling), IndexOfMax (greedy)    │
│   SIMD-accelerated token selection in decoder loop       │
├─────────────────────────────────────────────────────────┤
│ ONNX Runtime: InferenceSession                           │
│   Encoder session + Decoder session (with KV cache)      │
├─────────────────────────────────────────────────────────┤
│ Audio Primitives (MLNet.Audio.Core)                      │
│   WhisperFeatureExtractor (mel spectrogram)              │
│   WhisperTokenizer (BPE + 1700 special tokens)           │
└─────────────────────────────────────────────────────────┘
```

This is the deepest sample in the repo — it touches every layer from raw audio to decoded text. The `TensorPrimitives` usage is architecturally significant here: the decoder loop calls `TensorPrimitives.SoftMax()` and `TensorPrimitives.IndexOfMax()` on every generated token, making SIMD acceleration a real performance factor.

### KV Cache Explained

The KV (Key-Value) cache is the single most important optimization for autoregressive generation. Here's why:

**Without cache (naïve approach):**

At step N, the decoder must compute attention over all N previous tokens. This means:
- Step 1: attend to 1 token
- Step 2: attend to 2 tokens (recomputing step 1's keys/values)
- Step 3: attend to 3 tokens (recomputing steps 1–2's keys/values)
- Step N: attend to N tokens → **O(N²)** total computation across all steps

**With cache:**

Store the Key and Value projection matrices from each step. At step N, only compute the **new** token's K/V and concatenate with the cached ones:
- Step 1: compute K/V for token 1, store in cache
- Step 2: compute K/V for token 2 only, append to cache, attend to all 2
- Step N: compute K/V for token N only, append to cache → **O(N)** total new computation

**Cache structure in Whisper:**

Each decoder layer has **4 cache tensors**:

| Tensor | Shape | Behavior |
|--------|-------|----------|
| `past_key_values.{i}.decoder.key` | `[1, num_heads, seq_len, head_dim]` | **Grows** each step (new token appended) |
| `past_key_values.{i}.decoder.value` | `[1, num_heads, seq_len, head_dim]` | **Grows** each step (new token appended) |
| `past_key_values.{i}.encoder.key` | `[1, num_heads, 1500, head_dim]` | **Fixed** after step 1 (encoder output doesn't change) |
| `past_key_values.{i}.encoder.value` | `[1, num_heads, 1500, head_dim]` | **Fixed** after step 1 (encoder output doesn't change) |

The `use_cache_branch` flag controls this:
- **Step 1** (`use_cache_branch = false`): Full computation. Decoder processes all prompt tokens, encoder cross-attention keys/values are computed and cached.
- **Steps 2+** (`use_cache_branch = true`): Incremental. Only the new token is processed; past KV tensors are passed in and extended.

`WhisperKvCacheManager` handles all of this automatically — detecting the architecture from the ONNX model metadata and building/updating the cache tensors each step.

### Token Sampling Strategies

After the decoder produces logits (raw scores for every token in the vocabulary), we need to pick one:

**Greedy (argmax) — `Temperature = 0`:**
```
logits = [1.2, 5.8, 0.3, -1.0, ...]
selected = argmax(logits) = index 1
```
Always picks the highest-probability token. Fast, deterministic, and usually good enough for speech recognition. Implemented via `TensorPrimitives.IndexOfMax`.

**Temperature sampling — `Temperature > 0`:**
```
scaled_logits = logits / temperature
probabilities = softmax(scaled_logits)    // via TensorPrimitives.SoftMax
selected = multinomial_sample(probabilities)
```
- **Temperature < 1** → sharper distribution (more confident, less random)
- **Temperature = 1** → original model distribution
- **Temperature > 1** → flatter distribution (more random, more diverse)

Temperature sampling is useful when you want varied outputs (e.g., running transcription multiple times to find the best result) or when the model is uncertain and greedy decoding gets stuck in repetitive loops.

---

## What This Sample Demonstrates

### 1. Direct Transcription — `Transcribe()`

The simplest usage: audio in, text out.

```csharp
var results = transformer.Transcribe([audio]);
Console.WriteLine(results[0]); // "Hello world"
```

Internally runs the full pipeline: mel extraction → encoder → decoder loop → tokenizer decode.

### 2. Transcription with Timestamps — `TranscribeWithTimestamps()`

Whisper has **1501 special timestamp tokens** (`<|0.00|>` through `<|30.00|>` in 0.02s increments). The decoder naturally produces these tokens to mark when each word or phrase occurs in the audio.

```csharp
var detailed = transformer.TranscribeWithTimestamps([audio]);

Console.WriteLine(detailed[0].Text);       // Full text
Console.WriteLine(detailed[0].Language);   // "en"
Console.WriteLine(detailed[0].TokenIds);   // Raw token IDs

foreach (var seg in detailed[0].Segments)
    Console.WriteLine($"[{seg.Start:mm\\:ss\\.ff} → {seg.End:mm\\:ss\\.ff}] {seg.Text}");
// [00:00.00 → 00:02.40] Hello world
// [00:02.40 → 00:05.00] this is a test
```

`WhisperTokenizer.DecodeToSegments()` parses the token stream, pairing timestamp tokens as segment boundaries and extracting the text between them.

### 3. ML.NET Pipeline — `Fit`/`Transform`

Compose Whisper with other ML.NET transforms:

```csharp
var pipeline = mlContext.Transforms.OnnxWhisper(new OnnxWhisperOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    Language = "en",
});

// Compose with downstream transforms
pipeline.Append(mlContext.Transforms.OnnxTextEmbedding(...));
pipeline.Append(mlContext.Transforms.OnnxTextClassification(...));
```

This is the same Whisper logic wrapped as an `IEstimator<ITransformer>`, fitting naturally into ML.NET's pipeline model.

### 4. Batch Transcription

Process multiple audio files in one call:

```csharp
var batch = new[] { audio1, audio2, audio3 };
var results = transformer.Transcribe(batch);
// results[0], results[1], results[2]
```

Each audio input runs through the full encoder-decoder pipeline independently.

### 5. Temperature Sampling

Compare deterministic vs stochastic decoding:

```csharp
// Greedy: TensorPrimitives.IndexOfMax on logits
var greedy = new OnnxWhisperOptions { ..., Temperature = 0f };

// Temperature sampling: TensorPrimitives.SoftMax + multinomial sample
var sampling = new OnnxWhisperOptions { ..., Temperature = 0.6f };
```

---

## Why This Exists Alongside ORT GenAI

The repo provides **three ASR approaches**. Here's when to choose raw ONNX:

| | Provider-Agnostic | ORT GenAI | **Raw ONNX (this sample)** |
|---|---|---|---|
| **Backend** | Any `ISpeechToTextClient` | ORT GenAI runtime | OnnxRuntime only |
| **Model format** | N/A (cloud API) | ORT GenAI export | Standard ONNX (optimum-cli) |
| **Decoder loop** | Handled by provider | Handled by ORT GenAI | **You manage it** |
| **KV cache** | Hidden | Hidden | **Explicit** |
| **Custom sampling** | ✗ | Limited | **Full control** |
| **Dependencies** | MEAI client | Platform-specific GenAI pkg | OnnxRuntime only |

**Choose raw ONNX when you:**
- Need **custom sampling strategies** (temperature, top-k, nucleus, beam search)
- Want to **understand or debug** every step of the encoder-decoder pipeline
- Are using the **audio primitives directly** (`WhisperFeatureExtractor`, `WhisperTokenizer`, `WhisperKvCacheManager`)
- Have a model that **isn't available** in ORT GenAI format (only standard ONNX)
- Are doing **educational or research** work on autoregressive models

**Choose ORT GenAI** ([WhisperTranscription](../WhisperTranscription/)) when you want the simplest local transcription with minimal code.

**Choose provider-agnostic** ([SpeechToText](../SpeechToText/)) when you want cloud-scale transcription or don't want to manage models locally.

---

## Prerequisites

### 1. Export a Whisper model

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base models/whisper-base/
```

### 2. Verify the exported directory

```
models/whisper-base/
├── encoder_model.onnx                # Encoder (mel → hidden states)
├── decoder_model_merged.onnx         # Merged decoder (prefill + decode-with-past)
├── config.json                       # Model architecture (layers, heads, dims)
├── tokenizer.json                    # Vocabulary and special tokens
├── generation_config.json            # Default generation parameters
└── preprocessor_config.json          # Feature extraction config
```

The `decoder_model_merged.onnx` is key — it's a single model that handles both the initial full-computation step and subsequent incremental steps via the `use_cache_branch` input.

---

## Running It

### With model files (full transcription)

```bash
cd samples/WhisperRawOnnx

# Default: uses models/whisper-base directory
dotnet run -- "models/whisper-base"

# With a specific audio file
dotnet run -- "models/whisper-base" "path/to/audio.wav"
```

### Without model files (pattern demonstration)

```bash
cd samples/WhisperRawOnnx
dotnet run
```

Shows API patterns for all three ASR approaches and generates a synthetic mel spectrogram to demonstrate feature extraction — no model download required.

### Exporting the Model

Raw ONNX Whisper requires encoder and decoder models exported via [Optimum](https://huggingface.co/docs/optimum):

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx --model openai/whisper-base models/whisper-base/
```

This creates:
| File | Purpose | Size |
|------|---------|------|
| `encoder_model.onnx` | Audio features → hidden states | ~90 MB |
| `decoder_model_merged.onnx` | Autoregressive token generation | ~180 MB |
| `config.json` | Model configuration | <1 KB |
| `tokenizer.json` | BPE vocabulary + special tokens | ~2 MB |

---

## Code Walkthrough

### OnnxWhisperOptions

```csharp
var options = new OnnxWhisperOptions
{
    EncoderModelPath = encoderPath,     // Path to encoder_model.onnx
    DecoderModelPath = decoderPath,     // Path to decoder_model_merged.onnx
    Language = "en",                    // ISO language code (99 languages supported)
    MaxTokens = 256,                    // Maximum tokens to generate before stopping
    NumMelBins = 80,                    // 80 for whisper-base/small, 128 for large-v3
    Temperature = 0f,                   // 0 = greedy argmax, >0 = temperature sampling
    SampleRate = 16000,                 // Audio sample rate (Whisper expects 16kHz)
};
```

`NumDecoderLayers` and `NumAttentionHeads` can be set explicitly but are **auto-detected** by `WhisperKvCacheManager.DetectFromModel()` — it inspects the ONNX model's input metadata, finding inputs named `past_key_values.{i}.decoder.key` and parsing the tensor shapes to determine the number of layers, attention heads, and head dimension.

### WhisperKvCacheManager.DetectFromModel()

This is one of the most clever parts of the implementation. Instead of hardcoding model architecture parameters, the cache manager:

1. Opens the decoder ONNX model's metadata
2. Scans for input names matching `past_key_values.{i}.decoder.key`
3. Counts the layers (number of distinct `{i}` values)
4. Reads the tensor shape to get `num_heads` and `head_dim`

This means the same code works for whisper-tiny (4 layers, 6 heads) through whisper-large-v3 (32 layers, 20 heads) without any configuration changes.

### The Decoder Loop (Simplified)

```
1. Build initial prompt: [<|startoftranscript|>, <|en|>, <|transcribe|>]
2. Run encoder once → encoder_hidden_states
3. For each step until <|endoftext|> or MaxTokens:
   a. kvCache.BuildDecoderInputs(tokens, encoder_hidden_states)
      - Step 1: use_cache_branch=false, empty KV tensors
      - Step 2+: use_cache_branch=true, past KV tensors
   b. Run decoder session → logits + present KV tensors
   c. Sample next token from logits:
      - Temperature=0: TensorPrimitives.IndexOfMax(logits)
      - Temperature>0: TensorPrimitives.SoftMax(logits/T) → multinomial sample
   d. kvCache.UpdateFromOutputs(decoder_outputs)
      - Extract present.{i}.decoder.key/value (grows each step)
      - Extract present.{i}.encoder.key/value (cached once, reused)
   e. Append token to sequence
4. WhisperTokenizer.Decode(tokens) → text
```

### Timestamp Tokens

Whisper's vocabulary includes 1501 timestamp tokens covering 0.00s to 30.00s in 0.02s increments. During decoding, the model naturally interleaves these with text tokens:

```
<|0.00|> Hello world <|2.40|> <|2.40|> this is a test <|5.00|>
```

`WhisperTokenizer.DecodeToSegments()` parses this stream:
- Identifies timestamp token pairs as segment boundaries
- Extracts text tokens between each pair
- Returns `TranscriptionSegment[]` with `Start` (TimeSpan), `End` (TimeSpan), and `Text`

### Three ASR Approaches Comparison

The sample's `ShowApiPatterns()` method demonstrates all three approaches side-by-side:

```csharp
// 1. Provider-agnostic (any ISpeechToTextClient)
var pipeline = mlContext.Transforms.SpeechToText(sttClient);

// 2. ORT GenAI (easiest local option)
var pipeline = mlContext.Transforms.OnnxSpeechToText(ortGenAiOptions);

// 3. Raw ONNX (full control — THIS sample)
var pipeline = mlContext.Transforms.OnnxWhisper(rawOnnxOptions);
```

---

## Key Takeaways

1. **Encoder-decoder is fundamentally different from encoder-only.** Encoder-only models (BERT, AST) process input in parallel and produce fixed-size output. Encoder-decoder models (Whisper, T5) generate variable-length output sequentially — each token depends on all previous tokens.

2. **KV cache is a critical optimization.** Without it, autoregressive generation is O(n²) in sequence length. With it, each new step is O(n) — the cache stores previously computed attention keys and values so they don't need to be recomputed.

3. **Whisper uses special tokens extensively.** Language (`<|en|>`), task (`<|transcribe|>`, `<|translate|>`), timestamps (`<|0.00|>` – `<|30.00|>`), and control tokens (`<|startoftranscript|>`, `<|endoftext|>`) are all part of the vocabulary and generated naturally by the decoder.

4. **Raw ONNX gives full control but requires understanding the decoder loop.** You manage the KV cache, choose the sampling strategy, and process the token stream. This is more work than ORT GenAI but gives you complete visibility and customizability.

5. **All three ASR approaches are available in this repo.** Provider-agnostic for cloud APIs, ORT GenAI for easy local inference, and raw ONNX for full control — choose based on your needs.

---

## Troubleshooting

### "ONNX model not found" error
Raw ONNX Whisper requires models exported via Optimum (not the standard HuggingFace format). The model directory must contain both `encoder_model.onnx` and `decoder_model_merged.onnx`.

### Transcription is empty or just special tokens
- The decoder may be generating only `<|endoftext|>` immediately. This usually means:
  - Audio is silence or too quiet (try a real speech recording)
  - The mel spectrogram is all zeros (check audio loading)
  - Model files are corrupted (re-export with Optimum)

### Transcription contains repeated words
This is a known issue with greedy decoding in small Whisper models. Try:
- Use temperature sampling: higher temperature (0.5-0.8) reduces repetition
- Use a larger model (whisper-small or whisper-medium)
- The KV cache ensures efficient decoding but doesn't prevent repetition — that's a model behavior

### "KV cache dimension mismatch" or shape errors
- The `WhisperKvCacheManager` auto-detects dimensions from the ONNX model metadata
- If you exported with a different tool (not Optimum), the cache tensor names may differ
- Ensure `decoder_model_merged.onnx` contains both the initial (no-past) and with-past paths

### When should I use this vs ORT GenAI vs cloud API?

| Approach | Use When |
|----------|----------|
| **Cloud API** (Azure, OpenAI) | Production; highest accuracy; don't want to manage models |
| **ORT GenAI** ([WhisperTranscription](../WhisperTranscription/)) | Local inference with simple API; "just works" |
| **Raw ONNX** (this sample) | Learning how Whisper works; custom sampling strategies; debugging; non-standard models |

This sample is the **educational deep-dive**. If you just want transcription, start with [WhisperTranscription](../WhisperTranscription/).

---

## Going Further

- **[TextToSpeech](../TextToSpeech/)** — The reverse direction: text → encoder → decoder (with KV cache) → vocoder → audio. Same encoder-decoder pattern, different modality. SpeechT5 uses the same KV cache concepts.

- **[WhisperTranscription](../WhisperTranscription/)** — Same speech-to-text task with ORT GenAI handling the decoder loop. Compare the two to see what ORT GenAI abstracts away.

- **[Architecture docs](../../docs/architecture.md)** — The layered design (Layer 0 primitives → Layer 1 transforms → Layer 2 MEAI) and the KV cache section explaining how `WhisperKvCacheManager` fits into the broader system.

- **[SpeechToText](../SpeechToText/)** — Provider-agnostic ASR using `ISpeechToTextClient`. No local models, no decoder loop — just send audio and get text back from any provider.
