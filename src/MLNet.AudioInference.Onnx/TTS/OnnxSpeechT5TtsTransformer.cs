using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using MLNet.Audio.Core;
using MLNet.Audio.Tokenizers;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// SpeechT5 text-to-speech transformer — raw ONNX, full encoder-decoder-vocoder pipeline.
///
/// Pipeline stages (mirrors Whisper ASR in reverse):
///   1. SentencePiece tokenizer → token IDs (Microsoft.ML.Tokenizers)
///   2. Encoder session → encoder hidden states
///   3. Decoder loop with KV cache → mel spectrogram frames
///   4. Vocoder session (postnet + HiFi-GAN) → PCM waveform
///   5. Output as AudioData (our audio primitive)
///
/// Uses standard HuggingFace optimum-exported ONNX models (NeuML/txtai-speecht5-onnx).
/// KV cache management follows the same pattern as OnnxWhisperTransformer.
/// </summary>
public sealed class OnnxSpeechT5TtsTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxSpeechT5Options _options;
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _decoderSession;
    private readonly InferenceSession _vocoderSession;
    private readonly Tokenizer _tokenizer;
    private readonly float[]? _defaultSpeakerEmbedding;

    // Model dimensions (auto-detected or configured)
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _hiddenDim;
    private readonly int _speakerEmbeddingDim;

    public bool IsRowToRowMapper => true;

    public OnnxSpeechT5TtsTransformer(MLContext mlContext, OnnxSpeechT5Options options)
    {
        _mlContext = mlContext;
        _options = options;

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _encoderSession = new InferenceSession(options.EncoderModelPath, sessionOptions);
        _decoderSession = new InferenceSession(options.DecoderModelPath, sessionOptions);
        _vocoderSession = new InferenceSession(options.VocoderModelPath, sessionOptions);

        // Load SentencePiece tokenizer (with fallback for Char model type)
        var tokenizerPath = options.TokenizerModelPath
            ?? Path.Combine(Path.GetDirectoryName(options.EncoderModelPath)!, "spm_char.model");
        _tokenizer = LoadTokenizer(tokenizerPath);

        // Auto-detect decoder dimensions from model metadata
        (_numLayers, _numHeads, _headDim) = DetectDecoderDimensions(_decoderSession);
        if (options.NumDecoderLayers > 0) _numLayers = options.NumDecoderLayers;
        if (options.NumAttentionHeads > 0) _numHeads = options.NumAttentionHeads;
        _hiddenDim = _numHeads * _headDim;

        // Detect speaker embedding dimension from decoder metadata
        _speakerEmbeddingDim = DetectSpeakerEmbeddingDim(_decoderSession);

        // Load default speaker embedding
        var speakerPath = options.SpeakerEmbeddingPath
            ?? Path.Combine(Path.GetDirectoryName(options.EncoderModelPath)!, "speaker.npy");
        if (File.Exists(speakerPath))
            _defaultSpeakerEmbedding = LoadNpyFloat32(speakerPath);
    }

    /// <summary>
    /// Synthesize speech from text.
    /// </summary>
    public AudioData Synthesize(string text, float[]? speakerEmbedding = null)
    {
        var speaker = speakerEmbedding ?? _defaultSpeakerEmbedding
            ?? throw new InvalidOperationException(
                "No speaker embedding provided and no default speaker.npy found. " +
                "Provide a speaker embedding or place speaker.npy in the model directory.");

        // Stage 1: Tokenize text (Microsoft.ML.Tokenizers SentencePiece)
        var encoded = _tokenizer.EncodeToIds(text);
        var tokenIds = encoded.ToArray();

        // Stage 2: Run encoder
        var encoderHidden = RunEncoder(tokenIds);

        // Stage 3: Autoregressive decoder loop → mel frames
        var melFrames = RunDecoderLoop(encoderHidden, tokenIds.Length, speaker);

        // Stage 4: Vocoder → waveform
        var waveform = RunVocoder(melFrames);

        return new AudioData(waveform, _options.SampleRate);
    }

    /// <summary>
    /// Synthesize speech for multiple texts.
    /// </summary>
    public AudioData[] SynthesizeBatch(IReadOnlyList<string> texts, float[]? speakerEmbedding = null)
    {
        var results = new AudioData[texts.Count];
        for (int i = 0; i < texts.Count; i++)
            results[i] = Synthesize(texts[i], speakerEmbedding);
        return results;
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation: text column → audio column.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var texts = ReadTextColumn(input);
        var audioOutputs = texts.Select(t =>
        {
            var audio = Synthesize(t);
            return new TtsOutput { Audio = audio.Samples };
        }).ToArray();

        return _mlContext.Data.LoadFromEnumerable(audioOutputs);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "OnnxSpeechT5TtsTransformer wraps ONNX sessions and cannot be serialized.");

    public void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _vocoderSession?.Dispose();
    }

    // ========================================================================
    // Stage 2: Text encoder
    // ========================================================================

    private float[] RunEncoder(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;

        // input_ids: [1, seq_len]
        var idsTensor = new DenseTensor<long>(new[] { 1, seqLen });
        for (int i = 0; i < seqLen; i++)
            idsTensor[0, i] = tokenIds[i];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", idsTensor)
        };

        // Only add attention_mask if the model expects it
        if (_encoderSession.InputMetadata.ContainsKey("attention_mask"))
        {
            var maskTensor = new DenseTensor<long>(new[] { 1, seqLen });
            for (int i = 0; i < seqLen; i++)
                maskTensor[0, i] = 1;
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor));
        }

        using var results = _encoderSession.Run(inputs);
        var hiddenTensor = results.First().AsTensor<float>();
        return hiddenTensor.ToArray();
    }

    // ========================================================================
    // Stage 3: Autoregressive decoder loop with KV cache
    // ========================================================================

    private List<float[]> RunDecoderLoop(float[] encoderHidden, int encoderSeqLen, float[] speakerEmbedding)
    {
        var melFrames = new List<float[]>();
        int numMelBins = _options.NumMelBins;

        // KV cache state — same pattern as WhisperKvCacheManager
        var decoderKeys = new float[_numLayers][];
        var decoderValues = new float[_numLayers][];
        var encoderKeys = new float[_numLayers][];
        var encoderValues = new float[_numLayers][];
        int decoderCacheSeqLen = 0;
        int encoderCacheSeqLen = 0;
        bool crossAttentionCached = false;

        // Initial decoder input: zeros (start of sequence)
        var currentMelInput = new float[numMelBins];

        for (int step = 0; step < _options.MaxMelFrames; step++)
        {
            var decoderInputs = BuildDecoderInputs(
                currentMelInput, encoderHidden, encoderSeqLen, speakerEmbedding,
                decoderKeys, decoderValues, encoderKeys, encoderValues,
                decoderCacheSeqLen, encoderCacheSeqLen, crossAttentionCached,
                isFirstStep: step == 0);

            using var outputs = _decoderSession.Run(decoderInputs);
            var outputDict = outputs.ToDictionary(o => o.Name, o => o);

            // Extract mel frame from output_sequence (last frame)
            var outputSeq = outputDict.ContainsKey("output_sequence_out")
                ? outputDict["output_sequence_out"].AsTensor<float>()
                : outputs.First().AsTensor<float>();

            var melFrame = new float[numMelBins];
            var seqDims = outputSeq.Dimensions.ToArray();
            int lastFrameIdx = seqDims.Length >= 2 ? seqDims[1] - 1 : 0;
            for (int m = 0; m < numMelBins; m++)
                melFrame[m] = seqDims.Length >= 3 ? outputSeq[0, lastFrameIdx, m] : outputSeq[0, m];

            melFrames.Add(melFrame);

            // Check stop probability
            if (outputDict.ContainsKey("prob"))
            {
                var probTensor = outputDict["prob"].AsTensor<float>();
                float stopProb = probTensor.Length > 0 ? probTensor.ToArray().Last() : 0f;
                if (stopProb > _options.StopThreshold && step > 0)
                    break;
            }

            // Update KV cache from present.* outputs
            for (int i = 0; i < _numLayers; i++)
            {
                if (outputDict.TryGetValue($"present.{i}.decoder.key", out var dk))
                    decoderKeys[i] = dk.AsTensor<float>().ToArray();
                if (outputDict.TryGetValue($"present.{i}.decoder.value", out var dv))
                    decoderValues[i] = dv.AsTensor<float>().ToArray();

                if (!crossAttentionCached)
                {
                    if (outputDict.TryGetValue($"present.{i}.encoder.key", out var ek))
                        encoderKeys[i] = ek.AsTensor<float>().ToArray();
                    if (outputDict.TryGetValue($"present.{i}.encoder.value", out var ev))
                        encoderValues[i] = ev.AsTensor<float>().ToArray();
                }
            }

            // Update cache lengths
            if (decoderKeys[0] != null)
                decoderCacheSeqLen = decoderKeys[0].Length / (_numHeads * _headDim);
            if (!crossAttentionCached && encoderKeys[0] != null)
            {
                encoderCacheSeqLen = encoderKeys[0].Length / (_numHeads * _headDim);
                crossAttentionCached = true;
            }

            // Next input is the mel frame we just generated
            currentMelInput = melFrame;
        }

        return melFrames;
    }

    private List<NamedOnnxValue> BuildDecoderInputs(
        float[] melInput, float[] encoderHidden, int encoderSeqLen, float[] speakerEmbedding,
        float[][] decoderKeys, float[][] decoderValues,
        float[][] encoderKeys, float[][] encoderValues,
        int decoderCacheSeqLen, int encoderCacheSeqLen, bool crossAttentionCached,
        bool isFirstStep)
    {
        var inputs = new List<NamedOnnxValue>();
        int numMelBins = _options.NumMelBins;

        // output_sequence: [1, 1, num_mel_bins] — the previous mel frame
        var melTensor = new DenseTensor<float>(new[] { 1, 1, numMelBins });
        for (int m = 0; m < numMelBins; m++)
            melTensor[0, 0, m] = melInput[m];
        inputs.Add(NamedOnnxValue.CreateFromTensor("output_sequence", melTensor));

        // encoder_hidden_states: [1, encoder_seq_len, hidden_dim]
        var encTensor = new DenseTensor<float>(new[] { 1, encoderSeqLen, _hiddenDim });
        for (int s = 0; s < encoderSeqLen; s++)
            for (int d = 0; d < _hiddenDim; d++)
                encTensor[0, s, d] = encoderHidden[s * _hiddenDim + d];
        inputs.Add(NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encTensor));

        // encoder_attention_mask: [1, encoder_seq_len]
        var maskTensor = new DenseTensor<long>(new[] { 1, encoderSeqLen });
        for (int i = 0; i < encoderSeqLen; i++)
            maskTensor[0, i] = 1;
        inputs.Add(NamedOnnxValue.CreateFromTensor("encoder_attention_mask", maskTensor));

        // speaker_embeddings: [1, speaker_dim]
        var spkTensor = new DenseTensor<float>(new[] { 1, speakerEmbedding.Length });
        for (int i = 0; i < speakerEmbedding.Length; i++)
            spkTensor[0, i] = speakerEmbedding[i];
        inputs.Add(NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkTensor));

        // use_cache_branch: scalar bool
        bool useCache = !isFirstStep;
        var cacheTensor = new DenseTensor<bool>(new[] { 1 });
        cacheTensor[0] = useCache;
        inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", cacheTensor));

        // Past key values — same pattern as WhisperKvCacheManager
        for (int i = 0; i < _numLayers; i++)
        {
            if (useCache && decoderKeys[i] != null)
            {
                var dk = new DenseTensor<float>(new[] { 1, _numHeads, decoderCacheSeqLen, _headDim });
                decoderKeys[i].AsSpan().CopyTo(dk.Buffer.Span);
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.decoder.key", dk));

                var dv = new DenseTensor<float>(new[] { 1, _numHeads, decoderCacheSeqLen, _headDim });
                decoderValues[i].AsSpan().CopyTo(dv.Buffer.Span);
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.decoder.value", dv));

                var ek = new DenseTensor<float>(new[] { 1, _numHeads, encoderCacheSeqLen, _headDim });
                encoderKeys[i].AsSpan().CopyTo(ek.Buffer.Span);
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.encoder.key", ek));

                var ev = new DenseTensor<float>(new[] { 1, _numHeads, encoderCacheSeqLen, _headDim });
                encoderValues[i].AsSpan().CopyTo(ev.Buffer.Span);
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.encoder.value", ev));
            }
            else
            {
                // First step: empty KV tensors
                var emptyDec = new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim });
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.decoder.key", emptyDec));
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.decoder.value",
                    new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim })));

                var emptyEnc = new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim });
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.encoder.key", emptyEnc));
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{i}.encoder.value",
                    new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim })));
            }
        }

        return inputs;
    }

    // ========================================================================
    // Stage 4: Vocoder (postnet + HiFi-GAN) — mel → waveform
    // ========================================================================

    private float[] RunVocoder(List<float[]> melFrames)
    {
        int numFrames = melFrames.Count;
        int numMelBins = _options.NumMelBins;

        // Detect expected rank from vocoder model metadata
        var inputName = _vocoderSession.InputMetadata.Keys.First();
        var expectedRank = _vocoderSession.InputMetadata[inputName].Dimensions.Length;

        DenseTensor<float> melTensor;
        if (expectedRank == 2)
        {
            // [num_frames, num_mel_bins] — NeuML/txtai-speecht5-onnx format
            melTensor = new DenseTensor<float>(new[] { numFrames, numMelBins });
            for (int f = 0; f < numFrames; f++)
                for (int m = 0; m < numMelBins; m++)
                    melTensor[f, m] = melFrames[f][m];
        }
        else
        {
            // [1, num_frames, num_mel_bins] — batched format
            melTensor = new DenseTensor<float>(new[] { 1, numFrames, numMelBins });
            for (int f = 0; f < numFrames; f++)
                for (int m = 0; m < numMelBins; m++)
                    melTensor[0, f, m] = melFrames[f][m];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, melTensor)
        };

        using var results = _vocoderSession.Run(inputs);
        var waveformTensor = results.First().AsTensor<float>();

        // Flatten to 1D waveform
        var waveform = waveformTensor.ToArray();

        // Normalize to [-1, 1] using TensorPrimitives
        float maxAbs = TensorPrimitives.MaxMagnitude(waveform.AsSpan());
        if (maxAbs > 1.0f)
            TensorPrimitives.Divide(waveform.AsSpan(), maxAbs, waveform.AsSpan());

        return waveform;
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private List<string> ReadTextColumn(IDataView data)
    {
        var result = new List<string>();
        var col = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' not found in schema.");

        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);
        var buffer = default(ReadOnlyMemory<char>);

        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.ToString());
        }
        return result;
    }

    /// <summary>
    /// Auto-detect decoder dimensions from model metadata.
    /// SpeechT5 decoder has past_key_values.{i}.decoder.key inputs like Whisper.
    /// </summary>
    private static (int numLayers, int numHeads, int headDim) DetectDecoderDimensions(
        InferenceSession decoderSession)
    {
        // Count layers
        int numLayers = 0;
        while (decoderSession.InputMetadata.ContainsKey($"past_key_values.{numLayers}.decoder.key"))
            numLayers++;

        // Fallback: SpeechT5 default is 6 layers, 12 heads, 64 head_dim
        if (numLayers == 0)
            return (6, 12, 64);

        var meta = decoderSession.InputMetadata[$"past_key_values.0.decoder.key"];
        var dims = meta.Dimensions;
        int numHeads = dims[1] > 0 ? dims[1] : 12;
        int headDim = dims[3] > 0 ? dims[3] : 64;

        return (numLayers, numHeads, headDim);
    }

    /// <summary>
    /// Detect speaker embedding dimension from decoder input metadata.
    /// </summary>
    private static int DetectSpeakerEmbeddingDim(InferenceSession decoderSession)
    {
        if (decoderSession.InputMetadata.TryGetValue("speaker_embeddings", out var meta))
        {
            var dims = meta.Dimensions;
            if (dims.Length >= 2 && dims[1] > 0)
                return dims[1];
        }
        return 512; // SpeechT5 default x-vector dimension
    }

    /// <summary>
    /// Load a SentencePiece tokenizer, falling back to SentencePieceCharTokenizer
    /// if the model uses the Char type (not supported by Microsoft.ML.Tokenizers).
    /// </summary>
    private static Tokenizer LoadTokenizer(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            return SentencePieceTokenizer.Create(stream);
        }
        catch (ArgumentException ex) when (ex.Message.Contains("Char"))
        {
            // SentencePiece Char model (e.g., SpeechT5's spm_char.model)
            // not supported by Microsoft.ML.Tokenizers — use our custom implementation
            return SentencePieceCharTokenizer.Create(path);
        }
    }

    /// <summary>
    /// Load a NumPy .npy file containing a float32 array.
    /// Minimal parser for the simple case (1D or 2D float32 array).
    /// </summary>
    internal static float[] LoadNpyFloat32(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // NumPy .npy format:
        // 6 bytes magic: \x93NUMPY
        // 1 byte major version
        // 1 byte minor version
        // 2 bytes (v1) or 4 bytes (v2) header length
        // ASCII header dict (shape, dtype, order)
        // Raw data

        var magic = reader.ReadBytes(6);
        if (magic[0] != 0x93 || magic[1] != (byte)'N')
            throw new InvalidOperationException("Not a valid .npy file.");

        byte major = reader.ReadByte();
        byte minor = reader.ReadByte();

        int headerLen;
        if (major == 1)
            headerLen = reader.ReadUInt16();
        else
            headerLen = (int)reader.ReadUInt32();

        var headerBytes = reader.ReadBytes(headerLen);
        // Skip header parsing — just read remaining bytes as float32
        var remaining = stream.Length - stream.Position;
        int numFloats = (int)(remaining / 4);
        var data = new float[numFloats];
        for (int i = 0; i < numFloats; i++)
            data[i] = reader.ReadSingle();

        return data;
    }
}

internal class TtsOutput
{
    [ColumnName("Audio")]
    public float[] Audio { get; set; } = [];
}
