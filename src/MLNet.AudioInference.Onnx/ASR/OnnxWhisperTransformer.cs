using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Raw ONNX Whisper speech-to-text transformer — no ORT GenAI dependency.
///
/// Uses standard HuggingFace optimum-exported ONNX models (encoder_model.onnx +
/// decoder_model_merged.onnx) and manages the full autoregressive decode loop manually:
///   1. WhisperFeatureExtractor → mel spectrogram (our primitive)
///   2. Encoder InferenceSession → encoder hidden states
///   3. Decoder loop with KV cache → token IDs
///   4. WhisperTokenizer → text with timestamps (our primitive)
///   5. TensorPrimitives → softmax for token sampling (our primitive)
///
/// This is the "show every stage" approach — maximum control, maximum use of our
/// audio primitives, and the same KV cache pattern transfers to TTS later.
///
/// Contrast with:
///   - OnnxSpeechToTextTransformer (MLNet.ASR.OnnxGenAI): ORT GenAI handles decode loop
///   - SpeechToTextClientTransformer: wraps any ISpeechToTextClient provider
/// </summary>
public sealed class OnnxWhisperTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxWhisperOptions _options;
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _decoderSession;
    private readonly WhisperFeatureExtractor _featureExtractor;
    private readonly WhisperTokenizer _whisperTokenizer;
    private readonly WhisperKvCacheManager _kvCache;
    private readonly int _hiddenDim;
    private readonly int _encoderOutputSeqLen; // 1500 for 30s chunks

    public bool IsRowToRowMapper => true;

    public OnnxWhisperTransformer(MLContext mlContext, OnnxWhisperOptions options)
    {
        _mlContext = mlContext;
        _options = options;

        // Load ONNX sessions
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _encoderSession = new InferenceSession(options.EncoderModelPath, sessionOptions);
        _decoderSession = new InferenceSession(options.DecoderModelPath, sessionOptions);

        // Auto-detect model dimensions from decoder metadata
        var (numLayers, numHeads, headDim) = WhisperKvCacheManager.DetectFromModel(_decoderSession);
        if (options.NumDecoderLayers > 0) numLayers = options.NumDecoderLayers;
        if (options.NumAttentionHeads > 0) numHeads = options.NumAttentionHeads;

        _kvCache = new WhisperKvCacheManager(numLayers, numHeads, headDim);
        _hiddenDim = numHeads * headDim;

        // Detect encoder output sequence length from encoder output metadata
        var encOutMeta = _encoderSession.OutputMetadata.Values.First();
        _encoderOutputSeqLen = encOutMeta.Dimensions.Length >= 2 ? encOutMeta.Dimensions[1] : 1500;
        // Dynamic dims may show -1; default to 1500 (Whisper standard for 30s chunks)
        if (_encoderOutputSeqLen <= 0) _encoderOutputSeqLen = 1500;

        _featureExtractor = new WhisperFeatureExtractor(numMelBins: options.NumMelBins);
        _whisperTokenizer = new WhisperTokenizer(isMultilingual: options.IsMultilingual);
    }

    /// <summary>
    /// Transcribe audio samples directly.
    /// </summary>
    public string[] Transcribe(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new string[audioInputs.Count];
        for (int i = 0; i < audioInputs.Count; i++)
            results[i] = TranscribeSingle(audioInputs[i]);
        return results;
    }

    /// <summary>
    /// Transcribe with structured timestamp output.
    /// </summary>
    public WhisperTranscriptionResult[] TranscribeWithTimestamps(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new WhisperTranscriptionResult[audioInputs.Count];
        for (int i = 0; i < audioInputs.Count; i++)
        {
            var tokenIds = RunFullPipeline(audioInputs[i]);
            var text = _whisperTokenizer.Decode(tokenIds);
            var segments = _whisperTokenizer.DecodeToSegments(tokenIds);

            results[i] = new WhisperTranscriptionResult
            {
                Text = text,
                Segments = segments.ToArray(),
                TokenIds = tokenIds,
                Language = _options.Language
            };
        }
        return results;
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var audioInputs = audioSamples
            .Select(s => new AudioData(s, _options.SampleRate))
            .ToList();
        var transcriptions = Transcribe(audioInputs);

        var outputRows = transcriptions
            .Select(t => new WhisperTranscriptionOutput { Text = t })
            .ToArray();

        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "OnnxWhisperTransformer wraps ONNX sessions and cannot be serialized.");

    public void Dispose()
    {
        _decoderSession?.Dispose();
        _encoderSession?.Dispose();
        _whisperTokenizer?.Dispose();
    }

    // ========================================================================
    // Core pipeline: audio → mel → encoder → decoder loop → tokens
    // ========================================================================

    private string TranscribeSingle(AudioData audio)
    {
        var tokenIds = RunFullPipeline(audio);
        return _whisperTokenizer.Decode(tokenIds);
    }

    private int[] RunFullPipeline(AudioData audio)
    {
        // ── Stage 1: Feature extraction (our WhisperFeatureExtractor) ──
        var features = _featureExtractor.Extract(audio);
        int numFrames = features.GetLength(0);
        int numMels = features.GetLength(1);

        // ── Stage 2: Encoder pass ──
        var encoderHidden = RunEncoder(features, numFrames, numMels);
        int encoderSeqLen = encoderHidden.Length / _hiddenDim;

        // ── Stage 3: Build decoder prompt (our WhisperTokenizer) ──
        var promptTokens = _whisperTokenizer.GetStartOfTranscriptSequence(
            _options.Language, _options.Translate);

        // ── Stage 4: Autoregressive decode loop with KV cache ──
        _kvCache.Reset();
        var allTokens = new List<int>(promptTokens);
        int[] currentInput = promptTokens;

        for (int step = 0; step < _options.MaxTokens; step++)
        {
            // Build decoder inputs with KV cache state
            var decoderInputs = _kvCache.BuildDecoderInputs(
                currentInput, encoderHidden, encoderSeqLen, _hiddenDim);

            // Run decoder
            using var decoderOutputs = _decoderSession.Run(decoderInputs);

            // Extract logits for the last position
            int nextToken = SampleNextToken(decoderOutputs);

            // Check for end of text
            if (nextToken == _whisperTokenizer.EndOfTextId)
                break;

            allTokens.Add(nextToken);

            // Update KV cache from decoder "present" outputs
            _kvCache.UpdateFromOutputs(decoderOutputs);

            // Next input is just the new token
            currentInput = [nextToken];
        }

        return allTokens.ToArray();
    }

    /// <summary>
    /// Run encoder: mel spectrogram → encoder hidden states.
    /// </summary>
    private float[] RunEncoder(float[,] features, int numFrames, int numMels)
    {
        // Reshape to [1, numMels, numFrames] for encoder input
        var melTensor = new DenseTensor<float>(new[] { 1, numMels, numFrames });
        for (int m = 0; m < numMels; m++)
            for (int f = 0; f < numFrames; f++)
                melTensor[0, m, f] = features[f, m];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_features", melTensor)
        };

        using var results = _encoderSession.Run(inputs);
        var hiddenTensor = results.First().AsTensor<float>();
        return hiddenTensor.ToArray();
    }

    /// <summary>
    /// Sample next token from decoder logits.
    /// Uses TensorPrimitives for softmax (greedy) or temperature sampling.
    /// </summary>
    private int SampleNextToken(IReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        // logits shape: [1, seq_len, vocab_size]
        var logitsTensor = outputs.First(o => o.Name == "logits").AsTensor<float>();
        var shape = logitsTensor.Dimensions.ToArray();
        int seqLen = shape[1];
        int vocabSize = shape[2];

        // Extract logits for the LAST position: [vocab_size]
        var lastLogits = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++)
            lastLogits[v] = logitsTensor[0, seqLen - 1, v];

        if (_options.Temperature <= 0f)
        {
            // Greedy: argmax using TensorPrimitives
            return TensorPrimitives.IndexOfMax(lastLogits.AsSpan());
        }
        else
        {
            // Temperature sampling using TensorPrimitives
            var scaled = new float[vocabSize];
            TensorPrimitives.Divide(lastLogits.AsSpan(), _options.Temperature, scaled.AsSpan());
            var probs = new float[vocabSize];
            TensorPrimitives.SoftMax(scaled.AsSpan(), probs.AsSpan());

            // Multinomial sample
            float rand = Random.Shared.NextSingle();
            float cumSum = 0;
            for (int i = 0; i < vocabSize; i++)
            {
                cumSum += probs[i];
                if (rand < cumSum) return i;
            }
            return vocabSize - 1;
        }
    }

    private List<float[]> ReadAudioColumn(IDataView data)
    {
        var result = new List<float[]>();
        var col = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' not found in schema.");

        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        var buffer = default(VBuffer<float>);

        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.DenseValues().ToArray());
        }

        return result;
    }
}

/// <summary>
/// Structured transcription result from raw ONNX Whisper.
/// </summary>
public class WhisperTranscriptionResult
{
    public required string Text { get; init; }
    public TranscriptionSegment[] Segments { get; init; } = [];
    public int[] TokenIds { get; init; } = [];
    public string? Language { get; init; }
}

internal class WhisperTranscriptionOutput
{
    [ColumnName("Text")]
    public string Text { get; set; } = "";
}
