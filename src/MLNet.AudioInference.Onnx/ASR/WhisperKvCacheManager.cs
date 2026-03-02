using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Manages the key-value cache for autoregressive Whisper decoding with raw ONNX Runtime.
///
/// Whisper's decoder has two types of attention that need caching:
///   1. Decoder self-attention: grows each step (new token attends to all previous tokens)
///   2. Cross-attention: fixed after first step (decoder attends to encoder output)
///
/// For each decoder layer, we maintain 4 tensors:
///   - past_key_values.{i}.decoder.key   [1, num_heads, seq_len, head_dim]
///   - past_key_values.{i}.decoder.value  [1, num_heads, seq_len, head_dim]
///   - past_key_values.{i}.encoder.key   [1, num_heads, 1500, head_dim]  (fixed)
///   - past_key_values.{i}.encoder.value  [1, num_heads, 1500, head_dim]  (fixed)
///
/// DESIGN NOTE: This pattern is reusable for ANY encoder-decoder model (not just Whisper).
/// The same KV cache management is needed for TTS models like QwenTTS and VibeVoice.
/// A future generalization could be KvCacheManager{T} or an IKvCacheManager interface.
/// </summary>
internal sealed class WhisperKvCacheManager
{
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Decoder self-attention KV cache (grows each step)
    private readonly float[]?[] _decoderKeys;
    private readonly float[]?[] _decoderValues;
    private int _decoderSeqLen;

    // Cross-attention KV cache (set once from first decode step)
    private readonly float[]?[] _encoderKeys;
    private readonly float[]?[] _encoderValues;
    private int _encoderSeqLen;
    private bool _crossAttentionCached;

    public int NumLayers => _numLayers;
    public int NumHeads => _numHeads;
    public int HeadDim => _headDim;
    public bool HasCache => _decoderSeqLen > 0;

    public WhisperKvCacheManager(int numLayers, int numHeads, int headDim)
    {
        _numLayers = numLayers;
        _numHeads = numHeads;
        _headDim = headDim;

        _decoderKeys = new float[numLayers][];
        _decoderValues = new float[numLayers][];
        _encoderKeys = new float[numLayers][];
        _encoderValues = new float[numLayers][];
    }

    /// <summary>
    /// Build decoder input list for the merged decoder model.
    /// On first call (no cache): use_cache_branch=false, no past_key_values.
    /// Subsequent calls: use_cache_branch=true, with past_key_values from previous step.
    /// </summary>
    public List<NamedOnnxValue> BuildDecoderInputs(
        int[] inputIds,
        float[] encoderHiddenStates,
        int encoderSeqLen,
        int hiddenDim)
    {
        var inputs = new List<NamedOnnxValue>();

        // input_ids: [1, seq_len]
        int seqLen = inputIds.Length;
        var idsTensor = new DenseTensor<long>(new[] { 1, seqLen });
        for (int i = 0; i < seqLen; i++)
            idsTensor[0, i] = inputIds[i];
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", idsTensor));

        // encoder_hidden_states: [1, encoder_seq_len, hidden_dim]
        var encTensor = new DenseTensor<float>(new[] { 1, encoderSeqLen, hiddenDim });
        Buffer.BlockCopy(encoderHiddenStates, 0, encTensor.Buffer.ToArray(), 0,
            encoderHiddenStates.Length * sizeof(float));
        // Manual copy since BlockCopy to DenseTensor memory is tricky
        for (int s = 0; s < encoderSeqLen; s++)
            for (int d = 0; d < hiddenDim; d++)
                encTensor[0, s, d] = encoderHiddenStates[s * hiddenDim + d];
        inputs.Add(NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encTensor));

        // use_cache_branch: scalar bool
        var useCacheTensor = new DenseTensor<bool>(new[] { 1 });
        useCacheTensor[0] = HasCache;
        inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheTensor));

        // Past key values — provide for all layers
        for (int i = 0; i < _numLayers; i++)
        {
            if (HasCache)
            {
                // Decoder self-attention KV: [1, num_heads, past_seq_len, head_dim]
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.decoder.key",
                    CreateKvTensor(_decoderKeys[i]!, _decoderSeqLen)));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.decoder.value",
                    CreateKvTensor(_decoderValues[i]!, _decoderSeqLen)));

                // Cross-attention KV: [1, num_heads, encoder_seq_len, head_dim]
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.encoder.key",
                    CreateKvTensor(_encoderKeys[i]!, _encoderSeqLen)));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.encoder.value",
                    CreateKvTensor(_encoderValues[i]!, _encoderSeqLen)));
            }
            else
            {
                // First step: provide empty KV tensors (seq_len = 0)
                var emptyDecKv = new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim });
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.decoder.key", emptyDecKv));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.decoder.value", emptyDecKv));

                var emptyEncKv = new DenseTensor<float>(new[] { 1, _numHeads, 0, _headDim });
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.encoder.key", emptyEncKv));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.encoder.value", emptyEncKv));
            }
        }

        return inputs;
    }

    /// <summary>
    /// Update KV cache from decoder outputs after a decode step.
    /// Extracts "present.{i}.decoder.key/value" and "present.{i}.encoder.key/value".
    /// </summary>
    public void UpdateFromOutputs(IReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        var outputDict = outputs.ToDictionary(o => o.Name, o => o);

        for (int i = 0; i < _numLayers; i++)
        {
            // Decoder self-attention — these GROW each step
            _decoderKeys[i] = ExtractKvData(outputDict[$"present.{i}.decoder.key"]);
            _decoderValues[i] = ExtractKvData(outputDict[$"present.{i}.decoder.value"]);

            // Cross-attention — cache once, reuse forever
            if (!_crossAttentionCached)
            {
                _encoderKeys[i] = ExtractKvData(outputDict[$"present.{i}.encoder.key"]);
                _encoderValues[i] = ExtractKvData(outputDict[$"present.{i}.encoder.value"]);
            }
        }

        // Infer sequence lengths from the shape of the stored data
        if (_decoderKeys[0] != null)
            _decoderSeqLen = _decoderKeys[0]!.Length / (_numHeads * _headDim);

        if (!_crossAttentionCached && _encoderKeys[0] != null)
        {
            _encoderSeqLen = _encoderKeys[0]!.Length / (_numHeads * _headDim);
            _crossAttentionCached = true;
        }
    }

    /// <summary>Reset cache for a new transcription.</summary>
    public void Reset()
    {
        _decoderSeqLen = 0;
        _encoderSeqLen = 0;
        _crossAttentionCached = false;
        Array.Clear(_decoderKeys);
        Array.Clear(_decoderValues);
        Array.Clear(_encoderKeys);
        Array.Clear(_encoderValues);
    }

    /// <summary>
    /// Auto-detect model dimensions by inspecting the decoder session's input metadata.
    /// Returns (numLayers, numHeads, headDim).
    /// </summary>
    public static (int numLayers, int numHeads, int headDim) DetectFromModel(InferenceSession decoderSession)
    {
        // Count layers by finding how many past_key_values.{i}.decoder.key inputs exist
        int numLayers = 0;
        while (decoderSession.InputMetadata.ContainsKey($"past_key_values.{numLayers}.decoder.key"))
            numLayers++;

        if (numLayers == 0)
            throw new InvalidOperationException(
                "Could not detect decoder layers. Expected 'past_key_values.0.decoder.key' in model inputs.");

        // Get shape from first KV input: [1, num_heads, seq_len, head_dim]
        var meta = decoderSession.InputMetadata[$"past_key_values.0.decoder.key"];
        var dims = meta.Dimensions;
        int numHeads = dims[1];
        int headDim = dims[3];

        return (numLayers, numHeads, headDim);
    }

    private DenseTensor<float> CreateKvTensor(float[] data, int seqLen)
    {
        var tensor = new DenseTensor<float>(new[] { 1, _numHeads, seqLen, _headDim });
        data.AsSpan().CopyTo(tensor.Buffer.Span);
        return tensor;
    }

    private static float[] ExtractKvData(DisposableNamedOnnxValue value)
    {
        var tensor = value.AsTensor<float>();
        return tensor.ToArray();
    }
}
