using Microsoft.ML.OnnxRuntime;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// EnCodec audio codec tokenizer — converts audio waveforms to/from discrete codes
/// using Meta's EnCodec neural audio codec via ONNX.
///
/// This is the concrete implementation of AudioCodecTokenizer for EnCodec models.
/// It demonstrates how audio can be "tokenized" into discrete representations,
/// just like text tokenizers convert strings to token IDs.
///
/// EnCodec uses Residual Vector Quantization (RVQ) with multiple codebooks:
///   - 1.5 kbps: 2 codebooks
///   - 3.0 kbps: 4 codebooks
///   - 6.0 kbps: 8 codebooks
///   - 12.0 kbps: 16 codebooks
///   - 24.0 kbps: 32 codebooks
///
/// Each codebook has 1024 entries, producing codes in [0, 1023].
/// At 24kHz with 75Hz frame rate: 320 samples per frame.
///
/// Usage:
///   var tokenizer = new EnCodecTokenizer("encodec.onnx");
///   int[][] codes = tokenizer.Encode(audio);      // audio → discrete codes
///   AudioData reconstructed = tokenizer.Decode(codes);  // codes → audio
///   int[] flat = tokenizer.EncodeFlat(audio);      // for LM input (MusicGen, etc.)
///
/// MODELS:
///   - facebook/encodec_24khz (24kHz, mono, 1.5-24 kbps)
///   - facebook/encodec_48khz (48kHz, stereo, 3-24 kbps)
///
/// DESIGN NOTE: This prototypes what could become a first-class audio tokenizer in
/// Microsoft.ML.Tokenizers, alongside BpeTokenizer and SentencePieceTokenizer.
/// The key insight: tokenization is not text-specific — it's the conversion of
/// continuous signals to discrete representations, and that concept generalizes.
/// </summary>
public class EnCodecTokenizer : AudioCodecTokenizer
{
    private readonly InferenceSession? _encoderSession;
    private readonly InferenceSession? _decoderSession;
    private readonly int _numCodebooks;
    private readonly int _sampleRate;

    public override int NumCodebooks => _numCodebooks;
    public override int CodebookSize => 1024;
    public override int SampleRate => _sampleRate;
    public override int HopLength => _sampleRate == 24000 ? 320 : 640;

    /// <summary>
    /// Create an EnCodec tokenizer from ONNX model files.
    /// </summary>
    /// <param name="encoderModelPath">Path to the EnCodec encoder ONNX model (audio → codes).</param>
    /// <param name="decoderModelPath">Path to the EnCodec decoder ONNX model (codes → audio).</param>
    /// <param name="numCodebooks">Number of RVQ codebooks to use. Default: 8 (6 kbps at 24kHz).</param>
    /// <param name="sampleRate">Sample rate. Default: 24000.</param>
    public EnCodecTokenizer(
        string? encoderModelPath = null,
        string? decoderModelPath = null,
        int numCodebooks = 8,
        int sampleRate = 24000)
    {
        _numCodebooks = numCodebooks;
        _sampleRate = sampleRate;

        if (encoderModelPath != null && File.Exists(encoderModelPath))
            _encoderSession = new InferenceSession(encoderModelPath);

        if (decoderModelPath != null && File.Exists(decoderModelPath))
            _decoderSession = new InferenceSession(decoderModelPath);
    }

    /// <summary>
    /// Encode audio to discrete codec codes.
    /// Shape: [NumCodebooks, numFrames] where each value is in [0, 1023].
    /// </summary>
    public override int[][] Encode(AudioData audio)
    {
        if (_encoderSession == null)
            throw new InvalidOperationException("Encoder model not loaded.");

        // Resample if needed
        if (audio.SampleRate != _sampleRate)
            audio = AudioIO.Resample(audio, _sampleRate);

        // Prepare input: [batch=1, channels=1, samples]
        var inputData = audio.Samples;
        using var inputOrt = OrtValue.CreateTensorValueFromMemory(
            inputData, [1, 1, inputData.Length]);

        var inputs = new Dictionary<string, OrtValue>
        {
            [_encoderSession.InputNames[0]] = inputOrt
        };

        // Run encoder — output shape: [batch, numCodebooks, numFrames]
        using var results = _encoderSession.Run(
            new RunOptions(), inputs, _encoderSession.OutputNames.ToArray());

        // Extract codes from the quantized output
        var outputShape = results[0].GetTensorTypeAndShape().Shape;
        var outputData = results[0].GetTensorDataAsSpan<long>();

        int codebooks = (int)outputShape[1];
        int numFrames = (int)outputShape[2];
        int usedCodebooks = Math.Min(codebooks, _numCodebooks);

        var codes = new int[usedCodebooks][];
        for (int c = 0; c < usedCodebooks; c++)
        {
            codes[c] = new int[numFrames];
            for (int f = 0; f < numFrames; f++)
            {
                codes[c][f] = (int)outputData[c * numFrames + f];
            }
        }

        return codes;
    }

    /// <summary>
    /// Decode discrete codec codes back to audio.
    /// </summary>
    public override AudioData Decode(int[][] codes)
    {
        if (_decoderSession == null)
            throw new InvalidOperationException("Decoder model not loaded.");

        int numCodebooks = codes.Length;
        int numFrames = codes[0].Length;

        // Prepare input: [batch=1, numCodebooks, numFrames] as int64
        var inputData = new long[numCodebooks * numFrames];
        for (int c = 0; c < numCodebooks; c++)
            for (int f = 0; f < numFrames; f++)
                inputData[c * numFrames + f] = codes[c][f];

        using var inputOrt = OrtValue.CreateTensorValueFromMemory(
            inputData, [1, numCodebooks, numFrames]);

        var inputs = new Dictionary<string, OrtValue>
        {
            [_decoderSession.InputNames[0]] = inputOrt
        };

        // Run decoder — output shape: [batch, channels, samples]
        using var results = _decoderSession.Run(
            new RunOptions(), inputs, _decoderSession.OutputNames.ToArray());

        var outputData = results[0].GetTensorDataAsSpan<float>();
        var samples = outputData.ToArray();

        return new AudioData(samples, _sampleRate);
    }

    public override void Dispose()
    {
        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        base.Dispose();
    }
}
