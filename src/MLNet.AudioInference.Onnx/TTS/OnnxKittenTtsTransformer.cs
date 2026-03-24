using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Audio.Core;
using MLNet.Audio.DataIngestion;
using Microsoft.ML.Tokenizers;
using MLNet.Audio.Tokenizers;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// KittenTTS text-to-speech transformer — single ONNX model with espeak-ng phonemization.
///
/// Pipeline stages:
///   1. Sentence chunking via <see cref="TtsSentenceChunker"/> (token-aware)
///   2. Phonemization via <see cref="EspeakPhonemizationProcessor"/> (IPA with stress marks)
///   3. Tokenization via <see cref="KittenTtsTokenizer"/> (IPA symbol → integer token ID)
///   4. ONNX inference — (input_ids, voice_style, speed) → raw audio
///   5. Trim trailing samples, concatenate chunks
///   6. Output as AudioData at 24 kHz
///
/// Models: KittenML/kitten-tts-mini-0.8, kitten-tts-micro-0.8, kitten-tts-nano-0.8
/// Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
/// </summary>
public sealed class OnnxKittenTtsTransformer : ITransformer, IOnnxTtsSynthesizer
{
    private readonly MLContext _mlContext;
    private readonly OnnxKittenTtsOptions _options;
    private readonly InferenceSession _session;
    private readonly Dictionary<string, float[,]> _voices;
    private readonly Tokenizer _tokenizer;
    private readonly EspeakPhonemizationProcessor _phonemizer;
    private readonly TtsSentenceChunker _chunker;

    public bool IsRowToRowMapper => true;

    /// <summary>Gets the names of available voices loaded from voices.npz.</summary>
    public IReadOnlyCollection<string> AvailableVoices => _voices.Keys;

    public OnnxKittenTtsTransformer(MLContext mlContext, OnnxKittenTtsOptions options)
    {
        _mlContext = mlContext;
        _options = options;

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _session = new InferenceSession(options.ModelPath, sessionOptions);

        var voicesPath = options.VoicesPath
            ?? Path.Combine(Path.GetDirectoryName(options.ModelPath) ?? Directory.GetCurrentDirectory(), "voices.npz");
        _voices = LoadVoicesNpz(voicesPath);

        _tokenizer = options.Tokenizer ?? KittenTtsTokenizer.Create();

        _phonemizer = new EspeakPhonemizationProcessor(options.EspeakPath);

#pragma warning disable CS0618 // Obsolete MaxChunkLength used for backward compatibility
        var effectiveMaxChunk = options.MaxTokensPerChunk ?? options.MaxChunkLength;
#pragma warning restore CS0618

        _chunker = new TtsSentenceChunker(
            maxLengthPerChunk: effectiveMaxChunk,
            measureLength: s => _tokenizer.CountTokens(s));
    }

    /// <summary>Synthesize speech from text using the specified voice and speed.</summary>
    public AudioData Synthesize(string text, string? voice = null, float speed = 1.0f)
    {
        voice ??= _options.DefaultVoice;

        if (!_voices.TryGetValue(voice, out var voiceEmbeddings))
        {
            // Fall back to first available voice if the requested one doesn't exist
            // (voice names vary by model variant — mini uses Bella/Jasper/Luna, nano uses expr-voice-*)
            var firstVoice = _voices.Keys.FirstOrDefault()
                ?? throw new InvalidOperationException("No voices loaded from voices.npz.");
            System.Diagnostics.Trace.TraceWarning(
                $"Voice '{voice}' not found. Using '{firstVoice}'. " +
                $"Available: {string.Join(", ", _voices.Keys)}");
            voice = firstVoice;
            voiceEmbeddings = _voices[voice];
        }

        var sentenceChunks = _chunker.ChunkTextAsync(text).ToBlockingEnumerable().ToList();
        var allSamples = new List<float>();

        foreach (var sentenceChunk in sentenceChunks)
        {
            var sentenceText = sentenceChunk.Content;

            // Stage 2: Phonemize
            var phonemes = _phonemizer.PhonemizeAsync(sentenceText).GetAwaiter().GetResult();

            // Stage 3: Tokenize
            var tokenIds = _tokenizer.EncodeToIds(phonemes);

            // Build token sequence: [0, ...tokens, 10, 0]
            var tokenSequence = new List<int> { 0 };
            tokenSequence.AddRange(tokenIds);
            tokenSequence.Add(10); // ellipsis token
            tokenSequence.Add(0);  // pad token

            // Select voice embedding row based on original chunk length
            var rowIdx = Math.Min(sentenceText.Length, voiceEmbeddings.GetLength(0) - 1);
            var embDim = voiceEmbeddings.GetLength(1);
            var style = new float[embDim];
            for (int i = 0; i < embDim; i++)
                style[i] = voiceEmbeddings[rowIdx, i];

            var audio = RunInference(tokenSequence.ToArray(), style, speed);

            // Trim trailing samples
            var trimCount = Math.Min(_options.TrimEndSamples, audio.Length);
            var trimmed = audio.AsSpan(0, audio.Length - trimCount).ToArray();
            allSamples.AddRange(trimmed);
        }

        return new AudioData(allSamples.ToArray(), _options.SampleRate);
    }

    /// <summary>Synthesize speech for multiple texts.</summary>
    public AudioData[] SynthesizeBatch(IReadOnlyList<string> texts, string? voice = null, float speed = 1.0f)
    {
        var results = new AudioData[texts.Count];
        for (int i = 0; i < texts.Count; i++)
            results[i] = Synthesize(texts[i], voice, speed);
        return results;
    }

    // ========================================================================
    // IOnnxTtsSynthesizer — bridges to OnnxTextToSpeechClient
    // ========================================================================

    string IOnnxTtsSynthesizer.ProviderName => "OnnxKittenTts";

    Uri? IOnnxTtsSynthesizer.ProviderUri =>
        new("https://github.com/KittenML/KittenTTS");

    string? IOnnxTtsSynthesizer.ModelId =>
        Path.GetFileName(Path.GetDirectoryName(_options.ModelPath)) ?? "kittentts";

    AudioData IOnnxTtsSynthesizer.Synthesize(string text, TextToSpeechOptions? options)
    {
        var voice = options?.VoiceId ?? _options.DefaultVoice;
        var speed = options?.Speed ?? _options.DefaultSpeed;
        return Synthesize(text, voice, speed);
    }

    // ========================================================================
    // ML.NET ITransformer
    // ========================================================================

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
            "OnnxKittenTtsTransformer wraps ONNX sessions and cannot be serialized.");

    public void Dispose()
    {
        _session?.Dispose();
    }

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

    // ========================================================================
    // Stage 4: ONNX inference
    // ========================================================================

    private float[] RunInference(int[] tokenIds, float[] style, float speed)
    {
        var seqLen = tokenIds.Length;
        var embDim = style.Length;

        var inputIdsTensor = new DenseTensor<long>(new[] { 1, seqLen });
        for (int i = 0; i < seqLen; i++)
            inputIdsTensor[0, i] = tokenIds[i];

        var styleTensor = new DenseTensor<float>(new[] { 1, embDim });
        for (int i = 0; i < embDim; i++)
            styleTensor[0, i] = style[i];

        var speedTensor = new DenseTensor<float>(new[] { 1 });
        speedTensor[0] = speed;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("style", styleTensor),
            NamedOnnxValue.CreateFromTensor("speed", speedTensor)
        };

        using var results = _session.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();

        // Flatten to 1D array
        var audio = new float[outputTensor.Length];
        int idx = 0;
        foreach (var val in outputTensor)
            audio[idx++] = val;

        return audio;
    }

    // ========================================================================
    // NPZ / NPY reader — loads voice embeddings from voices.npz
    // ========================================================================

    private static Dictionary<string, float[,]> LoadVoicesNpz(string npzPath)
    {
        if (!File.Exists(npzPath))
            throw new FileNotFoundException(
                $"Voices file not found: {npzPath}. " +
                "Download the model with voices.npz from HuggingFace.");

        var voices = new Dictionary<string, float[,]>();

        using var zip = ZipFile.OpenRead(npzPath);
        foreach (var entry in zip.Entries)
        {
            var name = Path.GetFileNameWithoutExtension(entry.Name);
            using var stream = entry.Open();
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            ms.Position = 0;
            voices[name] = ReadNpyFloat2D(ms);
        }

        return voices;
    }

    private static float[,] ReadNpyFloat2D(MemoryStream ms)
    {
        using var reader = new BinaryReader(ms, Encoding.ASCII, leaveOpen: true);

        // Magic: \x93NUMPY
        var magic = reader.ReadBytes(6);
        if (magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
            throw new InvalidDataException("Invalid NPY file magic.");

        var majorVersion = reader.ReadByte();
        var minorVersion = reader.ReadByte();

        int headerLen;
        if (majorVersion == 1)
            headerLen = reader.ReadUInt16();
        else
            headerLen = (int)reader.ReadUInt32();

        var headerBytes = reader.ReadBytes(headerLen);
        var header = Encoding.ASCII.GetString(headerBytes).Trim();

        // Parse shape from header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (rows, cols), }
        var shapeMatch = Regex.Match(header, @"'shape':\s*\((\d+),\s*(\d+)\)");
        if (!shapeMatch.Success)
            throw new InvalidDataException($"Cannot parse shape from NPY header: {header}");

        var rows = int.Parse(shapeMatch.Groups[1].Value);
        var cols = int.Parse(shapeMatch.Groups[2].Value);

        // Verify dtype is float32
        if (!header.Contains("'<f4'") && !header.Contains("'float32'"))
            throw new InvalidDataException($"Expected float32 dtype in NPY header: {header}");

        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r, c] = reader.ReadSingle();

        return result;
    }
}
