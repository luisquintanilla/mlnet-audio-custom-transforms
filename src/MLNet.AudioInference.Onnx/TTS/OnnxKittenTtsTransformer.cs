using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Audio.Core;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// KittenTTS text-to-speech transformer — single ONNX model with espeak-ng phonemization.
///
/// Pipeline stages:
///   1. Text chunking (max 400 chars, sentence boundaries)
///   2. Phonemization via espeak-ng subprocess (IPA with stress marks)
///   3. TextCleaner — IPA symbol → integer token ID mapping
///   4. ONNX inference — (input_ids, voice_style, speed) → raw audio
///   5. Trim trailing 5000 samples, concatenate chunks
///   6. Output as AudioData at 24 kHz
///
/// Models: KittenML/kitten-tts-mini-0.8, kitten-tts-micro-0.8, kitten-tts-nano-0.8
/// Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
/// </summary>
public sealed partial class OnnxKittenTtsTransformer : ITransformer, IOnnxTtsSynthesizer
{
    private readonly MLContext _mlContext;
    private readonly OnnxKittenTtsOptions _options;
    private readonly InferenceSession _session;
    private readonly Dictionary<string, float[,]> _voices;
    private readonly string _espeakPath;

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
            ?? Path.Combine(Path.GetDirectoryName(options.ModelPath)!, "voices.npz");
        _voices = LoadVoicesNpz(voicesPath);

        _espeakPath = ResolveEspeakPath(options.EspeakPath);
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
            Console.Error.WriteLine(
                $"Warning: voice '{voice}' not found. Using '{firstVoice}'. " +
                $"Available: {string.Join(", ", _voices.Keys)}");
            voice = firstVoice;
            voiceEmbeddings = _voices[voice];
        }

        var chunks = ChunkText(text, _options.MaxChunkLength);
        var allSamples = new List<float>();

        foreach (var chunk in chunks)
        {
            var phonemes = Phonemize(chunk);
            var tokens = TextClean(phonemes);

            // Build token sequence: [0, ...tokens, 10, 0]
            var tokenSequence = new List<int> { 0 };
            tokenSequence.AddRange(tokens);
            tokenSequence.Add(10); // ellipsis token
            tokenSequence.Add(0);  // pad token

            // Select voice embedding row based on original chunk length
            var rowIdx = Math.Min(chunk.Length, voiceEmbeddings.GetLength(0) - 1);
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
    // Stage 1: Text chunking
    // ========================================================================

    private static List<string> ChunkText(string text, int maxLen)
    {
        var sentences = SentenceSplitRegex().Split(text);
        var chunks = new List<string>();

        foreach (var sentence in sentences)
        {
            var trimmed = sentence.Trim();
            if (string.IsNullOrEmpty(trimmed))
                continue;

            if (trimmed.Length <= maxLen)
            {
                chunks.Add(EnsurePunctuation(trimmed));
            }
            else
            {
                // Split long sentences by words
                var words = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var sb = new StringBuilder();
                foreach (var word in words)
                {
                    if (sb.Length + word.Length + 1 <= maxLen)
                    {
                        if (sb.Length > 0) sb.Append(' ');
                        sb.Append(word);
                    }
                    else
                    {
                        if (sb.Length > 0)
                        {
                            chunks.Add(EnsurePunctuation(sb.ToString()));
                            sb.Clear();
                        }
                        sb.Append(word);
                    }
                }
                if (sb.Length > 0)
                    chunks.Add(EnsurePunctuation(sb.ToString()));
            }
        }

        if (chunks.Count == 0)
            chunks.Add(EnsurePunctuation(text));

        return chunks;
    }

    private static string EnsurePunctuation(string text)
    {
        if (text.Length == 0) return text;
        var last = text[^1];
        if (last is '.' or '!' or '?' or ',' or ';' or ':')
            return text;
        return text + ",";
    }

    [GeneratedRegex(@"[.!?]+")]
    private static partial Regex SentenceSplitRegex();

    // ========================================================================
    // Stage 2: Phonemization via espeak-ng
    // ========================================================================

    private string Phonemize(string text)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _espeakPath,
            ArgumentList = { "--ipa", "-q", "--sep= ", "-v", "en-us", text },
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start espeak-ng process.");

        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit(10_000);

        if (process.ExitCode != 0)
        {
            var error = process.StandardError.ReadToEnd();
            throw new InvalidOperationException(
                $"espeak-ng failed (exit {process.ExitCode}): {error}");
        }

        return output.Trim();
    }

    private static string ResolveEspeakPath(string? configuredPath)
    {
        if (!string.IsNullOrEmpty(configuredPath))
        {
            if (File.Exists(configuredPath))
                return configuredPath;
            throw new FileNotFoundException(
                $"espeak-ng not found at configured path: {configuredPath}");
        }

        // Auto-detect from PATH
        var names = OperatingSystem.IsWindows()
            ? new[] { "espeak-ng.exe", "espeak.exe" }
            : new[] { "espeak-ng", "espeak" };

        var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator) ?? [];
        foreach (var dir in paths)
        {
            foreach (var name in names)
            {
                var candidate = Path.Combine(dir, name);
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        // Common install locations
        var commonPaths = OperatingSystem.IsWindows()
            ? new[] { @"C:\Program Files\eSpeak NG\espeak-ng.exe", @"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" }
            : new[] { "/usr/bin/espeak-ng", "/usr/local/bin/espeak-ng" };

        foreach (var path in commonPaths)
        {
            if (File.Exists(path))
                return path;
        }

        throw new FileNotFoundException(
            "espeak-ng not found. Install espeak-ng: " +
            "Windows: 'winget install espeak-ng' or https://github.com/espeak-ng/espeak-ng/releases | " +
            "Linux: 'apt install espeak-ng' | " +
            "macOS: 'brew install espeak-ng'");
    }

    // ========================================================================
    // Stage 3: TextCleaner — IPA symbol → token ID
    // ========================================================================

    // Symbol table: pad + punctuation + letters + IPA characters (176 total)
    private const string Pad = "$";
    private const string Punctuation = ";:,.!?¡¿—…\"«»\u201C\u201D ";
    private const string Letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    private const string LettersIpa =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u0329ᵻ";

    private static readonly Dictionary<char, int> SymbolToId = BuildSymbolTable();

    private static Dictionary<char, int> BuildSymbolTable()
    {
        var all = Pad + Punctuation + Letters + LettersIpa;
        var dict = new Dictionary<char, int>();
        for (int i = 0; i < all.Length; i++)
            dict[all[i]] = i;
        return dict;
    }

    private static List<int> TextClean(string phonemes)
    {
        // Tokenize: split on word boundaries and punctuation
        var matches = PhonemeTokenRegex().Matches(phonemes);
        var sb = new StringBuilder();
        foreach (Match m in matches)
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(m.Value);
        }

        var text = sb.ToString();
        var tokens = new List<int>();
        foreach (var ch in text)
        {
            if (SymbolToId.TryGetValue(ch, out var id))
                tokens.Add(id);
            // Unknown chars are silently skipped (matches Python behavior)
        }
        return tokens;
    }

    [GeneratedRegex(@"\w+|[^\w\s]")]
    private static partial Regex PhonemeTokenRegex();

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
