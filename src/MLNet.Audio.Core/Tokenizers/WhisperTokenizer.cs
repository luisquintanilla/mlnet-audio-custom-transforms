namespace MLNet.Audio.Core;

/// <summary>
/// Whisper-specific BPE tokenizer for decoding ASR output tokens to text.
///
/// Whisper uses a GPT-2 BPE tokenizer (English-only) or a multilingual BPE tokenizer.
/// On top of standard BPE, it defines ~1700 special tokens:
///   - <|startoftranscript|>, <|endoftext|>
///   - Language codes: <|en|>, <|zh|>, <|de|>, ... (99 languages)
///   - Task tokens: <|transcribe|>, <|translate|>
///   - Timestamp tokens: <|0.00|>, <|0.02|>, ..., <|30.00|> (1501 timestamps)
///   - <|nospeech|>, <|notimestamps|>, <|startoflm|>, <|startofprev|>
///
/// DESIGN NOTE: This class demonstrates what Microsoft.ML.Tokenizers should support natively.
/// The current BpeTokenizer handles standard BPE encoding/decoding but has no concept of:
///   1. Special token registries with semantic meaning (timestamps, language codes)
///   2. Domain-specific decode modes (with/without timestamps)
///   3. Token ID ranges with programmatic meaning (timestamp tokens → TimeSpan conversion)
///
/// A future Microsoft.ML.Tokenizers design could include:
///   - SpecialTokenRegistry: named groups of special tokens with metadata
///   - Tokenizer.DecodeWithMetadata(): returns tokens + their semantic annotations
///   - Domain-specific tokenizer extensions (audio timestamps, code spans, etc.)
/// </summary>
public class WhisperTokenizer : IDisposable
{
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<string, int> _tokenToId;

    // Special token IDs (Whisper multilingual defaults)
    public int EndOfTextId { get; }
    public int StartOfTranscriptId { get; }
    public int TranscribeId { get; }
    public int TranslateId { get; }
    public int NoSpeechId { get; }
    public int NoTimestampsId { get; }
    public int TimestampBeginId { get; }

    /// <summary>Language code → token ID mapping.</summary>
    public IReadOnlyDictionary<string, int> LanguageTokens { get; }

    /// <summary>Whether this is a multilingual tokenizer.</summary>
    public bool IsMultilingual { get; }

    /// <summary>
    /// Creates a WhisperTokenizer with the standard Whisper special token layout.
    /// </summary>
    /// <param name="vocabulary">Base BPE vocabulary (token → ID mapping).</param>
    /// <param name="isMultilingual">Whether to use multilingual token layout.</param>
    public WhisperTokenizer(
        IReadOnlyDictionary<string, int>? vocabulary = null,
        bool isMultilingual = true)
    {
        IsMultilingual = isMultilingual;

        // Initialize vocabulary
        _tokenToId = vocabulary != null
            ? new Dictionary<string, int>(vocabulary)
            : new Dictionary<string, int>();
        _idToToken = _tokenToId.ToDictionary(kv => kv.Value, kv => kv.Key);

        // Whisper special token layout (after the base vocabulary)
        // Base vocab size is 50257 (GPT-2) or 50258 (multilingual)
        int specialBase = isMultilingual ? 50258 : 50257;

        EndOfTextId = isMultilingual ? 50257 : 50256;
        StartOfTranscriptId = specialBase;

        // Language tokens (multilingual only): 99 languages
        var langTokens = new Dictionary<string, int>();
        if (isMultilingual)
        {
            var languages = new[]
            {
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
                "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
            };
            for (int i = 0; i < languages.Length; i++)
            {
                int tokenId = specialBase + 1 + i;
                langTokens[languages[i]] = tokenId;
                _tokenToId[$"<|{languages[i]}|>"] = tokenId;
                _idToToken[tokenId] = $"<|{languages[i]}|>";
            }
        }
        LanguageTokens = langTokens;

        // Task tokens
        int taskBase = isMultilingual ? specialBase + 1 + 99 : specialBase + 1;
        TranscribeId = taskBase;
        TranslateId = taskBase + 1;

        // Control tokens
        NoSpeechId = taskBase + 2;
        NoTimestampsId = taskBase + 3;

        // Timestamp tokens: <|0.00|> through <|30.00|> in 0.02s steps = 1501 tokens
        TimestampBeginId = taskBase + 4;

        // Register all special tokens
        _tokenToId["<|endoftext|>"] = EndOfTextId;
        _idToToken[EndOfTextId] = "<|endoftext|>";
        _tokenToId["<|startoftranscript|>"] = StartOfTranscriptId;
        _idToToken[StartOfTranscriptId] = "<|startoftranscript|>";
        _tokenToId["<|transcribe|>"] = TranscribeId;
        _idToToken[TranscribeId] = "<|transcribe|>";
        _tokenToId["<|translate|>"] = TranslateId;
        _idToToken[TranslateId] = "<|translate|>";
        _tokenToId["<|nospeech|>"] = NoSpeechId;
        _idToToken[NoSpeechId] = "<|nospeech|>";
        _tokenToId["<|notimestamps|>"] = NoTimestampsId;
        _idToToken[NoTimestampsId] = "<|notimestamps|>";

        for (int i = 0; i <= 1500; i++)
        {
            float time = i * 0.02f;
            string token = $"<|{time:F2}|>";
            int tokenId = TimestampBeginId + i;
            _tokenToId[token] = tokenId;
            _idToToken[tokenId] = token;
        }
    }

    /// <summary>
    /// Build the "start of transcript" token sequence for a given language and task.
    /// This is the initial prompt fed to the Whisper decoder.
    /// </summary>
    public int[] GetStartOfTranscriptSequence(string? language = null, bool translate = false)
    {
        var tokens = new List<int> { StartOfTranscriptId };

        if (language != null && IsMultilingual && LanguageTokens.TryGetValue(language, out int langId))
            tokens.Add(langId);

        tokens.Add(translate ? TranslateId : TranscribeId);

        return tokens.ToArray();
    }

    /// <summary>
    /// Decode token IDs to text, stripping all special tokens.
    /// Standard decode mode for clean transcription output.
    /// </summary>
    public string Decode(IReadOnlyList<int> tokenIds)
    {
        var textTokens = new List<string>();
        foreach (var id in tokenIds)
        {
            // Skip all special tokens (timestamps, control tokens, etc.)
            if (id >= StartOfTranscriptId)
                continue;
            if (id == EndOfTextId)
                break;

            if (_idToToken.TryGetValue(id, out var token))
                textTokens.Add(token);
        }
        return string.Join("", textTokens);
    }

    /// <summary>
    /// Decode token IDs to text, preserving timestamp annotations.
    /// Returns text with inline timestamps like "[0.00s] Hello world [1.52s]".
    /// </summary>
    public string DecodeWithTimestamps(IReadOnlyList<int> tokenIds)
    {
        var parts = new List<string>();
        foreach (var id in tokenIds)
        {
            if (id == EndOfTextId)
                break;
            if (id == StartOfTranscriptId || id == TranscribeId ||
                id == TranslateId || id == NoSpeechId || id == NoTimestampsId)
                continue;

            // Check if it's a language token
            if (IsMultilingual && id > StartOfTranscriptId && id < TranscribeId)
                continue;

            // Timestamp token → human-readable annotation
            if (id >= TimestampBeginId && id <= TimestampBeginId + 1500)
            {
                float time = (id - TimestampBeginId) * 0.02f;
                parts.Add($"[{time:F2}s]");
                continue;
            }

            if (_idToToken.TryGetValue(id, out var token))
                parts.Add(token);
        }
        return string.Join("", parts);
    }

    /// <summary>
    /// Decode token IDs to structured segments with timestamps.
    /// Returns (text, startTime, endTime) tuples.
    /// </summary>
    public IReadOnlyList<TranscriptionSegment> DecodeToSegments(IReadOnlyList<int> tokenIds)
    {
        var segments = new List<TranscriptionSegment>();
        var currentTokens = new List<int>();
        TimeSpan? segmentStart = null;

        foreach (var id in tokenIds)
        {
            if (id == EndOfTextId)
                break;

            if (id >= TimestampBeginId && id <= TimestampBeginId + 1500)
            {
                float time = (id - TimestampBeginId) * 0.02f;
                var ts = TimeSpan.FromSeconds(time);

                if (!segmentStart.HasValue)
                {
                    segmentStart = ts;
                }
                else
                {
                    // End of segment — decode accumulated tokens
                    var text = Decode(currentTokens);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        segments.Add(new TranscriptionSegment(
                            text.Trim(), segmentStart.Value, ts));
                    }
                    currentTokens.Clear();
                    segmentStart = null;
                }
                continue;
            }

            // Skip non-text special tokens
            if (id >= StartOfTranscriptId)
                continue;

            currentTokens.Add(id);
        }

        return segments;
    }

    /// <summary>
    /// Check if a token ID is a timestamp token.
    /// </summary>
    public bool IsTimestamp(int tokenId)
        => tokenId >= TimestampBeginId && tokenId <= TimestampBeginId + 1500;

    /// <summary>
    /// Convert a timestamp token ID to a TimeSpan.
    /// </summary>
    public TimeSpan? TokenToTimestamp(int tokenId)
    {
        if (!IsTimestamp(tokenId)) return null;
        float seconds = (tokenId - TimestampBeginId) * 0.02f;
        return TimeSpan.FromSeconds(seconds);
    }

    /// <summary>
    /// Convert a TimeSpan to the nearest timestamp token ID.
    /// </summary>
    public int TimestampToToken(TimeSpan time)
    {
        int index = (int)Math.Round(time.TotalSeconds / 0.02);
        index = Math.Clamp(index, 0, 1500);
        return TimestampBeginId + index;
    }

    public void Dispose() { }
}

/// <summary>
/// A timestamped segment from Whisper transcription output.
/// </summary>
public record TranscriptionSegment(string Text, TimeSpan Start, TimeSpan End)
{
    public TimeSpan Duration => End - Start;
}
