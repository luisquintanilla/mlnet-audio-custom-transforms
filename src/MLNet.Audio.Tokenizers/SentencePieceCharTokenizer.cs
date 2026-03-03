using System.Buffers;
using Microsoft.ML.Tokenizers;

namespace MLNet.Audio.Tokenizers;

/// <summary>
/// SentencePiece Char model tokenizer — fills a gap in Microsoft.ML.Tokenizers
/// which only supports BPE and Unigram SentencePiece model types.
///
/// Character-level SentencePiece models (used by SpeechT5, among others) tokenize
/// text by mapping each character to a vocabulary ID. This is the simplest possible
/// tokenization: prepend ▁ (U+2581), replace spaces with ▁, then look up each char.
///
/// Extends Microsoft.ML.Tokenizers.Tokenizer so it can be used anywhere a Tokenizer
/// is expected (e.g., OnnxSpeechT5TtsTransformer._tokenizer).
///
/// DESIGN NOTE: This is a workaround until Microsoft.ML.Tokenizers adds native
/// SentencePiece Char model support. When that happens, this class can be deprecated.
/// </summary>
public sealed class SentencePieceCharTokenizer : Tokenizer
{
    private const char SentencePieceSpace = '\u2581'; // ▁

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _reverseVocab;
    private readonly int _unknownId;

    private SentencePieceCharTokenizer(
        Dictionary<string, int> vocab,
        Dictionary<int, string> reverseVocab,
        int unknownId)
    {
        _vocab = vocab;
        _reverseVocab = reverseVocab;
        _unknownId = unknownId;
    }

    /// <summary>
    /// Create a SentencePieceCharTokenizer from a .model file stream.
    /// Validates that the model is actually a Char-type SentencePiece model.
    /// </summary>
    public static SentencePieceCharTokenizer Create(Stream modelStream)
    {
        var parsed = SentencePieceModelParser.Parse(modelStream);

        if (parsed.Type != SentencePieceModelParser.ModelType.Char)
            throw new ArgumentException(
                $"Expected a SentencePiece Char model but got '{parsed.Type}'. " +
                "Use SentencePieceTokenizer.Create() for BPE/Unigram models.",
                nameof(modelStream));

        var vocab = new Dictionary<string, int>();
        var reverseVocab = new Dictionary<int, string>();
        int unknownId = 0;

        for (int i = 0; i < parsed.Pieces.Count; i++)
        {
            var piece = parsed.Pieces[i];
            vocab[piece.Token] = i;
            reverseVocab[i] = piece.Token;

            if (piece.Type == SentencePieceModelParser.PieceType.Unknown)
                unknownId = i;
        }

        return new SentencePieceCharTokenizer(vocab, reverseVocab, unknownId);
    }

    /// <summary>
    /// Create a SentencePieceCharTokenizer from a .model file path.
    /// </summary>
    public static SentencePieceCharTokenizer Create(string modelPath)
    {
        using var stream = File.OpenRead(modelPath);
        return Create(stream);
    }

    /// <summary>
    /// Normalize text for SentencePiece Char tokenization:
    /// prepend ▁ and replace all spaces with ▁.
    /// </summary>
    private static string Normalize(ReadOnlySpan<char> text)
    {
        // SentencePiece Char model: ▁ prefix + replace spaces with ▁
        return string.Concat(SentencePieceSpace.ToString(), text.ToString().Replace(' ', SentencePieceSpace));
    }

    private int MapCharToId(char c)
    {
        return _vocab.TryGetValue(c.ToString(), out var id) ? id : _unknownId;
    }

    // ========================================================================
    // Tokenizer abstract method implementations
    // ========================================================================

    protected override EncodeResults<EncodedToken> EncodeToTokens(
        string? text,
        ReadOnlySpan<char> textSpan,
        EncodeSettings settings)
    {
        var normalized = Normalize(text is not null ? text.AsSpan() : textSpan);
        var tokens = new List<EncodedToken>();
        int maxTokenCount = settings.MaxTokenCount;

        int charOffset = 0;
        foreach (char c in normalized)
        {
            if (tokens.Count >= maxTokenCount)
                break;

            int id = MapCharToId(c);
            tokens.Add(new EncodedToken(id, c.ToString(), new Range(charOffset, charOffset + 1)));
            charOffset++;
        }

        return new EncodeResults<EncodedToken>
        {
            Tokens = tokens,
            NormalizedText = normalized,
            CharsConsumed = text?.Length ?? textSpan.Length
        };
    }

    public override OperationStatus Decode(
        IEnumerable<int> ids,
        Span<char> destination,
        out int idsConsumed,
        out int charsWritten)
    {
        idsConsumed = 0;
        charsWritten = 0;

        foreach (int id in ids)
        {
            if (!_reverseVocab.TryGetValue(id, out var piece))
            {
                idsConsumed++;
                continue;
            }

            // Skip control tokens (<unk>, <s>, </s>, <pad>)
            if (_vocab.TryGetValue(piece, out _) &&
                (piece == "<unk>" || piece == "<s>" || piece == "</s>" || piece == "<pad>"))
            {
                idsConsumed++;
                continue;
            }

            // Replace ▁ with space
            var decoded = piece.Replace(SentencePieceSpace, ' ');

            if (charsWritten + decoded.Length > destination.Length)
                return OperationStatus.DestinationTooSmall;

            decoded.AsSpan().CopyTo(destination.Slice(charsWritten));
            charsWritten += decoded.Length;
            idsConsumed++;
        }

        return OperationStatus.Done;
    }

    /// <summary>
    /// Decode token IDs back to text.
    /// </summary>
    public override string Decode(IEnumerable<int> ids)
    {
        var parts = new List<string>();
        foreach (int id in ids)
        {
            if (!_reverseVocab.TryGetValue(id, out var piece))
                continue;

            // Skip control tokens
            if (piece == "<unk>" || piece == "<s>" || piece == "</s>" || piece == "<pad>")
                continue;

            parts.Add(piece);
        }

        var result = string.Join("", parts);
        // Replace ▁ with space and trim leading space
        result = result.Replace(SentencePieceSpace, ' ');
        if (result.StartsWith(' '))
            result = result[1..];

        return result;
    }
}
