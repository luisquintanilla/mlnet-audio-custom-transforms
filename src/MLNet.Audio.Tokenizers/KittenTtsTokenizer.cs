using System.Buffers;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace MLNet.Audio.Tokenizers;

/// <summary>
/// IPA character-level tokenizer for KittenTTS.
///
/// Maps each character in a 176-symbol IPA alphabet (pad + punctuation + letters +
/// IPA extensions) to a token ID. Input text is first normalized by splitting on
/// word/punctuation boundaries with <c>\w+|[^\w\s]</c> and rejoining with spaces,
/// replicating the TextClean preprocessing used by KittenTTS.
///
/// Extends <see cref="Tokenizer"/> so it can be used anywhere a Tokenizer is expected
/// (e.g., OnnxKittenTtsTransformer).
/// </summary>
public sealed partial class KittenTtsTokenizer : Tokenizer
{
    private const string Pad = "$";
    private const string Punctuation = ";:,.!?¡¿—…\"«»\u201C\u201D ";
    private const string Letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    private const string LettersIpa =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u0329ᵻ";

    private static readonly string AllSymbols = Pad + Punctuation + Letters + LettersIpa;

    private readonly Dictionary<char, int> _symbolToId;
    private readonly Dictionary<int, char> _idToSymbol;

    private KittenTtsTokenizer(Dictionary<char, int> symbolToId, Dictionary<int, char> idToSymbol)
    {
        _symbolToId = symbolToId;
        _idToSymbol = idToSymbol;
    }

    /// <summary>
    /// Create a KittenTtsTokenizer. No model file is needed — the 176-symbol
    /// IPA vocabulary is hardcoded.
    /// </summary>
    public static KittenTtsTokenizer Create()
    {
        var symbolToId = new Dictionary<char, int>();
        var idToSymbol = new Dictionary<int, char>();

        for (int i = 0; i < AllSymbols.Length; i++)
        {
            symbolToId[AllSymbols[i]] = i;
            idToSymbol[i] = AllSymbols[i];
        }

        return new KittenTtsTokenizer(symbolToId, idToSymbol);
    }

    /// <summary>
    /// Normalize IPA text: split on word/punctuation boundaries and rejoin with spaces.
    /// This replicates the TextClean preprocessing used by KittenTTS.
    /// </summary>
    private static string Normalize(ReadOnlySpan<char> text)
    {
        var matches = PhonemeTokenRegex().Matches(text.ToString());
        var sb = new StringBuilder();
        foreach (Match m in matches)
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(m.Value);
        }
        return sb.ToString();
    }

    [GeneratedRegex(@"\w+|[^\w\s]")]
    private static partial Regex PhonemeTokenRegex();

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

            if (_symbolToId.TryGetValue(c, out int id))
            {
                tokens.Add(new EncodedToken(id, c.ToString(), new Range(charOffset, charOffset + 1)));
            }
            // Unknown chars are silently skipped (matches Python behavior)
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
            if (!_idToSymbol.TryGetValue(id, out char c))
            {
                idsConsumed++;
                continue;
            }

            // Skip pad token ($, id=0)
            if (id == 0)
            {
                idsConsumed++;
                continue;
            }

            if (charsWritten + 1 > destination.Length)
                return OperationStatus.DestinationTooSmall;

            destination[charsWritten] = c;
            charsWritten++;
            idsConsumed++;
        }

        return OperationStatus.Done;
    }

    /// <summary>
    /// Decode token IDs back to text. Pad tokens ($, id=0) are skipped.
    /// </summary>
    public override string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            // Skip pad token
            if (id == 0)
                continue;

            if (_idToSymbol.TryGetValue(id, out char c))
                sb.Append(c);
        }
        return sb.ToString();
    }
}
