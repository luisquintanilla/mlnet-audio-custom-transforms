using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.DataIngestion;

namespace MLNet.Audio.DataIngestion;

/// <summary>
/// Splits text into sentence-sized chunks for TTS processing.
/// Respects sentence boundaries (.!?) and enforces length limits
/// using either character count or a caller-supplied measurement function
/// (e.g., <c>tokenizer.CountTokens</c>).
/// </summary>
public sealed partial class TtsSentenceChunker : IngestionChunker<string>
{
    private readonly Func<string, int> _measureLength;
    private readonly int _maxLengthPerChunk;

    /// <param name="maxLengthPerChunk">
    /// Maximum length per chunk measured by <paramref name="measureLength"/> (default: 400).
    /// When using a tokenizer this represents max tokens; otherwise max characters.
    /// </param>
    /// <param name="measureLength">
    /// Optional function that returns the length of a string.
    /// Pass <c>tokenizer.CountTokens</c> for token-aware chunking.
    /// When null, falls back to <see cref="string.Length"/>.
    /// </param>
    public TtsSentenceChunker(int maxLengthPerChunk = 400, Func<string, int>? measureLength = null)
    {
        _maxLengthPerChunk = maxLengthPerChunk;
        _measureLength = measureLength ?? (s => s.Length);
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IngestionDocument document,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var fullText = ExtractText(document);
        if (string.IsNullOrWhiteSpace(fullText))
            yield break;

        var chunks = ChunkText(fullText, _maxLengthPerChunk);
        for (int i = 0; i < chunks.Count; i++)
        {
            var chunk = new IngestionChunk<string>(chunks[i], document, $"sentence-{i}");
            chunk.Metadata["chunkIndex"] = i.ToString();
            chunk.Metadata["chunkCount"] = chunks.Count.ToString();
            yield return chunk;
        }

        await Task.CompletedTask;
    }

    private static string ExtractText(IngestionDocument document)
    {
        var sb = new StringBuilder();

        foreach (var section in document.Sections)
        {
            if (!string.IsNullOrWhiteSpace(section.Text))
            {
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(section.Text.Trim());
            }

            foreach (var element in section.Elements)
            {
                if (element is IngestionDocumentParagraph para
                    && !string.IsNullOrWhiteSpace(para.Text))
                {
                    if (sb.Length > 0) sb.Append(' ');
                    sb.Append(para.Text.Trim());
                }
            }
        }

        return sb.ToString();
    }

    private List<string> ChunkText(string text, int maxLen)
    {
        var sentences = SentenceSplitRegex().Split(text);
        var chunks = new List<string>();

        foreach (var sentence in sentences)
        {
            var trimmed = sentence.Trim();
            if (string.IsNullOrEmpty(trimmed))
                continue;

            if (_measureLength(EnsurePunctuation(trimmed)) <= maxLen)
            {
                chunks.Add(EnsurePunctuation(trimmed));
            }
            else
            {
                // Split long sentences by words to fit under the limit
                var words = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                var sb = new StringBuilder();
                foreach (var word in words)
                {
                    var candidate = sb.Length > 0 ? $"{sb} {word}" : word;
                    if (_measureLength(EnsurePunctuation(candidate)) <= maxLen)
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
}
