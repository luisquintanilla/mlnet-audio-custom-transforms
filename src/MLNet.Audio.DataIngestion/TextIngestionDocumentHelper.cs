using Microsoft.Extensions.DataIngestion;

namespace MLNet.Audio.DataIngestion;

/// <summary>
/// Bridges plain text to the MEDI IngestionDocument model.
/// Use this when you have raw text (e.g., TTS input) and need to feed it
/// to an IngestionChunker that expects IngestionDocument.
/// </summary>
public static class TextIngestionDocumentHelper
{
    /// <summary>
    /// Wraps a plain text string as an IngestionDocument with paragraph elements.
    /// Each non-empty paragraph (split by double newlines) becomes an
    /// IngestionDocumentParagraph in a single section.
    /// </summary>
    public static IngestionDocument FromText(string text, string? identifier = null)
    {
        var doc = new IngestionDocument(identifier ?? "text-input");
        var section = new IngestionDocumentSection("content");

        // Normalize line endings (CRLF → LF) before splitting paragraphs
        var normalized = text.Replace("\r\n", "\n").Replace("\r", "\n");

        var paragraphs = normalized.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        foreach (var para in paragraphs)
        {
            var trimmed = para.Trim();
            if (!string.IsNullOrEmpty(trimmed))
            {
                // IngestionDocumentParagraph(string) is a markdown constructor;
                // .Text must be set explicitly for ExtractText to find it.
                section.Elements.Add(new IngestionDocumentParagraph(trimmed) { Text = trimmed });
            }
        }

        // If no paragraphs found (single block of text), add the whole text
        if (section.Elements.Count == 0)
            section.Elements.Add(new IngestionDocumentParagraph(text.Trim()) { Text = text.Trim() });

        doc.Sections.Add(section);
        return doc;
    }
}

/// <summary>
/// Extension methods for using IngestionChunker with plain text input.
/// </summary>
public static class IngestionChunkerTextExtensions
{
    /// <summary>
    /// Chunks a plain text string by wrapping it as an IngestionDocument first.
    /// Convenience method for TTS and other text-processing scenarios.
    /// </summary>
    public static IAsyncEnumerable<IngestionChunk<string>> ChunkTextAsync(
        this IngestionChunker<string> chunker,
        string text,
        string? identifier = null,
        CancellationToken cancellationToken = default)
    {
        var doc = TextIngestionDocumentHelper.FromText(text, identifier);
        return chunker.ProcessAsync(doc, cancellationToken);
    }
}
