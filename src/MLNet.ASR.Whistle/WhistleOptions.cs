using System.Runtime.InteropServices;
using MLNet.ASR.Whistle.Native;

namespace MLNet.ASR.Whistle;

/// <summary>
/// Configuration for the Cactus Compute Whistle speech-to-text runtime.
/// </summary>
public class WhistleOptions
{
    /// <summary>Path to a Whistle <c>.cact</c> model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// Optional path to a native Cactus Needle shared library. If omitted, the
    /// package searches the configured environment and application directories.
    /// </summary>
    public string? NativeLibraryPath { get; init; }

    /// <summary>Language code, or <see langword="null"/> to detect the language.</summary>
    public string? Language { get; init; }

    /// <summary>
    /// Words or phrases to bias during decoding. These are sent to Needle as
    /// newline-separated keywords.
    /// </summary>
    public IReadOnlyList<string> Keywords { get; init; } = [];

    /// <summary>Whether native word timestamps should be requested.</summary>
    public bool WordTimestamps { get; init; }

    /// <summary>Sample rate expected by Whistle. This must remain 16,000 Hz.</summary>
    public int SampleRate { get; init; } = 16_000;

    /// <summary>Maximum duration accepted by a single Whistle request.</summary>
    public TimeSpan MaxDuration { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Capacity of the JSON response buffer passed to the native API.</summary>
    public int OutputBufferSize { get; init; } = 64 * 1024;

    internal ValidatedWhistleOptions Validate()
    {
        if ((!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux()) ||
            !Environment.Is64BitProcess ||
            RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            throw new PlatformNotSupportedException(
                "MLNet.ASR.Whistle supports only Windows x64 and Linux x64 processes.");
        }

        if (string.IsNullOrWhiteSpace(ModelPath))
            throw new ArgumentException("A Whistle .cact model path is required.", nameof(ModelPath));

        var modelPath = Path.GetFullPath(ModelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Whistle model file was not found: {modelPath}", modelPath);

        if (!string.Equals(Path.GetExtension(modelPath), ".cact", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Whistle ModelPath must point to a .cact model file.", nameof(ModelPath));

        if (SampleRate != 16_000)
            throw new ArgumentOutOfRangeException(nameof(SampleRate), SampleRate, "Whistle requires 16,000 Hz audio.");

        if (MaxDuration <= TimeSpan.Zero || MaxDuration > TimeSpan.FromSeconds(30))
            throw new ArgumentOutOfRangeException(nameof(MaxDuration), "Whistle accepts clips up to 30 seconds.");

        if (OutputBufferSize < 1024)
            throw new ArgumentOutOfRangeException(nameof(OutputBufferSize), "The native response buffer must be at least 1024 bytes.");

        var language = WhistleLanguages.Normalize(Language);
        var keywords = (Keywords ?? [])
            .Where(k => !string.IsNullOrWhiteSpace(k))
            .Select(k => k.Trim())
            .ToArray();

        var nativePath = NativeLibraryLoader.ResolvePath(NativeLibraryPath);
        return new ValidatedWhistleOptions(
            modelPath,
            nativePath,
            language,
            keywords,
            WordTimestamps,
            SampleRate,
            MaxDuration,
            OutputBufferSize);
    }
}

internal sealed record ValidatedWhistleOptions(
    string ModelPath,
    string? NativeLibraryPath,
    string? Language,
    string[] Keywords,
    bool WordTimestamps,
    int SampleRate,
    TimeSpan MaxDuration,
    int OutputBufferSize);

internal static class WhistleLanguages
{
    private static readonly HashSet<string> Supported = new(StringComparer.OrdinalIgnoreCase)
    {
        "en", "de", "fr", "es", "it", "nl", "pl"
    };

    public static string? Normalize(string? language)
    {
        if (string.IsNullOrWhiteSpace(language))
            return null;

        var normalized = language.Trim().ToLowerInvariant();
        if (!Supported.Contains(normalized))
            throw new ArgumentException(
                $"Unsupported Whistle language '{language}'. Supported languages: {string.Join(", ", Supported.Order())}.",
                nameof(language));

        return normalized;
    }
}

/// <summary>Per-request transcription overrides.</summary>
public sealed class WhistleTranscriptionOptions
{
    /// <summary>Language code, or null to use the configured language/detection.</summary>
    public string? Language { get; init; }

    /// <summary>Optional request-specific keyword bias list.</summary>
    public IReadOnlyList<string>? Keywords { get; init; }

    /// <summary>Overrides the configured timestamp setting when non-null.</summary>
    public bool? WordTimestamps { get; init; }
}
