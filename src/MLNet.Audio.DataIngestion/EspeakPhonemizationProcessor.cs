using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DataIngestion;

namespace MLNet.Audio.DataIngestion;

/// <summary>
/// MEDI pipeline processor that phonemizes text chunks using espeak-ng.
/// Transforms chunk Content from natural language to IPA phonemes.
/// Original text is preserved in Metadata["original_text"].
/// </summary>
public sealed class EspeakPhonemizationProcessor : IngestionChunkProcessor<string>
{
    private readonly string _espeakPath;
    private readonly string _language;
    private readonly int _timeoutMs;

    public EspeakPhonemizationProcessor(
        string? espeakPath = null,
        string language = "en-us",
        int timeoutMs = 10_000)
    {
        _espeakPath = ResolveEspeakPath(espeakPath);
        _language = language;
        _timeoutMs = timeoutMs;
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var chunk in chunks.WithCancellation(cancellationToken))
        {
            var ipa = await PhonemizeAsync(chunk.Content, cancellationToken);

            var newChunk = new IngestionChunk<string>(ipa, chunk.Document, chunk.Context);

            if (chunk.HasMetadata)
                foreach (var kv in chunk.Metadata)
                    newChunk.Metadata[kv.Key] = kv.Value;

            newChunk.Metadata["original_text"] = chunk.Content;
            newChunk.Metadata["language"] = _language;

            yield return newChunk;
        }
    }

    /// <summary>
    /// Phonemizes the given text using espeak-ng, returning IPA output.
    /// Can be used standalone outside the MEDI pipeline.
    /// </summary>
    public async Task<string> PhonemizeAsync(string text, CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _espeakPath,
            ArgumentList = { "--ipa", "-q", "--sep= ", "-v", _language, text },
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start espeak-ng process.");

        // Read stdout and stderr asynchronously to avoid deadlocks on full pipe buffers.
        var stdOutTask = process.StandardOutput.ReadToEndAsync(ct);
        var stdErrTask = process.StandardError.ReadToEndAsync(ct);

        try
        {
            await process.WaitForExitAsync(ct).WaitAsync(TimeSpan.FromMilliseconds(_timeoutMs), ct);
        }
        catch (TimeoutException)
        {
            try { process.Kill(true); } catch { /* best-effort cleanup */ }
            throw new TimeoutException($"espeak-ng did not complete within {_timeoutMs} ms.");
        }

        var output = await stdOutTask;
        var error = await stdErrTask;

        if (process.ExitCode != 0)
            throw new InvalidOperationException($"espeak-ng failed (exit {process.ExitCode}): {error}");

        return output.Trim();
    }

    /// <summary>
    /// Resolves the path to the espeak-ng executable.
    /// Checks the configured path, then PATH, then common install locations.
    /// </summary>
    public static string ResolveEspeakPath(string? configuredPath)
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
}
