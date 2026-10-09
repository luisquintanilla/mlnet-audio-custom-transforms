using System.Runtime.InteropServices;

namespace MLNet.ASR.Whistle.Native;

internal static class NativeLibraryLoader
{
    private static readonly string[] LibraryNames =
    [
        "libneedle3",
        "needle3",
        "libneedle",
        "needle"
    ];

    public static string? ResolvePath(string? configuredPath)
    {
        if (!string.IsNullOrWhiteSpace(configuredPath))
        {
            var path = Path.GetFullPath(configuredPath);
            if (!File.Exists(path))
                throw new FileNotFoundException($"Whistle native library was not found: {path}", path);

            return path;
        }

        var environmentPath = Environment.GetEnvironmentVariable("MLNET_WHISTLE_NATIVE_PATH");
        if (!string.IsNullOrWhiteSpace(environmentPath))
        {
            var path = Path.GetFullPath(environmentPath);
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"MLNET_WHISTLE_NATIVE_PATH points to a missing native library: {path}",
                    path);

            return path;
        }

        var rid = RuntimeInformation.RuntimeIdentifier;
        var candidates = LibraryNames
            .SelectMany(name => new[]
            {
                Path.Combine(AppContext.BaseDirectory, name),
                Path.Combine(AppContext.BaseDirectory, GetFileName(name)),
                Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", GetFileName(name))
            })
            .Distinct(StringComparer.OrdinalIgnoreCase);

        return candidates.FirstOrDefault(File.Exists);
    }

    public static NativeLibraryHandle Load(string? configuredPath)
    {
        var resolvedPath = ResolvePath(configuredPath);
        var candidates = resolvedPath is not null
            ? [resolvedPath]
            : LibraryNames.AsEnumerable();

        foreach (var candidate in candidates)
        {
            if (NativeLibrary.TryLoad(candidate, out var handle))
                return new NativeLibraryHandle(handle, candidate);
        }

        throw new DllNotFoundException(
            "Unable to load the Cactus Needle native library. Set WhistleOptions.NativeLibraryPath " +
            "or MLNET_WHISTLE_NATIVE_PATH to a shared library exposing the needle_* C ABI.");
    }

    private static string GetFileName(string baseName)
    {
        if (OperatingSystem.IsWindows())
            return baseName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) ? baseName : $"{baseName}.dll";

        if (OperatingSystem.IsMacOS())
            return baseName.EndsWith(".dylib", StringComparison.OrdinalIgnoreCase) ? baseName : $"{baseName}.dylib";

        return baseName.EndsWith(".so", StringComparison.OrdinalIgnoreCase) ? baseName : $"{baseName}.so";
    }
}

internal sealed class NativeLibraryHandle : IDisposable
{
    private IntPtr _handle;

    public NativeLibraryHandle(IntPtr handle, string identity)
    {
        _handle = handle;
        Identity = identity;
    }

    public IntPtr Handle => _handle;

    public string Identity { get; }

    public void Dispose()
    {
        var handle = Interlocked.Exchange(ref _handle, IntPtr.Zero);
        if (handle != IntPtr.Zero)
            NativeLibrary.Free(handle);
    }
}
