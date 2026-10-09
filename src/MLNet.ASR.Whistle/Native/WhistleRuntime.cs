namespace MLNet.ASR.Whistle.Native;

internal static class WhistleRuntime
{
    private static readonly object Gate = new();
    private static RuntimeState? _state;

    internal static Lease Acquire(
        ValidatedWhistleOptions options,
        Func<string?, IWhistleNativeApi> apiFactory)
    {
        Monitor.Enter(Gate);
        try
        {
            var modelIdentity = Path.GetFullPath(options.ModelPath);
            if (_state is not null)
            {
                if (!PathEquals(_state.ModelPath, modelIdentity) ||
                    !PathEquals(_state.NativeLibraryPath, options.NativeLibraryPath))
                {
                    throw new InvalidOperationException(
                        "The Cactus Needle runtime is process-global and already has a different Whistle model " +
                        "or native library loaded. Use one model/native library per process.");
                }

                return new Lease(_state.Api);
            }

            var api = apiFactory(options.NativeLibraryPath);
            try
            {
                api.Load(File.ReadAllBytes(modelIdentity));
                _state = new RuntimeState(modelIdentity, options.NativeLibraryPath, api);
                return new Lease(api);
            }
            catch
            {
                api.Dispose();
                throw;
            }
        }
        catch
        {
            Monitor.Exit(Gate);
            throw;
        }
    }

    internal static void ResetForTests()
    {
        lock (Gate)
        {
            _state?.Api.Dispose();
            _state = null;
        }
    }

    private sealed record RuntimeState(string ModelPath, string? NativeLibraryPath, IWhistleNativeApi Api);

    private static bool PathEquals(string? left, string? right)
        => string.Equals(
            left,
            right,
            OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal);

    internal sealed class Lease : IDisposable
    {
        private bool _disposed;

        internal Lease(IWhistleNativeApi api) => Api = api;

        internal IWhistleNativeApi Api { get; }

        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;
            Monitor.Exit(Gate);
        }
    }
}
