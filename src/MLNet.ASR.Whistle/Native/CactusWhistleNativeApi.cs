using System.Runtime.InteropServices;
using System.Text;

namespace MLNet.ASR.Whistle.Native;

internal sealed class CactusWhistleNativeApi : IWhistleNativeApi
{
    private readonly NativeLibraryHandle _library;
    private readonly NeedleLoad _load;
    private readonly NeedleLastError _lastError;
    private readonly NeedleTranscribe _transcribe;

    private CactusWhistleNativeApi(NativeLibraryHandle library)
    {
        _library = library;
        _load = GetExport<NeedleLoad>("needle_load");
        _lastError = GetExport<NeedleLastError>("needle_last_error");
        _transcribe = GetExport<NeedleTranscribe>("needle_transcribe");
    }

    public static CactusWhistleNativeApi Create(string? configuredPath)
    {
        var library = NativeLibraryLoader.Load(configuredPath);
        try
        {
            return new CactusWhistleNativeApi(library);
        }
        catch (EntryPointNotFoundException ex)
        {
            library.Dispose();
            throw new InvalidOperationException(
                "The configured Needle native library does not expose the Whistle speech ABI. " +
                "Use a Whistle-capable Needle 3.1 or later library exporting " +
                "needle_load, needle_last_error, and needle_transcribe.",
                ex);
        }
        catch
        {
            library.Dispose();
            throw;
        }
    }

    public void Load(ReadOnlyMemory<byte> model)
    {
        if (model.IsEmpty)
            throw new InvalidDataException("The Whistle model file is empty.");

        var modelBytes = model.ToArray();
        var handle = GCHandle.Alloc(modelBytes, GCHandleType.Pinned);
        try
        {
            var result = _load(handle.AddrOfPinnedObject(), (ulong)modelBytes.Length);
            if (result < 0)
                ThrowNativeError("needle_load");
        }
        finally
        {
            handle.Free();
        }
    }

    public NativeTranscriptionResult Transcribe(
        ReadOnlyMemory<float> samples,
        string? language,
        string? keywords,
        bool wordTimestamps,
        int outputBufferSize)
    {
        var pcm = samples.ToArray();
        var output = new byte[outputBufferSize];
        var pcmHandle = GCHandle.Alloc(pcm, GCHandleType.Pinned);
        var outputHandle = GCHandle.Alloc(output, GCHandleType.Pinned);
        var languagePointer = IntPtr.Zero;
        var keywordsPointer = IntPtr.Zero;

        try
        {
            languagePointer = AllocUtf8(language);
            keywordsPointer = AllocUtf8(keywords);
            var tokenCount = _transcribe(
                pcmHandle.AddrOfPinnedObject(),
                pcm.Length,
                languagePointer,
                keywordsPointer,
                wordTimestamps ? 1 : 0,
                outputHandle.AddrOfPinnedObject(),
                output.Length);

            if (tokenCount < 0)
                ThrowNativeError("needle_transcribe");

            var length = Array.IndexOf(output, (byte)0);
            if (length < 0)
                throw new InvalidOperationException(
                    $"Whistle native output exceeded the configured {output.Length}-byte response buffer.");

            return new NativeTranscriptionResult(
                tokenCount,
                Encoding.UTF8.GetString(output, 0, length));
        }
        finally
        {
            FreeUtf8(languagePointer);
            FreeUtf8(keywordsPointer);
            outputHandle.Free();
            pcmHandle.Free();
        }
    }

    public void Dispose() => _library.Dispose();

    private T GetExport<T>(string name) where T : Delegate
    {
        var export = NativeLibrary.GetExport(_library.Handle, name);
        return Marshal.GetDelegateForFunctionPointer<T>(export);
    }

    private void ThrowNativeError(string operation)
    {
        var pointer = _lastError();
        var error = pointer == IntPtr.Zero
            ? "No native error was provided."
            : Marshal.PtrToStringUTF8(pointer) ?? "Unknown native error.";

        throw new InvalidOperationException($"Whistle {operation} failed: {error}");
    }

    private static IntPtr AllocUtf8(string? value)
        => value is null ? IntPtr.Zero : Marshal.StringToCoTaskMemUTF8(value);

    private static void FreeUtf8(IntPtr pointer)
    {
        if (pointer != IntPtr.Zero)
            Marshal.FreeCoTaskMem(pointer);
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int NeedleLoad(IntPtr cact, ulong size);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr NeedleLastError();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int NeedleTranscribe(
        IntPtr pcm,
        int samples,
        IntPtr language,
        IntPtr keywords,
        int wordTimestamps,
        IntPtr output,
        int outputCapacity);
}
