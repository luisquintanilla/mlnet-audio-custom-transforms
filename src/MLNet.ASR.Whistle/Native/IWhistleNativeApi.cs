namespace MLNet.ASR.Whistle.Native;

internal interface IWhistleNativeApi : IDisposable
{
    void Load(ReadOnlyMemory<byte> model);

    NativeTranscriptionResult Transcribe(
        ReadOnlyMemory<float> samples,
        string? language,
        string? keywords,
        bool wordTimestamps,
        int outputBufferSize);
}

internal readonly record struct NativeTranscriptionResult(int TokenCount, string Json);
