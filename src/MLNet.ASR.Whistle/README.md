# MLNet.ASR.Whistle

Local speech-to-text for ML.NET and `Microsoft.Extensions.AI` using Cactus
Compute's Whistle model and Needle native runtime.

## Requirements

1. Download a compatible `whistle.cact` model.
2. Provide a shared Needle library exporting `needle_load`,
   `needle_last_error`, and `needle_transcribe`.
3. Set `WhistleOptions.NativeLibraryPath` or
   `MLNET_WHISTLE_NATIVE_PATH` to the shared library.

The NuGet package contains managed bindings only. Cactus publishes static
archives and platform-specific Python wheel artifacts, so applications must
obtain or build a compatible shared library for their operating system and
architecture. Use a Whistle-capable Needle 3.1 or later engine; older Needle
3.0 engines do not export the speech ABI. See the repository's
[`docs/whistle.md`](https://github.com/luisquintanilla/mlnet-audio-custom-transforms/blob/main/docs/whistle.md)
and [`native/README.md`](https://github.com/luisquintanilla/mlnet-audio-custom-transforms/blob/main/native/README.md)
for setup instructions.

## Direct transcription

```csharp
using MLNet.ASR.Whistle;
using MLNet.Audio.Core;

var options = new WhistleOptions
{
    ModelPath = "models/whistle.cact",
    NativeLibraryPath = "native/libneedle3.so",
    WordTimestamps = true
};

using var transcriber = new WhistleTranscriber(options);
var result = transcriber.Transcribe(AudioIO.LoadWav("speech.wav"));
Console.WriteLine(result.Text);
```
