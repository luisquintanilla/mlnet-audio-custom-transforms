# Whistle speech-to-text

`MLNet.ASR.Whistle` integrates [Cactus Compute Whistle](https://huggingface.co/Cactus-Compute/whistle)
with ML.NET and `Microsoft.Extensions.AI`. Whistle is a 16.9 MB `.cact` speech model
executed by the Cactus Needle native runtime; it is not an ONNX model.

## Runtime prerequisites

The package supports Windows x64 and Linux x64 when a Whistle-capable shared
library exposing the Needle C ABI is available. Use Needle 3.1 or later; the
3.0 engine predates the speech exports. The published `needle3` repository
contains platform static libraries and headers, and its Python wheel artifacts
contain the corresponding dynamic engine libraries. A static archive cannot be
loaded with .NET directly, so the native shared library is intentionally
supplied by the application rather than silently downloaded or bundled by this
package.

The official wheel artifact can be extracted without installing a native
toolchain. The following commands fetch the current Windows x64 or Linux x64
library into `native`:

```powershell
python -m pip install cactus-needle
python -c "from needle.agent.fetch import fetch_library; fetch_library(version='3.2.0', dest_dir='native', tag='win_amd64', generation=3)"
```

```bash
python -m pip install cactus-needle
python -c "from needle.agent.fetch import fetch_library; fetch_library(version='3.2.0', dest_dir='native', tag='manylinux2014_x86_64', generation=3)"
```

If Cactus publishes a newer Needle 3 engine, use a version whose library
exports `needle_load`, `needle_last_error`, and `needle_transcribe`. The
alternative static-library build is documented in [`native/README.md`](../native/README.md).

Set the path explicitly:

```csharp
var options = new WhistleOptions
{
    ModelPath = @"models\whistle.cact",
    NativeLibraryPath = @"native\libneedle.dll", // Windows wheel artifact
    // NativeLibraryPath = "/opt/whistle/libneedle.so", // Linux wheel artifact
    Language = null,        // auto-detect
    WordTimestamps = true
};
```

Alternatively set `MLNET_WHISTLE_NATIVE_PATH`. The shared library must export
`needle_load`, `needle_last_error`, and `needle_transcribe` with the signatures
from `needle.h`. The loader also searches the application directory and the
RID-specific `runtimes/<rid>/native` directory for `libneedle3`/`needle3`.

Download the model separately:

```bash
huggingface-cli download Cactus-Compute/whistle whistle.cact --local-dir models
```

Do not commit model weights or native binaries to the repository.

## Direct API

Whistle accepts mono 16 kHz float PCM and clips up to 30 seconds. WAV input is
normalized automatically by the direct API.

```csharp
using MLNet.ASR.Whistle;
using MLNet.Audio.Core;

using var transcriber = new WhistleTranscriber(options);
var result = transcriber.Transcribe(AudioIO.LoadWav("speech.wav"));

Console.WriteLine(result.Text);
Console.WriteLine($"Language: {result.Language}");
foreach (var word in result.Words)
    Console.WriteLine($"{word.Start} - {word.End}: {word.Word}");
```

Whistle supports `en`, `de`, `fr`, `es`, `it`, `nl`, and `pl`. `null` enables
language detection. `Keywords` are sent to the native runtime as newline-separated
phrases.

## Microsoft.Extensions.AI

```csharp
using Microsoft.Extensions.AI;
using MLNet.ASR.Whistle;

using ISpeechToTextClient client = new WhistleSpeechToTextClient(options);
await using var stream = File.OpenRead("speech.wav");
var response = await client.GetTextAsync(stream);

Console.WriteLine(response.Text);
Console.WriteLine(response.AdditionalProperties["language"]);
```

The response includes language, time-to-first-token, decoder throughput, and
word records in `AdditionalProperties`. The streaming MEAI method currently
provides a cancellation-aware response update sequence over a complete WAV
request; native incremental streaming remains a separate API because Needle
holds stream state globally.

## ML.NET

```csharp
var mlContext = new MLContext();
var pipeline = mlContext.Transforms.WhistleSpeechToText(new WhistleTransformOptions
{
    ModelPath = @"models\whistle.cact",
    NativeLibraryPath = @"native\libneedle3.dll",
    InputColumnName = "Audio",
    OutputColumnName = "Text"
});
```

The transformer reads a `VBuffer<float>` audio column, normalizes each row, and
eagerly returns a text column. Native model state is process-global and
non-thread-safe, so calls are serialized. One model/native library identity may
be loaded per process; attempting to use a different identity fails explicitly.
The transformer cannot be serialized into an ML.NET model file.

## Licensing

Whistle and Needle are published by Cactus Compute under Apache-2.0. Applications
redistributing the model or native runtime must preserve the applicable license
and attribution notices. Review the current Cactus repository and model card
before redistributing native artifacts.
