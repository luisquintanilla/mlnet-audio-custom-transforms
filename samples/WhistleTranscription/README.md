# WhistleTranscription

Demonstrates local speech-to-text with Cactus Compute Whistle through the
`MLNet.ASR.Whistle` direct API. The sample is intentionally runnable without
model files: it prints setup instructions and exits when prerequisites are
missing.

```bash
huggingface-cli download Cactus-Compute/whistle whistle.cact --local-dir models
dotnet run -- "models/whistle.cact" "speech.wav" "path/to/libneedle3.so" "en" "ML.NET,Whistle" "true"
```

Arguments after the model, WAV, and native library paths are optional:

1. language (`en`, `de`, `fr`, `es`, `it`, `nl`, or `pl`; omit for detection);
2. comma-separated keyword phrases;
3. whether to request word timestamps (`true` by default).

Use `MLNET_WHISTLE_NATIVE_PATH` instead of the third argument when preferred.
The native runtime is not bundled; see [`native/README.md`](../../native/README.md)
for the shared-library build boundary.
