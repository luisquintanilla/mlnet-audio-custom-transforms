# Obtaining a Needle shared library

`MLNet.ASR.Whistle` loads a shared library through `NativeLibrary` and requires
the exported functions in Cactus's `needle.h`. The Cactus `needle3` repository
publishes `libneedle.a` static archives and platform-specific Python wheel
artifacts for Windows x64 and Linux x64. Use a Whistle-capable Needle 3.1 or
later engine. The 3.0 engine does not export the speech ABI.

## Official dynamic artifacts

The Python wheel is the simplest way to obtain a compatible shared library
without a native toolchain:

```powershell
python -m pip install cactus-needle
python -c "from needle.agent.fetch import fetch_library; fetch_library(version='3.2.0', dest_dir='native', tag='win_amd64', generation=3)"
```

```bash
python -m pip install cactus-needle
python -c "from needle.agent.fetch import fetch_library; fetch_library(version='3.2.0', dest_dir='native', tag='manylinux2014_x86_64', generation=3)"
```

The commands write `native/libneedle.dll` on Windows and
`native/libneedle.so` on Linux. Copy the resulting file to the deployment
directory or pass its absolute path through `WhistleOptions.NativeLibraryPath`.
Do not copy the Python package into the .NET application.

## Linux x64

Download `linux-x86_64/libneedle.a` and `linux-x86_64/needle.h` from
[`Cactus-Compute/needle3`](https://huggingface.co/Cactus-Compute/needle3), then
build:

```bash
c++ -shared -fPIC \
  -Wl,--whole-archive libneedle.a -Wl,--no-whole-archive \
  -ldl -lpthread -lm \
  -o libneedle3.so
```

Confirm that the exports are present:

```bash
nm -D --defined-only libneedle3.so |
  grep -E 'needle_(load|last_error|transcribe)$'
```

## Windows x64

Use a compiler/linker compatible with the published Windows x64 archive (for
example, the LLVM/MinGW toolchain used to produce the archive):

```powershell
clang++ -shared -Wl,--export-all-symbols -o libneedle3.dll `
  -Wl,--whole-archive libneedle.a -Wl,--no-whole-archive
```

Inspect exports with `dumpbin /exports libneedle3.dll` or the equivalent LLVM
tool and verify `needle_load`, `needle_last_error`, and `needle_transcribe`.
The exact system libraries may vary with the selected toolchain.

Set `MLNET_WHISTLE_NATIVE_PATH` to the resulting `.so`/`.dll`, or pass the
absolute path as `WhistleOptions.NativeLibraryPath`. Do not commit generated
native binaries or `.cact` weights to this repository. Preserve Cactus's
Apache-2.0 license and attribution when redistributing either artifact.
