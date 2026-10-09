using MLNet.ASR.Whistle;

// ============================================================================
// Whistle transcription sample — Cactus Compute's native Whistle runtime
// ============================================================================
//
// Prerequisites:
//   1. Download whistle.cact from:
//      https://huggingface.co/Cactus-Compute/whistle
//   2. Build or obtain a shared Cactus Needle library for your RID:
//      https://huggingface.co/Cactus-Compute/needle3
//   3. Set MLNET_WHISTLE_NATIVE_PATH or pass the library path as argument 3.
//
// Usage:
//   dotnet run -- "models/whistle.cact" "speech.wav" "path/to/libneedle3.so" "en" "ML.NET,Whistle" "true"
//
// The sample exits with API instructions when model/audio files are absent.

var modelPath = args.Length > 0 ? args[0] : @"models\whistle.cact";
var audioPath = args.Length > 1 ? args[1] : "speech.wav";
var nativePath = args.Length > 2 ? args[2] : null;
var language = args.Length > 3 ? args[3] : null;
var keywords = args.Length > 4
    ? args[4].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
    : [];
var wordTimestamps = true;
if (args.Length > 5 && !bool.TryParse(args[5], out wordTimestamps))
{
    Console.Error.WriteLine("The timestamps argument must be 'true' or 'false'.");
    return;
}

if (!File.Exists(modelPath) || !File.Exists(audioPath))
{
    Console.WriteLine("Whistle model or audio file not found.");
    Console.WriteLine();
    Console.WriteLine("Download whistle.cact and provide a 16 kHz WAV file:");
    Console.WriteLine("  dotnet run -- \"models/whistle.cact\" \"speech.wav\" \"path/to/libneedle3.so\" \"en\" \"ML.NET,Whistle\" \"true\"");
    Console.WriteLine();
    Console.WriteLine("The native library can also be configured with MLNET_WHISTLE_NATIVE_PATH.");
    return;
}

var options = new WhistleOptions
{
    ModelPath = modelPath,
    NativeLibraryPath = nativePath,
    Language = language,
    Keywords = keywords,
    WordTimestamps = wordTimestamps
};

using var transcriber = new WhistleTranscriber(options);
var result = transcriber.TranscribeFile(audioPath);

Console.WriteLine($"Language: {result.Language ?? "unknown"}");
Console.WriteLine($"Text: {result.Text}");
Console.WriteLine($"TTFT: {result.TimeToFirstToken.TotalMilliseconds:F1} ms");

foreach (var word in result.Words)
    Console.WriteLine($"  [{word.Start.TotalSeconds:F2}s - {word.End.TotalSeconds:F2}s] {word.Word} ({word.Probability:F2})");

Console.WriteLine();
Console.WriteLine("MEAI and ML.NET usage:");
Console.WriteLine("  using var client = new WhistleSpeechToTextClient(options);");
Console.WriteLine("  var response = await client.GetTextAsync(File.OpenRead(\"speech.wav\"));");
Console.WriteLine("  var pipeline = mlContext.Transforms.WhistleSpeechToText(new WhistleTransformOptions { ... });");
