using Microsoft.ML;
using MLNet.Audio.Core;
using MLNet.AudioInference.Onnx;

// ============================================================================
// SpeechT5 Text-to-Speech Sample — Local TTS using raw ONNX
// ============================================================================
//
// This sample uses OnnxSpeechT5TtsTransformer which manages the FULL TTS pipeline:
//   1. SentencePieceCharTokenizer (MLNet.Audio.Tokenizers) → token IDs
//   2. Encoder ONNX session → hidden states
//   3. Decoder loop with KV cache → mel spectrogram frames
//   4. Vocoder ONNX session (postnet + HiFi-GAN) → PCM waveform
//   5. Output as AudioData (our audio primitive)
//
// PREREQUISITES:
//   1. Download the SpeechT5 ONNX model:
//      git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5
//
//   2. The model directory should contain:
//      - encoder_model.onnx           (343 MB)
//      - decoder_model_merged.onnx    (244 MB)
//      - decoder_postnet_and_vocoder.onnx (55 MB)
//      - spm_char.model               (tokenizer)
//      - speaker.npy                  (default speaker embedding)
//
//   3. Run:
//      dotnet run -- "models/speecht5"
//      dotnet run -- "models/speecht5" "Hello, I am a speech synthesis model."
//
// ============================================================================

Console.WriteLine("=== SpeechT5 Text-to-Speech — Raw ONNX Pipeline ===\n");

// --- Resolve model path ---
var modelDir = args.Length > 0 ? args[0] : @"models\speecht5";
var encoderPath = Path.Combine(modelDir, "encoder_model.onnx");
var decoderPath = Path.Combine(modelDir, "decoder_model_merged.onnx");
var vocoderPath = Path.Combine(modelDir, "decoder_postnet_and_vocoder.onnx");

if (!File.Exists(encoderPath) || !File.Exists(decoderPath) || !File.Exists(vocoderPath))
{
    Console.WriteLine($"Model directory: {Path.GetFullPath(modelDir)}");
    Console.WriteLine($"  encoder_model.onnx:              {(File.Exists(encoderPath) ? "found" : "MISSING")}");
    Console.WriteLine($"  decoder_model_merged.onnx:       {(File.Exists(decoderPath) ? "found" : "MISSING")}");
    Console.WriteLine($"  decoder_postnet_and_vocoder.onnx: {(File.Exists(vocoderPath) ? "found" : "MISSING")}");
    Console.WriteLine();
    Console.WriteLine("To download the SpeechT5 ONNX model:");
    Console.WriteLine("  git clone https://huggingface.co/NeuML/txtai-speecht5-onnx models/speecht5");
    Console.WriteLine();
    Console.WriteLine("Showing API patterns instead...\n");

    ShowApiPatterns();
    return;
}

// --- Setup ---
var mlContext = new MLContext();

var options = new OnnxSpeechT5Options
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    VocoderModelPath = vocoderPath,
    MaxMelFrames = 500,
    StopThreshold = 0.5f,
};

using var transformer = new OnnxSpeechT5TtsTransformer(mlContext, options);
Console.WriteLine("SpeechT5 model loaded (encoder + decoder + vocoder)\n");

// --- 1. Basic synthesis ---
Console.WriteLine("--- 1. Direct Synthesis ---\n");

var text = args.Length > 1 ? args[1] : "Hello, this is a text to speech synthesis test.";
Console.WriteLine($"  Input: \"{text}\"");

var audio = transformer.Synthesize(text);
Console.WriteLine($"  Output: {audio.Duration.TotalSeconds:F2}s, {audio.SampleRate}Hz, {audio.Samples.Length} samples");

// Save to WAV
var outputPath = "output.wav";
AudioIO.SaveWav(outputPath, audio);
Console.WriteLine($"  Saved: {outputPath}");

// --- 2. ITextToSpeechClient (MEAI-style) ---
Console.WriteLine("\n--- 2. ITextToSpeechClient ---\n");

using var ttsClient = new OnnxTextToSpeechClient(options);
Console.WriteLine($"  Provider: {ttsClient.Metadata.ProviderName}");
Console.WriteLine($"  Model: {ttsClient.Metadata.DefaultModelId}");

var response = await ttsClient.GetAudioAsync("This is the MEAI client.");
Console.WriteLine($"  Result: {response.Duration.TotalSeconds:F2}s audio, voice={response.Voice}");

// --- 3. ML.NET Pipeline ---
Console.WriteLine("\n--- 3. ML.NET Pipeline ---\n");

var estimator = mlContext.Transforms.SpeechT5Tts(options);
Console.WriteLine("  Pipeline: Text → SentencePiece → Encoder → Decoder (KV Cache) → Vocoder → Audio");
Console.WriteLine("  Can compose with ASR for voice round-trip:");
Console.WriteLine("    var pipeline = mlContext.Transforms");
Console.WriteLine("        .OnnxWhisper(whisperOptions)    // Audio → Text");
Console.WriteLine("        .Append(.SpeechT5Tts(ttsOptions)); // Text → Audio");

// --- 4. Batch synthesis ---
Console.WriteLine("\n--- 4. Batch Synthesis ---\n");

var texts = new[] { "Good morning.", "How are you today?" };
var batch = transformer.SynthesizeBatch(texts);
for (int i = 0; i < batch.Length; i++)
    Console.WriteLine($"  [{i}] \"{texts[i]}\" → {batch[i].Duration.TotalSeconds:F2}s");

Console.WriteLine("\n=== Done ===");

// --- Helpers ---

static void ShowApiPatterns()
{
    Console.WriteLine("--- Pattern 1: Direct Synthesis ---");
    Console.WriteLine("""
        var options = new OnnxSpeechT5Options
        {
            EncoderModelPath = "models/speecht5/encoder_model.onnx",
            DecoderModelPath = "models/speecht5/decoder_model_merged.onnx",
            VocoderModelPath = "models/speecht5/decoder_postnet_and_vocoder.onnx",
        };
    
        using var transformer = new OnnxSpeechT5TtsTransformer(mlContext, options);
        var audio = transformer.Synthesize("Hello world!");
        AudioIO.SaveWav("output.wav", audio);
    """);

    Console.WriteLine("\n--- Pattern 2: ITextToSpeechClient (MEAI-style) ---");
    Console.WriteLine("""
        using var client = new OnnxTextToSpeechClient(options);
        var response = await client.GetAudioAsync("Say something");
        AudioIO.SaveWav("output.wav", response.Audio);
    """);

    Console.WriteLine("\n--- Pattern 3: ML.NET Pipeline ---");
    Console.WriteLine("""
        var pipeline = mlContext.Transforms.SpeechT5Tts(options);
        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);
    """);

    Console.WriteLine("\n--- Pattern 4: Custom Speaker Embedding ---");
    Console.WriteLine("""
        // Extract x-vector from reference audio using speechbrain/ECAPA-TDNN
        var speakerEmbedding = LoadSpeakerEmbedding("reference.npy");
        var audio = transformer.Synthesize("Clone this voice", speakerEmbedding);
    """);

    Console.WriteLine("\n--- Pattern 5: Voice Round-Trip (STT → TTS) ---");
    Console.WriteLine("""
        // Transcribe audio, then synthesize with a different voice
        var pipeline = mlContext.Transforms
            .OnnxWhisper(whisperOptions)         // Audio → Text
            .Append(.SpeechT5Tts(ttsOptions));   // Text → Audio (different voice!)
    """);
}
