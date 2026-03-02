using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntimeGenAI;
using MLNet.Audio.Core;

namespace MLNet.ASR.OnnxGenAI;

/// <summary>
/// ML.NET transformer for local speech-to-text using Whisper via ORT GenAI.
/// Handles the full pipeline: audio → mel spectrogram → encoder → autoregressive decoder → text.
///
/// This is the audio equivalent of OnnxTextGenerationTransformer from the text-inference repo.
/// Like text generation, this is a standalone transformer (not part of the shared 3-stage pipeline)
/// because encoder-decoder models don't fit the linear Feature→Score→PostProcess pattern.
///
/// Uses eager evaluation: reads all audio rows, transcribes each, returns results.
/// </summary>
public sealed class OnnxSpeechToTextTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxSpeechToTextOptions _options;
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly AudioFeatureExtractor _featureExtractor;

    public bool IsRowToRowMapper => true;

    public OnnxSpeechToTextTransformer(MLContext mlContext, OnnxSpeechToTextOptions options)
    {
        _mlContext = mlContext;
        _options = options;

        _model = new Model(options.ModelPath);
        _tokenizer = new Tokenizer(_model);

        // Use provided feature extractor or create a default WhisperFeatureExtractor
        _featureExtractor = options.FeatureExtractor ?? new WhisperFeatureExtractor(
            numMelBins: options.NumMelBins);
    }

    /// <summary>
    /// Transcribe audio samples directly (outside ML.NET pipeline).
    /// </summary>
    public string[] Transcribe(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new string[audioInputs.Count];

        for (int i = 0; i < audioInputs.Count; i++)
        {
            results[i] = TranscribeSingle(audioInputs[i]);
        }

        return results;
    }

    /// <summary>
    /// Transcribe audio with structured output including timestamps.
    /// </summary>
    public TranscriptionResult[] TranscribeWithTimestamps(IReadOnlyList<AudioData> audioInputs)
    {
        var results = new TranscriptionResult[audioInputs.Count];

        for (int i = 0; i < audioInputs.Count; i++)
        {
            var tokenIds = RunGeneration(audioInputs[i]);
            var text = _tokenizer.Decode(tokenIds);

            // Use WhisperTokenizer for structured timestamp output
            var whisperTokenizer = new WhisperTokenizer(isMultilingual: _options.IsMultilingual);
            var segments = whisperTokenizer.DecodeToSegments(tokenIds.ToList());

            results[i] = new TranscriptionResult
            {
                Text = CleanTranscription(text),
                Segments = segments.ToArray(),
                TokenIds = tokenIds,
                Language = _options.Language
            };
        }

        return results;
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation. Reads all audio, transcribes, returns results.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var audioInputs = audioSamples
            .Select(s => new AudioData(s, _options.SampleRate))
            .ToList();
        var transcriptions = Transcribe(audioInputs);

        var outputRows = transcriptions
            .Select(t => new TranscriptionOutput { Text = t })
            .ToArray();

        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "OnnxSpeechToTextTransformer wraps ORT GenAI and cannot be serialized. " +
            "The model must be loaded from disk at runtime.");

    public void Dispose()
    {
        _tokenizer?.Dispose();
        _model?.Dispose();
    }

    private string TranscribeSingle(AudioData audio)
    {
        var tokenIds = RunGeneration(audio);
        var text = _tokenizer.Decode(tokenIds);
        return CleanTranscription(text);
    }

    /// <summary>
    /// Core generation: audio → mel features → ORT GenAI encoder-decoder → token IDs.
    /// </summary>
    private int[] RunGeneration(AudioData audio)
    {
        // Stage 1: Feature extraction — audio → mel spectrogram
        var features = _featureExtractor.Extract(audio);
        int numFrames = features.GetLength(0);
        int numMels = features.GetLength(1);

        // Reshape to [1, numMels, numFrames] — ORT GenAI expects (batch, mels, frames)
        var melData = new float[numMels * numFrames];
        for (int m = 0; m < numMels; m++)
            for (int f = 0; f < numFrames; f++)
                melData[m * numFrames + f] = features[f, m];

        // Stage 2: Set up ORT GenAI generation
        using var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", _options.MaxLength);

        // Pin mel data and create ORT GenAI Tensor from raw pointer
        var melHandle = GCHandle.Alloc(melData, GCHandleType.Pinned);
        try
        {
            using var melTensor = new Tensor(
                melHandle.AddrOfPinnedObject(),
                [1, numMels, numFrames],
                ElementType.float32);

            // Stage 3: Create generator and set audio input
            using var generator = new Generator(_model, generatorParams);
            generator.SetModelInput("input_features", melTensor);

            // Append decoder prompt tokens (SOT + language + task)
            var promptTokens = BuildDecoderPrompt();
            if (promptTokens.Length > 0)
            {
                generator.AppendTokens(promptTokens);
            }

            // Stage 4: Autoregressive decode loop
            var outputTokens = new List<int>();

            while (!generator.IsDone())
            {
                generator.GenerateNextToken();

                var sequence = generator.GetSequence(0);
                if (sequence.Length > 0)
                {
                    int latestToken = sequence[sequence.Length - 1];
                    outputTokens.Add(latestToken);
                }
            }

            return outputTokens.ToArray();
        }
        finally
        {
            melHandle.Free();
        }
    }

    /// <summary>
    /// Build the Whisper decoder prompt token sequence.
    /// For transcription: [SOT, language, transcribe/translate]
    /// </summary>
    private int[] BuildDecoderPrompt()
    {
        var whisperTokenizer = new WhisperTokenizer(isMultilingual: _options.IsMultilingual);
        return whisperTokenizer.GetStartOfTranscriptSequence(
            _options.Language,
            _options.Translate);
    }

    /// <summary>
    /// Clean up Whisper's raw text output — strip special tokens and normalize whitespace.
    /// </summary>
    private static string CleanTranscription(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        // Remove common Whisper special token markers that may leak through
        text = text.Replace("<|endoftext|>", "")
                   .Replace("<|startoftranscript|>", "")
                   .Replace("<|transcribe|>", "")
                   .Replace("<|translate|>", "")
                   .Replace("<|nospeech|>", "")
                   .Replace("<|notimestamps|>", "");

        // Remove timestamp tokens: <|0.00|> through <|30.00|>
        int idx;
        while ((idx = text.IndexOf("<|")) >= 0)
        {
            int end = text.IndexOf("|>", idx);
            if (end > idx)
                text = text.Remove(idx, end - idx + 2);
            else
                break;
        }

        return text.Trim();
    }

    private List<float[]> ReadAudioColumn(IDataView data)
    {
        var result = new List<float[]>();
        var col = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' not found in schema.");

        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<VBuffer<float>>(col);
        var buffer = default(VBuffer<float>);

        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.DenseValues().ToArray());
        }

        return result;
    }
}

/// <summary>
/// Structured result from Whisper transcription including timestamps.
/// </summary>
public class TranscriptionResult
{
    /// <summary>Full transcribed text.</summary>
    public required string Text { get; init; }

    /// <summary>Timestamped segments (if timestamps were generated).</summary>
    public TranscriptionSegment[] Segments { get; init; } = [];

    /// <summary>Raw token IDs from the decoder.</summary>
    public int[] TokenIds { get; init; } = [];

    /// <summary>Detected or specified language.</summary>
    public string? Language { get; init; }
}

internal class TranscriptionOutput
{
    [ColumnName("Text")]
    public string Text { get; set; } = "";
}
