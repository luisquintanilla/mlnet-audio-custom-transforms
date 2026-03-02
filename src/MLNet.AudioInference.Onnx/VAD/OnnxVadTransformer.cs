using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Audio.Core;
using System.Runtime.CompilerServices;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET transformer for Voice Activity Detection using ONNX models (e.g., Silero VAD).
/// Processes audio in fixed-size windows, scores each with the VAD model,
/// then merges adjacent speech frames into segments with timestamps.
/// Silero VAD is stateful — it maintains hidden state (h/c) across frames.
/// </summary>
public sealed class OnnxVadTransformer : ITransformer, IVoiceActivityDetector, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxVadOptions _options;
    private readonly InferenceSession _session;

    public bool IsRowToRowMapper => true;

    internal OnnxVadTransformer(MLContext mlContext, OnnxVadOptions options)
    {
        _mlContext = mlContext;
        _options = options;

        var sessionOptions = new SessionOptions();
        if (options.GpuDeviceId.HasValue)
        {
            try { sessionOptions.AppendExecutionProvider_CUDA(options.GpuDeviceId.Value); }
            catch { /* fallback to CPU */ }
        }

        _session = new InferenceSession(options.ModelPath, sessionOptions);
    }

    /// <summary>
    /// Detect speech segments in audio samples directly (outside ML.NET pipeline).
    /// </summary>
    public SpeechSegment[] DetectSpeech(AudioData audio)
    {
        var probabilities = ScoreFrames(audio);
        return MergeSegments(probabilities, audio.SampleRate);
    }

    /// <summary>
    /// Detect speech segments from a stream (IVoiceActivityDetector implementation).
    /// </summary>
    public async IAsyncEnumerable<SpeechSegment> DetectSpeechAsync(
        Stream audioStream,
        VadOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var audio = AudioIO.LoadWav(audioStream);
        var threshold = options?.Threshold ?? _options.Threshold;
        var minSpeech = options?.MinSpeechDuration ?? _options.MinSpeechDuration;
        var minSilence = options?.MinSilenceDuration ?? _options.MinSilenceDuration;
        var speechPad = options?.SpeechPad ?? _options.SpeechPad;

        var probabilities = ScoreFrames(audio);
        var segments = MergeSegments(probabilities, audio.SampleRate, threshold, minSpeech, minSilence, speechPad);

        foreach (var segment in segments)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return segment;
        }

        await Task.CompletedTask; // Ensure async signature is valid
    }

    /// <summary>
    /// ML.NET Transform — eager evaluation. Detects speech segments for each audio row.
    /// Outputs the number of speech segments detected and total speech duration.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var audioSamples = ReadAudioColumn(input);
        var outputRows = new List<VadOutput>();

        foreach (var samples in audioSamples)
        {
            var audio = new AudioData(samples, _options.SampleRate);
            var segments = DetectSpeech(audio);

            outputRows.Add(new VadOutput
            {
                SegmentCount = segments.Length,
                TotalSpeechSeconds = (float)segments.Sum(s => s.Duration.TotalSeconds),
                SegmentStarts = segments.Select(s => (float)s.Start.TotalSeconds).ToArray(),
                SegmentEnds = segments.Select(s => (float)s.End.TotalSeconds).ToArray(),
                SegmentConfidences = segments.Select(s => s.Confidence).ToArray()
            });
        }

        return _mlContext.Data.LoadFromEnumerable(outputRows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn("SegmentCount", NumberDataViewType.Int32);
        builder.AddColumn("TotalSpeechSeconds", NumberDataViewType.Single);
        builder.AddColumn("SegmentStarts", new VectorDataViewType(NumberDataViewType.Single));
        builder.AddColumn("SegmentEnds", new VectorDataViewType(NumberDataViewType.Single));
        builder.AddColumn("SegmentConfidences", new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use ModelPackager.Save() instead.");

    public void Dispose()
    {
        _session?.Dispose();
    }

    /// <summary>
    /// Score each audio frame with the VAD model.
    /// Returns speech probability per frame.
    /// Silero VAD expects: input [1, windowSize], sr [1], h [2, 1, 64], c [2, 1, 64]
    /// Returns: output [1, 1] (speech probability), hn, cn (updated hidden state)
    /// </summary>
    private (float probability, int frameIndex)[] ScoreFrames(AudioData audio)
    {
        int windowSize = _options.WindowSize;
        int totalSamples = audio.Samples.Length;
        int numFrames = totalSamples / windowSize;

        var results = new List<(float probability, int frameIndex)>();

        // Initialize hidden states for Silero VAD (h and c are LSTM states)
        var h = new float[2 * 1 * 64]; // [2, 1, 64] flattened
        var c = new float[2 * 1 * 64]; // [2, 1, 64] flattened
        var sr = new long[] { _options.SampleRate };

        for (int frame = 0; frame < numFrames; frame++)
        {
            int offset = frame * windowSize;
            var frameData = new float[windowSize];
            Array.Copy(audio.Samples, offset, frameData, 0, windowSize);

            // Create input tensors
            using var inputOrt = OrtValue.CreateTensorValueFromMemory(frameData, [1, windowSize]);
            using var srOrt = OrtValue.CreateTensorValueFromMemory(sr, [1]);
            using var hOrt = OrtValue.CreateTensorValueFromMemory(h.ToArray(), [2, 1, 64]);
            using var cOrt = OrtValue.CreateTensorValueFromMemory(c.ToArray(), [2, 1, 64]);

            var inputs = new Dictionary<string, OrtValue>
            {
                ["input"] = inputOrt,
                ["sr"] = srOrt,
                ["h"] = hOrt,
                ["c"] = cOrt
            };

            // Determine output names (Silero VAD outputs: output, hn, cn)
            var outputNames = _session.OutputNames.ToArray();

            using var runResults = _session.Run(new RunOptions(), inputs, outputNames);

            // First output: speech probability
            float prob = runResults[0].GetTensorDataAsSpan<float>()[0];
            results.Add((prob, frame));

            // Update hidden states for next frame (stateful processing)
            if (runResults.Count > 1)
            {
                var hnSpan = runResults[1].GetTensorDataAsSpan<float>();
                hnSpan.CopyTo(h);
            }
            if (runResults.Count > 2)
            {
                var cnSpan = runResults[2].GetTensorDataAsSpan<float>();
                cnSpan.CopyTo(c);
            }
        }

        return results.ToArray();
    }

    /// <summary>
    /// Merge frame-level speech probabilities into speech segments.
    /// Applies thresholding, minimum duration filtering, and padding.
    /// </summary>
    private SpeechSegment[] MergeSegments(
        (float probability, int frameIndex)[] frameProbabilities,
        int sampleRate,
        float? threshold = null,
        TimeSpan? minSpeechDuration = null,
        TimeSpan? minSilenceDuration = null,
        TimeSpan? speechPad = null)
    {
        float thresh = threshold ?? _options.Threshold;
        var minSpeech = minSpeechDuration ?? _options.MinSpeechDuration;
        var minSilence = minSilenceDuration ?? _options.MinSilenceDuration;
        var pad = speechPad ?? _options.SpeechPad;

        int windowSize = _options.WindowSize;
        double frameDuration = (double)windowSize / sampleRate;

        // Step 1: Identify speech frames
        var speechFrames = new List<(int startFrame, int endFrame, float avgConfidence)>();
        int? currentStart = null;
        float confidenceSum = 0;
        int confidenceCount = 0;

        for (int i = 0; i < frameProbabilities.Length; i++)
        {
            var (prob, _) = frameProbabilities[i];

            if (prob >= thresh)
            {
                if (!currentStart.HasValue)
                    currentStart = i;
                confidenceSum += prob;
                confidenceCount++;
            }
            else if (currentStart.HasValue)
            {
                // Check if silence is long enough to split
                int silenceFrames = 0;
                for (int j = i; j < frameProbabilities.Length && frameProbabilities[j].probability < thresh; j++)
                    silenceFrames++;

                var silenceDuration = TimeSpan.FromSeconds(silenceFrames * frameDuration);
                if (silenceDuration >= minSilence)
                {
                    speechFrames.Add((currentStart.Value, i - 1, confidenceSum / confidenceCount));
                    currentStart = null;
                    confidenceSum = 0;
                    confidenceCount = 0;
                }
                else
                {
                    // Silence is too short, continue the segment
                    confidenceSum += prob;
                    confidenceCount++;
                }
            }
        }

        // Close any open segment
        if (currentStart.HasValue)
        {
            speechFrames.Add((currentStart.Value, frameProbabilities.Length - 1,
                confidenceCount > 0 ? confidenceSum / confidenceCount : 0));
        }

        // Step 2: Convert frames to time segments with padding and filter by min duration
        var segments = new List<SpeechSegment>();
        var audioDuration = TimeSpan.FromSeconds((double)frameProbabilities.Length * frameDuration);

        foreach (var (startFrame, endFrame, avgConf) in speechFrames)
        {
            var start = TimeSpan.FromSeconds(startFrame * frameDuration) - pad;
            var end = TimeSpan.FromSeconds((endFrame + 1) * frameDuration) + pad;

            // Clamp to audio bounds
            if (start < TimeSpan.Zero) start = TimeSpan.Zero;
            if (end > audioDuration) end = audioDuration;

            var duration = end - start;
            if (duration >= minSpeech)
            {
                segments.Add(new SpeechSegment(start, end, avgConf));
            }
        }

        return segments.ToArray();
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

internal class VadOutput
{
    [ColumnName("SegmentCount")]
    public int SegmentCount { get; set; }

    [ColumnName("TotalSpeechSeconds")]
    public float TotalSpeechSeconds { get; set; }

    [ColumnName("SegmentStarts")]
    [VectorType]
    public float[] SegmentStarts { get; set; } = [];

    [ColumnName("SegmentEnds")]
    [VectorType]
    public float[] SegmentEnds { get; set; } = [];

    [ColumnName("SegmentConfidences")]
    [VectorType]
    public float[] SegmentConfidences { get; set; } = [];
}
