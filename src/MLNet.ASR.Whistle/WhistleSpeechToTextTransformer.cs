using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Audio.Core;

namespace MLNet.ASR.Whistle;

/// <summary>ML.NET estimator for Whistle speech-to-text.</summary>
public sealed class WhistleSpeechToTextEstimator : IEstimator<WhistleSpeechToTextTransformer>
{
    private readonly MLContext _mlContext;
    private readonly WhistleTransformOptions _options;

    /// <summary>Creates an estimator using the supplied ML.NET context and Whistle options.</summary>
    public WhistleSpeechToTextEstimator(MLContext mlContext, WhistleTransformOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);
        _mlContext = mlContext;
        _options = options;
    }

    /// <inheritdoc />
    public WhistleSpeechToTextTransformer Fit(IDataView input)
        => new(_mlContext, _options);

    /// <inheritdoc />
    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);
        var ctor = typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)ctor.Invoke(
        [
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}

/// <summary>ML.NET options for the Whistle transformer.</summary>
public sealed class WhistleTransformOptions : WhistleOptions
{
    /// <summary>Name of the input vector column containing PCM samples.</summary>
    public string InputColumnName { get; init; } = "Audio";

    /// <summary>Name of the output text column.</summary>
    public string OutputColumnName { get; init; } = "Text";
}

/// <summary>ML.NET transformer that eagerly transcribes each audio row.</summary>
public sealed class WhistleSpeechToTextTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly WhistleTransformOptions _options;
    private readonly WhistleTranscriber _transcriber;

    /// <inheritdoc />
    public bool IsRowToRowMapper => true;

    /// <summary>Creates a transformer using the supplied ML.NET context and Whistle options.</summary>
    public WhistleSpeechToTextTransformer(MLContext mlContext, WhistleTransformOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);
        _mlContext = mlContext;
        _options = options;
        _transcriber = new WhistleTranscriber(options);
    }

    /// <inheritdoc />
    public IDataView Transform(IDataView input)
    {
        var audio = ReadAudioColumn(input);
        var output = audio
            .Select(samples => new WhistleOutput
            {
                Text = _transcriber.Transcribe(new AudioData(samples, _options.SampleRate)).Text
            })
            .ToArray();

        var outputData = _mlContext.Data.LoadFromEnumerable(output);
        if (string.Equals(_options.OutputColumnName, "Text", StringComparison.Ordinal))
            return outputData;

        var copied = _mlContext.Transforms
            .CopyColumns(_options.OutputColumnName, "Text")
            .Fit(outputData)
            .Transform(outputData);

        return _mlContext.Transforms
            .DropColumns("Text")
            .Fit(copied)
            .Transform(copied);
    }

    /// <inheritdoc />
    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    /// <inheritdoc />
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Use Transform() directly.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "WhistleSpeechToTextTransformer wraps a native runtime and cannot be serialized.");

    /// <inheritdoc />
    public void Dispose() => _transcriber.Dispose();

    private List<float[]> ReadAudioColumn(IDataView data)
    {
        var column = data.Schema.GetColumnOrNull(_options.InputColumnName)
            ?? throw new InvalidOperationException(
                $"Input column '{_options.InputColumnName}' was not found in the data schema.");

        var result = new List<float[]>();
        using var cursor = data.GetRowCursor([column]);
        var getter = cursor.GetGetter<VBuffer<float>>(column);
        var buffer = default(VBuffer<float>);
        while (cursor.MoveNext())
        {
            getter(ref buffer);
            result.Add(buffer.DenseValues().ToArray());
        }

        return result;
    }

    private sealed class WhistleOutput
    {
        [ColumnName("Text")]
        public string Text { get; set; } = string.Empty;
    }
}
