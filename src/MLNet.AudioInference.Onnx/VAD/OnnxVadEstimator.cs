using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// ML.NET estimator for Voice Activity Detection using ONNX models (e.g., Silero VAD).
/// Call Fit() to produce an OnnxVadTransformer.
/// </summary>
public sealed class OnnxVadEstimator : IEstimator<OnnxVadTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxVadOptions _options;

    public OnnxVadEstimator(MLContext mlContext, OnnxVadOptions options)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(options);

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX VAD model not found: {options.ModelPath}");

        _mlContext = mlContext;
        _options = options;
    }

    public OnnxVadTransformer Fit(IDataView input)
    {
        return new OnnxVadTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name, c => c);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns["SegmentCount"] = (SchemaShape.Column)colCtor.Invoke([
            "SegmentCount",
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)NumberDataViewType.Int32,
            false,
            (SchemaShape?)null
        ]);

        columns["TotalSpeechSeconds"] = (SchemaShape.Column)colCtor.Invoke([
            "TotalSpeechSeconds",
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        columns["SegmentStarts"] = (SchemaShape.Column)colCtor.Invoke([
            "SegmentStarts",
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        columns["SegmentEnds"] = (SchemaShape.Column)colCtor.Invoke([
            "SegmentEnds",
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        columns["SegmentConfidences"] = (SchemaShape.Column)colCtor.Invoke([
            "SegmentConfidences",
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);

        return new SchemaShape(columns.Values);
    }
}
