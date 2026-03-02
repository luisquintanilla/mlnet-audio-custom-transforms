using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace MLNet.AudioInference.Onnx;

/// <summary>
/// Options for the ONNX audio model scorer sub-transform.
/// </summary>
public class OnnxAudioScoringOptions
{
    /// <summary>Path to ONNX model file.</summary>
    public required string ModelPath { get; set; }

    /// <summary>Name of the input column containing features (VBuffer&lt;float&gt;).</summary>
    public string InputColumnName { get; set; } = "Features";

    /// <summary>Name of the output column for raw model scores (VBuffer&lt;float&gt;).</summary>
    public string OutputColumnName { get; set; } = "Scores";

    /// <summary>ONNX input tensor name. Auto-detected if null.</summary>
    public string? InputTensorName { get; set; }

    /// <summary>ONNX output tensor name. Auto-detected if null.</summary>
    public string? OutputTensorName { get; set; }

    /// <summary>GPU device ID. Null for CPU.</summary>
    public int? GpuDeviceId { get; set; }
}

/// <summary>
/// Estimator that creates an OnnxAudioScoringTransformer.
/// Stage 2 of the composed audio pipeline: features → raw ONNX output scores.
/// </summary>
public sealed class OnnxAudioScoringEstimator : IEstimator<OnnxAudioScoringTransformer>
{
    private readonly IHostEnvironment _env;
    private readonly OnnxAudioScoringOptions _options;

    public OnnxAudioScoringEstimator(IHostEnvironment env, OnnxAudioScoringOptions options)
    {
        _env = env;
        _options = options;
    }

    public OnnxAudioScoringTransformer Fit(IDataView input)
    {
        return new OnnxAudioScoringTransformer(_env, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        var colCtor = typeof(SchemaShape.Column)
            .GetConstructors(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public)
            .First(c => c.GetParameters().Length == 5);

        columns[_options.OutputColumnName] = (SchemaShape.Column)colCtor.Invoke(
            [_options.OutputColumnName, SchemaShape.Column.VectorKind.VariableVector,
             NumberDataViewType.Single, false, null]);

        return new SchemaShape(columns.Values);
    }
}
