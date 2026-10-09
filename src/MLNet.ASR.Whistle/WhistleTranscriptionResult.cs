namespace MLNet.ASR.Whistle;

/// <summary>Structured result returned by Whistle transcription.</summary>
public sealed class WhistleTranscriptionResult
{
    /// <summary>Recognized text.</summary>
    public required string Text { get; init; }

    /// <summary>Detected or requested language.</summary>
    public string? Language { get; init; }

    /// <summary>Time to first token reported by Needle.</summary>
    public TimeSpan TimeToFirstToken { get; init; }

    /// <summary>Decoder throughput reported by Needle.</summary>
    public double DecodeTokensPerSecond { get; init; }

    /// <summary>Word-level timestamps returned by Needle.</summary>
    public IReadOnlyList<WhistleWord> Words { get; init; } = [];
}

/// <summary>A word and its timing/probability metadata.</summary>
public sealed class WhistleWord
{
    /// <summary>Recognized word text.</summary>
    public required string Word { get; init; }

    /// <summary>Word start time from the beginning of the clip.</summary>
    public TimeSpan Start { get; init; }

    /// <summary>Word end time from the beginning of the clip.</summary>
    public TimeSpan End { get; init; }

    /// <summary>Native confidence estimate.</summary>
    public double Probability { get; init; }
}
