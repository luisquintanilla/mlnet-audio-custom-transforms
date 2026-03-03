# MLNet.Audio.Tokenizers

Audio-specific tokenizer extensions for [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers).

## Purpose

Fills gaps in `Microsoft.ML.Tokenizers` for audio/speech model tokenization. Extends the `Tokenizer` base class so custom tokenizers integrate seamlessly with the .NET AI tokenizer ecosystem.

## Tokenizers

### SentencePieceCharTokenizer

SentencePiece **Char** model tokenizer. `Microsoft.ML.Tokenizers.SentencePieceTokenizer` supports BPE and Unigram models but throws `ArgumentException` for Char models (used by SpeechT5 and other character-level models).

```csharp
// Create from a SentencePiece .model file
var tokenizer = SentencePieceCharTokenizer.Create("spm_char.model");

// Standard Tokenizer API — works anywhere a Tokenizer is expected
IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello world");
string? decoded = tokenizer.Decode(ids);
```

Character-level tokenization is the simplest scheme: each character maps to a vocabulary ID. The ▁ (U+2581) character represents word boundaries.

## Dependencies

- `Microsoft.ML.Tokenizers` 2.0.0 — base `Tokenizer` class
- `Google.Protobuf` (transitive) — SentencePiece `.model` file parsing
