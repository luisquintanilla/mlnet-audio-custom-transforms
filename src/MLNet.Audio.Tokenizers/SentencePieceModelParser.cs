using Google.Protobuf;

namespace MLNet.Audio.Tokenizers;

/// <summary>
/// Minimal parser for SentencePiece .model protobuf files.
/// Extracts only the fields needed for Char model tokenization:
/// vocabulary pieces and trainer spec model type.
///
/// SentencePiece ModelProto schema (simplified):
///   message ModelProto {
///     repeated SentencePiece pieces = 1;
///     TrainerSpec trainer_spec = 2;
///   }
///   message SentencePiece {
///     string piece = 1;
///     float score = 2;
///     Type type = 3;  // NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
///   }
///   message TrainerSpec {
///     repeated string input = 1;
///     string model_prefix = 2;
///     ModelType model_type = 3;  // UNIGRAM=1, BPE=2, WORD=3, CHAR=4
///   }
/// </summary>
internal static class SentencePieceModelParser
{
    internal enum ModelType
    {
        Unigram = 1,
        Bpe = 2,
        Word = 3,
        Char = 4
    }

    internal enum PieceType
    {
        Normal = 1,
        Unknown = 2,
        Control = 3,
        UserDefined = 4,
        Unused = 5,
        Byte = 6
    }

    internal record Piece(string Token, float Score, PieceType Type);

    internal record ParsedModel(List<Piece> Pieces, ModelType Type);

    /// <summary>
    /// Parse a SentencePiece .model file, extracting vocabulary and model type.
    /// </summary>
    internal static ParsedModel Parse(Stream stream)
    {
        var pieces = new List<Piece>();
        var modelType = ModelType.Unigram; // default

        using var input = new CodedInputStream(stream, leaveOpen: true);
        uint tag;
        while ((tag = input.ReadTag()) != 0)
        {
            int fieldNumber = WireFormat.GetTagFieldNumber(tag);
            var wireType = WireFormat.GetTagWireType(tag);

            switch (fieldNumber)
            {
                case 1 when wireType == WireFormat.WireType.LengthDelimited:
                    // repeated SentencePiece pieces
                    var pieceBytes = input.ReadBytes();
                    pieces.Add(ParsePiece(pieceBytes));
                    break;

                case 2 when wireType == WireFormat.WireType.LengthDelimited:
                    // TrainerSpec trainer_spec
                    var specBytes = input.ReadBytes();
                    modelType = ParseTrainerSpecModelType(specBytes);
                    break;

                default:
                    input.SkipLastField();
                    break;
            }
        }

        return new ParsedModel(pieces, modelType);
    }

    private static Piece ParsePiece(ByteString data)
    {
        string token = "";
        float score = 0f;
        var type = PieceType.Normal;

        using var input = new CodedInputStream(data.ToByteArray());
        uint tag;
        while ((tag = input.ReadTag()) != 0)
        {
            int fieldNumber = WireFormat.GetTagFieldNumber(tag);
            switch (fieldNumber)
            {
                case 1:
                    token = input.ReadString();
                    break;
                case 2:
                    score = input.ReadFloat();
                    break;
                case 3:
                    type = (PieceType)input.ReadInt32();
                    break;
                default:
                    input.SkipLastField();
                    break;
            }
        }

        return new Piece(token, score, type);
    }

    private static ModelType ParseTrainerSpecModelType(ByteString data)
    {
        using var input = new CodedInputStream(data.ToByteArray());
        uint tag;
        while ((tag = input.ReadTag()) != 0)
        {
            int fieldNumber = WireFormat.GetTagFieldNumber(tag);
            var wireType = WireFormat.GetTagWireType(tag);
            // model_type is field 3 in TrainerSpec (field 1 = input, field 2 = model_prefix)
            if (fieldNumber == 3 && wireType == WireFormat.WireType.Varint)
                return (ModelType)input.ReadInt32();
            input.SkipLastField();
        }

        return ModelType.Unigram; // default
    }
}
