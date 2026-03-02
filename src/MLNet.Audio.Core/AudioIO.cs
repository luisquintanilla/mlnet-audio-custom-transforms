using System.Buffers.Binary;

namespace MLNet.Audio.Core;

/// <summary>
/// Reads and writes WAV audio files and performs basic audio transformations
/// (resampling, mono conversion).
/// </summary>
public static class AudioIO
{
    /// <summary>
    /// Loads a WAV file from disk and returns mono AudioData.
    /// Supports 16-bit PCM and 32-bit float WAV formats.
    /// </summary>
    public static AudioData LoadWav(string path)
    {
        using var stream = File.OpenRead(path);
        return LoadWav(stream);
    }

    /// <summary>
    /// Loads a WAV file from a stream and returns mono AudioData.
    /// </summary>
    public static AudioData LoadWav(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        // RIFF header
        var riff = reader.ReadBytes(4);
        if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F')
            throw new InvalidDataException("Not a valid WAV file: missing RIFF header.");

        reader.ReadInt32(); // file size

        var wave = reader.ReadBytes(4);
        if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E')
            throw new InvalidDataException("Not a valid WAV file: missing WAVE marker.");

        // Find fmt chunk
        int audioFormat = 0, numChannels = 0, sampleRate = 0, bitsPerSample = 0;
        bool foundFmt = false;

        while (stream.Position < stream.Length)
        {
            var chunkId = reader.ReadBytes(4);
            var chunkSize = reader.ReadInt32();
            var chunkIdStr = System.Text.Encoding.ASCII.GetString(chunkId);

            if (chunkIdStr == "fmt ")
            {
                audioFormat = reader.ReadInt16();   // 1 = PCM, 3 = IEEE float
                numChannels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32();                 // byte rate
                reader.ReadInt16();                 // block align
                bitsPerSample = reader.ReadInt16();

                // Skip any extra fmt bytes
                var extraBytes = chunkSize - 16;
                if (extraBytes > 0)
                    reader.ReadBytes(extraBytes);

                foundFmt = true;
            }
            else if (chunkIdStr == "data")
            {
                if (!foundFmt)
                    throw new InvalidDataException("WAV file has data chunk before fmt chunk.");

                var samples = ReadSamples(reader, chunkSize, audioFormat, bitsPerSample, numChannels);
                return new AudioData(samples, sampleRate, channels: 1);
            }
            else
            {
                // Skip unknown chunks
                reader.ReadBytes(chunkSize);
            }
        }

        throw new InvalidDataException("WAV file missing data chunk.");
    }

    /// <summary>
    /// Saves AudioData to a 16-bit PCM WAV file.
    /// </summary>
    public static void SaveWav(string path, AudioData audio)
    {
        using var stream = File.Create(path);
        SaveWav(stream, audio);
    }

    /// <summary>
    /// Writes AudioData as 16-bit PCM WAV to a stream.
    /// </summary>
    public static void SaveWav(Stream stream, AudioData audio)
    {
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        int bitsPerSample = 16;
        int numChannels = audio.Channels;
        int byteRate = audio.SampleRate * numChannels * bitsPerSample / 8;
        int blockAlign = numChannels * bitsPerSample / 8;
        int dataSize = audio.Samples.Length * bitsPerSample / 8;

        // RIFF header
        writer.Write("RIFF"u8);
        writer.Write(36 + dataSize);
        writer.Write("WAVE"u8);

        // fmt chunk
        writer.Write("fmt "u8);
        writer.Write(16);                   // chunk size
        writer.Write((short)1);             // PCM format
        writer.Write((short)numChannels);
        writer.Write(audio.SampleRate);
        writer.Write(byteRate);
        writer.Write((short)blockAlign);
        writer.Write((short)bitsPerSample);

        // data chunk
        writer.Write("data"u8);
        writer.Write(dataSize);

        foreach (var sample in audio.Samples)
        {
            var clamped = Math.Clamp(sample, -1.0f, 1.0f);
            var value = (short)(clamped * short.MaxValue);
            writer.Write(value);
        }
    }

    /// <summary>
    /// Resamples audio to a target sample rate using linear interpolation.
    /// </summary>
    public static AudioData Resample(AudioData audio, int targetSampleRate)
    {
        if (audio.SampleRate == targetSampleRate)
            return audio;

        double ratio = (double)audio.SampleRate / targetSampleRate;
        int newLength = (int)(audio.Samples.Length / ratio);
        var resampled = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            double srcIndex = i * ratio;
            int idx0 = (int)srcIndex;
            int idx1 = Math.Min(idx0 + 1, audio.Samples.Length - 1);
            double frac = srcIndex - idx0;
            resampled[i] = (float)(audio.Samples[idx0] * (1.0 - frac) + audio.Samples[idx1] * frac);
        }

        return new AudioData(resampled, targetSampleRate, audio.Channels);
    }

    /// <summary>
    /// Converts multi-channel audio to mono by averaging channels.
    /// </summary>
    public static AudioData ToMono(AudioData audio)
    {
        if (audio.Channels == 1)
            return audio;

        int numSamples = audio.Samples.Length / audio.Channels;
        var mono = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            float sum = 0;
            for (int ch = 0; ch < audio.Channels; ch++)
                sum += audio.Samples[i * audio.Channels + ch];
            mono[i] = sum / audio.Channels;
        }

        return new AudioData(mono, audio.SampleRate, channels: 1);
    }

    private static float[] ReadSamples(BinaryReader reader, int dataSize, int audioFormat, int bitsPerSample, int numChannels)
    {
        float[] rawSamples;

        if (audioFormat == 1 && bitsPerSample == 16)
        {
            int numSamples = dataSize / 2;
            rawSamples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
                rawSamples[i] = reader.ReadInt16() / (float)short.MaxValue;
        }
        else if (audioFormat == 1 && bitsPerSample == 8)
        {
            rawSamples = new float[dataSize];
            for (int i = 0; i < dataSize; i++)
                rawSamples[i] = (reader.ReadByte() - 128) / 128.0f;
        }
        else if (audioFormat == 3 && bitsPerSample == 32)
        {
            int numSamples = dataSize / 4;
            rawSamples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
                rawSamples[i] = reader.ReadSingle();
        }
        else if (audioFormat == 1 && bitsPerSample == 24)
        {
            int numSamples = dataSize / 3;
            rawSamples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                var b = reader.ReadBytes(3);
                int value = b[0] | (b[1] << 8) | (b[2] << 16);
                if ((value & 0x800000) != 0)
                    value |= unchecked((int)0xFF000000); // sign extend
                rawSamples[i] = value / 8388608.0f; // 2^23
            }
        }
        else
        {
            throw new NotSupportedException(
                $"Unsupported WAV format: audioFormat={audioFormat}, bitsPerSample={bitsPerSample}");
        }

        // Mix to mono if multi-channel
        if (numChannels > 1)
        {
            int monoLength = rawSamples.Length / numChannels;
            var mono = new float[monoLength];
            for (int i = 0; i < monoLength; i++)
            {
                float sum = 0;
                for (int ch = 0; ch < numChannels; ch++)
                    sum += rawSamples[i * numChannels + ch];
                mono[i] = sum / numChannels;
            }
            return mono;
        }

        return rawSamples;
    }
}
