using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.LossyWAV
{
    public class LossyWAVReader : IAudioSource
    {
        private IAudioSource _audioSource, _lwcdfSource;
        private AudioBuffer lwcdfBuffer;
        private double scaling_factor;

        public IAudioDecoderSettings Settings => null;

        public long Length
        {
            get
            {
                return _audioSource.Length;
            }
        }

        public long Position
        {
            get
            {
                return _audioSource.Position;
            }
            set
            {
                _audioSource.Position = value;
                _lwcdfSource.Position = value;
            }
        }

        public long Remaining
        {
            get
            {
                return _audioSource.Remaining;
            }
        }

        public AudioPCMConfig PCM
        {
            get
            {
                return _audioSource.PCM;
            }
        }

        public string Path
        {
            get
            {
                return _audioSource.Path;
            }
        }

        public LossyWAVReader(IAudioSource audioSource, IAudioSource lwcdfSource)
        {
            _audioSource = audioSource;
            _lwcdfSource = lwcdfSource;

            if (_audioSource.Length != _lwcdfSource.Length)
                throw new Exception("Data not same length");
            if (_audioSource.PCM.BitsPerSample != _lwcdfSource.PCM.BitsPerSample
                || _audioSource.PCM.ChannelCount != _lwcdfSource.PCM.ChannelCount
                || _audioSource.PCM.SampleRate != _lwcdfSource.PCM.SampleRate)
                throw new Exception("FMT Data mismatch");

            scaling_factor = 1.0; // !!!! Need to read 'fact' chunks or tags here
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            if (lwcdfBuffer == null || lwcdfBuffer.Size < buff.Size)
                lwcdfBuffer = new AudioBuffer(_lwcdfSource, buff.Size);
            int sampleCount = _audioSource.Read(buff, maxLength);
            if (sampleCount != _lwcdfSource.Read(lwcdfBuffer, maxLength))
                throw new Exception("size mismatch"); // Very likely to happen (depending on lwcdfSource implementation)
            for (uint i = 0; i < sampleCount; i++)
                for (int c = 0; c < buff.PCM.ChannelCount; c++)
                    buff.Samples[i, c] = (int)Math.Round(buff.Samples[i, c] / scaling_factor + lwcdfBuffer.Samples[i, c]);
            return sampleCount;
        }


        public void Close()
        {
            _audioSource.Close();
            _lwcdfSource.Close();
        }
    }
}
