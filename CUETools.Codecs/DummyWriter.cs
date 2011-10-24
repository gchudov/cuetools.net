using System;

namespace CUETools.Codecs
{
    public class DummyWriter : IAudioDest
    {
        AudioPCMConfig _pcm;

        public DummyWriter(string path, AudioPCMConfig pcm)
        {
            _pcm = pcm;
        }

        public void Close()
        {
        }

        public void Delete()
        {
        }

        public long FinalSampleCount
        {
            set { }
        }

        public int CompressionLevel
        {
            get { return 0; }
            set { }
        }

        public object Settings
        {
            get
            {
                return null;
            }
            set
            {
                if (value != null && value.GetType() != typeof(object))
                    throw new Exception("Unsupported options " + value);
            }
        }

        public long Padding
        {
            set { }
        }

        public long BlockSize
        {
            set { }
        }

        public AudioPCMConfig PCM
        {
            get { return _pcm; }
        }

        public void Write(AudioBuffer buff)
        {
        }

        public string Path { get { return null; } }
    }
}
