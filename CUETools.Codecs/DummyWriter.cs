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

        public AudioEncoderSettings Settings
        {
            get
            {
                return new AudioEncoderSettings();
            }
            set
            {
                if (value != null && value.GetType() != typeof(AudioEncoderSettings))
                    throw new Exception("Unsupported options " + value);
            }
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
