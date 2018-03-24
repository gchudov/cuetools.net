using System;

namespace CUETools.Codecs.NULL
{
    public class AudioEncoder : IAudioDest
    {
        AudioEncoderSettings m_settings;

        public AudioEncoder(string path, AudioEncoderSettings settings)
        {
            m_settings = settings;
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

        public AudioEncoderSettings Settings => m_settings;

        public void Write(AudioBuffer buff)
        {
        }

        public string Path => null;
    }
}
