using System;

namespace CUETools.Codecs.NULL
{
    public class AudioEncoder : IAudioDest
    {
        IAudioEncoderSettings m_settings;

        public AudioEncoder(string path, IAudioEncoderSettings settings)
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

        public IAudioEncoderSettings Settings => m_settings;

        public void Write(AudioBuffer buff)
        {
        }

        public string Path => null;
    }
}
