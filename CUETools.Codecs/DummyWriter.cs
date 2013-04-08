using System;

namespace CUETools.Codecs
{
    public class DummyWriter : IAudioDest
    {
        AudioEncoderSettings m_settings;

        public DummyWriter(string path, AudioEncoderSettings settings)
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

        public AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
        }

        public void Write(AudioBuffer buff)
        {
        }

        public string Path { get { return null; } }
    }
}
