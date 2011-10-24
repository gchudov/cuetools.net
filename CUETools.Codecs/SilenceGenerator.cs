namespace CUETools.Codecs
{
    public class SilenceGenerator : IAudioSource
    {
        private long _sampleOffset, _sampleCount;
        private AudioPCMConfig pcm;
        private int _sampleVal;

        public long Length
        {
            get { return _sampleCount; }
        }

        public long Remaining
        {
            get { return _sampleCount - _sampleOffset; }
        }

        public long Position
        {
            get { return _sampleOffset; }
            set { _sampleOffset = value; }
        }

        public AudioPCMConfig PCM { get { return pcm; } }

        public string Path { get { return null; } }

        public SilenceGenerator(AudioPCMConfig pcm, long sampleCount, int sampleVal)
        {
            this._sampleVal = sampleVal;
            this._sampleOffset = 0;
            this._sampleCount = sampleCount;
            this.pcm = pcm;
        }

        public SilenceGenerator(long sampleCount)
            : this(AudioPCMConfig.RedBook, sampleCount, 0)
        {
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            int[,] samples = buff.Samples;
            for (int i = 0; i < buff.Length; i++)
                for (int j = 0; j < PCM.ChannelCount; j++)
                    samples[i, j] = _sampleVal;

            _sampleOffset += buff.Length;
            return buff.Length;
        }

        public void Close()
        {
        }
    }
}
