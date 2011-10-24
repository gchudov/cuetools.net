namespace CUETools.Codecs
{
    public class AudioPCMConfig
    {
        public static readonly AudioPCMConfig RedBook = new AudioPCMConfig(16, 2, 44100);

        private int _bitsPerSample;
        private int _channelCount;
        private int _sampleRate;

        public int BitsPerSample { get { return _bitsPerSample; } }
        public int ChannelCount { get { return _channelCount; } }
        public int SampleRate { get { return _sampleRate; } }
        public int BlockAlign { get { return _channelCount * ((_bitsPerSample + 7) / 8); } }
        public bool IsRedBook { get { return _bitsPerSample == 16 && _channelCount == 2 && _sampleRate == 44100; } }

        public AudioPCMConfig(int bitsPerSample, int channelCount, int sampleRate)
        {
            _bitsPerSample = bitsPerSample;
            _channelCount = channelCount;
            _sampleRate = sampleRate;
        }
    }
}
