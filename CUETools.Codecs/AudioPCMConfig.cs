namespace CUETools.Codecs
{
    public class AudioPCMConfig
    {
        public static readonly AudioPCMConfig RedBook = new AudioPCMConfig(16, 2, 44100);
        public enum SpeakerConfig
        {
            SPEAKER_FRONT_LEFT = 0x1,
            SPEAKER_FRONT_RIGHT = 0x2,
            SPEAKER_FRONT_CENTER = 0x4,
            SPEAKER_LOW_FREQUENCY = 0x8,
            SPEAKER_BACK_LEFT = 0x10,
            SPEAKER_BACK_RIGHT = 0x20,
            SPEAKER_FRONT_LEFT_OF_CENTER = 0x40,
            SPEAKER_FRONT_RIGHT_OF_CENTER = 0x80,
            SPEAKER_BACK_CENTER = 0x100,
            SPEAKER_SIDE_LEFT = 0x200,
            SPEAKER_SIDE_RIGHT = 0x400,
            SPEAKER_TOP_CENTER = 0x800,
            SPEAKER_TOP_FRONT_LEFT = 0x1000,
            SPEAKER_TOP_FRONT_CENTER = 0x2000,
            SPEAKER_TOP_FRONT_RIGHT = 0x4000,
            SPEAKER_TOP_BACK_LEFT = 0x8000,
            SPEAKER_TOP_BACK_CENTER = 0x10000,
            SPEAKER_TOP_BACK_RIGHT = 0x20000,

            DIRECTOUT = 0,
            KSAUDIO_SPEAKER_MONO = (SPEAKER_FRONT_CENTER),
            KSAUDIO_SPEAKER_STEREO = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT),
            KSAUDIO_SPEAKER_QUAD = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT),
            KSAUDIO_SPEAKER_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_BACK_CENTER),
            KSAUDIO_SPEAKER_5POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT),
            KSAUDIO_SPEAKER_5POINT1_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT),
            KSAUDIO_SPEAKER_7POINT1 = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER),
            KSAUDIO_SPEAKER_7POINT1_SURROUND = (SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT)
        }

        private int _bitsPerSample;
        private int _channelCount;
        private int _sampleRate;
        private SpeakerConfig _channelMask;

        public int BitsPerSample { get { return _bitsPerSample; } }
        public int ChannelCount { get { return _channelCount; } }
        public int SampleRate { get { return _sampleRate; } }
        public int BlockAlign { get { return _channelCount * ((_bitsPerSample + 7) / 8); } }
        public SpeakerConfig ChannelMask { get { return _channelMask; } }
        public bool IsRedBook { get { return _bitsPerSample == 16 && _channelCount == 2 && _sampleRate == 44100; } }
        public static SpeakerConfig GetDefaultChannelMask(int channelCount)
        {
            switch (channelCount)
            {
                case 1:
                    return SpeakerConfig.KSAUDIO_SPEAKER_MONO;
                case 2:
                    return SpeakerConfig.KSAUDIO_SPEAKER_STEREO;
                case 3:
                    return SpeakerConfig.KSAUDIO_SPEAKER_STEREO | SpeakerConfig.SPEAKER_LOW_FREQUENCY;
                case 4:
                    return SpeakerConfig.KSAUDIO_SPEAKER_QUAD;
                case 5:
                    //return SpeakerConfig.KSAUDIO_SPEAKER_5POINT1 & ~SpeakerConfig.SPEAKER_LOW_FREQUENCY;
                    return SpeakerConfig.KSAUDIO_SPEAKER_5POINT1_SURROUND & ~SpeakerConfig.SPEAKER_LOW_FREQUENCY;
                case 6:
                    //return SpeakerConfig.KSAUDIO_SPEAKER_5POINT1;
                    return SpeakerConfig.KSAUDIO_SPEAKER_5POINT1_SURROUND;
                case 7:
                    return SpeakerConfig.KSAUDIO_SPEAKER_5POINT1_SURROUND | SpeakerConfig.SPEAKER_BACK_CENTER;
                case 8:
                    return SpeakerConfig.KSAUDIO_SPEAKER_7POINT1_SURROUND; 
            }
            return (SpeakerConfig)((1 << channelCount) - 1);
        }

        public AudioPCMConfig(int bitsPerSample, int channelCount, int sampleRate, SpeakerConfig channelMask = SpeakerConfig.DIRECTOUT)
        {
            _bitsPerSample = bitsPerSample;
            _channelCount = channelCount;
            _sampleRate = sampleRate;
            _channelMask = channelMask == 0 ? GetDefaultChannelMask(channelCount) : channelMask;
        }
    }
}
