using CUETools.Codecs;

namespace CUETools.DSP.Mixer
{
    public class MixingBuffer
    {
        public AudioBuffer[] source;
        public float[] volume;
        public bool[] filled;

        public MixingBuffer(AudioPCMConfig pcm, int size, int sources)
        {
            source = new AudioBuffer[sources];
            volume = new float[sources];
            filled = new bool[sources];
            for (int i = 0; i < sources; i++)
            {
                source[i] = new AudioBuffer(pcm, size);
            }
        }
    }
}
