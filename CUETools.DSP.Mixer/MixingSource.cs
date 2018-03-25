using System;
using System.Threading;
using CUETools.Codecs;

namespace CUETools.DSP.Mixer
{
    public class MixingSource : IAudioSource
    {
        private AudioPCMConfig pcm;
        private MixingBuffer[] buf;
        private bool[] playing;
        private float[] volume;
        private long samplePos;
        private int size;
        private AudioReadEventArgs audioReadArgs = new AudioReadEventArgs();
        private MixingBuffer mixbuff = null;
        private int mixoffs = 0;
        private int current = 0;

        public IAudioDecoderSettings Settings => null;

        public void Close()
        {
        }

        public long Position
        {
            get { return samplePos; }
            set { throw new NotSupportedException(); }
        }

        public long Length
        {
            get { throw new NotSupportedException(); }
        }

        public long Remaining
        {
            get { throw new NotSupportedException(); }
        }

        public AudioPCMConfig PCM
        {
            get { return pcm; }
        }

        public string Path { get { return ""; } }

        public int BufferSize
        {
            get
            {
                return buf[0].source[0].Size;
            }
        }

        public MixingSource(AudioPCMConfig pcm, int delay, int sources)
        {
            if (pcm.BitsPerSample != 32)
                throw new NotSupportedException("please use 32 bits per sample (float)");
            this.pcm = pcm;
            this.size = delay * pcm.SampleRate / 1000;
            this.buf = new MixingBuffer[2];
            this.buf[0] = new MixingBuffer(pcm, size, sources);
            this.buf[1] = new MixingBuffer(pcm, size, sources);
            this.playing = new bool[sources];
            this.volume = new float[sources];
            this.samplePos = 0;
        }

        public int Read(AudioBuffer result, int maxLength)
        {
            if (maxLength > (BufferSize - mixoffs) || maxLength < 0)
                maxLength = (BufferSize - mixoffs);

            result.Prepare(maxLength);

            if (mixbuff == null)
                mixbuff = LockFilledBuffer();

            float sumVolume = 0.0f;
            for (int iSource = 0; iSource < mixbuff.source.Length; iSource++)
                if (mixbuff.filled[iSource])
                    sumVolume += mixbuff.volume[iSource];
            for (int iSource = 0; iSource < mixbuff.source.Length; iSource++)
                volume[iSource] = mixbuff.filled[iSource] ? mixbuff.volume[iSource] / Math.Max(1.0f, sumVolume) : 0.0f;
            for (int iSmp = 0; iSmp < result.Length; iSmp++)
            {
                for (int iChan = 0; iChan < result.PCM.ChannelCount; iChan++)
                {
                    float sample = 0.0f;
                    for (int iSource = 0; iSource < mixbuff.source.Length; iSource++)
                        sample += mixbuff.source[iSource].Float[mixoffs + iSmp, iChan] * volume[iSource];
                    result.Float[iSmp, iChan] = sample;
                }
            }
            mixoffs += result.Length;
            if (mixoffs == BufferSize)
            {
                UnlockFilledBuffer(mixbuff);
                mixbuff = null;
                mixoffs = 0;
            }
            samplePos += result.Length;

            if (AudioRead != null)
            {
                audioReadArgs.source = this;
                audioReadArgs.buffer = result;
                AudioRead(this, audioReadArgs);
            }

            return result.Length;
        }

        private bool IsFilled(MixingBuffer buf)
        {
            bool res = true;
            for (int i = 0; i < buf.filled.Length; i++)
                res &= buf.filled[i] || !this.playing[i];
            return res;
        }

        internal MixingBuffer LockFilledBuffer()
        {
            lock (this)
            {
                //Trace.WriteLine(string.Format("LockFilledBuffer: 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1]));
                int no = current;
                while (!IsFilled(buf[no]))
                    Monitor.Wait(this);
                current = 1 - no;
                return buf[no];
            }
        }

        internal void UnlockFilledBuffer(MixingBuffer mixbuff)
        {
            lock (this)
            {
                //Trace.WriteLine(string.Format("UnockFilledBuffer: 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1]));
                for (int i = 0; i < mixbuff.filled.Length; i++)
                    mixbuff.filled[i] = false;
                Monitor.PulseAll(this);
            }
        }

        public void BufferPlaying(int iSource, bool playing)
        {
            lock (this)
            {
                //Trace.WriteLine(string.Format("BufferPlaying{8}< 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1], iSource));

                this.playing[iSource] = playing;
                //if (!playing) buf0.filled[iSource] = false;
                //if (!playing) buf1.filled[iSource] = false;

                //Trace.WriteLine(string.Format("BufferPlaying{8}> 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1], iSource));
                Monitor.PulseAll(this);
            }
        }

        internal MixingBuffer LockEmptyBuffer(int iSource)
        {
            lock (this)
            {
                //Trace.WriteLine(string.Format("LockEmptyBuffer{8}: 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1], iSource));

                while (!playing[iSource] || buf[current].filled[iSource])
                    Monitor.Wait(this);

                return buf[current];
            }
        }

        internal void UnlockEmptyBuffer(MixingBuffer mixbuff, int iSource, float volume)
        {
            lock (this)
            {
                //Trace.WriteLine(string.Format("UnlockEmptyBuffer{8}< 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1], iSource));

                mixbuff.volume[iSource] = volume;
                mixbuff.filled[iSource] = true;

                //Trace.WriteLine(string.Format("UnlockEmptyBuffer{8}> 0.0: {0} {1}; 0.1: {2} {3}; 1.0: {4} {5}; 1.1: {6} {7};",
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1],
                //    buf0.playing[0], buf0.filled[0], buf0.playing[1], buf0.filled[1], iSource));

                Monitor.PulseAll(this);
            }
        }

        public event EventHandler<AudioReadEventArgs> AudioRead;
    }
}
