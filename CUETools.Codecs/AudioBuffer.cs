using System;

namespace CUETools.Codecs
{
    public class AudioBuffer
    {
        #region Static Methods

        public static unsafe void FLACSamplesToBytes_16(int[,] inSamples, int inSampleOffset,
            byte* outSamples, int sampleCount, int channelCount)
        {
            int loopCount = sampleCount * channelCount;

            if (inSamples.GetLength(0) - inSampleOffset < sampleCount)
                throw new IndexOutOfRangeException();

            fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0])
            {
                int* pInSamples = pInSamplesFixed;
                short* pOutSamples = (short*)outSamples;
                for (int i = 0; i < loopCount; i++)
                    pOutSamples[i] = (short)pInSamples[i];
                //*(pOutSamples++) = (short)*(pInSamples++);
            }
        }

        public static unsafe void FLACSamplesToBytes_16(int[,] inSamples, int inSampleOffset,
            byte[] outSamples, int outByteOffset, int sampleCount, int channelCount)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
                (outSamples.Length - outByteOffset < loopCount * 2))
            {
                throw new IndexOutOfRangeException();
            }

            fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
                FLACSamplesToBytes_16(inSamples, inSampleOffset, pOutSamplesFixed, sampleCount, channelCount);
        }

        public static unsafe void FLACSamplesToBytes_24(int[,] inSamples, int inSampleOffset,
            byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int wastedBits)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
                (outSamples.Length - outByteOffset < loopCount * 3))
            {
                throw new IndexOutOfRangeException();
            }

            fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0])
            {
                fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
                {
                    int* pInSamples = pInSamplesFixed;
                    byte* pOutSamples = pOutSamplesFixed;

                    for (int i = 0; i < loopCount; i++)
                    {
                        uint sample_out = (uint)*(pInSamples++) << wastedBits;
                        *(pOutSamples++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(pOutSamples++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(pOutSamples++) = (byte)(sample_out & 0xFF);
                    }
                }
            }
        }

        public static unsafe void FloatToBytes_16(float[,] inSamples, int inSampleOffset,
            byte[] outSamples, int outByteOffset, int sampleCount, int channelCount)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
                (outSamples.Length - outByteOffset < loopCount * 2))
            {
                throw new IndexOutOfRangeException();
            }

            fixed (float* pInSamplesFixed = &inSamples[inSampleOffset, 0])
            {
                fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
                {
                    float* pInSamples = pInSamplesFixed;
                    short* pOutSamples = (short*)pOutSamplesFixed;

                    for (int i = 0; i < loopCount; i++)
                    {
                        *(pOutSamples++) = (short)(32758 * (*(pInSamples++)));
                    }
                }
            }
        }

        public static unsafe void FloatToBytes(float[,] inSamples, int inSampleOffset,
            byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int bitsPerSample)
        {
            if (bitsPerSample == 16)
                FloatToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
            //else if (bitsPerSample > 16 && bitsPerSample <= 24)
            //    FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount, 24 - bitsPerSample);
            else if (bitsPerSample == 32)
                Buffer.BlockCopy(inSamples, inSampleOffset * 4 * channelCount, outSamples, outByteOffset, sampleCount * 4 * channelCount);
            else
                throw new Exception("Unsupported bitsPerSample value");
        }

        public static unsafe void FLACSamplesToBytes(int[,] inSamples, int inSampleOffset,
            byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int bitsPerSample)
        {
            if (bitsPerSample == 16)
                FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
            else if (bitsPerSample > 16 && bitsPerSample <= 24)
                FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount, 24 - bitsPerSample);
            else
                throw new Exception("Unsupported bitsPerSample value");
        }

        public static unsafe void FLACSamplesToBytes(int[,] inSamples, int inSampleOffset,
            byte* outSamples, int sampleCount, int channelCount, int bitsPerSample)
        {
            if (bitsPerSample == 16)
                FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, sampleCount, channelCount);
            else
                throw new Exception("Unsupported bitsPerSample value");
        }

        public static unsafe void Bytes16ToFloat(byte[] inSamples, int inByteOffset,
            float[,] outSamples, int outSampleOffset, int sampleCount, int channelCount)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.Length - inByteOffset < loopCount * 2) ||
                (outSamples.GetLength(0) - outSampleOffset < sampleCount))
                throw new IndexOutOfRangeException();

            fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
            {
                fixed (float* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
                {
                    short* pInSamples = (short*)pInSamplesFixed;
                    float* pOutSamples = pOutSamplesFixed;
                    for (int i = 0; i < loopCount; i++)
                        *(pOutSamples++) = *(pInSamples++) / 32768.0f;
                }
            }
        }

        public static unsafe void BytesToFLACSamples_16(byte[] inSamples, int inByteOffset,
            int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.Length - inByteOffset < loopCount * 2) ||
                (outSamples.GetLength(0) - outSampleOffset < sampleCount))
            {
                throw new IndexOutOfRangeException();
            }

            fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
            {
                fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
                {
                    short* pInSamples = (short*)pInSamplesFixed;
                    int* pOutSamples = pOutSamplesFixed;

                    for (int i = 0; i < loopCount; i++)
                    {
                        *(pOutSamples++) = (int)*(pInSamples++);
                    }
                }
            }
        }

        public static unsafe void BytesToFLACSamples_24(byte[] inSamples, int inByteOffset,
            int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount, int wastedBits)
        {
            int loopCount = sampleCount * channelCount;

            if ((inSamples.Length - inByteOffset < loopCount * 3) ||
                (outSamples.GetLength(0) - outSampleOffset < sampleCount))
                throw new IndexOutOfRangeException();

            fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
            {
                fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
                {
                    byte* pInSamples = (byte*)pInSamplesFixed;
                    int* pOutSamples = pOutSamplesFixed;
                    for (int i = 0; i < loopCount; i++)
                    {
                        int sample = (int)*(pInSamples++);
                        sample += (int)*(pInSamples++) << 8;
                        sample += (int)*(pInSamples++) << 16;
                        *(pOutSamples++) = (sample << 8) >> (8 + wastedBits);
                    }
                }
            }
        }

        public static unsafe void BytesToFLACSamples(byte[] inSamples, int inByteOffset,
            int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount, int bitsPerSample)
        {
            if (bitsPerSample == 16)
                BytesToFLACSamples_16(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount);
            else if (bitsPerSample > 16 && bitsPerSample <= 24)
                BytesToFLACSamples_24(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount, 24 - bitsPerSample);
            else
                throw new Exception("Unsupported bitsPerSample value");
        }

        #endregion

        private int[,] samples;
        private float[,] fsamples;
        private byte[] bytes;
        private int length;
        private int size;
        private AudioPCMConfig pcm;
        private bool dataInSamples = false;
        private bool dataInBytes = false;
        private bool dataInFloat = false;

        public int Length
        {
            get { return length; }
            set { length = value; }
        }

        public int Size
        {
            get { return size; }
        }

        public AudioPCMConfig PCM { get { return pcm; } }

        public int ByteLength
        {
            get
            {
                return length * pcm.BlockAlign;
            }
        }

        public int[,] Samples
        {
            get
            {
                if (samples == null || samples.GetLength(0) < length)
                    samples = new int[size, pcm.ChannelCount];
                if (!dataInSamples && dataInBytes && length != 0)
                    BytesToFLACSamples(bytes, 0, samples, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
                dataInSamples = true;
                return samples;
            }
        }

        public float[,] Float
        {
            get
            {
                if (fsamples == null || fsamples.GetLength(0) < length)
                    fsamples = new float[size, pcm.ChannelCount];
                if (!dataInFloat && dataInBytes && length != 0)
                {
                    if (pcm.BitsPerSample == 16)
                        Bytes16ToFloat(bytes, 0, fsamples, 0, length, pcm.ChannelCount);
                    //else if (pcm.BitsPerSample > 16 && PCM.BitsPerSample <= 24)
                    //    BytesToFLACSamples_24(bytes, 0, fsamples, 0, length, pcm.ChannelCount, 24 - pcm.BitsPerSample);
                    else if (pcm.BitsPerSample == 32)
                        Buffer.BlockCopy(bytes, 0, fsamples, 0, length * 4 * pcm.ChannelCount);
                    else
                        throw new Exception("Unsupported bitsPerSample value");
                }
                dataInFloat = true;
                return fsamples;
            }
        }

        public byte[] Bytes
        {
            get
            {
                if (bytes == null || bytes.Length < length * pcm.BlockAlign)
                    bytes = new byte[size * pcm.BlockAlign];
                if (!dataInBytes && length != 0)
                {
                    if (dataInSamples)
                        FLACSamplesToBytes(samples, 0, bytes, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
                    else if (dataInFloat)
                        FloatToBytes(fsamples, 0, bytes, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
                }
                dataInBytes = true;
                return bytes;
            }
        }

        public AudioBuffer(AudioPCMConfig _pcm, int _size)
        {
            pcm = _pcm;
            size = _size;
            length = 0;
        }

        public AudioBuffer(AudioPCMConfig _pcm, int[,] _samples, int _length)
        {
            pcm = _pcm;
            // assert _samples.GetLength(1) == pcm.ChannelCount
            Prepare(_samples, _length);
        }

        public AudioBuffer(AudioPCMConfig _pcm, byte[] _bytes, int _length)
        {
            pcm = _pcm;
            Prepare(_bytes, _length);
        }

        public AudioBuffer(IAudioSource source, int _size)
        {
            pcm = source.PCM;
            size = _size;
        }

        public void Prepare(IAudioDest dest)
        {
            if (dest.Settings.PCM.ChannelCount != pcm.ChannelCount || dest.Settings.PCM.BitsPerSample != pcm.BitsPerSample)
                throw new Exception("AudioBuffer format mismatch");
        }

        public void Prepare(IAudioSource source, int maxLength)
        {
            if (source.PCM.ChannelCount != pcm.ChannelCount || source.PCM.BitsPerSample != pcm.BitsPerSample)
                throw new Exception("AudioBuffer format mismatch");
            length = size;
            if (maxLength >= 0)
                length = Math.Min(length, maxLength);
            if (source.Remaining >= 0)
                length = (int)Math.Min((long)length, source.Remaining);
            dataInBytes = false;
            dataInSamples = false;
            dataInFloat = false;
        }

        public void Prepare(int maxLength)
        {
            length = size;
            if (maxLength >= 0)
                length = Math.Min(length, maxLength);
            dataInBytes = false;
            dataInSamples = false;
            dataInFloat = false;
        }

        public void Prepare(int[,] _samples, int _length)
        {
            length = _length;
            size = _samples.GetLength(0);
            samples = _samples;
            dataInSamples = true;
            dataInBytes = false;
            dataInFloat = false;
            if (length > size)
                throw new Exception("Invalid length");
        }

        public void Prepare(byte[] _bytes, int _length)
        {
            length = _length;
            size = _bytes.Length / PCM.BlockAlign;
            bytes = _bytes;
            dataInSamples = false;
            dataInBytes = true;
            dataInFloat = false;
            if (length > size)
                throw new Exception("Invalid length");
        }

        internal unsafe void Load(int dstOffset, AudioBuffer src, int srcOffset, int copyLength)
        {
            if (dataInBytes)
                Buffer.BlockCopy(src.Bytes, srcOffset * pcm.BlockAlign, Bytes, dstOffset * pcm.BlockAlign, copyLength * pcm.BlockAlign);
            if (dataInSamples)
                Buffer.BlockCopy(src.Samples, srcOffset * pcm.ChannelCount * 4, Samples, dstOffset * pcm.ChannelCount * 4, copyLength * pcm.ChannelCount * 4);
            if (dataInFloat)
                Buffer.BlockCopy(src.Float, srcOffset * pcm.ChannelCount * 4, Float, dstOffset * pcm.ChannelCount * 4, copyLength * pcm.ChannelCount * 4);
        }

        public unsafe void Prepare(AudioBuffer _src, int _offset, int _length)
        {
            length = Math.Min(size, _src.Length - _offset);
            if (_length >= 0)
                length = Math.Min(length, _length);
            dataInBytes = false;
            dataInFloat = false;
            dataInSamples = false;
            if (_src.dataInBytes)
                dataInBytes = true;
            else if (_src.dataInSamples)
                dataInSamples = true;
            else if (_src.dataInFloat)
                dataInFloat = true;
            Load(0, _src, _offset, length);
        }

        public void Swap(AudioBuffer buffer)
        {
            if (pcm.BitsPerSample != buffer.PCM.BitsPerSample || pcm.ChannelCount != buffer.PCM.ChannelCount)
                throw new Exception("AudioBuffer format mismatch");

            int[,] samplesTmp = samples;
            float[,] floatsTmp = fsamples;
            byte[] bytesTmp = bytes;

            fsamples = buffer.fsamples;
            samples = buffer.samples;
            bytes = buffer.bytes;
            length = buffer.length;
            size = buffer.size;
            dataInSamples = buffer.dataInSamples;
            dataInBytes = buffer.dataInBytes;
            dataInFloat = buffer.dataInFloat;

            buffer.samples = samplesTmp;
            buffer.bytes = bytesTmp;
            buffer.fsamples = floatsTmp;
            buffer.length = 0;
            buffer.dataInSamples = false;
            buffer.dataInBytes = false;
            buffer.dataInFloat = false;
        }

        unsafe public void Interlace(int pos, int* src1, int* src2, int n)
        {
            if (PCM.ChannelCount != 2)
            {
                throw new Exception("Must be stereo");
            }
            if (PCM.BitsPerSample == 16)
            {
                fixed (byte* bs = Bytes)
                {
                    int* res = ((int*)bs) + pos;
                    for (int i = n; i > 0; i--)
                        *(res++) = (*(src1++) & 0xffff) ^ (*(src2++) << 16);
                }
            }
            else if (PCM.BitsPerSample == 24)
            {
                fixed (byte* bs = Bytes)
                {
                    byte* res = bs + pos * 6;
                    for (int i = n; i > 0; i--)
                    {
                        uint sample_out = (uint)*(src1++);
                        *(res++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(res++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(res++) = (byte)(sample_out & 0xFF);
                        sample_out = (uint)*(src2++);
                        *(res++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(res++) = (byte)(sample_out & 0xFF);
                        sample_out >>= 8;
                        *(res++) = (byte)(sample_out & 0xFF);
                    }
                }
            }
            else
            {
                throw new Exception("Unsupported BPS");
            }
        }

        //public void Clear()
        //{
        //    length = 0;
        //}
    }
}
