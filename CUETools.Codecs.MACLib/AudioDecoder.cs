using System;
using System.IO;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.MACLib
{
    public unsafe class AudioDecoder : IAudioSource, IDisposable
    {
        public AudioDecoder(DecoderSettings settings, string path, Stream IO)
        {
            m_settings = settings;

			_path = path;

			m_stream = (IO != null) ? IO : new FileStream (path, FileMode.Open, FileAccess.Read, FileShare.Read);
            m_StreamIO = new StreamIO(m_stream);

            int errorCode = 0;
            pAPEDecompress = MACLibDll.c_APEDecompress_CreateEx(m_StreamIO.Callbacks, out errorCode);
			if (pAPEDecompress == null) {
				throw new Exception("Unable to initialize the decoder: " + errorCode);
			}

			pcm = new AudioPCMConfig(
                MACLibDll.c_APEDecompress_GetInfo(pAPEDecompress, APE_DECOMPRESS_FIELDS.APE_INFO_BITS_PER_SAMPLE, 0, 0),
                MACLibDll.c_APEDecompress_GetInfo(pAPEDecompress, APE_DECOMPRESS_FIELDS.APE_INFO_CHANNELS, 0, 0),
                MACLibDll.c_APEDecompress_GetInfo(pAPEDecompress, APE_DECOMPRESS_FIELDS.APE_INFO_SAMPLE_RATE, 0, 0),
				(AudioPCMConfig.SpeakerConfig)0);
            _samplesBuffer = new byte[16384 * pcm.BlockAlign];
            _sampleCount = MACLibDll.c_APEDecompress_GetInfo(pAPEDecompress, APE_DECOMPRESS_FIELDS.APE_DECOMPRESS_TOTAL_BLOCKS, 0, 0);
            _sampleOffset = 0;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (m_StreamIO != null)
                {
                    m_StreamIO.Dispose();
                    m_StreamIO = null;
                }
                if (m_stream != null)
                {
                    m_stream.Dispose();
                    m_stream = null;
                }
                if (_samplesBuffer != null)
                {
                    _samplesBuffer = null;
                }
            }

            if (pAPEDecompress != null) MACLibDll.c_APEDecompress_Destroy(pAPEDecompress);
            pAPEDecompress = IntPtr.Zero;
        }

        ~AudioDecoder()
        {
            Dispose(false);
        }

        // TODO: intn should always be 64 bit? Or should we use different interface in 32-bit mode?

        private DecoderSettings m_settings;

        public IAudioDecoderSettings Settings => m_settings;

        public AudioPCMConfig PCM => pcm;

        public string Path => _path;

        public long Length => _sampleCount;

        internal long SamplesInBuffer => _bufferLength - _bufferOffset;

        public long Position
        {
            get => _sampleOffset - SamplesInBuffer;

            set
            {
                _bufferOffset = 0;
                _bufferLength = 0;
                _sampleOffset = value;
                int res = MACLibDll.c_APEDecompress_Seek(pAPEDecompress, (int)value);
                if (0 != res)
                    throw new Exception("unable to seek:" + res.ToString());
            }
        }

        public long Remaining => _sampleCount - _sampleOffset + SamplesInBuffer;

        public void Close()
        {
            Dispose(true);
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            long buffOffset = 0;
            long samplesNeeded = buff.Length;
            long _channelCount = pcm.ChannelCount;

            while (samplesNeeded != 0)
            {
                if (SamplesInBuffer == 0)
                {
                    long nBlocksRetrieved;
                    fixed (byte* pSampleBuffer = &_samplesBuffer[0])
                    {
                        int res = MACLibDll.c_APEDecompress_GetData(pAPEDecompress, (char*)pSampleBuffer, (IntPtr)16384, out nBlocksRetrieved);
                        if (res != 0)
                            throw new Exception("An error occurred while decoding: " + res.ToString());
                    }
                    _bufferOffset = 0;
                    _bufferLength = nBlocksRetrieved;
                    _sampleOffset += nBlocksRetrieved;
                }
                long copyCount = Math.Min(samplesNeeded, SamplesInBuffer);
                AudioBuffer.BytesToFLACSamples_16(_samplesBuffer, (int)(_bufferOffset * pcm.BlockAlign), buff.Samples, (int)buffOffset, (int)copyCount, (int)_channelCount);
                samplesNeeded -= copyCount;
                buffOffset += copyCount;
                _bufferOffset += copyCount;
            }

            return buff.Length;
        }

        IntPtr pAPEDecompress;
		long _sampleCount, _sampleOffset;
        AudioPCMConfig pcm;
        string _path;
        Stream m_stream;
        long _bufferOffset, _bufferLength;
        byte[] _samplesBuffer;
        StreamIO m_StreamIO;
    }
}
