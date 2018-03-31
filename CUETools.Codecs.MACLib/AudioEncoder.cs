using System;
using System.IO;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.MACLib
{
    public unsafe class AudioEncoder : IAudioDest, IDisposable
    {
        public AudioEncoder(EncoderSettings settings, string path, Stream output = null)
        {
            m_settings = settings;

            m_path = path;
            m_stream = output;
            m_settings = settings;
            m_streamGiven = output != null;
            m_initialized = false;
            m_finalSampleCount = 0;
            m_samplesWritten = 0;

            if (m_settings.PCM.ChannelCount != 1 && m_settings.PCM.ChannelCount != 2)
                throw new Exception("Only stereo and mono audio formats are allowed.");
            if (m_settings.PCM.BitsPerSample != 16 && m_settings.PCM.BitsPerSample != 24)
                throw new Exception("bits per sample must be 16 or 24");

            int nRetVal;
            pAPECompress = MACLibDll.c_APECompress_Create(out nRetVal);
            if (pAPECompress == null)
                throw new Exception("Unable to open APE compressor.");
        }

        public IAudioEncoderSettings Settings => m_settings;

        public string Path { get => m_path; }

        public long FinalSampleCount
        {
            get => m_finalSampleCount;
            set
            {
                if (value < 0)
                    throw new Exception("invalid final sample count");
                if (m_initialized)
                    throw new Exception("final sample count cannot be changed after encoding begins");
                m_finalSampleCount = value;
            }
        }

        public void Close()
        {
            try
            {
                if (pAPECompress != null)
                    MACLibDll.c_APECompress_Finish(pAPECompress, null, 0, 0);
            }
            catch (Exception)
            {
            }
            Dispose();
            if ((m_finalSampleCount != 0) && (m_samplesWritten != m_finalSampleCount))
                throw new Exception("samples written differs from the expected sample count");
        }

        public void Delete()
        {
            var path = m_path;
            Dispose(true);
            m_initialized = false;
            if (path != "")
                File.Delete(path);
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
            }

            if (pAPECompress != null) MACLibDll.c_APECompress_Destroy(pAPECompress);
            pAPECompress = IntPtr.Zero;
        }

        ~AudioEncoder()
        {
            Dispose(false);
        }

        public void Write(AudioBuffer sampleBuffer)
        {
            if (!m_initialized) Initialize();

            sampleBuffer.Prepare(this);

            fixed (byte* pSampleBuffer = &sampleBuffer.Bytes[0])
                if (0 != MACLibDll.c_APECompress_AddData(pAPECompress, pSampleBuffer, sampleBuffer.ByteLength))
                    throw new Exception("An error occurred while encoding");

            m_samplesWritten += sampleBuffer.Length;
        }

        void Initialize()
        {
            if (m_stream == null)
                m_stream = new FileStream(m_path, FileMode.Create, FileAccess.ReadWrite, FileShare.Read, 0x10000);
            m_StreamIO = new StreamIO(m_stream);

            WAVEFORMATEX* pWaveFormatEx = stackalloc WAVEFORMATEX[1];
            pWaveFormatEx->extraSize = 0;
            pWaveFormatEx->sampleRate = m_settings.PCM.SampleRate;
            pWaveFormatEx->bitsPerSample = (short)m_settings.PCM.BitsPerSample;
            pWaveFormatEx->channels = (short)m_settings.PCM.ChannelCount;
            pWaveFormatEx->waveFormatTag = 1;
            pWaveFormatEx->blockAlign = (short)m_settings.PCM.BlockAlign;
            pWaveFormatEx->averageBytesPerSecond = m_settings.PCM.BlockAlign * m_settings.PCM.SampleRate;

            int compressionLevel = (m_settings.GetEncoderModeIndex() + 1) * 1000;

            int res = MACLibDll.c_APECompress_StartEx(
                pAPECompress,
                m_StreamIO.CIO,
                pWaveFormatEx,
                (m_finalSampleCount == 0) ? -1 : (int) (m_finalSampleCount * m_settings.PCM.BlockAlign),
                compressionLevel,
                null,
                /*CREATE_WAV_HEADER_ON_DECOMPRESSION*/-1);
            if (res != 0)
                throw new Exception("Unable to create the encoder.");

			m_initialized = true;
        }


        IntPtr pAPECompress;
        EncoderSettings m_settings;
        Stream m_stream;
        bool m_streamGiven;
        bool m_initialized;
        string m_path;
        long m_finalSampleCount, m_samplesWritten;
        StreamIO m_StreamIO;
    }
}
