using System;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using System.IO;

namespace CUETools.Codecs.LAME
{
    [AudioEncoderClass(typeof(CBREncoderSettings))]
    [AudioEncoderClass(typeof(VBREncoderSettings))]
    public class AudioEncoder : IAudioDest
    {
        #region Unmanaged Functions

        private const string LameDll = "libmp3lame";
        private const CallingConvention LameCallingConvention = CallingConvention.Cdecl;

        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern IntPtr lame_init();
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_close(IntPtr handle);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_num_channels(IntPtr handle, int channels);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_in_samplerate(IntPtr handle, int sampleRate);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_quality(IntPtr handle, int quality);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR(IntPtr handle, int vbrMode);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR_mean_bitrate_kbps(IntPtr handle, int meanBitrate);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_init_params(IntPtr handle);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_num_samples(IntPtr handle, uint numSamples);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_encode_buffer_interleaved(IntPtr handle, IntPtr pcm, int num_samples, IntPtr mp3buf, int mp3buf_size);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_encode_flush(IntPtr handle, IntPtr mp3buf, int size);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern uint lame_get_lametag_frame(IntPtr handle, IntPtr buffer, uint size);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR_quality(IntPtr handle, float vbrQuality);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_brate(IntPtr handle, int bitrate);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_bWriteVbrTag(IntPtr handle, int writeVbrTag);
        [DllImport(LameDll, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_write_id3tag_automatic(IntPtr handle, int automaticWriteId3Tag);

        #endregion

        private string m_outputPath;
        private Stream m_outputStream;

        private bool m_closed = false, m_initialized = false;
        private IntPtr m_handle;
        private uint m_finalSampleCount;
        private byte[] m_outputBuffer;

        public long FinalSampleCount
        {
            set
            {
                if (value > uint.MaxValue)
                {
                    throw new ArgumentException("Input file too big.");
                }
                this.m_finalSampleCount = (uint)value;
            }
        }

        public string Path
        {
            get { return this.m_outputPath; }
        }

        private LameEncoderSettings m_settings;

        public virtual AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
        }

        public AudioEncoder(string path, Stream output, LameEncoderSettings settings)
        {
            this.CheckPCMConfig(settings.PCM);
            this.m_settings = settings;
            this.m_outputPath = path;
            this.m_outputStream = output != null ? output : File.Create(path);
        }

        public AudioEncoder(string path, LameEncoderSettings settings)
            : this(path, null, settings)
        {
        }

        private void CheckPCMConfig(AudioPCMConfig pcm)
        {
            if (pcm.BitsPerSample != 16)
            {
                throw new ArgumentException("LAME only supports 16 bits/sample.");
            }
        }

        private void FinalizeEncoding()
        {
            this.EnsureOutputBufferSize(7200);

            int flushResult;
            unsafe
            {
                fixed (byte* outputBufferPtr = m_outputBuffer)
                {
                    flushResult = lame_encode_flush(m_handle, (IntPtr)outputBufferPtr, m_outputBuffer.Length);
                }
            }
            if (flushResult < 0)
            {
                throw new LameException("Unknown flush error");
            }
            if (flushResult > 0)
            {
                this.m_outputStream.Write(this.m_outputBuffer, 0, flushResult);
            }

            int lametagFrameSize = this.GetLametagFrame();
            this.m_outputStream.Seek(0, SeekOrigin.Begin);
            this.m_outputStream.Write(this.m_outputBuffer, 0, lametagFrameSize);
        }

        public void Close()
        {
            if (!this.m_closed)
            {
                if (this.m_initialized)
                {
                    try
                    {
                        try
                        {
                            this.FinalizeEncoding();
                        }
                        finally
                        {
                            lame_close(m_handle);
                        }
                    }
                    finally
                    {
                        m_handle = IntPtr.Zero;
                        if (this.m_outputPath != null)
                        {
                            this.m_outputStream.Close();
                        }
                    }
                }

                this.m_closed = true;
            }
        }

        private int GetLametagFrame()
        {
            while (true)
            {
                uint lametagFrameResult;
                unsafe
                {
                    fixed (byte* outputBufferPtr = m_outputBuffer)
                    {
                        lametagFrameResult = lame_get_lametag_frame(this.m_handle, (IntPtr)outputBufferPtr, (uint)m_outputBuffer.Length);
                    }
                }
                if (lametagFrameResult < 0)
                {
                    throw new LameException("Error getting lametag frame.");
                }
                if (lametagFrameResult <= m_outputBuffer.Length)
                {
                    return (int)lametagFrameResult;
                }
                this.EnsureOutputBufferSize((int)lametagFrameResult);
            }
        }

        public uint GetLametagFrame(byte[] outputBuffer)
        {
            unsafe
            {
                fixed (byte* outputBufferPtr = outputBuffer)
                {
                    return lame_get_lametag_frame(m_handle, (IntPtr)outputBufferPtr, (uint)outputBuffer.Length);
                }
            }
        }

        private void EnsureInitialized()
        {
            if (!this.m_initialized)
            {
                m_handle = lame_init();

                lame_set_bWriteVbrTag(m_handle, 1);
                lame_set_write_id3tag_automatic(m_handle, 0);

                lame_set_num_channels(m_handle, this.Settings.PCM.ChannelCount);
                lame_set_in_samplerate(m_handle, this.Settings.PCM.SampleRate);

                if (this.m_finalSampleCount != 0)
                {
                    lame_set_num_samples(m_handle, this.m_finalSampleCount);
                }

                m_settings.Apply(m_handle);

                if (lame_init_params(m_handle) != 0)
                {
                    throw new LameException("lame_init_params failed");
                }

                this.m_initialized = true;
            }
        }

        public void Delete()
        {
            if (this.m_outputPath == null)
            {
                throw new InvalidOperationException("This writer was not created from file.");
            }

            if (!m_closed)
            {
                this.Close();
                File.Delete(this.m_outputPath);
            }
        }

        private void EnsureOutputBufferSize(int requiredSize)
        {
            if (this.m_outputBuffer == null || this.m_outputBuffer.Length < requiredSize)
            {
                this.m_outputBuffer = new byte[requiredSize];
            }
        }

        public void Write(AudioBuffer buffer)
        {
            if (this.m_closed)
            {
                throw new InvalidOperationException("Writer already closed.");
            }

            buffer.Prepare(this);

            this.EnsureInitialized();

            this.EnsureOutputBufferSize(buffer.Length * 5 / 4 + 7200);

            byte[] bytes = buffer.Bytes;

            int result;
            unsafe
            {
                fixed (byte* bytesPtr = bytes)
                {
                    fixed (byte* outputBufferPtr = this.m_outputBuffer)
                    {
                        result = lame_encode_buffer_interleaved(m_handle, (IntPtr)bytesPtr, buffer.Length, (IntPtr)outputBufferPtr, m_outputBuffer.Length);
                    }
                }
            }

            if (result < 0)
            {
                switch (result)
                {
                    case -1:
                        throw new LameException("Output buffer is too small");
                    case -2:
                        throw new LameException("malloc problem");
                    case -3:
                        throw new LameException("lame_init_params was not called");
                    case -4:
                        throw new LameException("Psycho acoustic problems");
                    default:
                        throw new LameException("Unknown error");
                }
            }

            if (result > 0)
            {
                this.m_outputStream.Write(this.m_outputBuffer, 0, result);
            }
        }
    }
}
