using System;
using System.IO;
using System.Text;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    public class LAMEEncoder : IAudioDest
    {
        private bool closed = false;
        private BE_CONFIG m_Mp3Config = null;
        private uint m_hLameStream = 0;
        private uint m_InputSamples = 0;
        private uint m_OutBufferSize = 0;
        private byte[] m_InBuffer = null;
        private int m_InBufferPos = 0;
        private byte[] m_OutBuffer = null;
        private AudioPCMConfig _pcm;
        private string _path;
        private Stream _IO;
        private long position = 0, sample_count = -1;
        private long bytesWritten = 0;
        private bool inited = false;

        public virtual int CompressionLevel
        {
            get
            {
                return 0;
            }
            set
            {
                if (value != 0)
                    throw new Exception("unsupported compression level");
            }
        }

        public virtual object Settings
        {
            get
            {
                return null;
            }
            set
            {
                if (value != null && value.GetType() != typeof(object))
                    throw new Exception("Unsupported options " + value);
            }
        }

        public long Padding
        {
            set { }
        }

        public long Position
        {
            get { return position; }
        }

        public long FinalSampleCount
        {
            set { sample_count = (int)value; }
        }

        public long BlockSize
        {
            set { }
            get { return 0; }
        }

        public AudioPCMConfig PCM
        {
            get { return _pcm; }
        }

        public string Path { get { return _path; } }

        public long BytesWritten
        {
            get { return bytesWritten; }
        }

        public LAMEEncoder(string path, Stream IO, AudioPCMConfig pcm)
        {
            if (pcm.BitsPerSample != 16)// && pcm.BitsPerSample != 32)
                throw new ArgumentOutOfRangeException("format", "Only 16 & 32 bits samples supported");
            _pcm = pcm;
            _path = path;
            _IO = IO;
        }

        public LAMEEncoder(string path, AudioPCMConfig pcm)
            : this(path, null, pcm)
        {
        }

        public void DeInit(bool flush)
        {
            if (!inited || closed)
                return;

            try
            {
                if (flush)
                {
                    uint EncodedSize = 0;
                    if (m_InBufferPos > 0)
                    {
                        if (Lame_encDll.EncodeChunk(m_hLameStream, m_InBuffer, 0, (uint)m_InBufferPos, m_OutBuffer, ref EncodedSize) == Lame_encDll.BE_ERR_SUCCESSFUL)
                        {
                            if (EncodedSize > 0)
                            {
                                _IO.Write(m_OutBuffer, 0, (int)EncodedSize);
                                bytesWritten += EncodedSize;
                            }
                        }
                    }
                    EncodedSize = 0;
                    if (Lame_encDll.beDeinitStream(m_hLameStream, m_OutBuffer, ref EncodedSize) == Lame_encDll.BE_ERR_SUCCESSFUL)
                    {
                        if (EncodedSize > 0)
                        {
                            _IO.Write(m_OutBuffer, 0, (int)EncodedSize);
                            bytesWritten += EncodedSize;
                        }
                    }
                }
            }
            finally
            {
                Lame_encDll.beCloseStream(m_hLameStream);
                _IO.Close();
                closed = true;
            }
        }

        public void Close()
        {
            bool needTag = !closed && _path != null && _path != "";
            DeInit(true);
            if (needTag)
            {
                bool utf8Required = Encoding.Default.GetString(Encoding.Default.GetBytes(_path)) != _path;
                var tempDir = System.IO.Path.Combine(System.IO.Path.GetPathRoot(_path), "Temp");
                var tempName = utf8Required ? System.IO.Path.Combine(tempDir, Guid.NewGuid().ToString()) : _path;
                try
                {
                    if (utf8Required && !Directory.Exists(tempDir)) Directory.CreateDirectory(tempDir);
                    if (utf8Required) File.Move(_path, tempName);
                    Lame_encDll.beWriteInfoTag(m_hLameStream, tempName);
                    if (utf8Required) File.Move(tempName, _path);
                }
                catch
                {
                    if (utf8Required) File.Move(tempName, _path);
                }
            }
        }

        public void Delete()
        {
            if (!closed)
            {
                DeInit(false);
                if (_path != "")
                    File.Delete(_path);
            }
        }

        protected virtual BE_CONFIG MakeConfig()
        {
            return new BE_CONFIG(_pcm);
        }

        private void Init()
        {
            if (inited)
                return;

            m_Mp3Config = MakeConfig();

            uint LameResult = Lame_encDll.beInitStream(m_Mp3Config, ref m_InputSamples, ref m_OutBufferSize, ref m_hLameStream);
            if (LameResult != Lame_encDll.BE_ERR_SUCCESSFUL)
                throw new ApplicationException(string.Format("Lame_encDll.beInitStream failed with the error code {0}", LameResult));

            m_InBuffer = new byte[m_InputSamples * 2]; //Input buffer is expected as short[]
            m_OutBuffer = new byte[Math.Max(65536, m_OutBufferSize)];

            if (_IO == null)
                _IO = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read);

            inited = true;
        }

        public unsafe void Write(AudioBuffer buff)
        {
            buff.Prepare(this);

            Init();

            byte[] buffer = buff.Bytes;
            int index = 0;
            int count = buff.ByteLength;

            int ToCopy = 0;
            uint EncodedSize = 0;
            uint LameResult;
            uint outBufferIndex = 0;
            fixed (byte* pBuffer = buffer, pOutBuffer = m_OutBuffer)
            {
                while (count > 0)
                {
                    if (m_InBufferPos > 0)
                    {
                        ToCopy = Math.Min(count, m_InBuffer.Length - m_InBufferPos);
                        Buffer.BlockCopy(buffer, index, m_InBuffer, m_InBufferPos, ToCopy);
                        m_InBufferPos += ToCopy;
                        index += ToCopy;
                        count -= ToCopy;
                        if (m_InBufferPos >= m_InBuffer.Length)
                        {
                            m_InBufferPos = 0;
                            if (outBufferIndex > 0)
                            {
                                _IO.Write(m_OutBuffer, 0, (int)outBufferIndex);
                                bytesWritten += outBufferIndex;
                                outBufferIndex = 0;
                            }

                            if ((LameResult = Lame_encDll.EncodeChunk(m_hLameStream, m_InBuffer, m_OutBuffer, ref EncodedSize)) == Lame_encDll.BE_ERR_SUCCESSFUL)
                            {
                                outBufferIndex += EncodedSize;
                            }
                            else
                            {
                                throw new ApplicationException(string.Format("Lame_encDll.EncodeChunk failed with the error code {0}", LameResult));
                            }
                        }
                    }
                    else
                    {
                        if (count >= m_InBuffer.Length)
                        {
                            if (outBufferIndex + m_OutBufferSize > m_OutBuffer.Length)
                            {
                                _IO.Write(m_OutBuffer, 0, (int)outBufferIndex);
                                bytesWritten += outBufferIndex;
                                outBufferIndex = 0;
                            }

                            if ((LameResult = Lame_encDll.EncodeChunk(m_hLameStream, pBuffer + index, (uint)m_InBuffer.Length, pOutBuffer + outBufferIndex, ref EncodedSize)) == Lame_encDll.BE_ERR_SUCCESSFUL)
                            {
                                outBufferIndex += EncodedSize;
                            }
                            else
                            {
                                throw new ApplicationException(string.Format("Lame_encDll.EncodeChunk failed with the error code {0}", LameResult));
                            }
                            count -= m_InBuffer.Length;
                            index += m_InBuffer.Length;
                        }
                        else
                        {
                            Buffer.BlockCopy(buffer, index, m_InBuffer, 0, count);
                            m_InBufferPos = count;
                            index += count;
                            count = 0;
                        }
                    }
                }
            }

            if (outBufferIndex > 0)
            {
                _IO.Write(m_OutBuffer, 0, (int)outBufferIndex);
                bytesWritten += outBufferIndex;
            }
        }
    }
}
