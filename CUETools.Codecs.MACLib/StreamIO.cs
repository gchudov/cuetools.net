using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace CUETools.Codecs.MACLib
{
    internal unsafe class StreamIO : IDisposable
    {
        public StreamIO(Stream stream)
        {
            m_stream = stream;

            // We keep the references to those callbacks to
            // prevent them from being garbage collected.
            m_read_bytes = ReadCallback;
            m_write_bytes = WriteCallback;
            m_get_pos = TellCallback;
            m_get_size = GetSizeCallback;
            m_seek = SeekRelativeCallback;

            m_hCIO = MACLibDll.c_APECIO_Create(null, m_read_bytes, m_write_bytes, m_seek, m_get_pos, m_get_size);
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
                m_stream = null;
                _readBuffer = null;
            }

            MACLibDll.c_APECIO_Destroy(m_hCIO);
            m_hCIO = IntPtr.Zero;
        }

        ~StreamIO()
        {
            Dispose(false);
        }

        private int ReadCallback(void* id, void* data, int bcount, out int pBytesRead)
        {
            if (_readBuffer == null || _readBuffer.Length < bcount)
                _readBuffer = new byte[Math.Max(bcount, 0x4000)];
            int len = m_stream.Read(_readBuffer, 0, bcount);
            if (len > 0) Marshal.Copy(_readBuffer, 0, (IntPtr)data, len);
            pBytesRead = len;
            return 0;
        }

        private int WriteCallback(void* id, void* data, int bcount, out int pBytesWritten)
        {
            if (_readBuffer == null || _readBuffer.Length < bcount)
                _readBuffer = new byte[Math.Max(bcount, 0x4000)];
            Marshal.Copy((IntPtr)data, _readBuffer, 0, bcount);
            m_stream.Write(_readBuffer, 0, bcount);
            pBytesWritten = bcount;
            return 0;
        }

        int TellCallback(void* id)
        {
            return (int)m_stream.Position;
        }

        uint GetSizeCallback(void* id)
        {
            return (uint)m_stream.Length;
        }

        long SeekRelativeCallback(void* id, long delta, int mode)
        {
            m_stream.Seek(delta, (SeekOrigin)(mode));
            return 0;
        }

        internal IntPtr CIO => m_hCIO;

        Stream m_stream;
        byte[] _readBuffer;
        CIO_ReadDelegate m_read_bytes;
        CIO_WriteDelegate m_write_bytes;
        CIO_GetPositionDelegate m_get_pos;
        CIO_GetSizeDelegate m_get_size;
        CIO_SeekDelegate m_seek;
        internal IntPtr m_hCIO;
    }
}
