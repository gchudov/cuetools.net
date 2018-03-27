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

            m_read_bytes = ReadCallback;
            m_write_bytes = WriteCallback;
            m_get_pos = TellCallback;
            m_get_size = GetSizeCallback;
            m_seek = SeekRelativeCallback;

            m_callbacks = (APE_CIO_Callbacks*)Marshal.AllocHGlobal(sizeof(APE_CIO_Callbacks)).ToPointer();
            m_callbacks->read_bytes = Marshal.GetFunctionPointerForDelegate(m_read_bytes);
            m_callbacks->write_bytes = Marshal.GetFunctionPointerForDelegate(m_write_bytes);
            m_callbacks->get_pos = Marshal.GetFunctionPointerForDelegate(m_get_pos);
            m_callbacks->get_size = Marshal.GetFunctionPointerForDelegate(m_get_size);
            m_callbacks->seek = Marshal.GetFunctionPointerForDelegate(m_seek);
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

            if (m_callbacks != null) Marshal.FreeHGlobal((IntPtr)m_callbacks);
            m_callbacks = null;
        }

        ~StreamIO()
        {
            Dispose(false);
        }

        private int ReadCallback(APE_CIO_Callbacks* id, void* data, int bcount, out int pBytesRead)
        {
            if (_readBuffer == null || _readBuffer.Length < bcount)
                _readBuffer = new byte[Math.Max(bcount, 0x4000)];
            int len = m_stream.Read(_readBuffer, 0, bcount);
            if (len > 0) Marshal.Copy(_readBuffer, 0, (IntPtr)data, len);
            pBytesRead = len;
            return 0;
        }

        private int WriteCallback(APE_CIO_Callbacks* id, void* data, int bcount, out int pBytesWritten)
        {
            if (_readBuffer == null || _readBuffer.Length < bcount)
                _readBuffer = new byte[Math.Max(bcount, 0x4000)];
            Marshal.Copy((IntPtr)data, _readBuffer, 0, bcount);
            m_stream.Write(_readBuffer, 0, bcount);
            pBytesWritten = bcount;
            return 0;
        }

        int TellCallback(APE_CIO_Callbacks* id)
        {
            return (int)m_stream.Position;
        }

        uint GetSizeCallback(APE_CIO_Callbacks* id)
        {
            return (uint)m_stream.Length;
        }

        int SeekRelativeCallback(APE_CIO_Callbacks* id, IntPtr delta, int mode)
        {
            m_stream.Seek((long)delta, (SeekOrigin)(mode));
            return 0;
        }

        internal APE_CIO_Callbacks* Callbacks => m_callbacks;

        APE_CIO_Callbacks* m_callbacks;
        Stream m_stream;
        byte[] _readBuffer;
        CIO_ReadDelegate m_read_bytes;
        CIO_WriteDelegate m_write_bytes;
        CIO_GetPositionDelegate m_get_pos;
        CIO_GetSizeDelegate m_get_size;
        CIO_SeekDelegate m_seek;
    }
}
