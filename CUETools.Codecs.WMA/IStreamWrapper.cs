using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices.ComTypes;
using System.IO;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.WMA
{
    public class StreamWrapper : IStream
    {
        public StreamWrapper(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException("stream", "Can't wrap null stream.");
            this.stream = stream;
        }

        private Stream stream;

        public void Read(byte[] pv, int cb, System.IntPtr pcbRead)
        {
            Marshal.WriteInt32(pcbRead, (Int32)stream.Read(pv, 0, cb));
        }

        public void Seek(long dlibMove, int dwOrigin, System.IntPtr plibNewPosition)
        {
            var res = stream.Seek(dlibMove, (SeekOrigin)dwOrigin);
            if (plibNewPosition != IntPtr.Zero) Marshal.WriteInt32(plibNewPosition, (int)res);
        }

        public void Stat(out System.Runtime.InteropServices.ComTypes.STATSTG pstatstg, int grfStatFlag)
        {
            if (grfStatFlag != 1) // STATFLAG_NONAME
                throw new NotSupportedException();
            var statstg = new System.Runtime.InteropServices.ComTypes.STATSTG();
            statstg.type = 2; // STGTY.STREAM
            statstg.cbSize = stream.Length;
            pstatstg = statstg;
        }

        public void Clone(out IStream ppstm)
        {
            throw new NotSupportedException();
        }

        public void Commit(int grfCommitFlags)
        {
            throw new NotSupportedException();
        }

        public void CopyTo(IStream pstm, long cb, IntPtr pcbRead, IntPtr pcbWritten)
        {
            throw new NotSupportedException();
        }

        public void LockRegion(long libOffset, long cb, int dwLockType)
        {
            throw new NotSupportedException();
        }

        public void Revert()
        {
            throw new NotSupportedException();
        }

        public void SetSize(long libNewSize)
        {
            throw new NotSupportedException();
        }

        public void UnlockRegion(long libOffset, long cb, int dwLockType)
        {
            throw new NotSupportedException();
        }

        public void Write(byte[] pv, int cb, IntPtr pcbWritten)
        {
            throw new NotSupportedException();
        }
    }
}
