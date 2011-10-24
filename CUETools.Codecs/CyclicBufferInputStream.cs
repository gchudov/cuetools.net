using System;
using System.IO;

namespace CUETools.Codecs
{
    public class CyclicBufferInputStream : Stream
    {
        private CyclicBuffer _buffer;

        public override bool CanRead
        {
            get { return true; }
        }

        public override bool CanSeek
        {
            get { return false; }
        }

        public override bool CanWrite
        {
            get { return false; }
        }

        public override long Length
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public override long Position
        {
            get { throw new NotSupportedException(); }
            set { throw new NotSupportedException(); }
        }

        public CyclicBufferInputStream(CyclicBuffer buffer)
        {
            _buffer = buffer;
        }

        public override void Close()
        {
            _buffer.Close();
        }

        public override void Flush()
        {
            throw new NotSupportedException();
        }

        public override void SetLength(long value)
        {
            throw new NotSupportedException();
        }

        public override int Read(byte[] array, int offset, int count)
        {
            _buffer.Write(array, offset, count);
            return count;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            throw new NotSupportedException();
        }

        public override void Write(byte[] array, int offset, int count)
        {
            throw new NotSupportedException();
        }
    }
}
