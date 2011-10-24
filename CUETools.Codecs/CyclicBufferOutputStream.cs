using System;
using System.IO;

namespace CUETools.Codecs
{
    public class CyclicBufferOutputStream : Stream
    {
        private CyclicBuffer _buffer;

        public override bool CanRead
        {
            get { return false; }
        }

        public override bool CanSeek
        {
            get { return false; }
        }

        public override bool CanWrite
        {
            get { return true; }
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

        public CyclicBufferOutputStream(CyclicBuffer buffer)
        {
            _buffer = buffer;
        }

        public CyclicBufferOutputStream(Stream output, int size)
        {
            _buffer = new CyclicBuffer(size);
            _buffer.WriteTo(output);
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
            throw new NotSupportedException();
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            throw new NotSupportedException();
        }

        public override void Write(byte[] array, int offset, int count)
        {
            _buffer.Read(array, offset, count);
        }
    }
}
