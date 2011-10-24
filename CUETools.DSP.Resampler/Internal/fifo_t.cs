using System;

namespace CUETools.DSP.Resampler.Internal
{
    class fifo_t
    {
        internal byte[] data;
        int item_size;    /* Size of each item in data */
        int begin;        /* Offset of the first byte to read. */
        int end;          /* 1 + Offset of the last byte byte to read. */

        const int FIFO_MIN = 0x4000;

        public int offset
        {
            get
            {
                return begin;
            }
        }

        public int occupancy
        {
            get
            {
                return (end - begin) / item_size;
            }
        }

        public fifo_t(int item_size)
        {
            this.data = new byte[FIFO_MIN];
            this.item_size = item_size;
            this.begin = 0;
            this.end = 0;
        }

        public void clear()
        {
            this.begin = 0;
            this.end = 0;
        }

        public int reserve(int n)
        {
            n *= item_size;

            if (begin == end)
                clear();

            while (true)
            {
                if (end + n <= data.Length)
                {
                    int pos = end;
                    end += n;
                    return pos;
                }
                if (begin > FIFO_MIN)
                {
                    Buffer.BlockCopy(data, begin, data, 0, end - begin);
                    end -= begin;
                    begin = 0;
                    continue;
                }
                byte[] data1 = new byte[data.Length + Math.Max(n, data.Length)];
                Buffer.BlockCopy(data, begin, data1, 0, end - begin);
                data = data1;
                end -= begin;
                begin = 0;
            }
        }

        public void trim_by(int n)
        {
            end -= n * item_size;
        }

        public int read(int n, byte[] buf)
        {
            n *= item_size;
            if (n > end - begin)
                throw new InvalidOperationException();
            if (buf != null)
                Buffer.BlockCopy(data, begin, buf, 0, n);
            begin += n;
            return begin - n;
        }
    }
}
