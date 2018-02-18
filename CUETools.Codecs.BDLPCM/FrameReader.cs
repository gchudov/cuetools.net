using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.BDLPCM
{
    internal unsafe class FrameReader
    {
        internal FrameReader(byte* ptr, byte* end)
        {
            ptr_m = ptr;
            end_m = end;
        }

        internal FrameReader(byte* ptr, long len)
        {
            ptr_m = ptr;
            end_m = ptr + len;
        }

        internal FrameReader(FrameReader src, long len)
        {
            if (src.ptr_m + len > src.end_m) throw new IndexOutOfRangeException();
            ptr_m = src.ptr_m;
            end_m = src.ptr_m + len;
        }

        internal void read_bytes(byte* dst, int len)
        {
            if (ptr_m + len > end_m) throw new IndexOutOfRangeException();
            AudioSamples.MemCpy(dst, ptr_m, len);
            ptr_m += len;
        }

        internal byte[] read_bytes(int len)
        {
            var res = new byte[len];
            fixed (byte* ptr = &res[0])
                read_bytes(ptr, len);
            return res;
        }

        internal string read_string(int len)
        {
            var res = new byte[len];
            fixed (byte* ptr = &res[0])
                read_bytes(ptr, len);
            return Encoding.UTF8.GetString(res, 0, res.Length);;
        }

        internal byte read_byte()
        {
            if (ptr_m + 1 > end_m) throw new IndexOutOfRangeException();
            return *(ptr_m++);
        }

        internal ushort read_ushort()
        {
            if (ptr_m + 2 > end_m) throw new IndexOutOfRangeException();
            ushort n = (ushort)(*(ptr_m++));
            n <<= 8; n += (ushort)(*(ptr_m++));
            return n;
        }

        internal uint read_uint()
        {
            if (ptr_m + 4 > end_m) throw new IndexOutOfRangeException();
            uint n = (uint)(*(ptr_m++));
            n <<= 8; n += (uint)(*(ptr_m++));
            n <<= 8; n += (uint)(*(ptr_m++));
            n <<= 8; n += (uint)(*(ptr_m++));
            return n;
        }

        internal ulong read_pts()
        {
            if (ptr_m + 5 > end_m) throw new IndexOutOfRangeException();
            ulong pts
                 = ((ulong)(*(ptr_m++) & 0x0e) << 29);
            pts |= ((ulong)(*(ptr_m++) & 0xff) << 22);
            pts |= ((ulong)(*(ptr_m++) & 0xfe) << 14);
            pts |= ((ulong)(*(ptr_m++) & 0xff) << 7);
            pts |= ((ulong)(*(ptr_m++) & 0xfe) >> 1);
            return pts;
        }

        internal void skip(long bytes)
        {
            if (ptr_m + bytes > end_m) throw new IndexOutOfRangeException();
            ptr_m += bytes;
        }

        internal long Length
        {
            get
            {
                return end_m - ptr_m;
            }
            set
            {
                end_m = ptr_m + value;
            }
        }

        internal byte* Ptr
        {
            get
            {
                return ptr_m;
            }
        }

        byte* ptr_m;
        byte* end_m;
    }
}
