using System;

namespace CUETools.Codecs
{
	unsafe public class BitReader
    {
        #region Static Methods

        public static int log2i(int v)
        {
            return log2i((uint)v);
        }

        public static int log2i(ulong v)
        {
            int n = 0;
            if (0 != (v & 0xffffffff00000000)) { v >>= 32; n += 32; }
            if (0 != (v & 0xffff0000)) { v >>= 16; n += 16; }
            if (0 != (v & 0xff00)) { v >>= 8; n += 8; }
            return n + byte_to_log2_table[v];
        }

        public static int log2i(uint v)
        {
            int n = 0;
            if (0 != (v & 0xffff0000)) { v >>= 16; n += 16; }
            if (0 != (v & 0xff00)) { v >>= 8; n += 8; }
            return n + byte_to_log2_table[v];
        }

        public static readonly byte[] byte_to_unary_table = new byte[] 
		{
			8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
			3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};

        public static readonly byte[] byte_to_log2_table = new byte[] 
		{
			0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
			4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
			5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
		};

        #endregion

        private byte* buffer_m;
        private byte* bptr_m;
        private int buffer_len_m;
        private int have_bits_m;
        private ulong cache_m;
        private ushort crc16_m;

		public int Position
		{
			get { return (int)(bptr_m - buffer_m - (have_bits_m >> 3)); }
		}

		public byte* Buffer
		{
			get
			{
				return buffer_m;
			}
		}

		public BitReader()
		{
			buffer_m = null;
            bptr_m = null;
			buffer_len_m = 0;
			have_bits_m = 0;
			cache_m = 0;
            crc16_m = 0;
		}

		public BitReader(byte* _buffer, int _pos, int _len)
		{
			Reset(_buffer, _pos, _len);
		}

		public void Reset(byte* _buffer, int _pos, int _len)
		{
			buffer_m = _buffer;
            bptr_m = _buffer + _pos;
			buffer_len_m = _len;
			have_bits_m = 0;
            cache_m = 0;
            crc16_m = 0;
			fill();
		}

		public void fill()
		{
            while (have_bits_m < 56)
            {
                have_bits_m += 8;
                byte b = *(bptr_m++);
                cache_m |= (ulong)b << (64 - have_bits_m);
                crc16_m = (ushort)((crc16_m << 8) ^ Crc16.table[(crc16_m >> 8) ^ b]);
            }
        }

		/* skip any number of bits */
		public void skipbits(int bits)
		{
            while (bits > have_bits_m)
            {
                bits -= have_bits_m;
                cache_m = 0;
                have_bits_m = 0;
                fill();
            }
            cache_m <<= bits;
            have_bits_m -= bits;
		}

        public long read_long()
        {
            return ((long)readbits(32) << 32) | readbits(32);
        }

        public ulong read_ulong()
        {
            return ((ulong)readbits(32) << 32) | readbits(32);
        }

        public int read_int()
        {
            return (int)readbits(sizeof(int));
        }
        
        public uint read_uint()
        {
            return (uint)readbits(sizeof(uint));
        }

        public short read_short()
        {
            return (short)readbits(16);
        }

        public ushort read_ushort()
        {
            return (ushort)readbits(16);
        }

		/* supports reading 1 to 32 bits, in big endian format */
        public uint readbits(int bits)
        {
            fill();
            uint result = (uint)(cache_m >> 64 - bits);
            skipbits(bits);
            return result;
        }

        /* supports reading 1 to 64 bits, in big endian format */
        public ulong readbits64(int bits)
		{
			if (bits <= 56)
				return readbits(bits);
			return ((ulong)readbits(32) << bits - 32) | readbits(bits - 32);
		}

		/* reads a single bit */
		public uint readbit()
		{
            return readbits(1);
		}

		public uint read_unary()
		{
            fill();
            uint val = 0;
			ulong result = cache_m >> 56;
			while (result == 0)
			{
				val += 8;
                cache_m <<= 8;
                byte b = *(bptr_m++);
                cache_m |= (ulong)b << (64 - have_bits_m);
                crc16_m = (ushort)((crc16_m << 8) ^ Crc16.table[(crc16_m >> 8) ^ b]);
                result = cache_m >> 56;
			}
			val += byte_to_unary_table[result];
			skipbits((int)(val & 7) + 1);
			return val;
		}

		public void flush()
		{
            if ((have_bits_m & 7) > 0)
            {
                cache_m <<= have_bits_m & 7;
                have_bits_m -= have_bits_m & 7;
            }
        }

        public ushort get_crc16()
        {
            if (have_bits_m == 0)
                return crc16_m;
            ushort crc = 0;
            int n = have_bits_m >> 3;
            for (int i = 0; i < n; i++)
                crc = (ushort)((crc << 8) ^ Crc16.table[(crc >> 8) ^ (byte)(cache_m >> (56 - (i << 3)))]);
            return Crc16.Substract(crc16_m, crc, n);
        }

        public int readbits_signed(int bits)
		{
			int val = (int)readbits(bits);
			val <<= (32 - bits);
			val >>= (32 - bits);
			return val;
		}

		public uint read_utf8()
		{
			uint x = readbits(8);
			uint v;
			int i;
			if (0 == (x & 0x80))
			{
				v = x;
				i = 0;
			}
			else if (0xC0 == (x & 0xE0)) /* 110xxxxx */
			{
				v = x & 0x1F;
				i = 1;
			}
			else if (0xE0 == (x & 0xF0)) /* 1110xxxx */
			{
				v = x & 0x0F;
				i = 2;
			}
			else if (0xF0 == (x & 0xF8)) /* 11110xxx */
			{
				v = x & 0x07;
				i = 3;
			}
			else if (0xF8 == (x & 0xFC)) /* 111110xx */
			{
				v = x & 0x03;
				i = 4;
			}
			else if (0xFC == (x & 0xFE)) /* 1111110x */
			{
				v = x & 0x01;
				i = 5;
			}
            else if (0xFE == x) /* 11111110 */
            {
                v = 0;
                i = 6;
            }
            else
            {
                throw new Exception("invalid utf8 encoding");
            }
			for (; i > 0; i--)
			{
				x = readbits(8);
				if (0x80 != (x & 0xC0))  /* 10xxxxxx */
					throw new Exception("invalid utf8 encoding");
				v <<= 6;
				v |= (x & 0x3F);
			}
			return v;
		}

		public void read_rice_block(int n, int k, int* r)
		{
            fill();
            fixed (byte* unary_table = byte_to_unary_table)
            fixed (ushort* t = Crc16.table)
            {
                uint mask = (1U << k) - 1;
                byte* bptr = bptr_m;
                int have_bits = have_bits_m;
                ulong cache = cache_m;
                ushort crc = crc16_m;
                for (int i = n; i > 0; i--)
                {
                    uint bits;
                    byte* orig_bptr = bptr;
                    while ((bits = unary_table[cache >> 56]) == 8)
                    {
                        cache <<= 8;
                        byte b = *(bptr++);
                        cache |= (ulong)b << (64 - have_bits);
                        crc = (ushort)((crc << 8) ^ t[(crc >> 8) ^ b]);
                    }
                    uint msbs = bits + ((uint)(bptr - orig_bptr) << 3);
                    // assumes k <= 41 (have_bits < 41 + 7 + 1 + 8 == 57, so we don't loose bits here)
                    while (have_bits < 56)
                    {
                        have_bits += 8;
                        byte b = *(bptr++);
                        cache |= (ulong)b << (64 - have_bits);
                        crc = (ushort)((crc << 8) ^ t[(crc >> 8) ^ b]);
                    }

                    int btsk = k + (int)bits + 1;
                    uint uval = (msbs << k) | (uint)((cache >> (64 - btsk)) & mask);
                    cache <<= btsk;
                    have_bits -= btsk;
                    *(r++) = (int)(uval >> 1 ^ -(int)(uval & 1));
                }
                have_bits_m = have_bits;
                cache_m = cache;
                bptr_m = bptr;
                crc16_m = crc;
            }
		}
	}
}
