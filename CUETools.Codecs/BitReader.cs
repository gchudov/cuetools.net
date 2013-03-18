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
        private int buffer_offset_m, buffer_len_m;
        private int bitaccumulator_m;
        private uint cache_m;

		public int Position
		{
			get { return buffer_offset_m; }
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
			buffer_offset_m = 0;
			buffer_len_m = 0;
			bitaccumulator_m = 0;
			cache_m = 0;
		}

		public BitReader(byte* _buffer, int _pos, int _len)
		{
			Reset(_buffer, _pos, _len);
		}

		public void Reset(byte* _buffer, int _pos, int _len)
		{
			buffer_m = _buffer;
			buffer_offset_m = _pos;
			buffer_len_m = _len;
			bitaccumulator_m = 0;
			cache_m = peek4();
		}

		public uint peek4()
		{
			//uint result = ((((uint)buffer[pos]) << 24) | (((uint)buffer[pos + 1]) << 16) | (((uint)buffer[pos + 2]) << 8) | ((uint)buffer[pos + 3])) << bitaccumulator_m;
			byte* b = buffer_m + buffer_offset_m;
			uint result = *(b++);
			result = (result << 8) + *(b++);
			result = (result << 8) + *(b++);
			result = (result << 8) + *(b++);
			result <<= bitaccumulator_m;
			return result;
		}

		/* skip any number of bits */
		public void skipbits(int bits)
		{
			int new_accumulator = (bitaccumulator_m + bits);
			buffer_offset_m += (new_accumulator >> 3);
			bitaccumulator_m = (new_accumulator & 7);
			cache_m = peek4();
		}

		/* skip up to 16 bits */
		public void skipbits16(int bits)
		{
			cache_m <<= bits;
			int new_accumulator = (bitaccumulator_m + bits);
			buffer_offset_m += (new_accumulator >> 3);
			bitaccumulator_m = (new_accumulator & 7);
			cache_m |= ((((uint)buffer_m[buffer_offset_m + 2] << 8) + (uint)buffer_m[buffer_offset_m + 3]) << bitaccumulator_m);
		}

		/* skip up to 8 bits */
		public void skipbits8(int bits)
		{
			cache_m <<= bits;
			int new_accumulator = (bitaccumulator_m + bits);
			buffer_offset_m += (new_accumulator >> 3);
			bitaccumulator_m = (new_accumulator & 7);
			cache_m |= ((uint)buffer_m[buffer_offset_m + 3] << bitaccumulator_m);
		}

		/* supports reading 1 to 24 bits, in big endian format */
		public uint readbits24(int bits)
		{
			//uint result = peek4() >> (32 - bits);
			uint result = cache_m >> (32 - bits);
			skipbits(bits);
			return result;
		}

		public uint peekbits24(int bits)
		{
			return cache_m >> 32 - bits;
		}

		/* supports reading 1 to 32 bits, in big endian format */
		public uint readbits(int bits)
		{
			uint result = cache_m >> 32 - bits;
			if (bits <= 24)
			{
				skipbits(bits);
				return result;
			}
			skipbits(24);
			result |= cache_m >> 56 - bits;
			skipbits(bits - 24);
			return result;
		}

		public ulong readbits64(int bits)
		{
			if (bits <= 24)
				return readbits24(bits);
			ulong result = readbits24(24);
			bits -= 24;
			if (bits <= 24)
				return (result << bits) | readbits24(bits);
			result = (result << 24) | readbits24(24);
			bits -= 24;
			return (result << bits) | readbits24(bits);
		}

		/* reads a single bit */
		public uint readbit()
		{
			uint result = cache_m >> 31;
			skipbits8(1);
			return result;
		}

		public uint read_unary()
		{
			uint val = 0;

			uint result = cache_m >> 24;
			while (result == 0)
			{
				val += 8;
				skipbits8(8);
				result = cache_m >> 24;
			}

			val += byte_to_unary_table[result];
			skipbits8((int)(val & 7) + 1);
			return val;
		}

		public void flush()
		{
			if (bitaccumulator_m > 0)
				skipbits8(8 - bitaccumulator_m);
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

		public int read_rice_signed(int k)
		{
			uint msbs = read_unary();
			uint lsbs = readbits24(k);
			uint uval = (msbs << k) | lsbs;
			return (int)(uval >> 1 ^ -(int)(uval & 1));
		}

		public int read_unary_signed()
		{
			uint uval = read_unary();
			return (int)(uval >> 1 ^ -(int)(uval & 1));
		}

		public void read_rice_block(int n, int k, int* r)
		{
            fixed (byte* unary_table = byte_to_unary_table)
            {
                uint mask = (1U << k) - 1;
                byte* bptr = &buffer_m[buffer_offset_m];
                int have_bits = 24 - bitaccumulator_m;
                ulong _lcache = ((ulong)cache_m) << 32;
                bptr += 3;
                for (int i = n; i > 0; i--)
                {
                    uint bits;
                    byte* orig_bptr = bptr;
                    while ((bits = unary_table[_lcache >> 56]) == 8)
                    {
                        _lcache <<= 8;
                        _lcache |= (ulong)*(bptr++) << (64 - have_bits);
                    }
                    uint msbs = bits + ((uint)(bptr - orig_bptr) << 3);
                    // assumes k <= 41 (have_bits < 41 + 7 + 1 + 8 == 57, so we don't loose bits here)
                    while (have_bits < 56)
                    {
                        have_bits += 8;
                        _lcache |= (ulong)*(bptr++) << (64 - have_bits);
                    }

                    int btsk = k + (int)bits + 1;
                    uint uval = (msbs << k) | (uint)((_lcache >> (64 - btsk)) & mask);
                    _lcache <<= btsk;
                    have_bits -= btsk;
                    *(r++) = (int)(uval >> 1 ^ -(int)(uval & 1));
                }
                while (have_bits <= 24)
                {
                    _lcache |= ((ulong)bptr[0] << 56) >> have_bits;
                    have_bits += 8;
                    bptr++;
                }
                while (have_bits > 32)
                {
                    have_bits -= 8;
                    bptr--;
                }
                bitaccumulator_m = 32 - have_bits;
                cache_m = (uint)(_lcache >> 32);
                bptr -= 4;
                buffer_offset_m = (int)(bptr - buffer_m);
            }
		}
	}
}
