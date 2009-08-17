using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.FLAKE
{
	class BitReader
	{
		byte[] buffer;
		byte[] byte_to_unary_table;
		int pos, len;
		int _bitaccumulator;

		public int Position
		{
			get { return pos; }
		}

		public byte[] Buffer
		{
			get
			{
				return buffer;
			}
		}

		public BitReader(byte[] _buffer, int _pos, int _len)
		{
			buffer = _buffer;
			pos = _pos;
			len = _len;
			_bitaccumulator = 0;

			byte_to_unary_table = new byte[] {
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
		}

		/* supports reading 1 to 24 bits, in big endian format */
		public uint readbits_24(int bits)
		{
			uint result = (((uint)buffer[pos]) << 24) | (((uint)buffer[pos + 1]) << 16) | (((uint)buffer[pos + 2]) << 8) | ((uint)buffer[pos + 3]);
			result <<= _bitaccumulator;
			result >>= 32 - bits;

			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return result;
		}

		public uint readbits_8(int bits)
		{
			uint result = (((uint)buffer[pos]) << 24) | (((uint)buffer[pos + 1]) << 16);
			result <<= _bitaccumulator;
			result >>= 32 - bits;

			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return result;
		}

		public uint peekbits_24(int bits)
		{
			uint result = (((uint)buffer[pos]) << 24) | (((uint)buffer[pos + 1]) << 16) | (((uint)buffer[pos + 2]) << 8) | ((uint)buffer[pos + 3]);
			result <<= _bitaccumulator;
			result >>= 32 - bits;
			return result;
		}

		///* supports reading 1 to 16 bits, in big endian format */
		//private unsafe uint peekbits_9(byte* buff, int pos)
		//{
		//    uint result = (((uint)buff[pos]) << 8) | (((uint)buff[pos + 1]));
		//    result <<= _bitaccumulator;
		//    result &= 0x0000ffff;
		//    result >>= 7;
		//    return result;
		//}

		/* supports reading 1 to 16 bits, in big endian format */
		public void skipbits(int bits)
		{
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
		}

		/* supports reading 1 to 32 bits, in big endian format */
		public uint readbits(int bits)
		{
			if (bits <= 24)
				return readbits_24(bits);

			ulong result = (((ulong)buffer[pos]) << 32) | (((ulong)buffer[pos + 1]) << 24) | (((ulong)buffer[pos + 2]) << 16) | (((ulong)buffer[pos + 3]) << 8) | ((ulong)buffer[pos + 4]);
			result <<= _bitaccumulator;
			result &= 0x00ffffffffff;
			result >>= 40 - bits;
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return (uint)result;
		}

		/* reads a single bit */
		public uint readbit()
		{
			int new_accumulator;
			uint result = buffer[pos];
			result <<= _bitaccumulator;
			result = result >> 7 & 1;
			new_accumulator = (_bitaccumulator + 1);
			pos += (new_accumulator / 8);
			_bitaccumulator = (new_accumulator % 8);
			return result;
		}

		public uint read_unary()
		{
			uint val = 0;

			int result = (buffer[pos] << _bitaccumulator) & 0xff;
			if (result == 0)
			{
				val = 8 - (uint)_bitaccumulator;
				_bitaccumulator = 0;
				pos++;
				return val + read_unary();
				// check eof
			}
			
			val = byte_to_unary_table[result];

			int new_accumulator = (_bitaccumulator + (int)val + 1);
			pos += (new_accumulator / 8);
			_bitaccumulator = (new_accumulator % 8);
			return val;
		}

		public void flush()
		{
			if (_bitaccumulator > 0)
				readbits(8 - _bitaccumulator);
		}

		public int readbits_signed(int bits)
		{
			int val = (int) readbits(bits);
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
				throw new Exception("invalid utf8 encoding");
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
			uint lsbs = readbits_24(k);
			uint uval = (msbs << k) | lsbs;
			return (int)(uval >> 1 ^ -(int)(uval & 1));
		}

		public int read_rice_signed8(int k)
		{
			uint msbs = read_unary();
			uint lsbs = readbits_8(k);
			uint uval = (msbs << k) | lsbs;
			return (int)(uval >> 1 ^ -(int)(uval & 1));
		}

		public int read_unary_signed()
		{
			uint uval = read_unary();
			return (int)(uval >> 1 ^ -(int)(uval & 1));
		}
	}
}
