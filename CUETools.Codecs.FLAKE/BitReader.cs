/**
 * CUETools.Flake: pure managed FLAC audio encoder
 * Copyright (c) 2009 Gregory S. Chudov
 * Based on Flake encoder, http://flake-enc.sourceforge.net/
 * Copyright (c) 2006-2009 Justin Ruggles
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.FLAKE
{
	unsafe class BitReader
	{
		byte* buffer;
		int pos, len;
		int _bitaccumulator;
		uint cache;

		static readonly byte[] byte_to_unary_table = new byte[] 
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

		public int Position
		{
			get { return pos; }
		}

		public byte* Buffer
		{
			get
			{
				return buffer;
			}
		}

		public BitReader(byte* _buffer, int _pos, int _len)
		{
			buffer = _buffer;
			pos = _pos;
			len = _len;
			_bitaccumulator = 0;
			cache = peek4();
		}

		public uint peek4()
		{
			//uint result = ((((uint)buffer[pos]) << 24) | (((uint)buffer[pos + 1]) << 16) | (((uint)buffer[pos + 2]) << 8) | ((uint)buffer[pos + 3])) << _bitaccumulator;
			byte* b = buffer + pos;
			uint result = *(b++);
			result = (result << 8) + *(b++);
			result = (result << 8) + *(b++);
			result = (result << 8) + *(b++);
			result <<= _bitaccumulator;
			return result;
		}

		/* skip any number of bits */
		public void skipbits(int bits)
		{
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			cache = peek4();
		}

		/* skip up to 8 bits */
		public void skipbits8(int bits)
		{
			cache <<= bits;
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			cache |= ((uint)buffer[pos + 3] << _bitaccumulator);
		}

		/* supports reading 1 to 24 bits, in big endian format */
		public uint readbits24(int bits)
		{
			//uint result = peek4() >> (32 - bits);
			uint result = cache >> (32 - bits);
			skipbits(bits);
			return result;
		}

		public uint peekbits24(int bits)
		{
			return cache >> 32 - bits;
		}

		/* supports reading 1 to 32 bits, in big endian format */
		public uint readbits(int bits)
		{
			uint result = cache >> 32 - bits;
			if (bits <= 24)
			{
				skipbits(bits);
				return result;
			}
			skipbits(24);
			result |= cache >> 56 - bits;
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
			uint result = cache >> 31;
			skipbits8(1);
			return result;
		}

		public uint read_unary()
		{
			uint val = 0;

			uint result = cache >> 24;
			while (result == 0)
			{
				val += 8;
				skipbits8(8);
				result = cache >> 24;
			}

			val += byte_to_unary_table[result];
			skipbits8((int)(val & 7) + 1);
			return val;
		}

		public void flush()
		{
			if (_bitaccumulator > 0)
				skipbits8(8 - _bitaccumulator);
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
				if (k == 0)
					for (int i = n; i > 0; i--)
						*(r++) = read_unary_signed();
				else if (k <= 8)
					for (int i = n; i > 0; i--)
					{
						//*(r++) = read_rice_signed((int)k);
						uint bits = unary_table[cache >> 24];
						uint msbs = bits;
						while (bits == 8)
						{
							skipbits8(8);
							bits = unary_table[cache >> 24];
							msbs += bits;
						}
						skipbits8((int)(msbs & 7) + 1);
						uint uval = (msbs << k) | (cache >> (32 - k));
						skipbits8(k);
						*(r++) = (int)(uval >> 1 ^ -(int)(uval & 1));
					}
				else
					for (int i = n; i > 0; i--)
					{
						//*(r++) = read_rice_signed((int)k);
						uint bits = unary_table[cache >> 24];
						uint msbs = bits;
						while (bits == 8)
						{
							skipbits8(8);
							bits = unary_table[cache >> 24];
							msbs += bits;
						}
						skipbits8((int)(msbs & 7) + 1);
						uint uval = (msbs << k) | (cache >> (32 - k));
						skipbits(k);
						*(r++) = (int)(uval >> 1 ^ -(int)(uval & 1));
					}
			}
		}
	}
}
