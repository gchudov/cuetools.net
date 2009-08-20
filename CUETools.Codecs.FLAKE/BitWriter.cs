using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.FLAKE
{
	class BitWriter
	{
		uint bit_buf;
		int bit_left;
		byte[] buffer;
		int buf_start, buf_ptr, buf_end;
		bool eof;

		public BitWriter(byte[] buf, int pos, int len)
		{
			buffer = buf;
			buf_start = pos;
			buf_ptr = pos;
			buf_end = pos + len;
			bit_left = 32;
			bit_buf = 0;
			eof = false;
		}

		public void writebits_signed(int bits, int val)
		{
			writebits(bits, val & ((1 << bits) - 1));
		}

		public void writebits_signed(uint bits, int val)
		{
			writebits((int) bits, val & ((1 << (int) bits) - 1));
		}

		public void writebits(int bits, int val)
		{
			writebits(bits, (uint)val);
		}

		public void writebits64(int bits, ulong val)
		{
			if (bits > 32)
			{
				writebits(bits - 32, (uint)(val >> 32));
				val &= 0xffffffffL;
				bits = 32;
			}
			writebits(bits, (uint)val);
		}

		public void writebits(int bits, uint val)
		{
			//assert(bits == 32 || val < (1U << bits));

			if (bits == 0 || eof) return;
			if ((buf_ptr + 3) >= buf_end)
			{
				eof = true;
				return;
			}
			if (bits < bit_left)
			{
				bit_buf = (bit_buf << bits) | val;
				bit_left -= bits;
			}
			else
			{
				uint bb = 0;
				if (bit_left == 32)
				{
					//assert(bits == 32);
					bb = val;
				}
				else
				{
					bb = (bit_buf << bit_left) | (val >> (bits - bit_left));
					bit_left += (32 - bits);
				}
				if (buffer != null)
				{
					buffer[buf_ptr + 3] = (byte)(bb & 0xFF); bb >>= 8;
					buffer[buf_ptr + 2] = (byte)(bb & 0xFF); bb >>= 8;
					buffer[buf_ptr + 1] = (byte)(bb & 0xFF); bb >>= 8;
					buffer[buf_ptr + 0] = (byte)(bb & 0xFF);
				}
				buf_ptr += 4;
				bit_buf = val;
			}
		}

		public void write_utf8(int val)
		{
			write_utf8((uint)val);
		}

		public void write_utf8(uint val)
		{
			if (val < 0x80)
			{
				writebits(8, val);
				return;
			}
			int bytes = (Flake.log2i(val) + 4) / 5;
			int shift = (bytes - 1) * 6;
			writebits(8, (256U - (256U >> bytes)) | (val >> shift));
			while (shift >= 6)
			{
				shift -= 6;
				writebits(8, 0x80 | ((val >> shift) & 0x3F));
			}
		}

		public void write_unary_signed(int val)
		{
			// convert signed to unsigned
			int v = -2 * val - 1;
			v ^= (v >> 31);

			// write quotient in unary
			int q = v + 1;
			while (q > 31)
			{
				writebits(31, 0);
				q -= 31;
			}
			writebits(q, 1);
		}

		public void write_rice_signed(int k, int val)
		{
			int v, q;

			// convert signed to unsigned
			v = -2 * val - 1;
			v ^= (v >> 31);

			// write quotient in unary
			q = (v >> k) + 1;
			while (q > 31)
			{
				writebits(31, 0);
				q -= 31;
			}
			writebits(q, 1);

			// write write remainder in binary using 'k' bits
			writebits(k, v & ((1 << k) - 1));
		}

		public void flush()
		{
			bit_buf <<= bit_left;
			while (bit_left < 32 && !eof)
			{
				if (buf_ptr >= buf_end)
				{
					eof = true;
					break;
				}
				if (buffer != null)
					buffer[buf_ptr] = (byte)(bit_buf >> 24);
				buf_ptr++;
				bit_buf <<= 8;
				bit_left += 8;
			}
			bit_left = 32;
			bit_buf = 0;
		}

		public int Length
		{
			get
			{
				return buf_ptr - buf_start;
			}
		}
	}
}
