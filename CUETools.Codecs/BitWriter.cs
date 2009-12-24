/**
 * CUETools.Codecs: common audio encoder/decoder routines
 * Copyright (c) 2009 Gregory S. Chudov
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

namespace CUETools.Codecs
{
	public class BitWriter
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

		public void Reset()
		{
			buf_ptr = buf_start;
			bit_left = 32;
			bit_buf = 0;
			eof = false;
		}

		public void writebytes(int bytes, byte c)
		{
			for (; bytes > 0; bytes--)
				writebits(8, c);
		}

		public void write(params char [] chars)
		{
			foreach (char c in chars)
				writebits(8, (byte)c);
		}

		public void write(string s)
		{
			for (int i = 0; i < s.Length; i++)
				writebits(8, (byte)s[i]);
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

		/// <summary>
		/// Assumes there's enough space, buffer != null and bits is in range 1..31
		/// </summary>
		/// <param name="bits"></param>
		/// <param name="val"></param>
		unsafe void writebits_fast(int bits, uint val, ref byte * buf)
		{
#if DEBUG
			if ((buf_ptr + 3) >= buf_end)
			{
				eof = true;
				return;
			}
#endif
			if (bits < bit_left)
			{
				bit_buf = (bit_buf << bits) | val;
				bit_left -= bits;
			}
			else
			{
				uint bb = (bit_buf << bit_left) | (val >> (bits - bit_left));
				bit_left += (32 - bits);

				*(buf++) = (byte)(bb >> 24);
				*(buf++) = (byte)(bb >> 16);
				*(buf++) = (byte)(bb >> 8);
				*(buf++) = (byte)(bb);

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
			int bytes = (BitReader.log2i(val) + 4) / 5;
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
			// convert signed to unsigned
			int v = -2 * val - 1;
			v ^= (v >> 31);

			// write quotient in unary
			int q = (v >> k) + 1;
			while (q + k > 31)
			{
				int b = Math.Min(q + k - 31, 31);
				writebits(b, 0);
				q -= b;
			}

			// write remainder in binary using 'k' bits
			writebits(k + q, (v & ((1 << k) - 1)) | (1 << k));
		}

		public unsafe void write_rice_block_signed(byte * fixedbuf, int k, int* residual, int count)
		{
			byte* buf = &fixedbuf[buf_ptr];
			//fixed (byte* fixbuf = &buffer[buf_ptr])
			{
				//byte* buf = fixbuf;
				for (int i = count; i > 0; i--)
				{
					int v = *(residual++);
					v = (v << 1) ^ (v >> 31);

					// write quotient in unary
					int q = (v >> k) + 1;
					int bits = k + q;
					while (bits > 31)
					{
#if DEBUG
						if (buf + 3 >= fixedbuf + buf_end)
						{
							eof = true;
							return;
						}
#endif
						int b = Math.Min(bits - 31, 31);
						if (b < bit_left)
						{
							bit_buf = (bit_buf << b);
							bit_left -= b;
						}
						else
						{
							uint bb = bit_buf << bit_left;
							bit_buf = 0;
							bit_left += (32 - b);
							*(buf++) = (byte)(bb >> 24);
							*(buf++) = (byte)(bb >> 16);
							*(buf++) = (byte)(bb >> 8);
							*(buf++) = (byte)(bb);
						}
						bits -= b;
					}

#if DEBUG
					if (buf + 3 >= fixedbuf + buf_end)
					{
						eof = true;
						return;
					}
#endif

					// write remainder in binary using 'k' bits
					//writebits_fast(k + q, (uint)((v & ((1 << k) - 1)) | (1 << k)), ref buf);
					uint val = (uint)((v & ((1 << k) - 1)) | (1 << k));
					if (bits < bit_left)
					{
						bit_buf = (bit_buf << bits) | val;
						bit_left -= bits;
					}
					else
					{
						uint bb = (bit_buf << bit_left) | (val >> (bits - bit_left));
						bit_buf = val;
						bit_left += (32 - bits);
						*(buf++) = (byte)(bb >> 24);
						*(buf++) = (byte)(bb >> 16);
						*(buf++) = (byte)(bb >> 8);
						*(buf++) = (byte)(bb);
					}
				}
				buf_ptr = (int)(buf - fixedbuf);
			}
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

		public byte[] Buffer
		{
			get
			{
				return buffer;
			}
		}

		public int Length
		{
			get
			{
				return buf_ptr - buf_start;
			}
			set
			{
				flush();
				buf_ptr = buf_start + value;
			}
		}
	}
}
