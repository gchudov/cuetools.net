using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
	public class Crc32
	{
		public uint[] table = new uint[256];

		public uint ComputeChecksum(uint crc, byte val)
		{
			return (crc >> 8) ^ table[(crc & 0xff) ^ val];
		}

		public unsafe uint ComputeChecksum(uint crc, byte* bytes, int count)
		{
			fixed (uint *t = table)
				for (int i = 0; i < count; i++)
					crc = (crc >> 8) ^ t[(crc ^ bytes[i]) & 0xff];
			return crc;
		}

		public unsafe uint ComputeChecksum(uint crc, byte[] bytes, int pos, int count)
		{
			fixed (byte* pbytes = &bytes[pos])
				return ComputeChecksum(crc, pbytes, count);
		}

		public uint ComputeChecksum(uint crc, uint s)
		{
			return ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
				crc, (byte)s), (byte)(s >> 8)), (byte)(s >> 16)), (byte)(s >> 24));
		}

		public unsafe uint ComputeChecksum(uint crc, int * samples, int count)
		{
			for (int i = 0; i < count; i++)
			{
				int s1 = samples[2 * i], s2 = samples[2 * i + 1];
				crc = ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
					crc, (byte)s1), (byte)(s1 >> 8)), (byte)s2), (byte)(s2 >> 8));
			}
			return crc;
		}

		uint Reflect(uint val, int ch)
		{
			uint value = 0;
			// Swap bit 0 for bit 7
			// bit 1 for bit 6, etc.
			for (int i = 1; i < (ch + 1); i++)
			{
				if (0 != (val & 1))
					value |= 1U << (ch - i);
				val >>= 1;
			}
			return value;
		}

		const uint ulPolynomial = 0x04c11db7;

		public Crc32()
		{
			for (uint i = 0; i < table.Length; i++)
			{
				table[i] = Reflect(i, 8) << 24;
				for (int j = 0; j < 8; j++)
					table[i] = (table[i] << 1) ^ ((table[i] & (1U << 31)) == 0 ? 0 : ulPolynomial);
				table[i] = Reflect(table[i], 32);
			}
		}

		const int GF2_DIM = 32;

		private unsafe uint gf2_matrix_times(uint* mat, uint vec)
		{
			return *(mat++) * (vec & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1) ^
			*(mat++) * ((vec >>= 1) & 1);
		}

		/* ========================================================================= */
		private unsafe void gf2_matrix_square(uint *square, uint *mat)
		{
			for (int n = 0; n < GF2_DIM; n++)
				square[n] = gf2_matrix_times(mat, mat[n]);
		}


		public unsafe uint Combine(uint crc1, uint crc2, long len2)
		{
			int n;
			uint row;
			uint* even = stackalloc uint[GF2_DIM];    /* even-power-of-two zeros operator */
			uint* odd = stackalloc uint[GF2_DIM];     /* odd-power-of-two zeros operator */

			/* degenerate case */
			if (len2 == 0)
				return crc1;

			/* put operator for one zero bit in odd */
			odd[0] = 0xedb88320;           /* CRC-32 polynomial */
			row = 1;
			for (n = 1; n < GF2_DIM; n++) {
				odd[n] = row;
				row <<= 1;
			}

			/* put operator for two zero bits in even */
			gf2_matrix_square(even, odd);

			/* put operator for four zero bits in odd */
			gf2_matrix_square(odd, even);

			/* apply len2 zeros to crc1 (first square will put the operator for one
			   zero byte, eight zero bits, in even) */
			do {
				/* apply zeros operator for this bit of len2 */
				gf2_matrix_square(even, odd);
				if ((len2 & 1) != 0)
					crc1 = gf2_matrix_times(even, crc1);
				len2 >>= 1;

				/* if no more bits set, then done */
				if (len2 == 0)
					break;

				/* another iteration of the loop with odd and even swapped */
				gf2_matrix_square(odd, even);
				if ((len2 & 1) != 0)
					crc1 = gf2_matrix_times(odd, crc1);
				len2 >>= 1;

				/* if no more bits set, then done */
			} while (len2 != 0);

			/* return combined crc */
			crc1 ^= crc2;
			return crc1;
		}
	}
}
