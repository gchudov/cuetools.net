using System;

namespace CUETools.Codecs
{
	public static class Crc32
	{
		public static readonly uint[] table;

		public static uint ComputeChecksum(uint crc, byte val)
		{
			return (crc >> 8) ^ table[(crc & 0xff) ^ val];
		}

		public static unsafe uint ComputeChecksum(uint crc, byte* bytes, int count)
		{
			fixed (uint *t = table)
				for (int i = 0; i < count; i++)
					crc = (crc >> 8) ^ t[(crc ^ bytes[i]) & 0xff];
			return crc;
		}

		public static unsafe uint ComputeChecksum(uint crc, byte[] bytes, int pos, int count)
		{
			fixed (byte* pbytes = &bytes[pos])
				return ComputeChecksum(crc, pbytes, count);
		}

		public static uint ComputeChecksum(uint crc, uint s)
		{
			return ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
				crc, (byte)s), (byte)(s >> 8)), (byte)(s >> 16)), (byte)(s >> 24));
		}

		public static unsafe uint ComputeChecksum(uint crc, int* samples, int count)
		{
			for (int i = 0; i < count; i++)
			{
				int s1 = samples[2 * i], s2 = samples[2 * i + 1];
				crc = ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
					crc, (byte)s1), (byte)(s1 >> 8)), (byte)s2), (byte)(s2 >> 8));
			}
			return crc;
		}

		static uint Reflect(uint val, int ch)
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

		private static readonly uint[,] combineTable;
		private static readonly uint[,] substractTable;

#if need_invert_binary_matrix
		static unsafe void invert_binary_matrix(uint* mat, uint* inv, int rows)
		{
			int cols, i, j;
			uint tmp;

			cols = rows;

			for (i = 0; i < rows; i++) inv[i] = (1U << i);

			/* First -- convert into upper triangular */

			for (i = 0; i < cols; i++)
			{

				/* Swap rows if we ave a zero i,i element.  If we can't swap, then the 
				   matrix was not invertible */

				if ((mat[i] & (1 << i)) == 0)
				{
					for (j = i + 1; j < rows && (mat[j] & (1 << i)) == 0; j++) ;
					if (j == rows)
						throw new Exception("Matrix not invertible");
					tmp = mat[i]; mat[i] = mat[j]; mat[j] = tmp;
					tmp = inv[i]; inv[i] = inv[j]; inv[j] = tmp;
				}

				/* Now for each j>i, add A_ji*Ai to Aj */
				for (j = i + 1; j != rows; j++)
				{
					if ((mat[j] & (1 << i)) != 0)
					{
						mat[j] ^= mat[i];
						inv[j] ^= inv[i];
					}
				}
			}

			/* Now the matrix is upper triangular.  Start at the top and multiply down */

			for (i = rows - 1; i >= 0; i--)
			{
				for (j = 0; j < i; j++)
				{
					if ((mat[j] & (1 << i)) != 0)
					{
						/*        mat[j] ^= mat[i]; */
						inv[j] ^= inv[i];
					}
				}
			}
		}
#endif

		static unsafe Crc32()
		{
			table = new uint[256];
			for (uint i = 0; i < table.Length; i++)
			{
				table[i] = Reflect(i, 8) << 24;
				for (int j = 0; j < 8; j++)
					table[i] = (table[i] << 1) ^ ((table[i] & (1U << 31)) == 0 ? 0 : ulPolynomial);
				table[i] = Reflect(table[i], 32);
			}
			combineTable = new uint[GF2_DIM, GF2_DIM];
			substractTable = new uint[GF2_DIM, GF2_DIM];
			combineTable[0, 0] = 0xedb88320;           /* CRC-32 polynomial */
			substractTable[0, 31] = 0xdb710641;
			for (int n = 1; n < GF2_DIM; n++)
			{
				combineTable[0, n] = 1U << (n - 1);
				substractTable[0, n - 1] = 1U << n;
			}
			fixed (uint* ct = &combineTable[0, 0], st = &substractTable[0, 0])
			{
				//for (int i = 0; i < GF2_DIM; i++)
				//	st[32 + i] = ct[i];
				//invert_binary_matrix(st + 32, st, GF2_DIM);

				for (int i = 1; i < GF2_DIM; i++)
				{
					gf2_matrix_square(ct + i * 32, ct + (i - 1) * 32);
					gf2_matrix_square(st + i * 32, st + (i - 1) * 32);
				}
			}
		}

		const int GF2_DIM = 32;
		//const int GF2_DIM2 = 67;

		private static unsafe uint gf2_matrix_times(uint* mat, uint vec)
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
		private static unsafe void gf2_matrix_square(uint *square, uint *mat)
		{
			for (int n = 0; n < GF2_DIM; n++)
				square[n] = gf2_matrix_times(mat, mat[n]);
		}

		public static unsafe uint Combine(uint crc1, uint crc2, long len2)
		{
			/* degenerate case */
			if (len2 == 0)
				return crc1;
			if (crc1 == 0)
				return crc2;
			if (len2 < 0)
				throw new ArgumentException("crc.Combine length cannot be negative", "len2");

			fixed (uint* ct = &combineTable[0, 0])
			{
				int n = 3;
				do
				{
					/* apply zeros operator for this bit of len2 */
					if ((len2 & 1) != 0)
						crc1 = gf2_matrix_times(ct + 32 * n, crc1);
					len2 >>= 1;
					n = (n + 1) & (GF2_DIM - 1);
					/* if no more bits set, then done */
				} while (len2 != 0);
			}

			/* return combined crc */
			crc1 ^= crc2;
			return crc1;
		}

		public static unsafe uint Substract(uint crc1, uint crc2, long len2)
		{
			/* degenerate case */
			if (len2 == 0)
				return crc1;
			if (len2 < 0)
				throw new ArgumentException("crc.Combine length cannot be negative", "len2");

			crc1 ^= crc2;

			fixed (uint* st = &substractTable[0, 0])
			{
				int n = 3;
				do
				{
					/* apply zeros operator for this bit of len2 */
					if ((len2 & 1) != 0)
						crc1 = gf2_matrix_times(st + 32 * n, crc1);
					len2 >>= 1;
					n = (n + 1) & (GF2_DIM - 1);
					/* if no more bits set, then done */
				} while (len2 != 0);
			}

			/* return combined crc */
			return crc1;
		}
	}
}
