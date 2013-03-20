using System;
namespace CUETools.Codecs
{
	public static class Crc16
	{
        const int GF2_DIM = 16;
        public static ushort[] table = new ushort[256];
        private static readonly ushort[,] combineTable = new ushort[GF2_DIM, GF2_DIM];
        private static readonly ushort[,] substractTable = new ushort[GF2_DIM, GF2_DIM];

		public static unsafe ushort ComputeChecksum(ushort crc, byte[] bytes, int pos, int count)
		{
			fixed (byte* bs = bytes)
				return ComputeChecksum(crc, bs + pos, count);
		}

        public static unsafe ushort ComputeChecksum(ushort crc, byte* bytes, int count)
		{
			fixed (ushort* t = table)
				for (int i = count; i > 0; i--)
					crc = (ushort)((crc << 8) ^ t[(crc >> 8) ^ *(bytes++)]);
			return crc;
		}

        const ushort polynomial = 0x8005;
        const ushort reversePolynomial = 0xa001;
        const ushort reversePolynomial2 = 0x4003;

		static unsafe Crc16()
		{
            int poly = (polynomial + (1 << GF2_DIM));
            for (ushort i = 0; i < table.Length; i++)
            {
                int crc = i;
                for (int j = 0; j < GF2_DIM; j++)
                {
                    if ((crc & (1U << (GF2_DIM - 1))) != 0)
                        crc = ((crc << 1) ^ poly);
                    else
                        crc <<= 1;
                }
                table[i] = (ushort)(crc & ((1 << GF2_DIM) - 1));
            }

            combineTable[0, 0] = reversePolynomial;
            substractTable[0, GF2_DIM - 1] = reversePolynomial2;
            for (int n = 1; n < GF2_DIM; n++)
            {
                combineTable[0, n] = (ushort)(1 << (n - 1));
                substractTable[0, n - 1] = (ushort)(1 << n);
            }
            
            fixed (ushort* ct = &combineTable[0, 0], st = &substractTable[0, 0])
            {
                //for (int i = 0; i < GF2_DIM; i++)
                //	st[32 + i] = ct[i];
                //invert_binary_matrix(st + 32, st, GF2_DIM);

                for (int i = 1; i < GF2_DIM; i++)
                {
                    gf2_matrix_square(ct + i * GF2_DIM, ct + (i - 1) * GF2_DIM);
                    gf2_matrix_square(st + i * GF2_DIM, st + (i - 1) * GF2_DIM);
                }
            }
        }

        private static unsafe ushort gf2_matrix_times(ushort* mat, ushort uvec)
        {
            int vec = ((int) uvec) << 16;
            return (ushort)(
                (*(mat++) & ((vec << 15) >> 31)) ^
                (*(mat++) & ((vec << 14) >> 31)) ^
                (*(mat++) & ((vec << 13) >> 31)) ^
                (*(mat++) & ((vec << 12) >> 31)) ^
                (*(mat++) & ((vec << 11) >> 31)) ^
                (*(mat++) & ((vec << 10) >> 31)) ^
                (*(mat++) & ((vec << 09) >> 31)) ^
                (*(mat++) & ((vec << 08) >> 31)) ^
                (*(mat++) & ((vec << 07) >> 31)) ^
                (*(mat++) & ((vec << 06) >> 31)) ^
                (*(mat++) & ((vec << 05) >> 31)) ^
                (*(mat++) & ((vec << 04) >> 31)) ^
                (*(mat++) & ((vec << 03) >> 31)) ^
                (*(mat++) & ((vec << 02) >> 31)) ^
                (*(mat++) & ((vec << 01) >> 31)) ^
                (*(mat++) & (vec >> 31)));
        }

        private static unsafe void gf2_matrix_square(ushort* square, ushort* mat)
        {
            for (int n = 0; n < GF2_DIM; n++)
                square[n] = gf2_matrix_times(mat, mat[n]);
        }

        public static unsafe ushort Combine(ushort crc1, ushort crc2, long len2)
        {
            crc1 = (ushort)Crc32.Reflect(crc1, 16);
            crc2 = (ushort)Crc32.Reflect(crc2, 16);

            /* degenerate case */
            if (len2 == 0)
                return crc1;
            if (crc1 == 0)
                return crc2;
            if (len2 < 0)
                throw new ArgumentException("crc.Combine length cannot be negative", "len2");

            fixed (ushort* ct = &combineTable[0, 0])
            {
                int n = 3;
                do
                {
                    /* apply zeros operator for this bit of len2 */
                    if ((len2 & 1) != 0)
                        crc1 = gf2_matrix_times(ct + GF2_DIM * n, crc1);
                    len2 >>= 1;
                    n = (n + 1) & (GF2_DIM - 1);
                    /* if no more bits set, then done */
                } while (len2 != 0);
            }

            /* return combined crc */
            crc1 ^= crc2;
            crc1 = (ushort)Crc32.Reflect(crc1, 16);
            return crc1;
        }

        public static unsafe ushort Substract(ushort crc1, ushort crc2, long len2)
        {
            crc1 = (ushort)Crc32.Reflect(crc1, 16);
            crc2 = (ushort)Crc32.Reflect(crc2, 16);
            /* degenerate case */
            if (len2 == 0)
                return crc1;
            if (len2 < 0)
                throw new ArgumentException("crc.Combine length cannot be negative", "len2");

            crc1 ^= crc2;

            fixed (ushort* st = &substractTable[0, 0])
            {
                int n = 3;
                do
                {
                    /* apply zeros operator for this bit of len2 */
                    if ((len2 & 1) != 0)
                        crc1 = gf2_matrix_times(st + GF2_DIM * n, crc1);
                    len2 >>= 1;
                    n = (n + 1) & (GF2_DIM - 1);
                    /* if no more bits set, then done */
                } while (len2 != 0);
            }

            /* return combined crc */
            crc1 = (ushort)Crc32.Reflect(crc1, 16);
            return crc1;
        }
    }
}
