using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Parity
{
    public class ParityToSyndrome
    {
        private int[] erasures_pos;
        private int[] erasure_loc_pol;
        private int[] erasure_diff;
        private int npar;
        const int w = 16;
        const int max = (1 << w) - 1;

        public ParityToSyndrome(int npar)
        {
            this.npar = npar;
            this.InitTables();
        }

        private void InitTables()
        {
            var num_erasures = this.npar;

            // Compute the erasure locator polynomial:
            erasures_pos = new int[num_erasures];
            for (int x = 0; x < num_erasures; x++)
                erasures_pos[x] = x;

            //%Compute the erasure-locator polynomial
            // Optimized version
            var erasure_loc_pol_exp = new int[num_erasures + 1];
            erasure_loc_pol_exp[0] = 1;
            for (int i = 0; i < num_erasures; i++)
                for (int x = num_erasures; x > 0; x--)
                    erasure_loc_pol_exp[x] ^= Galois16.instance.mulExp(erasure_loc_pol_exp[x - 1], erasures_pos[i]);
            erasure_loc_pol = Galois16.instance.toLog(erasure_loc_pol_exp);
            erasure_diff = Galois16.instance.gfdiff(erasure_loc_pol);
        }

        public static unsafe byte[] Syndrome2Bytes(ushort[,] inArray)
        {
            int stride = inArray.GetLength(0);
            int npar = inArray.GetLength(1);
            var outBuf = new byte[npar * stride * 2];
            fixed (byte* outPtr = outBuf)
            {
                var ppar = (ushort*)outPtr;
                for (int i = 0; i < npar; i++)
                    for (int j = 0; j < stride; j++)
                        ppar[j + i * stride] = inArray[j, i];
            }
            return outBuf;
        }

        public static unsafe ushort[,] Bytes2Syndrome(int stride, int npar, byte[] parity)
        {
            if (parity.Length < npar * stride * 2)
                 throw new Exception("invalid parity length");
            var syndrome = new ushort[stride, npar];
            fixed (byte* pbpar = parity)
            fixed (ushort* psyn = syndrome)
            {
                var ppar = (ushort*)pbpar;
                for (int i = 0; i < npar; i++)
                    for (int j = 0; j < stride; j++)
                        psyn[i + j * npar] = ppar[j + i * stride];
            }
            return syndrome;
        }

        public static unsafe byte[] Syndrome2Parity(ushort[,] syndrome, byte[] parity = null)
        {
            var stride = syndrome.GetLength(0);
            var npar = syndrome.GetLength(1);
            var converter = new ParityToSyndrome(npar);
            if (parity == null)
                parity = new byte[npar * stride * 2];
            fixed (byte* bpar = parity)
            fixed (ushort* syn = syndrome)
            {
                ushort* par = (ushort*)bpar;
                for (int i = 0; i < stride; i++)
                    converter.Syndrome2Parity(syn + i * npar, par + i * npar);
            }
            return parity;
        }

        public static unsafe ushort[,] Parity2Syndrome(int stride, int stride2, int npar, int npar2, byte[] parity, int pos = 0, int offset = 0)
        {
            if (npar > npar2 || stride > stride2)
                throw new InvalidOperationException();
            var syndrome = new ushort[stride, npar];
            fixed (byte* pbpar = &parity[pos])
            fixed (ushort* psyn = syndrome, plog = Galois16.instance.LogTbl, pexp = Galois16.instance.ExpTbl)
            {
                var ppar = (ushort*)pbpar;
                for (int y = 0; y < stride; y++)
                {
					int y1 = (y - offset + stride2) % stride2;
                    ushort* syn = psyn + y * npar;
                    ushort* par = ppar + y1 * npar2;
                    for (int x1 = 0; x1 < npar2; x1++)
                    {
                        ushort lo = par[x1];
                        if (lo != 0)
                        {
                            var llo = plog[lo] + 0xffff;
                            for (int x = 0; x < npar; x++)
                                syn[x] ^= pexp[llo - (1 + x1) * x];
                        }
                    }
                }
            }
            return syndrome;
        }

        public unsafe void Syndrome2Parity(ushort* syndrome, ushort* parity)
        {
            // Advance syndrome by npar zeroes (for npar 'erased' parity symbols)
            var S_pol = new int[npar + 1];
            S_pol[0] = -1;
            for (int i = 0; i < npar; i++)
            {
                if (syndrome[i] == 0)
                    S_pol[i + 1] = -1;
                else
                {
                    var exp = Galois16.instance.LogTbl[syndrome[i]] + npar * i;
                    S_pol[i + 1] = (exp & max) + (exp >> w);
                }
            }
            var mod_syn = Galois16.instance.gfconv(erasure_loc_pol, S_pol, npar + 1);

            //%Calculate remaining errors (non-erasures)
            //
            //S_M = [];
            //for i = 1:h - num_erasures
            //    S_M(i) = mod_syn(i + num_erasures + 1);
            //end		    
            //flag = 0;
            //if isempty(S_M) == 1
            //    flag = 0;
            //else
            //    for i = 1:length(S_M)
            //        if (S_M(i) ~= -Inf)
            //            flag = 1;     %Other errors occured in conjunction with erasures
            //        end
            //    end
            //end
            //%Find error-location polynomial sigma (Berlekamp's iterative algorithm - 
            //if (flag == 1)
            //{
            // ...
            //}
            //else
            //{
            //    sigma = 0;
            //    comp_error_locs = [];
            //}

#if kjsljdf
			var sigma = new int[1] { 0 };
			var omega = gfconv(sigma, gfadd(mod_syn, 0), npar + 1);
			var tsi = gfconv(sigma, erasure_loc_pol);
			var tsi_diff = gfdiff(tsi);
			//var e_e_places = [erasures_pos comp_error_locs];
#else
            var omega = mod_syn;
            var tsi_diff = erasure_diff;
            var e_e_places = erasures_pos;
#endif

            //%Calculate the error magnitudes
            //%Substitute the inverse into sigma_diff
            //var ERR = new int[e_e_places.Length];
            for (int ii = 0; ii < e_e_places.Length; ii++)
            {
                var point = max - e_e_places[ii];
                var ERR_DEN = Galois16.instance.gfsubstitute(tsi_diff, point, tsi_diff.Length);
                var ERR_NUM = Galois16.instance.gfsubstitute(omega, point, omega.Length);
                // Additional +ii because we use slightly different syndrome
                var pow = ERR_NUM + e_e_places[ii] + ii + max - ERR_DEN;

                //ERR[ii] = ERR_NUM == -1 ? 0 : expTbl[(pow & max) + (pow >> w)];
                parity[npar - 1 - ii] = ERR_NUM == -1 ? (ushort)0 : Galois16.instance.ExpTbl[(pow & max) + (pow >> w)];
            }
        }
    }
}
