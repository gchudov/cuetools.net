using System;

namespace CUETools.Codecs
{
	public class lpc
	{
		public const int MAX_LPC_ORDER = 32;
		public const int MAX_LPC_WINDOWS = 8;
		public const int MAX_LPC_PRECISIONS = 4;
        public const int MAX_LPC_SECTIONS = 32;

		public unsafe static void window_welch(float* window, int L)
		{
			int N = L - 1;
			double N2 = (double)N / 2.0;

			for (int n = 0; n <= N; n++)
			{
				double k = ((double)n - N2) / N2;
				k = 1.0 - k * k;
				window[n] = (float)(k);
			}
		}

        public unsafe static void window_bartlett(float* window, int L)
		{
			int N = L - 1;
			double N2 = (double)N / 2.0;
			for (int n = 0; n <= N; n++)
			{
				double k = ((double)n - N2) / N2;
				k = 1.0 - k * k;
				window[n] = (float)(k * k);
			}
		}

        public unsafe static void window_rectangle(float* window, int L)
		{
			for (int n = 0; n < L; n++)
				window[n] = 1.0F;
		}

        public unsafe static void window_flattop(float* window, int L)
		{
			int N = L - 1;
			for (int n = 0; n < L; n++)
				window[n] = (float)(1.0 - 1.93 * Math.Cos(2.0 * Math.PI * n / N) + 1.29 * Math.Cos(4.0 * Math.PI * n / N) - 0.388 * Math.Cos(6.0 * Math.PI * n / N) + 0.0322 * Math.Cos(8.0 * Math.PI * n / N));
		}

        public unsafe static void window_tukey(float* window, int L)
		{
			double p = 0.5;
            int z = 0;
            int Np = (int)(p / 2.0 * L) - z;
			if (Np > 0)
			{
                for (int n = 0; n < z; n++)
                    window[n] = window[L - n - 1] = 0;
                for (int n = 0; n < Np - 1; n++)
                    window[n + z] = window[L - n - 1 - z] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * (n + 1) / Np));
                for (int n = z + Np - 1; n < L - z - Np + 1; n++)
                    window[n] = 1.0F;
			}
		}

        public unsafe static void window_punchout_tukey(float* window, int L, double p, double start, double end)
        {
            int start_n = (int)(start * L);
            int end_n = (int)(end * L);
            int Np = (int)(p / 2.0 * L);
            int i, n = 0;

            if (start_n != 0)
            {
                for (i = 1; n < Np; n++, i++)
                    window[n] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / Np));
                for (; n < start_n - Np; n++)
                    window[n] = 1.0f;
                for (i = Np; n < start_n; n++, i--)
                    window[n] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / Np));
            }
            for (; n < end_n; n++)
		        window[n] = 0.0f;
            if (end_n != L)
            {
                for (i = 1; n < end_n + Np; n++, i++)
                    window[n] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / Np));
                for (; n < L - Np; n++)
                    window[n] = 1.0f;
                for (i = Np; n < L; n++, i--)
                    window[n] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / Np));
            }
        }

        public unsafe static void window_hann(float* window, int L)
		{
			int N = L - 1;
			for (int n = 0; n < L; n++)
				window[n] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / N));
		}

		private static short sign_only(int val)
		{
			return (short)((val >> 31) + ((val - 1) >> 31) + 1);
		}

#if XXX
		static public unsafe void
			compute_corr_int(/*const*/ short* data1, short* data2, int len, int min, int lag, int* autoc)
		{
			for (int i = min; i <= lag; ++i)
			{
				int temp = 0;
				int temp2 = 0;

				for (int j = 0; j <= lag - i; ++j)
					temp += data1[j + i] * data2[j];

				for (int j = lag + 1 - i; j < len - i; j += 2)
				{
					temp += data1[j + i] * data2[j];
					temp2 += data1[j + i + 1] * data2[j + 1];
				}
				autoc[i] = temp + temp2;
			}
		}
#endif

		/**
		 * Calculates autocorrelation data from audio samples
		 * A window function is applied before calculation.
		 */
		static public unsafe void
            compute_autocorr(/*const*/ int* data, float* window, int len, int min, int lag, double* autoc, int prev, int next)
		{
#if FPAC
			short* data1 = stackalloc short[len + 1];
			short* data2 = stackalloc short[len + 1];
			int* c1 = stackalloc int[lpc.MAX_LPC_ORDER + 1];
			int* c2 = stackalloc int[lpc.MAX_LPC_ORDER + 1];
			int* c3 = stackalloc int[lpc.MAX_LPC_ORDER + 1];
			int* c4 = stackalloc int[lpc.MAX_LPC_ORDER + 1];

			for (int i = 0; i < len; i++)
			{
				int val = (int)(data[i] * window[i]);
				data1[i] = (short)(sign_only(val) * (Math.Abs(val) >> 9));
				data2[i] = (short)(sign_only(val) * (Math.Abs(val) & 0x1ff));
			}
			data1[len] = 0;
			data2[len] = 0;

			compute_corr_int(data1, data1, len, min, lag, c1);
			compute_corr_int(data1, data2, len, min, lag, c2);
			compute_corr_int(data2, data1, len, min, lag, c3);
			compute_corr_int(data2, data2, len, min, lag, c4);
			
			for (int coeff = min; coeff <= lag; coeff++)
			    autoc[coeff] = (c1[coeff] * (double)(1 << 18) + (c2[coeff] + c3[coeff]) * (double)(1 << 9) + c4[coeff]);
#else
#if XXX
            if (min == 0 && lag >= 4)
            {
                int* pdata = data;
                float* pwindow = window;

                double temp0 = 1.0;
                double temp1 = 1.0;
                double temp2 = 1.0;
                double temp3 = 1.0;
                double temp4 = 1.0;

                double c0 = *(pdata++) * *(pwindow++);
                float c1 = *(pdata++) * *(pwindow++);
                float c2 = *(pdata++) * *(pwindow++);
                float c3 = *(pdata++) * *(pwindow++);
                float c4 = *(pdata++) * *(pwindow++);

                int* finish = data + len;

                while (pdata <= finish)
                {
                    temp0 += c0 * c0;
                    temp1 += c0 * c1;
                    temp2 += c0 * c2;
                    temp3 += c0 * c3;
                    temp4 += c0 * c4;

                    c0 = c1;
                    c1 = c2;
                    c2 = c3;
                    c3 = c4;
                    c4 = *(pdata++) * *(pwindow++);
                }

                temp0 += c0 * c0;
                temp1 += c0 * c1;
                temp2 += c0 * c2;
                temp3 += c0 * c3;
                c0 = c1;
                c1 = c2;
                c2 = c3;
                temp0 += c0 * c0;
                temp1 += c0 * c1;
                temp2 += c0 * c2;
                c0 = c1;
                c1 = c2;
                temp0 += c0 * c0;
                temp1 += c0 * c1;
                c0 = c1;
                temp0 += c0 * c0;
                
                autoc[0] += temp0;
                autoc[1] += temp1;
                autoc[2] += temp2;
                autoc[3] += temp3;
                autoc[4] += temp4;
                min = 5;

                if (lag < min) return;
            }
#endif
            double* data1 = stackalloc double[lag + len + lag];
            int i;

            for (i = 0; i < lag; i++)
                data1[i] = prev != 0 ? data[i - lag] : 0;
            for (i = 0; i < len; i++)
                data1[lag + i] = data[i] * window[i];
            for (i = 0; i < lag; i++)
                data1[lag + len + i] = next != 0 ? data[len + i] : 0;

            for (i = min; i <= lag; ++i)
			{
				double temp = 0;
				double temp2 = 0;
                double* pdata = data1 + lag - i;
				double* finish = data1 + lag + len - 1;

                while (pdata < finish)
                {
                    temp += pdata[i] * (*pdata++);
                    temp2 += pdata[i] * (*pdata++);
                }
                if (pdata <= finish)
                    temp += pdata[i] * (*pdata++);

                autoc[i] += temp + temp2;
			}
#endif
		}

        static public unsafe void
            compute_autocorr_windowless(/*const*/ int* data, int len, int min, int lag, double* autoc)
        {
            // if databits*2 + log2(len) <= 64
#if !XXX
#if XXX
            if (min == 0 && lag >= 4)
            {
                long temp0 = 0;
                long temp1 = 0;
                long temp2 = 0;
                long temp3 = 0;
                long temp4 = 0;
                int* pdata = data;
                int* finish = data + len - 4;
                while (pdata < finish)
                {
                    long c0 = *(pdata++);
                    temp0 += c0 * c0;
                    temp1 += c0 * pdata[0];
                    temp2 += c0 * pdata[1];
                    temp3 += c0 * pdata[2];
                    temp4 += c0 * pdata[3];
                }
                {
                    long c0 = *(pdata++);
                    temp0 += c0 * c0;
                    temp1 += c0 * pdata[0];
                    temp2 += c0 * pdata[1];
                    temp3 += c0 * pdata[2];
                }
                {
                    long c0 = *(pdata++);
                    temp0 += c0 * c0;
                    temp1 += c0 * pdata[0];
                    temp2 += c0 * pdata[1];
                }
                {
                    long c0 = *(pdata++);
                    temp0 += c0 * c0;
                    temp1 += c0 * pdata[0];
                }
                {
                    long c0 = *(pdata++);
                    temp0 += c0 * c0;
                }
                autoc[0] += temp0;
                autoc[1] += temp1;
                autoc[2] += temp2;
                autoc[3] += temp3;
                autoc[4] += temp4;
                min = 5;

                if (lag < min) return;
            }
#endif
            for (int i = min; i <= lag; ++i)
            {
                long temp = 0;
                long temp2 = 0;
                int* pdata = data;
                int* finish = data + len - i - 1;
                while (pdata < finish)
                {
                    temp += (long)pdata[i] * (*pdata++);
                    temp2 += (long)pdata[i] * (*pdata++);
                }
                if (pdata <= finish)
                    temp += (long)pdata[i] * (*pdata++);
                autoc[i] += temp + temp2;
            }
#else
            for (int i = min; i <= lag; ++i)
            {
                double temp = 0;
                double temp2 = 0;
                int* pdata = data;
                int* finish = data + len - i - 1;

                while (pdata < finish)
                {
                    temp += (double)pdata[i] * (double)(*pdata++);
                    temp2 += (double)pdata[i] * (double)(*pdata++);
                }
                if (pdata <= finish)
                    temp += (double)pdata[i] * (double)(*pdata++);
                autoc[i] += temp + temp2;
            }
#endif
        }

        static public unsafe void
            compute_autocorr_windowless_large(/*const*/ int* data, int len, int min, int lag, double* autoc)
        {
            for (int i = min; i <= lag; ++i)
            {
                double temp = 0;
                double temp2 = 0;
                int* pdata = data;
                int* finish = data + len - i - 1;
                while (pdata < finish)
                {
                    temp += (long)pdata[i] * (*pdata++);
                    temp2 += (long)pdata[i] * (*pdata++);
                }
                if (pdata <= finish)
                    temp += (long)pdata[i] * (*pdata++);
                autoc[i] += temp + temp2;
            }
        }

        static public unsafe void
            compute_autocorr_glue(/*const*/ int* data, int min, int lag, double* autoc)
        {
            for (int i = min; i <= lag; ++i)
            {
                long temp = 0;
                int* pdata = data - i;
                int* finish = data;
                while (pdata < finish)
                    temp += (long)pdata[i] * (*pdata++);
                autoc[i] += temp;
            }
        }

        /**
		 * Levinson-Durbin recursion.
		 * Produces LPC coefficients from autocorrelation data.
		 */
		public static unsafe void
		compute_lpc_coefs(uint max_order, double* reff, float* lpc/*[][MAX_LPC_ORDER]*/)
		{
			double* lpc_tmp = stackalloc double[MAX_LPC_ORDER];

			if (max_order > MAX_LPC_ORDER)
				throw new Exception("wierd");

			for (int i = 0; i < max_order; i++)
				lpc_tmp[i] = 0;

			for (int i = 0; i < max_order; i++)
			{
				double r = reff[i];
				int i2 = (i >> 1);
				lpc_tmp[i] = r;
				for (int j = 0; j < i2; j++)
				{
					double tmp = lpc_tmp[j];
					lpc_tmp[j] += r * lpc_tmp[i - 1 - j];
					lpc_tmp[i - 1 - j] += r * tmp;
				}

				if (0 != (i & 1))
					lpc_tmp[i2] += lpc_tmp[i2] * r;

				for (int j = 0; j <= i; j++)
					lpc[i * MAX_LPC_ORDER + j] = (float)-lpc_tmp[j];
			}
		}

		public static unsafe void
		compute_schur_reflection(/*const*/ double* autoc, uint max_order,
							  double* reff/*[][MAX_LPC_ORDER]*/, double * err)
		{
			double* gen0 = stackalloc double[MAX_LPC_ORDER];
			double* gen1 = stackalloc double[MAX_LPC_ORDER];

			// Schur recursion
			for (uint i = 0; i < max_order; i++)
				gen0[i] = gen1[i] = autoc[i + 1];

			double error = autoc[0];
			reff[0] = -gen1[0] / error;
			error += gen1[0] * reff[0];
			err[0] = error;
			for (uint i = 1; i < max_order; i++)
			{
				for (uint j = 0; j < max_order - i; j++)
				{
					gen1[j] = gen1[j + 1] + reff[i - 1] * gen0[j];
					gen0[j] = gen1[j + 1] * reff[i - 1] + gen0[j];
				}
				reff[i] = -gen1[0] / error;
				error += gen1[0] * reff[i];
				err[i] = error;
			}
		}

		/**
		 * Quantize LPC coefficients
		 */
		public static unsafe void
		quantize_lpc_coefs(float* lpc_in, int order, uint precision, int* lpc_out,
						   out int shift, int max_shift, int zero_shift)
		{
			int i;
			float d, cmax, error;
			int qmax;
			int sh, q;

			// define maximum levels
			qmax = (1 << ((int)precision - 1)) - 1;

			// find maximum coefficient value
			cmax = 0.0F;
			for (i = 0; i < order; i++)
			{
				d = Math.Abs(lpc_in[i]);
				if (d > cmax)
					cmax = d;
			}
			// if maximum value quantizes to zero, return all zeros
			if (cmax * (1 << max_shift) < 1.0)
			{
				shift = zero_shift;
				for (i = 0; i < order; i++)
					lpc_out[i] = 0;
				return;
			}

			// calculate level shift which scales max coeff to available bits
			sh = max_shift;
			while ((cmax * (1 << sh) > qmax) && (sh > 0))
			{
				sh--;
			}

			// since negative shift values are unsupported in decoder, scale down
			// coefficients instead
			if (sh == 0 && cmax > qmax)
			{
				float scale = ((float)qmax) / cmax;
				for (i = 0; i < order; i++)
				{
					lpc_in[i] *= scale;
				}
			}

			// output quantized coefficients and level shift
			error = 0;
			for (i = 0; i < order; i++)
			{
				error += lpc_in[i] * (1 << sh);
				q = (int)(error + 0.5);
				if (q < -(qmax+1)) q = -(qmax + 1);
				if (q > qmax) q = qmax;
				error -= q;
				lpc_out[i] = q;
			}
			shift = sh;
		}

		public static unsafe void
		encode_residual(int* res, int* smp, int n, int order,
			int* coefs, int shift)
		{
			for (int i = 0; i < order; i++)
				res[i] = smp[i];

			int* s = smp;
			int* r = res + order;
			int c0 = coefs[0];
			int c1 = coefs[1];
			switch (order)
			{
				case 1:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c0 * *(s++);
						*(r++) = *s - (pred >> shift);
					}
					break;
				case 2:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *(s--) - (pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[2] * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 7;
					}
					break;
				case 9:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + 
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 8;
					}
					break;
				case 10:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 9;
					}
					break;
				case 11:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 10;
					}
					break;
				case 12:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 11;
					}
					break;
				default:
					for (int i = order; i < n; i++)
					{
						s = smp + i - order;
						int pred = 0;
						int* c = coefs + order - 1;
						int* c11 = coefs + 11;
						while (c > c11)
							pred += *(c--) * *(s++);
						pred +=
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
					}
					break;
			}
		}

		public static unsafe void
		encode_residual_long(int* res, int* smp, int n, int order,
			int* coefs, int shift)
		{
			for (int i = 0; i < order; i++)
				res[i] = smp[i];

			int* s = smp;
			int* r = res + order;
			int c0 = coefs[0];
			int c1 = coefs[1];
			switch (order)
			{
				case 1:
					for (int i = n - order; i > 0; i--)
					{
						long pred = c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
					}
					break;
				case 2:
					for (int i = n - order; i > 0; i--)
					{
						long pred = c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *(s--) - (int)(pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[7] * (long)*(s++);
						pred += coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
						s -= 7;
					}
					break;
				default:
					for (int i = order; i < n; i++)
					{
						s = smp + i - order;
						long pred = 0;
						int* co = coefs + order - 1;
						int* c7 = coefs + 7;
						while (co > c7)
							pred += *(co--) * (long)*(s++);
						pred += coefs[7] * (long)*(s++);
						pred += coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(r++) = *s - (int)(pred >> shift);
					}
					break;
			}
		}

		public static unsafe void
		encode_residual2(int* res, int* smp, int n, int order,
			int* coefs, int shift)
		{
			int* s = smp;
			int* r = res;
			int c0 = coefs[0];
			int c1 = coefs[1];
			switch (order)
			{
				case 1:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c0 * *(s++);
						*(r++) = *s - (pred >> shift);
					}
					break;
				case 2:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *(s--) - (pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[2] * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 7;
					}
					break;
				case 9:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 8;
					}
					break;
				case 10:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 9;
					}
					break;
				case 11:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 10;
					}
					break;
				case 12:
					for (int i = n - order; i > 0; i--)
					{
						int* c = coefs + order - 1;
						int pred =
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 11;
					}
					break;
				default:
					for (int i = order; i < n; i++)
					{
						s = smp + i - order;
						int pred = 0;
						int* c = coefs + order - 1;
						int* c11 = coefs + 11;
						while (c > c11)
							pred += *(c--) * *(s++);
						pred +=
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							*(c--) * *(s++) + *(c--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*(r++) = *s - (pred >> shift);
					}
					break;
			}
		}

		public static unsafe void
		decode_residual(int* res, int* smp, int n, int order,
			int* coefs, int shift)
		{
			for (int i = 0; i < order; i++)
				smp[i] = res[i];

			int* s = smp;
			int* r = res + order;
			int c0 = coefs[0];
			int c1 = coefs[1];
			switch (order)
			{
				case 1:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c0 * *(s++);
						*s = *(r++) + (pred >> shift);
					}
					break;
				case 2:
					for (int i = n - order; i > 0; i--)
					{
						int pred = c1 * *(s++) + c0 * *(s++);
						*(s--) = *(r++) + (pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 7;
					}
					break;
				case 9:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 8;
					}
					break;
				case 10:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 9;
					}
					break;
				case 11:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred =
							*(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 10;
					}
					break;
				case 12:
					for (int i = n - order; i > 0; i--)
					{
						int* co = coefs + order - 1;
						int pred = 
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							*(co--) * *(s++) + *(co--) * *(s++) +
							c1 * *(s++) + c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 11;
					}
					break;
				default:
					for (int i = order; i < n; i++)
					{
						s = smp + i - order;
						int pred = 0;
						int* co = coefs + order - 1;
						int* c7 = coefs + 7;
						while (co > c7)
							pred += *(co--) * *(s++);
						pred += coefs[7] * *(s++);
						pred += coefs[6] * *(s++);
						pred += coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
					}
					break;
			}
		}
		public static unsafe void
		decode_residual_long(int* res, int* smp, int n, int order,
			int* coefs, int shift)
		{
			for (int i = 0; i < order; i++)
				smp[i] = res[i];

			int* s = smp;
			int* r = res + order;
			int c0 = coefs[0];
			int c1 = coefs[1];
			switch (order)
			{
				case 1:
					for (int i = n - order; i > 0; i--)
					{
						long pred = c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
					}
					break;
				case 2:
					for (int i = n - order; i > 0; i--)
					{
						long pred = c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*(s--) = *(r++) + (int)(pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						long pred = coefs[7] * (long)*(s++);
						pred += coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
						s -= 7;
					}
					break;
				default:
					for (int i = order; i < n; i++)
					{
						s = smp + i - order;
						long pred = 0;
						int* co = coefs + order - 1;
						int* c7 = coefs + 7;
						while (co > c7)
							pred += *(co--) * (long)*(s++);
						pred += coefs[7] * (long)*(s++);
						pred += coefs[6] * (long)*(s++);
						pred += coefs[5] * (long)*(s++);
						pred += coefs[4] * (long)*(s++);
						pred += coefs[3] * (long)*(s++);
						pred += coefs[2] * (long)*(s++);
						pred += c1 * (long)*(s++);
						pred += c0 * (long)*(s++);
						*s = *(r++) + (int)(pred >> shift);
					}
					break;
			}
		}
	}
}
