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
	class lpc
	{
		public const int MAX_LPC_ORDER = 32;
		public const int MAX_LPC_WINDOWS = 4;

		/**
		 * Apply Welch window function to audio block
		 */
		static unsafe void
			apply_welch_window(/*const*/int* data, uint len, double* w_data)
		{
			double c = (2.0 / (len - 1.0)) - 1.0;
			for (uint i = 0; i < (len >> 1); i++)
			{
				double w = 1.0 - ((c - i) * (c - i));
				w_data[i] = data[i] * w;
				w_data[len - 1 - i] = data[len - 1 - i] * w;
			}
		}

		/**
		 * Calculates autocorrelation data from audio samples
		 * A Welch window function is applied before calculation.
		 */
		static public unsafe void
			compute_autocorr(/*const*/ int* data, uint len, uint lag, double* autoc, double* window)
		{
			double* data1 = stackalloc double[(int)len + 16];
			uint i, j;
			double temp, temp2;

			if (window == null)
				apply_welch_window(data, len, data1);
			else
			{
				for (i = 0; i < len; i++)
					data1[i] = data[i] * window[i];
			}
			data1[len] = 0;

			for (i = 0; i <= lag; ++i)
			{
				temp = 1.0;
				temp2 = 1.0;
				for (j = 0; j <= lag - i; ++j)
					temp += data1[j + i] * data1[j];

				double* finish = data1 + len - i;
				for (double* pdata = data1 + lag + 1 - i; pdata < finish; pdata += 2)
				{
					temp += pdata[i] * pdata[0];
					temp2 += pdata[i + 1] * pdata[1];
				}
				autoc[i] = temp + temp2;
			}

			//int sample, coeff;
			//for (coeff = 0; coeff <= lag; coeff++)
			//    autoc[coeff] = 0.0;
			//int data_len = (int)len;
			//int limit = data_len - (int)lag - 1;
			//for (sample = 0; sample <= limit; sample++)
			//{
			//    double d = data1[sample];
			//    for (coeff = 0; coeff <= lag; coeff++)
			//        autoc[coeff] += d * data1[sample + coeff];
			//}
			//for (; sample < data_len; sample++)
			//{
			//    double d = data1[sample];
			//    for (coeff = 0; coeff < data_len - sample; coeff++)
			//        autoc[coeff] += d * data1[sample + coeff];
			//}
		}

		/**
		 * Levinson-Durbin recursion.
		 * Produces LPC coefficients from autocorrelation data.
		 */
		public static unsafe void
		compute_lpc_coefs(/*const*/ double* autoc, uint max_order, double* reff,
						  double* lpc/*[][MAX_LPC_ORDER]*/)
		{
			double* lpc_tmp = stackalloc double[MAX_LPC_ORDER];

			int i, j, i2;
			double r, err, tmp;

			if (max_order > MAX_LPC_ORDER)
				throw new Exception("wierd");

			for (i = 0; i < max_order; i++)
				lpc_tmp[i] = 0;

			err = 1.0;
			if (autoc != null)
				err = autoc[0];

			for (i = 0; i < max_order; i++)
			{
				if (reff != null)
				{
					r = reff[i];
				}
				else
				{
					r = -autoc[i + 1];
					for (j = 0; j < i; j++)
					{
						r -= lpc_tmp[j] * autoc[i - j];
					}
					r /= err;
					err *= 1.0 - (r * r);
				}

				i2 = (i >> 1);
				lpc_tmp[i] = r;
				for (j = 0; j < i2; j++)
				{
					tmp = lpc_tmp[j];
					lpc_tmp[j] += r * lpc_tmp[i - 1 - j];
					lpc_tmp[i - 1 - j] += r * tmp;
				}
				if (0 != (i & 1))
				{
					lpc_tmp[j] += lpc_tmp[j] * r;
				}

				for (j = 0; j <= i; j++)
				{
					lpc[i * MAX_LPC_ORDER + j] = -lpc_tmp[j];
				}
			}
		}

		public static unsafe void
		compute_schur_reflection(/*const*/ double* autoc, uint max_order,
							  double* reff/*[][MAX_LPC_ORDER]*/)
		{
			double* gen0 = stackalloc double[MAX_LPC_ORDER];
			double* gen1 = stackalloc double[MAX_LPC_ORDER];

			// Schur recursion
			for (uint i = 0; i < max_order; i++)
				gen0[i] = gen1[i] = autoc[i + 1];

			double error = autoc[0];
			reff[0] = -gen1[0] / error;
			error += gen1[0] * reff[0];
			for (uint i = 1; i < max_order; i++)
			{
				for (uint j = 0; j < max_order - i; j++)
				{
					gen1[j] = gen1[j + 1] + reff[i - 1] * gen0[j];
					gen0[j] = gen1[j + 1] * reff[i - 1] + gen0[j];
				}
				reff[i] = -gen1[0] / error;
				error += gen1[0] * reff[i];
			}
		}

		/**
		 * Compute LPC coefs for Flake.OrderMethod._EST
		 * Faster LPC coeff computation by first calculating the reflection coefficients
		 * using Schur recursion. That allows for estimating the optimal order before
		 * running Levinson recursion.
		 */
		public static unsafe uint
		compute_lpc_coefs_est(/*const*/ double* autoc, uint max_order,
							  double* lpc/*[][MAX_LPC_ORDER]*/)
		{
			double* reff = stackalloc double[MAX_LPC_ORDER];

			// Schur recursion
			compute_schur_reflection(autoc, max_order, reff);

			// Estimate optimal order using reflection coefficients
			uint order_est = 1;
			for (int i = (int)max_order - 1; i >= 0; i--)
			{
				if (Math.Abs(reff[i]) > 0.10)
				{
					order_est = (uint)i + 1;
					break;
				}
			}

			// Levinson recursion
			compute_lpc_coefs(null, order_est, reff, lpc);
			return order_est;
		}

		/**
		 * Quantize LPC coefficients
		 */
		public static unsafe void
		quantize_lpc_coefs(double* lpc_in, int order, uint precision, int* lpc_out,
						   out int shift)
		{
			int i;
			double d, cmax, error;
			int qmax;
			int sh, q;

			// define maximum levels
			qmax = (1 << ((int)precision - 1)) - 1;

			// find maximum coefficient value
			cmax = 0.0;
			for (i = 0; i < order; i++)
			{
				d = Math.Abs(lpc_in[i]);
				if (d > cmax)
					cmax = d;
			}
			// if maximum value quantizes to zero, return all zeros
			if (cmax * (1 << 15) < 1.0)
			{
				shift = 0;
				for (i = 0; i < order; i++)
					lpc_out[i] = 0;
				return;
			}

			// calculate level shift which scales max coeff to available bits
			sh = 15;
			while ((cmax * (1 << sh) > qmax) && (sh > 0))
			{
				sh--;
			}

			// since negative shift values are unsupported in decoder, scale down
			// coefficients instead
			if (sh == 0 && cmax > qmax)
			{
				double scale = ((double)qmax) / cmax;
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
				if (q <= -qmax) q = -qmax + 1;
				if (q > qmax) q = qmax;
				error -= q;
				lpc_out[i] = q;
			}
			shift = sh;
		}

		/**
		 * Calculate LPC coefficients for multiple orders
		 */
		public static unsafe uint
		calc_coefs(/*const*/ int* samples, uint blocksize, uint max_order, OrderMethod omethod, double* lpcs, double* window)
		{
			double* autoc = stackalloc double[MAX_LPC_ORDER + 1];

			compute_autocorr(samples, blocksize, max_order + 1, autoc, window);

			uint opt_order = max_order;
			if (omethod == OrderMethod.Estimate || omethod == OrderMethod.EstSearch)
				opt_order = compute_lpc_coefs_est(autoc, max_order, lpcs);
			else
				compute_lpc_coefs(autoc, max_order, null, lpcs);

			return opt_order;
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
						int pred = coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[6] * *(s++);
						pred += coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[7] * *(s++);
						pred += coefs[6] * *(s++);
						pred += coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*(r++) = *s - (pred >> shift);
						s -= 7;
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
						int pred = c1 * *(s++);
						pred += c0 * *(s++);
						*(s--) = *(r++) + (pred >> shift);
					}
					break;
				case 3:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 2;
					}
					break;
				case 4:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 3;
					}
					break;
				case 5:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 4;
					}
					break;
				case 6:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 5;
					}
					break;
				case 7:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[6] * *(s++);
						pred += coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 6;
					}
					break;
				case 8:
					for (int i = n - order; i > 0; i--)
					{
						int pred = coefs[7] * *(s++);
						pred += coefs[6] * *(s++);
						pred += coefs[5] * *(s++);
						pred += coefs[4] * *(s++);
						pred += coefs[3] * *(s++);
						pred += coefs[2] * *(s++);
						pred += c1 * *(s++);
						pred += c0 * *(s++);
						*s = *(r++) + (pred >> shift);
						s -= 7;
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
