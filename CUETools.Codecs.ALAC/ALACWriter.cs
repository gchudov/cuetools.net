/**
 * CUETools.Codecs.ALAC: pure managed ALAC audio encoder
 * Copyright (c) 2009 Gregory S. Chudov
 * Based on ffdshow ALAC audio encoder
 * Copyright (c) 2008  Jaikrishnan Menon, realityman@gmx.net
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
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;
//using System.Runtime.InteropServices;
using CUETools.Codecs;

namespace CUETools.Codecs.ALAC
{
	public class ALACWriter : IAudioDest
	{
		Stream _IO = null;
		string _path;
		long _position;

		// number of audio channels
		// valid values are 1 to 8
		int channels, ch_code;

		// audio sample rate in Hz
		int sample_rate;

		// sample size in bits
		// only 16-bit is currently supported
		uint bits_per_sample;

		// total stream samples
		// if 0, stream length is unknown
		int sample_count;

		ALACEncodeParams eparams;

		// maximum frame size in bytes
		// this can be used to allocate memory for output
		int max_frame_size;

		int initial_history, history_mult, k_modifier;

		byte[] frame_buffer = null;

		int frame_count = 0;

		long first_frame_offset = 0;

		TimeSpan _userProcessorTime;

		// header bytes
		byte[] header;

		uint[] _sample_byte_size;
		int[] samplesBuffer;
		int[] verifyBuffer;
		int[] residualBuffer;
		double[] windowBuffer;
		int samplesInBuffer = 0;

		int _compressionLevel = 4;
		int _blocksize = 0;
		int _totalSize = 0;
		int _windowsize = 0, _windowcount = 0;

		Crc8 crc8;
		Crc16 crc16;
		ALACFrame frame;
		ALACReader verify;

		int mdat_pos;

		bool inited = false;

		List<int> chunk_pos;

		public ALACWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO)
		{
			if (bitsPerSample != 16)
				throw new Exception("Bits per sample must be 16.");
			if (channelCount != 2)
				throw new Exception("ChannelCount must be 2.");

			channels = channelCount;
			sample_rate = sampleRate;
			bits_per_sample = (uint) bitsPerSample;

			_path = path;
			_IO = IO;

			samplesBuffer = new int[Alac.MAX_BLOCKSIZE * (channels == 2 ? 5 : channels)];
			residualBuffer = new int[Alac.MAX_BLOCKSIZE * (channels == 2 ? 6 : channels + 1)];
			windowBuffer = new double[Alac.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS];

			eparams.set_defaults(_compressionLevel);
			eparams.padding_size = 8192;

			crc8 = new Crc8();
			crc16 = new Crc16();
			frame = new ALACFrame(channels == 2 ? 5 : channels);
			chunk_pos = new List<int>();
		}

		public int TotalSize
		{
			get
			{
				return _totalSize;
			}
		}

		public int PaddingLength
		{
			get
			{
				return eparams.padding_size;
			}
			set
			{
				eparams.padding_size = value;
			}
		}

		public int CompressionLevel
		{
			get
			{
				return _compressionLevel;
			}
			set
			{
				if (value < 0 || value > 11)
					throw new Exception("unsupported compression level");
				_compressionLevel = value;
				eparams.set_defaults(_compressionLevel);
			}
		}

		//[DllImport("kernel32.dll")]
		//static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
		//[DllImport("kernel32.dll")]
		//static extern IntPtr GetCurrentThread();

		void chunk_start(BitWriter bitwriter)
		{
			bitwriter.flush();
			chunk_pos.Add(bitwriter.Length);
			bitwriter.writebits(32, 0); // length placeholder
		}

		void chunk_end(BitWriter bitwriter)
		{
			bitwriter.flush();
			int pos = chunk_pos[chunk_pos.Count - 1];
			chunk_pos.RemoveAt(chunk_pos.Count - 1);
			int chunk_end = bitwriter.Length;
			bitwriter.Length = pos;
			bitwriter.writebits(32, chunk_end - pos);
			bitwriter.Length = chunk_end;
		}

		void DoClose()
		{
			if (inited)
			{
				while (samplesInBuffer > 0)
					output_frame(samplesInBuffer);

				if (_IO.CanSeek)
				{
					int mdat_len = (int)_IO.Position - mdat_pos;
					_IO.Position = mdat_pos;
					BitWriter bitwriter = new BitWriter(header, 0, 4);
					bitwriter.writebits(32, mdat_len);
					bitwriter.flush();
					_IO.Write(header, 0, 4);

					_IO.Position = _IO.Length;
					int trailer_len = write_trailers();
					_IO.Write(header, 0, trailer_len);
				}
				_IO.Close();
				inited = false;
			}

			//long fake, KernelStart, UserStart;
			//GetThreadTimes(GetCurrentThread(), out fake, out fake, out KernelStart, out UserStart);
			//_userProcessorTime = new TimeSpan(UserStart);
		}

		public void Close()
		{
			DoClose();
			if (sample_count != 0 && _position != sample_count)
				throw new Exception("Samples written differs from the expected sample count.");
		}

		public void Delete()
		{
			if (inited)
			{
				_IO.Close();
				inited = false;
			}

			if (_path != "")
				File.Delete(_path);
		}

		public long Position
		{
			get
			{
				return _position;
			}
		}

		public long FinalSampleCount
		{
			set { sample_count = (int)value; }
		}

		public long BlockSize
		{
			set { _blocksize = (int)value; }
			get { return _blocksize == 0 ? eparams.block_size : _blocksize; }
		}

		public OrderMethod OrderMethod
		{
			get { return eparams.order_method; }
			set { eparams.order_method = value; }
		}

		public StereoMethod StereoMethod
		{
			get { return eparams.stereo_method; }
			set { eparams.stereo_method = value; }
		}

		public WindowFunction WindowFunction
		{
			get { return eparams.window_function; }
			set { eparams.window_function = value; }
		}

		public bool DoVerify
		{
			get { return eparams.do_verify; }
			set { eparams.do_verify = value; }
		}

		public bool DoSeekTable
		{
			get { return eparams.do_seektable; }
			set { eparams.do_seektable = value; }
		}

		public int MinLPCOrder
		{
			get
			{
				return eparams.min_prediction_order;
			}
			set
			{
				if (value < 1)
					throw new Exception("invalid MinLPCOrder " + value.ToString());
				eparams.min_prediction_order = value;
				if (eparams.max_prediction_order < value)
					eparams.max_prediction_order = value;
			}
		}

		public int MaxLPCOrder
		{
			get
			{
				return eparams.max_prediction_order;
			}
			set
			{
				if (value > 30 || value < eparams.min_prediction_order)
					throw new Exception("invalid MaxLPCOrder " + value.ToString());
				eparams.max_prediction_order = value;
				if (eparams.min_prediction_order > value)
					eparams.min_prediction_order = value;
			}
		}

		public int MinHistoryModifier
		{
			get
			{
				return eparams.min_modifier;
			}
			set
			{
				if (value < 1)
					throw new Exception("invalid MinHistoryModifier " + value.ToString());
				eparams.min_modifier = value;
				if (eparams.max_modifier < value)
					eparams.max_modifier = value;
			}
		}

		public int MaxHistoryModifier
		{
			get
			{
				return eparams.max_modifier;
			}
			set
			{
				if (value > 100)
					throw new Exception("invalid MaxHistoryModifier " + value.ToString());
				eparams.max_modifier = value;
				if (eparams.min_modifier > value)
					eparams.min_modifier = value;
			}
		}

		public int EstimationDepth
		{
			get
			{
				return eparams.estimation_depth;
			}
			set
			{
				if (value > 32 || value < 1)
					throw new Exception("invalid estimation_depth " + value.ToString());
				eparams.estimation_depth = value;
			}
		}

		public TimeSpan UserProcessorTime
		{
			get { return _userProcessorTime; }
		}

		public int BitsPerSample
		{
			get { return 16; }
		}

		/// <summary>
		/// Copy channel-interleaved input samples into separate subframes
		/// </summary>
		/// <param name="samples"></param>
		/// <param name="pos"></param>
		/// <param name="block"></param>
 		unsafe void copy_samples(int[,] samples, int pos, int block)
		{
			fixed (int* fsamples = samplesBuffer, src = &samples[pos, 0])
			{
				if (channels == 2)
					AudioSamples.Deinterlace(fsamples + samplesInBuffer, fsamples + Alac.MAX_BLOCKSIZE + samplesInBuffer, src, block);
				else
					for (int ch = 0; ch < channels; ch++)
					{
						int* psamples = fsamples + ch * Alac.MAX_BLOCKSIZE + samplesInBuffer;
						for (int i = 0; i < block; i++)
							psamples[i] = src[i * channels + ch];
					}
			}
			samplesInBuffer += block;
		}

		unsafe static void channel_decorrelation(int* leftS, int* rightS, int *leftM, int *rightM, int blocksize)
		{
			for (int i = 0; i < blocksize; i++)
			{
				leftM[i] = (leftS[i] + rightS[i]) >> 1;
				rightM[i] = leftS[i] - rightS[i];
			}
		}

		unsafe static void channel_decorrelation(int* leftS, int* rightS, int* leftM, int* rightM, int blocksize, int leftweight, int shift)
		{
			for (int i = 0; i < blocksize; i++)
			{
				leftM[i] = rightS[i] + ((leftS[i] - rightS[i]) * leftweight >> shift);
				rightM[i] = leftS[i] - rightS[i];
			}
		}

		private static int extend_sign32(int val, int bits)
		{
			return (val << (32 - bits)) >> (32 - bits);
		}

		private static short sign_only(int val)
		{
			return (short)((val >> 31) + ((val - 1) >> 31) + 1);
		}

		unsafe static void alac_encode_residual_31(int* res, int* smp, int n)
		{
			res[0] = smp[0];
			for (int i = 1; i < n; i++)
				res[i] = smp[i] - smp[i - 1];
		}

		unsafe static void alac_encode_residual_0(int* res, int* smp, int n)
		{
			AudioSamples.MemCpy(res, smp, n);
		}

		unsafe static void alac_encode_residual(int* res, int* smp, int n, int order, int* coefs, int shift, int bps)
		{
			int csum = 0;
			
			for (int i = order - 1; i >= 0; i--)
				csum += coefs[i];

			if (n <= order || order <= 0 || order > 30)
				throw new Exception("invalid output");

			/* generate warm-up samples */
			res[0] = smp[0];
			for (int i = 1; i <= order; i++)
				res[i] = smp[i] - smp[i - 1];

			/* general case */
			for (int i = order + 1; i < n; i++)
			{
				int sample = *(smp++);
				int/*long*/ sum = (1 << (shift - 1)) -csum * sample;
				int sum2 = 0;
				for (int j = 0; j < order; j+= 2)
				{
					sum += smp[j] * coefs[j];
					sum2 += smp[j+1] * coefs[j+1];
				}
				int resval = extend_sign32(smp[order] - (int)((sum + sum2) >> shift) - sample, bps);
				res[i] = resval;
				int error_sign = sign_only(resval);
				for (int j = 0; j < order && resval * error_sign > 0; j++)
				{
					int val = sample - smp[j];
					int sign = error_sign * sign_only(val);
					coefs[j] -= sign;
					csum -= sign;
					resval -= ((val * sign) >> shift) * (j + 1);
					//error_sign = (error_sign + sign_only(resval)) / 2;
				}
			}
			res[n] = 1; // Stop byte to help alac_entropy_coder;
		}

		unsafe static int encode_scalar(int x, int k, int bps)
		{
			int divisor = (1 << k) - 1;
			int q = x / divisor;
			int r = x % divisor;
			return q > 8 ? 9 + bps : q + k + (r - 1 >> 31) + 1;//== 0 ? 0 : 1);
		}

		unsafe void encode_scalar(BitWriter bitwriter, int x, int k, int bps)
		{
			k = Math.Min(k, k_modifier);
			int divisor = (1 << k) - 1;
			int q = x / divisor;
			int r = x % divisor;

			if (q > 8)
			{
				// write escape code and sample value directly
				bitwriter.writebits(9, 0x1ff);
				bitwriter.writebits(bps, x);
				return;
			}

			// q times one, then 1 zero, e.g. q == 3 is written as 1110
			int unary = ((1 << (q + 1)) - 2);
			if (r == 0)
			{
				bitwriter.writebits(q + k, unary << (k - 1));
				return;
			}

			bitwriter.writebits(q + 1 + k, (unary << k) + r + 1); 
		}

		unsafe int alac_entropy_coder(int* res, int n, int bps, out int modifier)
		{
			int size = 1 << 30;
			modifier = eparams.min_modifier;
			for (int i = eparams.min_modifier; i <= eparams.max_modifier; i++)
			{
				int newsize = alac_entropy_coder(res, n, bps, i);
				if (size > newsize)
				{
					size = newsize;
					modifier = i;
				}
			}
			return size;
		}

		unsafe int alac_entropy_coder(int* res, int n, int bps, int modifier)
		{
			int history = initial_history;
			int sign_modifier = 0;
			int rice_historymult = modifier * history_mult / 4;
			int size = 0;

			for (int i = 0; i < n; )
			{
				int k = BitReader.log2i((history >> 9) + 3);
				int x = -2 * (*res) - 1;
				x ^= (x >> 31);

				res++;
				i++;

				size += encode_scalar(x - sign_modifier, Math.Min(k, k_modifier), bps);

				history += x * rice_historymult - ((history * rice_historymult) >> 9);

				sign_modifier = 0;
				if (x > 0xFFFF)
					history = 0xFFFF;

				if (history < 128 && i < n)
				{
					k = 7 - BitReader.log2i(history) + ((history + 16) >> 6);
					int block_size = 0;
					while (res[block_size] == 0) // we have a stop byte, so need not check if i + blocksize < n
						block_size++;
					res += block_size;
					i += block_size;
					size += encode_scalar(block_size, Math.Min(k, k_modifier), 16);
					//sign_modifier = (block_size <= 0xFFFF) ? 1 : 0; //never happens
					sign_modifier = 1;
					history = 0;
				}
			}
			return size;
		}

		unsafe void alac_entropy_coder(BitWriter bitwriter, int* res, int n, int bps, int modifier)
		{
			int history = initial_history;
			int sign_modifier = 0;
			int rice_historymult = modifier * history_mult / 4;

			for (int i = 0; i < n; )
			{
				int k = BitReader.log2i((history >> 9) + 3);
				int x = -2 * (*res) - 1;
				x ^= (x >> 31);

				res++;
				i++;

				encode_scalar(bitwriter, x - sign_modifier, k, bps);

				history += x * rice_historymult - ((history * rice_historymult) >> 9);

				sign_modifier = 0;
				if (x > 0xFFFF)
					history = 0xFFFF;

				if (history < 128 && i < n)
				{
					k = 7 - BitReader.log2i(history) + ((history + 16) >> 6);
					int block_size = 0;
					while (res[block_size] == 0) // we have a stop byte, so need not check if i + blocksize < n
						block_size++;
					res += block_size;
					i += block_size;
					encode_scalar(bitwriter, block_size, k, 16);
					sign_modifier = (block_size <= 0xFFFF) ? 1 : 0;
					history = 0;
				}
			}
		}

		unsafe void encode_residual_lpc_sub(ALACFrame frame, double * lpcs, int iWindow, int order, int ch)
		{
			// select LPC precision based on block size
			uint lpc_precision = 15;
			int i_precision = 0;
			//if (frame.blocksize <= 192) lpc_precision = 7U;
			//else if (frame.blocksize <= 384) lpc_precision = 8U;
			//else if (frame.blocksize <= 576) lpc_precision = 9U;
			//else if (frame.blocksize <= 1152) lpc_precision = 10U;
			//else if (frame.blocksize <= 2304) lpc_precision = 11U;
			//else if (frame.blocksize <= 4608) lpc_precision = 12U;
			//else if (frame.blocksize <= 8192) lpc_precision = 13U;
			//else if (frame.blocksize <= 16384) lpc_precision = 14U;

			//for (int i_precision = eparams.lpc_min_precision_search; i_precision <= eparams.lpc_max_precision_search && lpc_precision + i_precision < 16; i_precision++)
				// check if we already calculated with this order, window and precision
				if ((frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[i_precision] & (1U << (order - 1))) == 0)
				{
					frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[i_precision] |= (1U << (order - 1));

					uint cbits = lpc_precision + (uint)i_precision;

					frame.current.order = order;
					frame.current.window = iWindow;

					int bps = (int)bits_per_sample + channels - 1;

					int* coefs = stackalloc int[lpc.MAX_LPC_ORDER];

					lpc.quantize_lpc_coefs(lpcs + (frame.current.order - 1) * lpc.MAX_LPC_ORDER,
						frame.current.order, cbits, coefs, out frame.current.shift, 15, 1);

					if (frame.current.shift < 0 || frame.current.shift > 15)
						throw new Exception("negative shift");

					for (int i = 0; i < frame.current.order; i++)
						frame.current.coefs[i] = coefs[i];

					for (int i = 0; i < frame.current.order; i++)
						coefs[i] = frame.current.coefs[frame.current.order - 1 - i];
					coefs[frame.current.order] = 0;

					alac_encode_residual(frame.current.residual, frame.subframes[ch].samples, frame.blocksize,
						frame.current.order, coefs, frame.current.shift, bps);

					frame.current.size = (uint)(alac_entropy_coder(frame.current.residual, frame.blocksize, bps, out frame.current.ricemodifier) + 16 + 16 * order);

					frame.ChooseBestSubframe(ch);
				}
		}

		unsafe void encode_residual(ALACFrame frame, int ch, OrderMethod omethod, int pass)
		{
			int* smp = frame.subframes[ch].samples;
			int i, n = frame.blocksize;
			int best_window = frame.subframes[ch].best.window;
			int bps = (int)bits_per_sample + channels - 1;

			// FIXED
			//if (0 == (2 & frame.subframes[ch].done_fixed) && (pass != 1 || n < eparams.max_prediction_order))
			//{
			//    frame.subframes[ch].done_fixed |= 2;
			//    frame.current.order = 31;
			//    frame.current.window = -1;
			//    alac_encode_residual_31(frame.current.residual, frame.subframes[ch].samples, frame.blocksize);
			//    frame.current.size = (uint)(alac_entropy_coder(frame.current.residual, frame.blocksize, bps, out frame.current.ricemodifier) + 16);
			//    frame.ChooseBestSubframe(ch);
			//}
			//if (0 == (1 & frame.subframes[ch].done_fixed) && (pass != 1 || n < eparams.max_prediction_order))
			//{
			//    frame.subframes[ch].done_fixed |= 1;
			//    frame.current.order = 0;
			//    frame.current.window = -1;
			//    alac_encode_residual_0(frame.current.residual, frame.subframes[ch].samples, frame.blocksize);
			//    frame.current.size = (uint)(alac_entropy_coder(frame.current.residual, frame.blocksize, bps, out frame.current.ricemodifier) + 16);
			//    frame.ChooseBestSubframe(ch);
			//}

			// LPC
			if (n < eparams.max_prediction_order)
				return;

			double* lpcs = stackalloc double[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
			int min_order = eparams.min_prediction_order;
			int max_order = eparams.max_prediction_order;

			for (int iWindow = 0; iWindow < _windowcount; iWindow++)
			{
				if (pass == 2 && iWindow != best_window)
					continue;

				LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[iWindow];

				lpc_ctx.GetReflection(max_order, smp, n, frame.window_buffer + iWindow * Alac.MAX_BLOCKSIZE * 2);
				lpc_ctx.ComputeLPC(lpcs);

				switch (omethod)
				{
					case OrderMethod.Max:
						// always use maximum order
						encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
						break;
					case OrderMethod.Estimate:
						// estimated orders
						// Search at reflection coeff thresholds (where they cross 0.10)
						{
							int found = 0;
							for (i = max_order; i >= min_order && found < eparams.estimation_depth; i--)
								if (lpc_ctx.IsInterestingOrder(i))
								{
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
									found++;
								}
							if (0 == found)
								encode_residual_lpc_sub(frame, lpcs, iWindow, min_order, ch);
						}
						break;
					case OrderMethod.EstSearch2:
						// Search at reflection coeff thresholds (where they cross 0.10)
						{
							int found = 0;
							for (i = min_order; i <= max_order && found < eparams.estimation_depth; i++)
								if (lpc_ctx.IsInterestingOrder(i))
								{
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
									found++;
								}
							if (0 == found)
								encode_residual_lpc_sub(frame, lpcs, iWindow, min_order, ch);
						}
						break;
					case OrderMethod.Search:
						// brute-force optimal order search
						for (i = max_order; i >= min_order; i--)
							encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
						break;
					case OrderMethod.LogFast:
						// Try max, est, 32,16,8,4,2,1
						encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
						for (i = lpc.MAX_LPC_ORDER; i >= min_order; i >>= 1)
							if (i < max_order)
								encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
						break;
					default:
						throw new Exception("unknown ordermethod");
				}
			}
		}

		unsafe void output_frame_header(ALACFrame frame, BitWriter bitwriter)
		{
			bitwriter.writebits(3, channels - 1);
			bitwriter.writebits(16, 0);
			bitwriter.writebits(1, frame.blocksize != eparams.block_size ? 1 : 0); // sample count is in the header
			bitwriter.writebits(2, 0); // wasted bytes
			bitwriter.writebits(1, frame.type == FrameType.Verbatim ? 1 : 0); // is verbatim
			if (frame.blocksize != eparams.block_size)
				bitwriter.writebits(32, frame.blocksize);
			if (frame.type != FrameType.Verbatim)
			{
				bitwriter.writebits(8, frame.interlacing_shift);
				bitwriter.writebits(8, frame.interlacing_leftweight);
				for (int ch = 0; ch < channels; ch++)
				{
					bitwriter.writebits(4, 0); // prediction type
					bitwriter.writebits(4, frame.subframes[ch].best.shift);
					bitwriter.writebits(3, frame.subframes[ch].best.ricemodifier);
					bitwriter.writebits(5, frame.subframes[ch].best.order);
					if (frame.subframes[ch].best.order != 31)
						for (int c = 0; c < frame.subframes[ch].best.order; c++)
							bitwriter.writebits_signed(16, frame.subframes[ch].best.coefs[c]);
				}
			}
		}

		void output_frame_footer(BitWriter bitwriter)
		{
			bitwriter.writebits(3, 7);
			bitwriter.flush();
		}

		unsafe void window_welch(double* window, int L)
		{
			int N = L - 1;
			double N2 = (double)N / 2.0;

			for (int n = 0; n <= N; n++)
			{
				double k = 1 / N2 - 1.0 - Math.Min(n, N - n);
				//double k = ((double)n - N2) / N2;
				window[n] = 1.0 - k * k;
			}
		}

		unsafe void window_rectangle(double* window, int L)
		{
			for (int n = 0; n < L; n++)
				window[n] = 1.0;
		}

		unsafe void window_flattop(double* window, int L)
		{
			int N = L - 1;
			for (int n = 0; n < L; n++)
				window[n] = 1.0 - 1.93 * Math.Cos(2.0 * Math.PI * n / N) + 1.29 * Math.Cos(4.0 * Math.PI * n / N) - 0.388 * Math.Cos(6.0 * Math.PI * n / N) + 0.0322 * Math.Cos(8.0 * Math.PI * n / N);
		}

		unsafe void window_tukey(double* window, int L)
		{
			window_rectangle(window, L);
			double p = 0.5;
			int Np = (int)(p / 2.0 * L) - 1;
			if (Np > 0)
			{
				for (int n = 0; n <= Np; n++)
				{
					window[n] = 0.5 - 0.5 * Math.Cos(Math.PI * n / Np);
					window[L - Np - 1 + n] = 0.5 - 0.5 * Math.Cos(Math.PI * (n + Np) / Np);
				}
			}
		}

		unsafe void window_hann(double* window, int L)
		{
			int N = L - 1;
			for (int n = 0; n < L; n++)
				window[n] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / N);
		}

		unsafe void encode_residual_pass1(ALACFrame frame, int ch)
		{
			int max_prediction_order = eparams.max_prediction_order;
			int estimation_depth = eparams.estimation_depth;
			int min_modifier = eparams.min_modifier;
			eparams.max_prediction_order = Math.Min(8,eparams.max_prediction_order);
			eparams.estimation_depth = 1;
			eparams.min_modifier = eparams.max_modifier;
			encode_residual(frame, ch, OrderMethod.Estimate, 1);
			eparams.max_prediction_order = max_prediction_order;
			eparams.estimation_depth = estimation_depth;
			eparams.min_modifier = min_modifier;
		}

		unsafe void encode_residual_pass2(ALACFrame frame, int ch)
		{
			encode_residual(frame, ch, eparams.order_method, 2);
		}

		unsafe void encode_residual_onepass(ALACFrame frame, int ch)
		{
			if (_windowcount > 1)
			{
				encode_residual_pass1(frame, ch);
				encode_residual_pass2(frame, ch);
			}
			else
				encode_residual(frame, ch, eparams.order_method, 0);
		}

		unsafe void estimate_frame(ALACFrame frame, bool do_midside)
		{
			int subframes = do_midside ? 5 : channels;

			switch (eparams.stereo_method)
			{
				case StereoMethod.Evaluate:
					for (int ch = 0; ch < subframes; ch++)
					{
						int windowcount = _windowcount;
						_windowcount = 1;
						encode_residual_pass1(frame, ch);
						_windowcount = windowcount;
					}
					break;
				case StereoMethod.Search:
					for (int ch = 0; ch < subframes; ch++)
						encode_residual_onepass(frame, ch);
					break;
			}
		}

		unsafe uint measure_frame_size(ALACFrame frame, bool do_midside)
		{
			// crude estimation of header/footer size
			uint total = 16 + 3;

			if (do_midside)
			{
				uint bitsBest = frame.subframes[0].best.size + frame.subframes[1].best.size;
				frame.interlacing_leftweight = 0;
				frame.interlacing_shift = 0;

				if (bitsBest > frame.subframes[3].best.size + frame.subframes[0].best.size) // leftside
				{
					bitsBest = frame.subframes[3].best.size + frame.subframes[0].best.size;
					frame.interlacing_leftweight = 1;
					frame.interlacing_shift = 0;
				}
				if (bitsBest > frame.subframes[3].best.size + frame.subframes[2].best.size) // midside
				{
					bitsBest = frame.subframes[3].best.size + frame.subframes[2].best.size;
					frame.interlacing_leftweight = 1;
					frame.interlacing_shift = 1;
				}
				if (bitsBest > frame.subframes[3].best.size + frame.subframes[4].best.size) // rightside
				{
					bitsBest = frame.subframes[3].best.size + frame.subframes[4].best.size;
					frame.interlacing_leftweight = 1;
					frame.interlacing_shift = 31;
				}

				return total + bitsBest;
			}

			for (int ch = 0; ch < channels; ch++)
				total += frame.subframes[ch].best.size;

			return total;
		}

		unsafe void encode_estimated_frame(ALACFrame frame)
		{
			switch (eparams.stereo_method)
			{
				case StereoMethod.Evaluate:
					for (int ch = 0; ch < channels; ch++)
					{
						if (_windowcount > 1)
							encode_residual_pass1(frame, ch);
						encode_residual_pass2(frame, ch);
					}
					break;
				case StereoMethod.Search:
					break;
			}
		}

		unsafe delegate void window_function(double* window, int size);

		unsafe void calculate_window(double* window, window_function func, WindowFunction flag)
		{
			if ((eparams.window_function & flag) == 0 || _windowcount == lpc.MAX_LPC_WINDOWS)
				return;
			int sz = _windowsize;
			double* pos = window + _windowcount * Alac.MAX_BLOCKSIZE * 2;
			do
			{
				func(pos, sz);
				if ((sz & 1) != 0)
					break;
				pos += sz;
				sz >>= 1;
			} while (sz >= 32);
			_windowcount++;
		}

		unsafe int encode_frame(ref int size)
		{
			fixed (int* s = samplesBuffer, r = residualBuffer)
			fixed (double* window = windowBuffer)
			{
				frame.InitSize(size);

				if (frame.blocksize != _windowsize && frame.blocksize > 4)
				{
					_windowsize = frame.blocksize;
					_windowcount = 0;
					calculate_window(window, window_welch, WindowFunction.Welch);
					calculate_window(window, window_tukey, WindowFunction.Tukey);
					calculate_window(window, window_hann, WindowFunction.Hann);
					calculate_window(window, window_flattop, WindowFunction.Flattop);
					if (_windowcount == 0)
						throw new Exception("invalid windowfunction");
				}
				frame.window_buffer = window;

				int bps = (int)bits_per_sample + channels - 1;
				if (channels != 2 || frame.blocksize <= 32 || eparams.stereo_method == StereoMethod.Independent)
				{
					frame.current.residual = r + channels * Alac.MAX_BLOCKSIZE;

					for (int ch = 0; ch < channels; ch++)
						frame.subframes[ch].Init(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE);

					for (int ch = 0; ch < channels; ch++)
						encode_residual_onepass(frame, ch);
				}
				else if (eparams.stereo_method == StereoMethod.Estimate || eparams.stereo_method == StereoMethod.Estimate2)
				{
					int* sl = s;
					int* sr = s + Alac.MAX_BLOCKSIZE;
					int n = frame.blocksize;
					ulong lsum = 0, rsum = 0, dsum = 0, s31 = 0, s1 = 0, s2 = 0, s3 = 0;
					if (eparams.stereo_method == StereoMethod.Estimate)
					for (int i = 2; i < n; i++)
					{
						int lt = sl[i] - 2 * sl[i - 1] + sl[i - 2];
						int rt = sr[i] - 2 * sr[i - 1] + sr[i - 2];
						int df = lt - rt;
						lsum += (ulong)Math.Abs(lt);
						rsum += (ulong)Math.Abs(rt);
						dsum += (ulong)Math.Abs(df);
						s1 += (ulong)Math.Abs(rt + (df >> 1));
						s31 += (ulong)Math.Abs(rt + (df >> 31));
					}
					else
					for (int i = 2; i < n; i++)
					{
						int lt = sl[i] - 2 * sl[i - 1] + sl[i - 2];
						int rt = sr[i] - 2 * sr[i - 1] + sr[i - 2];
						int df = lt - rt;
						lsum += (ulong)Math.Abs(lt);
						rsum += (ulong)Math.Abs(rt);
						dsum += (ulong)Math.Abs(df);
						s1 += (ulong)Math.Abs(rt + (df >> 1));
						s2 += (ulong)Math.Abs(rt + (df >> 2));
						s3 += (ulong)Math.Abs(rt + (df * 3 >> 2));
						s31 += (ulong)Math.Abs(rt + (df >> 31));
					}
					frame.interlacing_leftweight = 0;
					frame.interlacing_shift = 0;
					ulong score = lsum + rsum;
					if (lsum + dsum < score) //leftside
					{
						frame.interlacing_leftweight = 1;
						frame.interlacing_shift = 0;
						score = lsum + dsum;
					}
					if (s1 + dsum < score) // midside
					{
						frame.interlacing_leftweight = 1;
						frame.interlacing_shift = 1;
						score = s1 + dsum;
					}
					if (s31 + dsum < score) // rightside
					{
						frame.interlacing_leftweight = 1;
						frame.interlacing_shift = 31;
						score = s31 + dsum;
					}
					if (eparams.stereo_method == StereoMethod.Estimate2)
					{
						if (s2 + dsum < score) // close to rightside
						{
							frame.interlacing_leftweight = 1;
							frame.interlacing_shift = 2;
							score = s2 + dsum;
						}
						if (s3 + dsum < score) // close to leftside
						{
							frame.interlacing_leftweight = 3;
							frame.interlacing_shift = 2;
							score = s3 + dsum;
						}
					}
					if (frame.interlacing_leftweight == 0)
					{
						frame.current.residual = r + channels * Alac.MAX_BLOCKSIZE;
						for (int ch = 0; ch < channels; ch++)
							frame.subframes[ch].Init(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE);
					}
					else
					{
						frame.current.residual = r + 2 * channels * Alac.MAX_BLOCKSIZE;
						channel_decorrelation(s, s + Alac.MAX_BLOCKSIZE, s + 2 * Alac.MAX_BLOCKSIZE, s + 3 * Alac.MAX_BLOCKSIZE, frame.blocksize,
							frame.interlacing_leftweight, frame.interlacing_shift);
						for (int ch = 0; ch < channels; ch++)
							frame.subframes[ch].Init(s + (channels + ch) * Alac.MAX_BLOCKSIZE, r + (channels + ch) * Alac.MAX_BLOCKSIZE);
					}

					for (int ch = 0; ch < channels; ch++)
						encode_residual_onepass(frame, ch);
				}
				else
				{
					channel_decorrelation(s, s + Alac.MAX_BLOCKSIZE, s + 2 * Alac.MAX_BLOCKSIZE, s + 3 * Alac.MAX_BLOCKSIZE, frame.blocksize, 1, 1);
					channel_decorrelation(s, s + Alac.MAX_BLOCKSIZE, s + 4 * Alac.MAX_BLOCKSIZE, s + 3 * Alac.MAX_BLOCKSIZE, frame.blocksize, 1, 31);
					frame.current.residual = r + 5 * Alac.MAX_BLOCKSIZE;
					for (int ch = 0; ch < 5; ch++)
						frame.subframes[ch].Init(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE);
					estimate_frame(frame, true);
					measure_frame_size(frame, true);
					frame.ChooseSubframes();
					encode_estimated_frame(frame);
				}
				uint fs = measure_frame_size(frame, false);
				frame.type = ((int)fs > frame.blocksize * channels * bps) ? FrameType.Verbatim : FrameType.Compressed;
				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);
				output_frame_header(frame, bitwriter);
				if (frame.type == FrameType.Verbatim)
				{
					for (int i = 0; i < frame.blocksize; i++)
						for (int ch = 0; ch < channels; ch++)
							bitwriter.writebits_signed((int)bits_per_sample, frame.subframes[ch].samples[i]);
				}
				else if (frame.type == FrameType.Compressed)
				{
					for (int ch = 0; ch < channels; ch++)
						alac_entropy_coder(bitwriter, frame.subframes[ch].best.residual, frame.blocksize, 
							bps, frame.subframes[ch].best.ricemodifier);
				}
				output_frame_footer(bitwriter);

				_sample_byte_size[frame_count++] = (uint)bitwriter.Length;

				size = frame.blocksize;
				return bitwriter.Length;
			}
		}

		unsafe int output_frame(int blocksize)
		{
			if (verify != null)
			{
				fixed (int* s = verifyBuffer, r = samplesBuffer)
					for (int ch = 0; ch < channels; ch++)
						AudioSamples.MemCpy(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE, eparams.block_size);
			}

			//if (0 != eparams.variable_block_size && 0 == (eparams.block_size & 7) && eparams.block_size >= 128)
			//    fs = encode_frame_vbs();
			//else
			int bs = blocksize;
			int fs = encode_frame(ref bs);

			_position += bs;
			_IO.Write(frame_buffer, 0, fs);
			_totalSize += fs;

			if (verify != null)
			{
				int decoded = verify.DecodeFrame(frame_buffer, 0, fs);
				if (decoded != fs || verify.Remaining != (ulong)bs)
					throw new Exception("validation failed!");
				int [,] deinterlaced = new int[bs,channels];
				verify.deinterlace(deinterlaced, 0, (uint)bs);
				fixed (int* s = verifyBuffer, r = deinterlaced)
				{
					for (int i = 0; i < bs; i++)
						for (int ch = 0; ch < channels; ch++)
							if (r[i * channels + ch] != s[ch * Alac.MAX_BLOCKSIZE + i])
								throw new Exception("validation failed!");
				}
			}

			if (bs < blocksize)
			{
				fixed (int* s = samplesBuffer)
					for (int ch = 0; ch < channels; ch++)
						AudioSamples.MemCpy(s + ch * Alac.MAX_BLOCKSIZE, s + bs + ch * Alac.MAX_BLOCKSIZE, eparams.block_size - bs);
			}

			samplesInBuffer -= bs;

			return bs;
		}

		public void Write(int[,] buff, int pos, int sampleCount)
		{
			if (!inited)
			{
				if (_IO == null)
					_IO = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read);
				int header_size = encode_init();
				_IO.Write(header, 0, header_size);
				if (_IO.CanSeek)
					first_frame_offset = _IO.Position;
				inited = true;
			}

			int len = sampleCount;
			while (len > 0)
			{
				int block = Math.Min(len, eparams.block_size - samplesInBuffer);

				copy_samples(buff, pos, block);

				len -= block;
				pos += block;

				while (samplesInBuffer >= eparams.block_size)
					output_frame(eparams.block_size);
			}
		}

		public string Path { get { return _path; } }

		string vendor_string = "CUETools.2.05";

		int select_blocksize(int samplerate, int time_ms)
		{
			int target = (samplerate * time_ms) / 1000;
			int blocksize = 1024;
			while (target >= blocksize)
				blocksize <<= 1;
			return blocksize >> 1;
		}

		void write_chunk_mvhd(BitWriter bitwriter, TimeSpan UnixTime)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'v', 'h', 'd');
				bitwriter.writebits(32, 0);
				bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
				bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
				bitwriter.writebits(32, 1000);
				bitwriter.writebits(32, sample_count);
				bitwriter.writebits(32, 0x00010000); // reserved (preferred rate) 1.0 = normal
				bitwriter.writebits(16, 0x0100); // reserved (preferred volume) 1.0 = normal
				bitwriter.writebytes(10, 0); // reserved
				bitwriter.writebits(32, 0x00010000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00010000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0x40000000); // reserved (matrix structure)
				bitwriter.writebits(32, 0); // preview time
				bitwriter.writebits(32, 0); // preview duration
				bitwriter.writebits(32, 0); // poster time
				bitwriter.writebits(32, 0); // selection time
				bitwriter.writebits(32, 0); // selection duration
				bitwriter.writebits(32, 0); // current time
				bitwriter.writebits(32, 2); // next track ID
			}
			chunk_end(bitwriter);
		}

		void write_chunk_minf(BitWriter bitwriter)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'i', 'n', 'f');
				chunk_start(bitwriter);
				{
					bitwriter.write('s', 'm', 'h', 'd');
					bitwriter.writebits(32, 0); // version & flags
					bitwriter.writebits(16, 0); // reserved (balance)
					bitwriter.writebits(16, 0); // reserved
				}
				chunk_end(bitwriter);
				chunk_start(bitwriter);
				{
					bitwriter.write('d', 'i', 'n', 'f');
					chunk_start(bitwriter);
					{
						bitwriter.write('d', 'r', 'e', 'f');
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, 1); // entry count
						chunk_start(bitwriter);
						{
							bitwriter.write('u', 'r', 'l', ' ');
							bitwriter.writebits(32, 1); // version & flags
						}
						chunk_end(bitwriter);
					}
					chunk_end(bitwriter);
				}
				chunk_end(bitwriter);
				chunk_start(bitwriter);
				{
					bitwriter.write('s', 't', 'b', 'l');
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 's', 'd');
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, 1); // entry count
						chunk_start(bitwriter);
						{
							bitwriter.write('a', 'l', 'a', 'c');
							bitwriter.writebits(32, 0); // reserved
							bitwriter.writebits(16, 0); // reserved
							bitwriter.writebits(16, 1); // data reference index
							bitwriter.writebits(16, 0); // version
							bitwriter.writebits(16, 0); // revision
							bitwriter.writebits(32, 0); // reserved
							bitwriter.writebits(16, 2); // reserved channels
							bitwriter.writebits(16, 16); // reserved bps
							bitwriter.writebits(16, 0); // reserved compression ID
							bitwriter.writebits(16, 0); // packet size
							bitwriter.writebits(16, sample_rate); // time scale
							bitwriter.writebits(16, 0); // reserved
							chunk_start(bitwriter);
							{
								bitwriter.write('a', 'l', 'a', 'c');
								bitwriter.writebits(32, 0); // reserved
								bitwriter.writebits(32, eparams.block_size); // max frame size
								bitwriter.writebits(8, 0); // reserved
								bitwriter.writebits(8, bits_per_sample);
								bitwriter.writebits(8, history_mult);
								bitwriter.writebits(8, initial_history);
								bitwriter.writebits(8, k_modifier);
								bitwriter.writebits(8, channels); // channels
								bitwriter.writebits(16, 0); // reserved
								bitwriter.writebits(32, max_frame_size);
								bitwriter.writebits(32, sample_rate * channels * (int)bits_per_sample); // average bitrate
								bitwriter.writebits(32, sample_rate);
							}
							chunk_end(bitwriter);
						}
						chunk_end(bitwriter);
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 't', 's');
						bitwriter.writebits(32, 0); // version & flags
						if (sample_count % eparams.block_size == 0)
						{
							bitwriter.writebits(32, 1); // entries
							bitwriter.writebits(32, sample_count / eparams.block_size);
							bitwriter.writebits(32, eparams.block_size);
						}
						else
						{
							bitwriter.writebits(32, 2); // entries
							bitwriter.writebits(32, sample_count / eparams.block_size);
							bitwriter.writebits(32, eparams.block_size);
							bitwriter.writebits(32, 1);
							bitwriter.writebits(32, sample_count % eparams.block_size);
						}
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 's', 'c');
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, 1); // entry count
						bitwriter.writebits(32, 1); // first chunk
						bitwriter.writebits(32, 1); // samples in chunk
						bitwriter.writebits(32, 1); // sample description index
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 's', 'z');
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, 0); // sample size (0 == variable)
						bitwriter.writebits(32, frame_count); // entry count
						for (int i = 0; i < frame_count; i++)
							bitwriter.writebits(32, _sample_byte_size[i]);
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 'c', 'o');
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, frame_count); // entry count
						uint pos = (uint)mdat_pos + 8;
						for (int i = 0; i < frame_count; i++)
						{
							bitwriter.writebits(32, pos);
							pos += _sample_byte_size[i];
						}
					}
					chunk_end(bitwriter);
				}
				chunk_end(bitwriter);
			}
			chunk_end(bitwriter);
		}

		void write_chunk_mdia(BitWriter bitwriter, TimeSpan UnixTime)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'd', 'i', 'a');
				chunk_start(bitwriter);
				{
					bitwriter.write('m', 'd', 'h', 'd');
					bitwriter.writebits(32, 0); // version & flags
					bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
					bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
					bitwriter.writebits(32, sample_rate);
					bitwriter.writebits(32, sample_count);
					bitwriter.writebits(16, 0x55c4); // language
					bitwriter.writebits(16, 0); // quality
				}
				chunk_end(bitwriter);
				chunk_start(bitwriter);
				{
					bitwriter.write('h', 'd', 'l', 'r');
					bitwriter.writebits(32, 0); // version & flags
					bitwriter.writebits(32, 0); // hdlr
					bitwriter.write('s', 'o', 'u', 'n');
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(8, "SoundHandler".Length);
					bitwriter.write("SoundHandler");
				}
				chunk_end(bitwriter);
				write_chunk_minf(bitwriter);
			}
			chunk_end(bitwriter);
		}

		void write_chunk_trak(BitWriter bitwriter, TimeSpan UnixTime)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('t', 'r', 'a', 'k');
				chunk_start(bitwriter);
				{
					bitwriter.write('t', 'k', 'h', 'd');
					bitwriter.writebits(32, 15); // version
					bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
					bitwriter.writebits(32, (int)UnixTime.TotalSeconds);
					bitwriter.writebits(32, 1); // track ID
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, sample_count / sample_rate);
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, 0); // reserved (layer & alternate group)
					bitwriter.writebits(16, 0x0100); // reserved (preferred volume) 1.0 = normal
					bitwriter.writebits(16, 0); // reserved
					bitwriter.writebits(32, 0x00010000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00010000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x00000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0x40000000); // reserved (matrix structure)
					bitwriter.writebits(32, 0); // reserved (width)
					bitwriter.writebits(32, 0); // reserved (height)
				}
				chunk_end(bitwriter);
				write_chunk_mdia(bitwriter, UnixTime);
			}
			chunk_end(bitwriter);
		}

		void write_chunk_udta(BitWriter bitwriter)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('u', 'd', 't', 'a');
				chunk_start(bitwriter);
				{
					bitwriter.write('m', 'e', 't', 'a');
					bitwriter.writebits(32, 0);
					chunk_start(bitwriter);
					{
						bitwriter.write('h', 'd', 'l', 'r');
						bitwriter.writebits(32, 0);
						bitwriter.writebits(32, 0);
						bitwriter.write('m', 'd', 'i', 'r');
						bitwriter.write('a', 'p', 'p', 'l');
						bitwriter.writebits(32, 0);
						bitwriter.writebits(32, 0);
						bitwriter.writebits(16, 0);
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('i', 'l', 's', 't');
						chunk_start(bitwriter);
						{
							bitwriter.write((char)0xA9, 't', 'o', 'o');
							chunk_start(bitwriter);
							{
								bitwriter.write('d', 'a', 't', 'a');
								bitwriter.writebits(32, 1);
								bitwriter.writebits(32, 0);
								bitwriter.write(vendor_string);
							}
							chunk_end(bitwriter);
						}
						chunk_end(bitwriter);
					}
					chunk_end(bitwriter);
				}
				chunk_end(bitwriter);
			}
			chunk_end(bitwriter);
		}

		int write_trailers()
		{
			TimeSpan UnixTime = DateTime.Now - new DateTime(1970, 1, 1, 0, 0, 0, 0).ToLocalTime();
			header = new byte[0x1000 + frame_count * 8]; // FIXME!!! Possible buffer overrun
			BitWriter bitwriter = new BitWriter(header, 0, header.Length);
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'o', 'o', 'v');
				write_chunk_mvhd(bitwriter, UnixTime);
				write_chunk_trak(bitwriter, UnixTime);
				write_chunk_udta(bitwriter);
			}
			chunk_end(bitwriter);
			return bitwriter.Length;
		}

		int write_headers()
		{
			BitWriter bitwriter = new BitWriter(header, 0, header.Length);

			chunk_start(bitwriter);
			bitwriter.write('f', 't', 'y', 'p');
			bitwriter.write('M', '4', 'A', ' ');
			bitwriter.writebits(32, 0x200); // minor version
			bitwriter.write('M', '4', 'A', ' ');
			bitwriter.write('m', 'p', '4', '2');
			bitwriter.write('i', 's', 'o', 'm');
			bitwriter.writebits(32, 0);
			chunk_end(bitwriter);

			chunk_start(bitwriter); // padding in case we need extended mdat len
			bitwriter.write('f', 'r', 'e', 'e');
			chunk_end(bitwriter);

			mdat_pos = bitwriter.Length;

			chunk_start(bitwriter); // mdat len placeholder
			bitwriter.write('m', 'd', 'a', 't');
			chunk_end(bitwriter);

			return bitwriter.Length;
		}

		int encode_init()
		{
			int i, header_len;

			//if(flake_validate_params(s) < 0)

			ch_code = channels - 1;

			// FIXME: For now, only 44100 samplerate is supported
			if (sample_rate != 44100)
				throw new Exception("non-standard samplerate");

			// FIXME: For now, only 16-bit encoding is supported
			if (bits_per_sample != 16)
				throw new Exception("non-standard bps");

			if (_blocksize == 0)
			{
				if (eparams.block_size == 0)
					eparams.block_size = select_blocksize(sample_rate, eparams.block_time_ms);
				_blocksize = eparams.block_size;
			}
			else
				eparams.block_size = _blocksize;

			// set maximum encoded frame size (if larger, re-encodes in verbatim mode)
			if (channels == 2)
				max_frame_size = 16 + ((eparams.block_size * (int)(bits_per_sample + bits_per_sample + 1) + 7) >> 3);
			else
				max_frame_size = 16 + ((eparams.block_size * channels * (int)bits_per_sample + 7) >> 3);

			//if (_IO.CanSeek && eparams.do_seektable)
			//{
			//}

			// output header bytes
			header = new byte[eparams.padding_size + 0x1000];
			header_len = write_headers();

			frame_buffer = new byte[max_frame_size];
			_sample_byte_size = new uint[sample_count / eparams.block_size + 1];

			initial_history = 10;
			history_mult = 40;
			k_modifier = 14;

			if (eparams.do_verify)
			{
				verify = new ALACReader(channels, (int)bits_per_sample, history_mult, initial_history, k_modifier, eparams.block_size);
				verifyBuffer = new int[Alac.MAX_BLOCKSIZE * channels];
			}

			return header_len;
		}
	}

	struct ALACEncodeParams
	{
		// compression quality
		// set by user prior to calling encode_init
		// standard values are 0 to 8
		// 0 is lower compression, faster encoding
		// 8 is higher compression, slower encoding
		// extended values 9 to 12 are slower and/or use
		// higher prediction orders
		public int compression;

		// prediction order selection method
		// set by user prior to calling encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 5
		// 0 = use maximum order only
		// 1 = use estimation
		// 2 = 2-level
		// 3 = 4-level
		// 4 = 8-level
		// 5 = full search
		// 6 = log search
		public OrderMethod order_method;


		// stereo decorrelation method
		// set by user prior to calling encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 2
		// 0 = independent L+R channels
		// 1 = mid-side encoding
		public StereoMethod stereo_method;

		// block size in samples
		// set by the user prior to calling encode_init
		// if set to 0, a block size is chosen based on block_time_ms
		// can also be changed by user before encoding a frame
		public int block_size;

		// block time in milliseconds
		// set by the user prior to calling encode_init
		// used to calculate block_size based on sample rate
		// can also be changed by user before encoding a frame
		public int block_time_ms;

		// padding size in bytes
		// set by the user prior to calling encode_init
		// if set to less than 0, defaults to 4096
		public int padding_size;

		// minimum LPC order
		// set by user prior to calling encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 1 to 32
		public int min_prediction_order;

		// maximum LPC order
		// set by user prior to calling encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 1 to 32 
		public int max_prediction_order;

		// Number of LPC orders to try (for estimate mode)
		// set by user prior to calling encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 1 to 32 
		public int estimation_depth;

		public int min_modifier, max_modifier;

		public WindowFunction window_function;

		public bool do_verify;
		public bool do_seektable;

		public int set_defaults(int lvl)
		{
			compression = lvl;

			if ((lvl < 0 || lvl > 12) && (lvl != 99))
			{
				return -1;
			}

			// default to level 5 params
			window_function = WindowFunction.Flattop | WindowFunction.Tukey;
			order_method = OrderMethod.Estimate;
			stereo_method = StereoMethod.Evaluate;
			block_size = 0;
			block_time_ms = 105;
			min_modifier = 4;
			max_modifier = 4;
			min_prediction_order = 1;
			max_prediction_order = 12;
			estimation_depth = 1;
			do_verify = false;
			do_seektable = false;

			// differences from level 6
			switch (lvl)
			{
				case 0:
					block_time_ms = 53;
					stereo_method = StereoMethod.Independent;
					window_function = WindowFunction.Welch;
					max_prediction_order = 6;
					break;
				case 1:
					stereo_method = StereoMethod.Independent;
					window_function = WindowFunction.Welch;
					max_prediction_order = 8;
					break;
				case 2:
					stereo_method = StereoMethod.Estimate;
					window_function = WindowFunction.Welch;
					max_prediction_order = 6;
					break;
				case 3:
					stereo_method = StereoMethod.Estimate;
					window_function = WindowFunction.Welch;
					max_prediction_order = 8;
					break;
				case 4:
					stereo_method = StereoMethod.Estimate2;
					window_function = WindowFunction.Welch;
					break;
				case 5:
					stereo_method = StereoMethod.Estimate2;
					break;
				case 6:					
					break;
				case 7:
					estimation_depth = 3;
					min_modifier = 3;
					break;
				case 8:
					estimation_depth = 5;
					max_prediction_order = 30;
					min_modifier = 2;
					break;
			}

			return 0;
		}
	}
}
