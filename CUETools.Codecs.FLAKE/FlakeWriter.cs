using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using CUETools.Codecs;

namespace CUETools.Codecs.FLAKE
{
	public class FlakeWriter : IAudioDest
	{
		Stream _IO = null;
		string _path;
		long _position;

		// number of audio channels
		// set by user prior to calling flake_encode_init
		// valid values are 1 to 8
		int channels, ch_code;

		// audio sample rate in Hz
		// set by user prior to calling flake_encode_init
		int sample_rate, sr_code0, sr_code1;

		// sample size in bits
		// set by user prior to calling flake_encode_init
		// only 16-bit is currently supported
		uint bits_per_sample;
		int bps_code;

		// total stream samples
		// set by user prior to calling flake_encode_init
		// if 0, stream length is unknown
		int sample_count;

		FlakeEncodeParams eparams;

		// maximum frame size in bytes
		// set by flake_encode_init
		// this can be used to allocate memory for output
		int max_frame_size;

		byte[] frame_buffer = null;

		uint lpc_precision;

		int frame_count = 0;

		TimeSpan _userProcessorTime;

		// header bytes
		// allocated by flake_encode_init and freed by flake_encode_close
		byte[] header;

		int[] samplesBuffer;
		int[] verifyBuffer;
		int[] residualBuffer;
		double[] windowBuffer;
		int samplesInBuffer = 0;

		int _compressionLevel = 5;
		int _blocksize = 0;
		int _totalSize = 0;
		int _windowsize = 0, _windowcount = 0;

		Crc8 crc8;
		Crc16 crc16;
		MD5 md5;

		FlakeReader verify;

		bool inited = false;

		public FlakeWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO)
		{
			if (bitsPerSample != 16)
				throw new Exception("Bits per sample must be 16.");
			if (channelCount != 2)
				throw new Exception("ChannelCount must be 2.");

			flac_samplerates = new int[16] {
				0, 0, 0, 0,
				8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
				0, 0, 0, 0
			};
			flac_bitdepths = new int[8] { 0, 8, 12, 0, 16, 20, 24, 0 };
			flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };

			channels = channelCount;
			sample_rate = sampleRate;
			bits_per_sample = (uint) bitsPerSample;

			// flake_validate_params

			_path = path;
			_IO = IO;

			//verify = new FlakeReader(channels, bits_per_sample);

			samplesBuffer = new int[Flake.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels)];
			residualBuffer = new int[Flake.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels)];
			windowBuffer = new double[Flake.MAX_BLOCKSIZE * lpc.MAX_LPC_WINDOWS];
			if (verify != null)
				verifyBuffer = new int[Flake.MAX_BLOCKSIZE * channels];

			eparams.flake_set_defaults(_compressionLevel);
			eparams.padding_size = 8192;

			crc8 = new Crc8();
			crc16 = new Crc16();
			md5 = new MD5CryptoServiceProvider();
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
				_compressionLevel = value;
				eparams.flake_set_defaults(_compressionLevel);
			}
		}

		[DllImport("kernel32.dll")]
		static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
		[DllImport("kernel32.dll")]
		static extern IntPtr GetCurrentThread();

		void DoClose()
		{
			if (samplesInBuffer > 0)
			{
				eparams.block_size = samplesInBuffer;
				output_frame();
			}

			long fake, KernelStart, UserStart;
			GetThreadTimes(GetCurrentThread(), out fake, out fake, out KernelStart, out UserStart);
			_userProcessorTime = new TimeSpan(UserStart);

			md5.TransformFinalBlock(frame_buffer, 0, 0);
			if (_IO.CanSeek)
			{
				_IO.Position = 26;
				_IO.Write(md5.Hash, 0, md5.Hash.Length);
			}
			_IO.Close();
		}

		public void Close()
		{
			DoClose();
			if (sample_count != 0 && _position != sample_count)
				throw new Exception("Samples written differs from the expected sample count.");
		}

		public void Delete()
		{
			DoClose();
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
		}

		public OrderMethod OrderMethod
		{
			get { return eparams.order_method; }
			set { eparams.order_method = value; }
		}

		public PredictionType PredictionType
		{
			get { return eparams.prediction_type; }
			set { eparams.prediction_type = value; }
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

		public int MinPredictionOrder
		{
			get
			{
				return eparams.prediction_type == PredictionType.Fixed ?
					eparams.min_fixed_order : eparams.min_prediction_order;
			}
			set
			{
				if (eparams.prediction_type == PredictionType.Fixed)
				{
					if (value < 0 || value > eparams.max_fixed_order)
						throw new Exception("invalid order");
					eparams.min_fixed_order = value;
					return;
				}
				if (value < 0 || value > eparams.max_prediction_order)
					throw new Exception("invalid order");
				eparams.min_prediction_order = value;
			}
		}

		public int MaxPredictionOrder
		{
			get 
			{
				return eparams.prediction_type == PredictionType.Fixed ?
					eparams.max_fixed_order : eparams.max_prediction_order; 
			}
			set
			{
				if (eparams.prediction_type == PredictionType.Fixed)
				{
					if (value > 4 || value < eparams.min_fixed_order)
						throw new Exception("invalid order");
					eparams.max_fixed_order = value;
					return;
				}
				if (value > 32 || value < eparams.min_prediction_order)
					throw new Exception("invalid order");
				eparams.max_prediction_order = value;
			}
		}

		public int MaxPartitionOrder
		{
			get { return eparams.max_partition_order; }
			set
			{
				if (value > 8 || value < eparams.min_partition_order)
					throw new Exception("invalid order");
				eparams.max_partition_order = value;
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

		int encode_frame_vbs()
		{
			throw new Exception("vbs not supported");
		}

		unsafe int get_wasted_bits(int* signal, int samples)
		{
			int i, shift;
			int x = 0;

			for (i = 0; i < samples && 0 == (x & 1); i++)
				x |= signal[i];

			if (x == 0)
			{
				shift = 0;
			}
			else
			{
				for (shift = 0; 0 == (x & 1); shift++)
					x >>= 1;
			}

			if (shift > 0)
			{
				for (i = 0; i < samples; i++)
					signal[i] >>= shift;
			}

			return shift;
		}

		unsafe void init_frame(FlacFrame * frame)
		{
			//if (channels == 2)
			//max_frame_size =			

			int i = 15;
			if (eparams.variable_block_size == 0)
			{
				for (i = 0; i < 15; i++)
				{
					if (eparams.block_size == flac_blocksizes[i])
					{
						frame->blocksize = flac_blocksizes[i];
						frame->bs_code0 = i;
						frame->bs_code1 = -1;
						break;
					}
				}
			}
			if (i == 15)
			{
				frame->blocksize = eparams.block_size;
				if (frame->blocksize <= 256)
				{
					frame->bs_code0 = 6;
					frame->bs_code1 = frame->blocksize - 1;
				}
				else
				{
					frame->bs_code0 = 7;
					frame->bs_code1 = frame->blocksize - 1;
				}
			}
		}

		/**
		 * Copy channel-interleaved input samples into separate subframes
         */
		unsafe void copy_samples(int[,] samples, int pos, int block)
		{
			fixed (int* fsamples = samplesBuffer)
				for (int ch = 0; ch < channels; ch++)
				{
					int* psamples = fsamples + ch * Flake.MAX_BLOCKSIZE + samplesInBuffer;
					for (int i = 0; i < block; i++)
						psamples[i] = samples[pos + i, ch];
				}
			samplesInBuffer += block;
		}

		static uint rice_encode_count(uint sum, uint n, uint k)
		{
			return n*(k+1) + ((sum-(n>>1))>>(int)k);
		}

		//static unsafe uint find_optimal_rice_param(uint sum, uint n)
		//{
		//    uint* nbits = stackalloc uint[Flake.MAX_RICE_PARAM + 1];
		//    int k_opt = 0;

		//    nbits[0] = UINT32_MAX;
		//    for (int k = 0; k <= Flake.MAX_RICE_PARAM; k++)
		//    {
		//        nbits[k] = rice_encode_count(sum, n, (uint)k);
		//        if (nbits[k] < nbits[k_opt])
		//            k_opt = k;
		//    }
		//    return (uint)k_opt;
		//}

		static unsafe uint find_optimal_rice_param(uint sum, uint n, out uint nbits_best)
		{
			int k_opt = 0;
			uint a = n;
			uint b = sum - (n >> 1);
			uint nbits = a + b;
			for (int k = 1; k <= Flake.MAX_RICE_PARAM; k++)
			{
				a += n;
				b >>= 1;
				uint nbits_k = a + b;
				if (nbits_k < nbits)
				{
					k_opt = k;
					nbits = nbits_k;
				}
			}
			nbits_best = nbits;
			return (uint)k_opt;
		}

		unsafe uint calc_decorr_score(FlacFrame* frame, int ch, FlacSubframe* sub)
		{
			int* s = sub->samples;
			int n = frame->blocksize;
			ulong sum = 0;
			for (int i = 2; i < n; i++)
				sum += (ulong)Math.Abs(s[i] - 2 * s[i - 1] + s[i - 2]);
			uint nbits;
			find_optimal_rice_param((uint)(2 * sum), (uint)n, out nbits);
			return nbits;
		}

		unsafe void initialize_subframe(FlacFrame* frame, int ch, int *s, int * r, uint bps)
		{
			int w = get_wasted_bits(s, frame->blocksize);
			if (w > bps)
				throw new Exception("internal error");
			frame->subframes[ch].samples = s;
			frame->subframes[ch].residual = r;
			frame->subframes[ch].obits = bps - (uint)w;
			frame->subframes[ch].wbits = (uint)w;
			frame->subframes[ch].type = SubframeType.Verbatim;
			frame->subframes[ch].size = UINT32_MAX;
			for (int iWindow = 0; iWindow < lpc.MAX_LPC_WINDOWS; iWindow++)
				frame->subframes[ch].done_lpcs[iWindow] = 0;
			frame->subframes[ch].done_fixed = 0;
		}

		unsafe static void channel_decorrelation(int* leftS, int* rightS, int *leftM, int *rightM, int blocksize)
		{
			for (int i = 0; i < blocksize; i++)
			{
				leftM[i] = (leftS[i] + rightS[i]) >> 1;
				rightM[i] = leftS[i] - rightS[i];
			}
		}

		unsafe void encode_residual_verbatim(int* res, int* smp, uint n)
		{
			Flake.memcpy(res, smp, (int) n);
		}

		unsafe void encode_residual_fixed(int* res, int* smp, int n, int order)
		{
			int i;
			int s0, s1, s2;
			switch (order)
			{
				case 0:
					Flake.memcpy(res, smp, n);
					return;
				case 1:
					*(res++) = s1 = *(smp++);
					for (i = n - 1; i > 0; i--)
					{
						s0 = *(smp++);
						*(res++) = s0 - s1;
						s1 = s0;
					}
					return;
				case 2:
					*(res++) = s2 = *(smp++);
					*(res++) = s1 = *(smp++);
					for (i = n - 2; i > 0; i--)
					{
						s0 = *(smp++);
						*(res++) = s0 - 2 * s1 + s2;
						s2 = s1;
						s1 = s0;
					}
					return;
				case 3:
					res[0] = smp[0];
					res[1] = smp[1];
					res[2] = smp[2];
					for (i = 3; i < n; i++)
					{
						res[i] = smp[i] - 3 * smp[i - 1] + 3 * smp[i - 2] - smp[i - 3];
					}
					return;
				case 4:
					res[0] = smp[0];
					res[1] = smp[1];
					res[2] = smp[2];
					res[3] = smp[3];
					for (i = 4; i < n; i++)
					{
						res[i] = smp[i] - 4 * smp[i - 1] + 6 * smp[i - 2] - 4 * smp[i - 3] + smp[i - 4];
					}
					return;
				default:
					return;
			}
		}

		public const uint UINT32_MAX = 0xffffffff;

		static unsafe uint calc_optimal_rice_params(RiceContext* rc, int porder, uint* sums, uint n, uint pred_order)
		{
			uint part = (1U << porder);
			uint all_bits = 0;			
			rc->rparams[0] = find_optimal_rice_param(sums[0], (n >> porder) - pred_order, out all_bits);
			uint cnt = (n >> porder);
			for (uint i = 1; i < part; i++)
			{
				uint nbits;
				rc->rparams[i] = find_optimal_rice_param(sums[i], cnt, out nbits);
				all_bits += nbits;
			}
			all_bits += (4 * part);
			rc->porder = porder;
			return all_bits;
		}

		static unsafe void calc_sums(int pmin, int pmax, uint* data, uint n, uint pred_order, uint* sums)
		{
			uint* res = &data[pred_order];

			// sums for highest level
			int parts = (1 << pmax);
			uint cnt = (n >> pmax) - pred_order;
			for (int i = 0; i < parts; i++)
			{
				if (i == 1) cnt = (n >> pmax);
				if (i > 0) res = &data[i * cnt];
				sums[pmax * Flake.MAX_PARTITIONS + i] = 0;
				for (int j = 0; j < cnt; j++)
				{
					sums[pmax * Flake.MAX_PARTITIONS + i] += res[j];
				}
			}
			// sums for lower levels
			for (int i = pmax - 1; i >= pmin; i--)
			{
				parts = (1 << i);
				for (int j = 0; j < parts; j++)
				{
					sums[i * Flake.MAX_PARTITIONS + j] = 
						sums[(i + 1) * Flake.MAX_PARTITIONS + 2 * j] + 
						sums[(i + 1) * Flake.MAX_PARTITIONS + 2 * j + 1];
				}
			}
		}

		static unsafe uint calc_rice_params(RiceContext* rc, int pmin, int pmax, int* data, uint n, uint pred_order)
		{
			RiceContext tmp_rc;
			uint* udata = stackalloc uint[(int)n];
			uint* sums = stackalloc uint[(pmax + 1) * Flake.MAX_PARTITIONS];
			//uint* bits = stackalloc uint[Flake.MAX_PARTITION_ORDER];

			//assert(pmin >= 0 && pmin <= Flake.MAX_PARTITION_ORDER);
			//assert(pmax >= 0 && pmax <= Flake.MAX_PARTITION_ORDER);
			//assert(pmin <= pmax);

			for (uint i = 0; i < n; i++)
				udata[i] = (uint) ((2 * data[i]) ^ (data[i] >> 31));

			calc_sums(pmin, pmax, udata, n, pred_order, sums);

			int opt_porder = pmin;
			uint opt_bits = UINT32_MAX;
			for (int i = pmin; i <= pmax; i++)
			{
				uint bits = calc_optimal_rice_params(&tmp_rc, i, sums + i * Flake.MAX_PARTITIONS, n, pred_order);
				if (bits <= opt_bits)
				{
					opt_porder = i;
					opt_bits = bits;
					*rc = tmp_rc;
				}
			}

			return opt_bits;
		}

		static int get_max_p_order(int max_porder, int n, int order)
		{
			int porder = Math.Min(max_porder, Flake.log2i(n ^ (n - 1)));
			if (order > 0)
				porder = Math.Min(porder, Flake.log2i(n / order));
			return porder;
		}

		static unsafe uint calc_rice_params_fixed(RiceContext* rc, int pmin, int pmax,
			int* data, int n, int pred_order, uint bps)
		{
			pmin = get_max_p_order(pmin, n, pred_order);
			pmax = get_max_p_order(pmax, n, pred_order);
			uint bits = (uint)pred_order * bps + 6;
			bits += calc_rice_params(rc, pmin, pmax, data, (uint)n, (uint)pred_order);
			return bits;
		}

		static unsafe uint calc_rice_params_lpc(RiceContext* rc, int pmin, int pmax,
			int* data, int n, int pred_order, uint bps, uint precision)
		{
			pmin = get_max_p_order(pmin, n, pred_order);
			pmax = get_max_p_order(pmax, n, pred_order);
			uint bits = (uint)pred_order * bps + 4 + 5 + (uint)pred_order * precision + 6;
			bits += calc_rice_params(rc, pmin, pmax, data, (uint)n, (uint)pred_order);
			return bits;
		}

		unsafe void choose_best_subframe(FlacFrame* frame, int ch)
		{
			if (frame->current.size < frame->subframes[ch].size)
			{
				FlacSubframe tmp = frame->subframes[ch];
				frame->subframes[ch] = frame->current;
				for (int iWindow = 0; iWindow < lpc.MAX_LPC_WINDOWS; iWindow++)
					frame->subframes[ch].done_lpcs[iWindow] = tmp.done_lpcs[iWindow];
				frame->subframes[ch].done_fixed = tmp.done_fixed;
				frame->current = tmp;
			}
		}

		unsafe void encode_residual_lpc_sub(FlacFrame* frame, double* lpcs, int iWindow, int order, int ch)
		{
			if ((frame->subframes[ch].done_lpcs[iWindow] & (1U << (order - 1))) != 0)
				return; // already calculated;

			frame->subframes[ch].done_lpcs[iWindow] |= (1U << (order - 1));

			frame->current.type = SubframeType.LPC;
			frame->current.order = order;
			frame->current.window = iWindow;

			lpc.quantize_lpc_coefs(lpcs + (frame->current.order - 1) * lpc.MAX_LPC_ORDER,
				frame->current.order, lpc_precision, frame->current.coefs, out frame->current.shift);

			lpc.encode_residual(frame->current.residual, frame->current.samples, frame->blocksize, frame->current.order, frame->current.coefs, frame->current.shift);

			frame->current.size = calc_rice_params_lpc(&frame->current.rc, eparams.min_partition_order, eparams.max_partition_order,
				frame->current.residual, frame->blocksize, frame->current.order, frame->current.obits, lpc_precision);

			choose_best_subframe(frame, ch);
		}

		unsafe void encode_residual_fixed_sub(FlacFrame* frame, int order, int ch)
		{
			if ((frame->subframes[ch].done_fixed & (1U << order)) != 0)
				return; // already calculated;

			frame->current.order = order;
			frame->current.type = SubframeType.Fixed;

			encode_residual_fixed(frame->current.residual, frame->current.samples, frame->blocksize, frame->current.order);

			frame->current.size = calc_rice_params_fixed(&frame->current.rc, eparams.min_partition_order, eparams.max_partition_order,
				frame->current.residual, frame->blocksize, frame->current.order, frame->current.obits);

			frame->subframes[ch].done_fixed |= (1U << order);

			choose_best_subframe(frame, ch);
		}

		unsafe void encode_residual(FlacFrame* frame, int ch, PredictionType predict, OrderMethod omethod)
		{
			int i;
			FlacSubframe* sub = frame->subframes + ch;
			int* smp = sub->samples;
			int n = frame->blocksize;
			
			frame->current.samples = sub->samples;
			frame->current.obits = sub->obits;
			frame->current.wbits = sub->wbits;

			// CONSTANT
			for (i = 1; i < n; i++)
			{
				if (smp[i] != smp[0]) break;
			}
			if (i == n)
			{
				sub->type = SubframeType.Constant;
				sub->residual[0] = smp[0];
				sub->size = sub->obits;
				return;
			}

			// VERBATIM
			frame->current.type = SubframeType.Verbatim;
			frame->current.size = frame->current.obits * (uint)frame->blocksize;
			choose_best_subframe(frame, ch);

			if (n < 5 || predict == PredictionType.None)
				return;

			// FIXED
			if (predict == PredictionType.Fixed ||
				predict == PredictionType.Search ||
				(predict == PredictionType.Estimated && sub->type == SubframeType.Fixed) ||
				n <= eparams.max_prediction_order)
			{
				int max_fixed_order = Math.Min(eparams.max_fixed_order, 4);
				int min_fixed_order = Math.Min(eparams.min_fixed_order, max_fixed_order);

				for (i = min_fixed_order; i <= max_fixed_order; i++)
					encode_residual_fixed_sub(frame, i, ch);
			}

			// LPC
			if (n > eparams.max_prediction_order &&
			   (predict == PredictionType.Levinson ||
				predict == PredictionType.Search ||
				(predict == PredictionType.Estimated && sub->type == SubframeType.LPC)))
			{
				double* lpcs = stackalloc double[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
				int min_order = eparams.min_prediction_order;
				int max_order = eparams.max_prediction_order;

				fixed (double* window = windowBuffer)
					for (int iWindow = 0; iWindow < _windowcount; iWindow++)
					{
						if (predict == PredictionType.Estimated && sub->window != iWindow)
							continue;
						
						int est_order = (int)lpc.calc_coefs(smp, (uint)n, (uint)max_order, omethod, lpcs,
							window + iWindow * _windowsize);

						switch (omethod)
						{
							case OrderMethod.Max:
								// always use maximum order
								encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
								break;
							case OrderMethod.Estimate:
								// estimated order
								encode_residual_lpc_sub(frame, lpcs, iWindow, est_order, ch);
								break;
							case OrderMethod.Search:
								// brute-force optimal order search
								for (i = max_order; i > 0; i--)
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
								break;
							case OrderMethod.EstSearch:
								// brute-force search starting from estimate
								for (i = est_order; i > 0; i--)
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
								break;
							case OrderMethod.LogFast:
								// Try max, est, 32,16,8,4,2,1
								encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
								//encode_residual_lpc_sub(frame, lpcs, est_order, ch);
								for (i = lpc.MAX_LPC_ORDER; i > 0; i >>= 1)
									if (i < max_order)
										encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
								break;
							case OrderMethod.LogSearch:
								// do LogFast first
								encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
								//encode_residual_lpc_sub(frame, lpcs, est_order, ch);
								for (i = lpc.MAX_LPC_ORDER; i > 0; i >>= 1)
									if (i < max_order)
										encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
								// if found a good order, try to search around it
								if (frame->subframes[ch].type == SubframeType.LPC)
								{
									// log search (written by Michael Niedermayer for FFmpeg)
									for (int step = lpc.MAX_LPC_ORDER; step > 0; step >>= 1)
									{
										int last = frame->subframes[ch].order;
										if (step <= (last + 1) / 2)
											for (i = last - step; i <= last + step; i += step)
												if (i >= min_order && i <= max_order)
													encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
									}
								}
								break;
							default:
								throw new Exception("unknown ordermethod");
						}
					}
			}
		}

		unsafe void output_frame_header(FlacFrame* frame, BitWriter bitwriter)
		{
			bitwriter.writebits(16, 0xFFF8);
			bitwriter.writebits(4, frame->bs_code0);
			bitwriter.writebits(4, sr_code0);
			if (frame->ch_mode == ChannelMode.NotStereo)
				bitwriter.writebits(4, ch_code);
			else
				bitwriter.writebits(4, (int) frame->ch_mode);
			bitwriter.writebits(3, bps_code);
			bitwriter.writebits(1, 0);
			bitwriter.write_utf8(frame_count);

			// custom block size
			if (frame->bs_code1 >= 0)
			{
				if (frame->bs_code1 < 256)
					bitwriter.writebits(8, frame->bs_code1);
				else
					bitwriter.writebits(16, frame->bs_code1);
			}

			// custom sample rate
			if (sr_code1 > 0)
			{
				if (sr_code1 < 256)
					bitwriter.writebits(8, sr_code1);
				else
					bitwriter.writebits(16, sr_code1);
			}

			// CRC-8 of frame header
			bitwriter.flush();
			byte crc = crc8.ComputeChecksum(frame_buffer, 0, bitwriter.Length);
			bitwriter.writebits(8, crc);
		}

		unsafe void output_residual(FlacFrame* frame, BitWriter bitwriter, FlacSubframe* sub)
		{
			// rice-encoded block
			bitwriter.writebits(2, 0);

			// partition order
			int porder = sub->rc.porder;
			int psize = frame->blocksize >> porder;
			//assert(porder >= 0);
			bitwriter.writebits(4, porder);
			int res_cnt = psize - sub->order;

			// residual
			int j = sub->order;
			for (int p = 0; p < (1 << porder); p++)
			{
				int k = (int) sub->rc.rparams[p];
				bitwriter.writebits(4, k);
				if (p == 1) res_cnt = psize;
				for (int i = 0; i < res_cnt && j < frame->blocksize; i++, j++)
				{
					bitwriter.write_rice_signed(k, sub->residual[j]);
				}
			}
		}

		unsafe void 
		output_subframe_constant(FlacFrame* frame, BitWriter bitwriter, FlacSubframe* sub)
		{
			bitwriter.writebits_signed(sub->obits, sub->residual[0]);
		}

		unsafe void
		output_subframe_verbatim(FlacFrame* frame, BitWriter bitwriter, FlacSubframe* sub)
		{
			int n = frame->blocksize;
			for (int i = 0; i < n; i++)
				bitwriter.writebits_signed(sub->obits, sub->samples[i]); 
			// Don't use residual here, because we don't copy samples to residual for verbatim frames.
		}

		unsafe void
		output_subframe_fixed(FlacFrame* frame, BitWriter bitwriter, FlacSubframe* sub)
		{
			// warm-up samples
			for (int i = 0; i < sub->order; i++)
				bitwriter.writebits_signed(sub->obits, sub->residual[i]);

			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void
		output_subframe_lpc(FlacFrame* frame, BitWriter bitwriter, FlacSubframe* sub)
		{
			// warm-up samples
			for (int i = 0; i < sub->order; i++)
				bitwriter.writebits_signed(sub->obits, sub->residual[i]);

			// LPC coefficients
			uint cbits = lpc_precision;
			bitwriter.writebits(4, cbits - 1);
			bitwriter.writebits_signed(5, sub->shift);
			for (int i = 0; i < sub->order; i++)
				bitwriter.writebits_signed(cbits, sub->coefs[i]);
			
			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void output_subframes(FlacFrame* frame, BitWriter bitwriter)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				FlacSubframe* sub = frame->subframes + ch;
				// subframe header
				int type_code = (int) sub->type;
				if (sub->type == SubframeType.Fixed)
					type_code |= sub->order;
				if (sub->type == SubframeType.LPC)
					type_code |= sub->order - 1;
				bitwriter.writebits(1, 0);
				bitwriter.writebits(6, type_code);
				bitwriter.writebits(1, sub->wbits != 0 ? 1 : 0);
				if (sub->wbits > 0)
					bitwriter.writebits((int)sub->wbits, 1);

				// subframe
				switch (sub->type)
				{
					case SubframeType.Constant:
						output_subframe_constant(frame, bitwriter, sub);
						break;
					case SubframeType.Verbatim:
						output_subframe_verbatim(frame, bitwriter, sub);
						break;
					case SubframeType.Fixed:
						output_subframe_fixed(frame, bitwriter, sub);
						break;
					case SubframeType.LPC:
						output_subframe_lpc(frame, bitwriter, sub);
						break;
				}
			}
		}

		void output_frame_footer(BitWriter bitwriter)
		{
			bitwriter.flush();
			ushort crc = crc16.ComputeChecksum(frame_buffer, 0, bitwriter.Length);
			bitwriter.writebits(16, crc);
			bitwriter.flush();
		}

		unsafe void window_welch(double* window, int L)
		{
			int N = L - 1;
			double N2 = (double)N / 2.0;

			for (int n = 0; n <= N; n++)
			{
			    double k = 1 / N2 - 1.0 - Math.Min(n, N-n);
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
			int Np = (int) (p / 2.0 * L) - 1;
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

		unsafe int encode_frame()
		{
			FlacFrame* frame = stackalloc FlacFrame[1];
			FlacSubframe* sf = stackalloc FlacSubframe[channels * 3];

			fixed (int* s = samplesBuffer, r = residualBuffer)
			{
				frame->subframes = sf;
				init_frame(frame);
				int* current_residual = stackalloc int[frame->blocksize * 3];
				bool alreadyEncoded = false;
				if (frame->blocksize != _windowsize && frame->blocksize > 4)
					fixed (double* window = windowBuffer)
					{
						_windowsize = frame->blocksize;
						_windowcount = 0;
						if ((eparams.window_function & WindowFunction.Welch) != 0)
							window_welch(window + (_windowcount++)*_windowsize, _windowsize);
						if ((eparams.window_function & WindowFunction.Tukey) != 0)
							window_tukey(window + (_windowcount++) * _windowsize, _windowsize);
						if ((eparams.window_function & WindowFunction.Hann) != 0)
							window_hann(window + (_windowcount++) * _windowsize, _windowsize);
						if ((eparams.window_function & WindowFunction.Flattop) != 0)
							window_flattop(window + (_windowcount++) * _windowsize, _windowsize);
						if (_windowcount == 0)
							throw new Exception("invalid windowfunction");
					}

				if (channels != 2 || frame->blocksize <= 32 || eparams.stereo_method == StereoMethod.Independent)
				{
					frame->ch_mode = channels != 2 ? ChannelMode.NotStereo : ChannelMode.LeftRight;
					frame->current.residual = current_residual;
					for (int ch = 0; ch < channels; ch++)
						initialize_subframe(frame, ch, s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, bits_per_sample);
				}
				else
				{
					FlacFrame* frameM = stackalloc FlacFrame[1];
					FlacFrame* frameS = stackalloc FlacFrame[1];
					init_frame(frameS);
					frameS->subframes = sf + channels;
					frameS->ch_mode = ChannelMode.LeftRight;
					frameS->current.residual = current_residual;
					init_frame(frameM);
					frameM->subframes = sf + channels * 2;
					frameM->ch_mode = ChannelMode.MidSide;
					frameM->current.residual = current_residual + frame->blocksize;

					channel_decorrelation(s, s + Flake.MAX_BLOCKSIZE, s + 2 * Flake.MAX_BLOCKSIZE, s + 3 * Flake.MAX_BLOCKSIZE, frame->blocksize);

					for (int ch = 0; ch < channels; ch++)
					{
						initialize_subframe(frameS, ch, s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, bits_per_sample);
						initialize_subframe(frameM, ch, s + (channels + ch) * Flake.MAX_BLOCKSIZE, r + (channels + ch) * Flake.MAX_BLOCKSIZE, bits_per_sample + (uint)ch);
					}

					uint bitsBest = UINT32_MAX;
					ChannelMode modeBest = ChannelMode.LeftRight;

					if (eparams.stereo_method == StereoMethod.Estimate)
					{
						frameM->subframes[0].size = (uint)frame->blocksize * 32 + calc_decorr_score(frameM, 0, frameM->subframes + 0);
						frameM->subframes[1].size = (uint)frame->blocksize * 32 + calc_decorr_score(frameM, 1, frameM->subframes + 1);
						frameS->subframes[0].size = (uint)frame->blocksize * 32 + calc_decorr_score(frameS, 0, frameS->subframes + 0);
						frameS->subframes[1].size = (uint)frame->blocksize * 32 + calc_decorr_score(frameS, 1, frameS->subframes + 1);
					}
					else if (eparams.stereo_method == StereoMethod.Estimate2)
					{
						int max_prediction_order = eparams.max_prediction_order;
						int max_window = _windowcount;
						eparams.max_prediction_order /= 2;
						_windowcount = 1;
						encode_residual(frameM, 0, PredictionType.Levinson, OrderMethod.Estimate);
						encode_residual(frameM, 1, PredictionType.Levinson, OrderMethod.Estimate);
						encode_residual(frameS, 0, PredictionType.Levinson, OrderMethod.Estimate);
						encode_residual(frameS, 1, PredictionType.Levinson, OrderMethod.Estimate);
						_windowcount = max_window;
						eparams.max_prediction_order = max_prediction_order;
					}
					else if (eparams.stereo_method == StereoMethod.Estimate3)
					{
						int max_fixed_order = eparams.max_fixed_order;
						eparams.max_fixed_order = 2;
						encode_residual(frameM, 0, PredictionType.Fixed, OrderMethod.Estimate);
						encode_residual(frameM, 1, PredictionType.Fixed, OrderMethod.Estimate);
						encode_residual(frameS, 0, PredictionType.Fixed, OrderMethod.Estimate);
						encode_residual(frameS, 1, PredictionType.Fixed, OrderMethod.Estimate);
						eparams.max_fixed_order = max_fixed_order;
					}
					else if (eparams.stereo_method == StereoMethod.Estimate4)
					{
						int max_prediction_order = eparams.max_prediction_order;
						int max_fixed_order = eparams.max_fixed_order;
						int min_fixed_order = eparams.min_fixed_order;
						eparams.min_fixed_order = 2;
						eparams.max_fixed_order = 2;
						eparams.max_prediction_order = Math.Min(eparams.max_prediction_order, 8);
						encode_residual(frameM, 0, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameM, 1, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameS, 0, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameS, 1, eparams.prediction_type, OrderMethod.Estimate);
						eparams.min_fixed_order = min_fixed_order;
						eparams.max_fixed_order = max_fixed_order;
						eparams.max_prediction_order = max_prediction_order;
					}
					else if (eparams.stereo_method == StereoMethod.Estimate5)
					{
						int max_prediction_order = eparams.max_prediction_order;
						int max_fixed_order = eparams.max_fixed_order;
						int min_fixed_order = eparams.min_fixed_order;
						eparams.min_fixed_order = 2;
						eparams.max_fixed_order = 2;
						eparams.max_prediction_order = Math.Min(eparams.max_prediction_order, 12);
						encode_residual(frameM, 0, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameM, 1, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameS, 0, eparams.prediction_type, OrderMethod.Estimate);
						encode_residual(frameS, 1, eparams.prediction_type, OrderMethod.Estimate);
						eparams.min_fixed_order = min_fixed_order;
						eparams.max_fixed_order = max_fixed_order;
						eparams.max_prediction_order = max_prediction_order;
					}
					else // StereoMethod.Search
					{
						encode_residual(frameM, 0, eparams.prediction_type, eparams.order_method);
						encode_residual(frameM, 1, eparams.prediction_type, eparams.order_method);
						encode_residual(frameS, 0, eparams.prediction_type, eparams.order_method);
						encode_residual(frameS, 1, eparams.prediction_type, eparams.order_method);
						alreadyEncoded = true;
					}

					if (bitsBest > frameM->subframes[0].size + frameM->subframes[1].size)
					{
						bitsBest = frameM->subframes[0].size + frameM->subframes[1].size;
						modeBest = ChannelMode.MidSide;
						frame->subframes[0] = frameM->subframes[0];
						frame->subframes[1] = frameM->subframes[1];
					}
					if (bitsBest > frameM->subframes[1].size + frameS->subframes[1].size)
					{
						bitsBest = frameM->subframes[1].size + frameS->subframes[1].size;
						modeBest = ChannelMode.RightSide;
						frame->subframes[0] = frameM->subframes[1];
						frame->subframes[1] = frameS->subframes[1];
					}
					if (bitsBest > frameM->subframes[1].size + frameS->subframes[0].size)
					{
						bitsBest = frameM->subframes[1].size + frameS->subframes[0].size;
						modeBest = ChannelMode.LeftSide;
						frame->subframes[0] = frameS->subframes[0];
						frame->subframes[1] = frameM->subframes[1];
					}
					if (bitsBest > frameS->subframes[0].size + frameS->subframes[1].size)
					{
						bitsBest = frameS->subframes[0].size + frameS->subframes[1].size;
						modeBest = ChannelMode.LeftRight;
						frame->subframes[0] = frameS->subframes[0];
						frame->subframes[1] = frameS->subframes[1];
					}
					frame->ch_mode = modeBest;
					frame->current.residual = current_residual + 2 * frame->blocksize;
					if (eparams.stereo_method == StereoMethod.Estimate4 || eparams.stereo_method == StereoMethod.Estimate5)
					{
						encode_residual(frame, 0, PredictionType.Estimated, eparams.order_method);
						encode_residual(frame, 1, PredictionType.Estimated, eparams.order_method);
						alreadyEncoded = true;
					}
				}

				if (!alreadyEncoded)
					for (int ch = 0; ch < channels; ch++)
						encode_residual(frame, ch, eparams.prediction_type, eparams.order_method);

				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);

				output_frame_header(frame, bitwriter);
				output_subframes(frame, bitwriter);
				output_frame_footer(bitwriter);

				if (frame_buffer != null)
				{
					if (eparams.variable_block_size > 0)
						frame_count += eparams.block_size;
					else
						frame_count++;
				}
				return bitwriter.Length;
			}
		}

		unsafe void output_frame()
		{
			if (verify != null)
			{
				fixed (int* s = verifyBuffer, r = samplesBuffer)
					for (int ch = 0; ch < channels; ch++)
						Flake.memcpy(s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, eparams.block_size);
			}

			int fs;
			if (0 != eparams.variable_block_size && 0 == (eparams.block_size & 7) && eparams.block_size >= 128)
				fs = encode_frame_vbs();
			else
				fs = encode_frame();

			_position += eparams.block_size;
			_IO.Write(frame_buffer, 0, fs);
			_totalSize += fs;

			if (verify != null)
			{
				int decoded = verify.DecodeFrame(frame_buffer, 0, fs);
				if (decoded != fs || verify.Remaining != (ulong)eparams.block_size)
					throw new Exception("validation failed!");
				fixed (int* s = verifyBuffer, r = verify.Samples)
				{
					for (int ch = 0; ch < channels; ch++)
						if (Flake.memcmp(s + ch * Flake.MAX_BLOCKSIZE, r +ch * Flake.MAX_BLOCKSIZE, decoded))
							throw new Exception("validation failed!");
				}
			}
		}

		public void Write(int[,] buff, int pos, int sampleCount)
		{
			if (!inited)
			{
				int header_size = flake_encode_init();
				if (_IO == null)
					_IO = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read);
				_IO.Write(header, 0, header_size);
				inited = true;
			}

			int len = sampleCount;
			while (len > 0)
			{
				int block = Math.Min(len, eparams.block_size - samplesInBuffer);

				copy_samples(buff, pos, block);

				AudioSamples.FLACSamplesToBytes(buff, pos, frame_buffer, 0, block, channels, (int)bits_per_sample);
				md5.TransformBlock(frame_buffer, 0, block * channels * ((int)bits_per_sample >> 3), null, 0);

				if (samplesInBuffer < eparams.block_size)
					return;

				output_frame();

				samplesInBuffer = 0;
				len -= block;
				pos += block;
			}
		}

		public string Path { get { return _path; } }

		int[] flac_samplerates;
		int[] flac_bitdepths;
		int[] flac_blocksizes;
		string vendor_string = "Flake#0.1";

		int select_blocksize(int samplerate, int time_ms)
		{
			int blocksize = flac_blocksizes[1];
			int target = (samplerate * time_ms) / 1000;
			for (int i = 0; i < flac_blocksizes.Length; i++)
				if (target >= flac_blocksizes[i] && flac_blocksizes[i] > blocksize)
				{
					blocksize = flac_blocksizes[i];
				}
			return blocksize;
		}

		void write_streaminfo(byte[] header, int pos, int last)
		{
			Array.Clear(header, pos, 38);
			BitWriter bitwriter = new BitWriter(header, pos, 38);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, 0);
			bitwriter.writebits(24, 34);

			if (eparams.variable_block_size > 0)
				bitwriter.writebits(16, 0);
			else
				bitwriter.writebits(16, eparams.block_size);

			bitwriter.writebits(16, eparams.block_size);
			bitwriter.writebits(24, 0);
			bitwriter.writebits(24, max_frame_size);
			bitwriter.writebits(20, sample_rate);
			bitwriter.writebits(3, channels - 1);
			bitwriter.writebits(5, bits_per_sample - 1);

			// total samples
			if (sample_count > 0)
			{
				bitwriter.writebits(4, 0);
				bitwriter.writebits(32, sample_count);
			}
			else
			{
				bitwriter.writebits(4, 0);
				bitwriter.writebits(32, 0);
			}
			bitwriter.flush();
		}

		/**
		 * Write vorbis comment metadata block to byte array.
		 * Just writes the vendor string for now.
	     */
		int write_vorbis_comment(byte[] comment, int pos, int last)
		{
			BitWriter bitwriter = new BitWriter(comment, pos, 4);
			Encoding enc = new ASCIIEncoding();
			int vendor_len = enc.GetBytes(vendor_string, 0, vendor_string.Length, comment, pos + 8);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, 4);
			bitwriter.writebits(24, vendor_len + 8);

			comment[pos + 4] = (byte)(vendor_len & 0xFF);
			comment[pos + 5] = (byte)((vendor_len >> 8) & 0xFF);
			comment[pos + 6] = (byte)((vendor_len >> 16) & 0xFF);
			comment[pos + 7] = (byte)((vendor_len >> 24) & 0xFF);
			comment[pos + 8 + vendor_len] = 0;
			comment[pos + 9 + vendor_len] = 0;
			comment[pos + 10 + vendor_len] = 0;
			comment[pos + 11 + vendor_len] = 0;
			bitwriter.flush();
			return vendor_len + 12;
		}

		/**
		 * Write padding metadata block to byte array.
		 */
		int
		write_padding(byte[] padding, int pos, int last, int padlen)
		{
			BitWriter bitwriter = new BitWriter(padding, pos, 4);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, 1);
			bitwriter.writebits(24, padlen);

			return padlen + 4;
		}

		int write_headers()
		{
			int header_size = 0;
			int last = 0;

			// stream marker
			header[0] = 0x66;
			header[1] = 0x4C;
			header[2] = 0x61;
			header[3] = 0x43;
			header_size += 4;

			// streaminfo
			write_streaminfo(header, header_size, last);
			header_size += 38;

			// vorbis comment
			if (eparams.padding_size == 0) last = 1;
			header_size += write_vorbis_comment(header, header_size, last);

			// padding
			if (eparams.padding_size > 0)
			{
				last = 1;
				header_size += write_padding(header, header_size, last, eparams.padding_size);
			}

			return header_size;
		}

		int flake_encode_init()
		{
			int i, header_len;

			//if(flake_validate_params(s) < 0)

			ch_code = channels - 1;

			// find samplerate in table
			for (i = 4; i < 12; i++)
			{
				if (sample_rate == flac_samplerates[i])
				{
					sr_code0 = i;
					break;
				}
			}

			// if not in table, samplerate is non-standard
			if (i == 12)
				throw new Exception("non-standard samplerate");

			for (i = 1; i < 8; i++)
			{
				if (bits_per_sample == flac_bitdepths[i])
				{
					bps_code = i;
					break;
				}
			}
			if (i == 8)
				throw new Exception("non-standard bps");
			// FIXME: For now, only 16-bit encoding is supported
			if (bits_per_sample != 16)
				throw new Exception("non-standard bps");

			if (eparams.block_size == 0)
				if (_blocksize == 0)
					eparams.block_size = select_blocksize(sample_rate, eparams.block_time_ms);
				else
					eparams.block_size = _blocksize;

			// select LPC precision based on block size
			if (eparams.block_size <= 192) lpc_precision = 7U;
			else if (eparams.block_size <= 384) lpc_precision = 8U;
			else if (eparams.block_size <= 576) lpc_precision = 9U;
			else if (eparams.block_size <= 1152) lpc_precision = 10U;
			else if (eparams.block_size <= 2304) lpc_precision = 11U;
			else if (eparams.block_size <= 4608) lpc_precision = 12U;
			else if (eparams.block_size <= 8192) lpc_precision = 13U;
			else if (eparams.block_size <= 16384) lpc_precision = 14U;
			else lpc_precision = 15;

			// set maximum encoded frame size (if larger, re-encodes in verbatim mode)
			if (channels == 2)
				max_frame_size = 16 + ((eparams.block_size * (int)(bits_per_sample + bits_per_sample + 1) + 7) >> 3);
			else
				max_frame_size = 16 + ((eparams.block_size * channels * (int)bits_per_sample + 7) >> 3);

			// output header bytes
			header = new byte[eparams.padding_size + 1024];
			header_len = write_headers();

			// initialize CRC & MD5
			//crc_init();
			//md5_init(&ctx->md5ctx);

			frame_buffer = new byte[max_frame_size];

			return header_len;
		}
	}

	struct FlakeEncodeParams
	{
		// compression quality
		// set by user prior to calling flake_encode_init
		// standard values are 0 to 8
		// 0 is lower compression, faster encoding
		// 8 is higher compression, slower encoding
		// extended values 9 to 12 are slower and/or use
		// higher prediction orders
		public int compression;

		// prediction order selection method
		// set by user prior to calling flake_encode_init
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
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 2
		// 0 = independent L+R channels
		// 1 = mid-side encoding
		public StereoMethod stereo_method;

		// block size in samples
		// set by the user prior to calling flake_encode_init
		// if set to 0, a block size is chosen based on block_time_ms
		// can also be changed by user before encoding a frame
		public int block_size;

		// block time in milliseconds
		// set by the user prior to calling flake_encode_init
		// used to calculate block_size based on sample rate
		// can also be changed by user before encoding a frame
		public int block_time_ms;

		// padding size in bytes
		// set by the user prior to calling flake_encode_init
		// if set to less than 0, defaults to 4096
		public int padding_size;

		// maximum encoded frame size
		// this is set by flake_encode_init based on input audio format
		// it can be used by the user to allocate an output buffer
		public int max_frame_size;

		// minimum LPC order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 1 to 32
		public int min_prediction_order;

		// maximum LPC order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 1 to 32 
		public int max_prediction_order;

		// minimum fixed prediction order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 4
		public int min_fixed_order;

		// maximum fixed prediction order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 4
		public int max_fixed_order;

		// type of linear prediction
		// set by user prior to calling flake_encode_init
		public PredictionType prediction_type;

		// minimum partition order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 8
		public int min_partition_order;

		// maximum partition order
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 8
		public int max_partition_order;

		// whether to use variable block sizes
		// set by user prior to calling flake_encode_init
		// 0 = fixed block size
		// 1 = variable block size
		public int variable_block_size;


		public WindowFunction window_function;

		public int flake_set_defaults(int lvl)
		{
			compression = lvl;

			if ((lvl < 0 || lvl > 12) && (lvl != 99))
			{
				return -1;
			}

			// default to level 5 params
			window_function = WindowFunction.Flattop | WindowFunction.Tukey;
			order_method = OrderMethod.Estimate;
			stereo_method = StereoMethod.Estimate;
			block_size = 0;
			block_time_ms = 105;			
			prediction_type = PredictionType.Search;
			min_prediction_order = 1;
			max_prediction_order = 12;
			min_fixed_order = 0;
			max_fixed_order = 4;
			min_partition_order = 0;
			max_partition_order = 6;
			variable_block_size = 0;

			// differences from level 5
			switch (lvl)
			{
				case 0:
					stereo_method = StereoMethod.Independent;
					block_time_ms = 27;
					prediction_type = PredictionType.Fixed;
					min_fixed_order = 2;
					max_fixed_order = 2;
					min_partition_order = 4;
					max_partition_order = 4;
					break;
				case 1:
					block_time_ms = 27;
					prediction_type = PredictionType.Fixed;
					min_fixed_order = 2;
					max_fixed_order = 3;
					min_partition_order = 2;
					max_partition_order = 2;
					break;
				case 2:
					block_time_ms = 27;
					prediction_type = PredictionType.Fixed;
					//prediction_type = PredictionType.Fixed;
					min_fixed_order = 2;
					max_fixed_order = 4;
					//min_partition_order = 0;
					//max_partition_order = 3;
					break;
				case 3:
					prediction_type = PredictionType.Levinson;
					window_function = WindowFunction.Welch;
					max_prediction_order = 8;
					break;
				case 4:
					prediction_type = PredictionType.Levinson;
					window_function = WindowFunction.Welch;
					break;
				case 5:
					prediction_type = PredictionType.Levinson;
					break;
				case 6:
					stereo_method = StereoMethod.Estimate4;
					break;
				case 7:
					order_method = OrderMethod.LogSearch;
					stereo_method = StereoMethod.Estimate4;
					break;
				case 8:
					order_method = OrderMethod.Search;
					stereo_method = StereoMethod.Estimate4;
					break;
				case 9:
					stereo_method = StereoMethod.Estimate4;
					max_prediction_order = 32;
					break;
				case 10:
					order_method = OrderMethod.LogFast;
					stereo_method = StereoMethod.Estimate5;
					max_prediction_order = 32;
					break;
				case 11:
					order_method = OrderMethod.LogSearch;
					stereo_method = StereoMethod.Estimate5;
					max_prediction_order = 32;
					max_partition_order = 8;
					break;
				case 99:
					order_method = OrderMethod.Search;
					block_time_ms = 186;
					max_prediction_order = 32;
					max_partition_order = 8;
					variable_block_size = 2;
					break;
			}

			return 0;
		}
	}
}
