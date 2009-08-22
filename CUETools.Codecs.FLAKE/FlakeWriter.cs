using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Security.Cryptography;
//using System.Runtime.InteropServices;
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

		int frame_count = 0;

		long first_frame_offset = 0;

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

		SeekPoint[] seek_table;
		int seek_table_offset = -1;

		bool inited = false;

		public FlakeWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO)
		{
			if (bitsPerSample != 16)
				throw new Exception("Bits per sample must be 16.");
			if (channelCount != 2)
				throw new Exception("ChannelCount must be 2.");

			channels = channelCount;
			sample_rate = sampleRate;
			bits_per_sample = (uint) bitsPerSample;

			// flake_validate_params

			_path = path;
			_IO = IO;

			samplesBuffer = new int[Flake.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels)];
			residualBuffer = new int[Flake.MAX_BLOCKSIZE * (channels == 2 ? 10 : channels + 1)];
			windowBuffer = new double[Flake.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS];

			eparams.flake_set_defaults(_compressionLevel);
			eparams.padding_size = 8192;

			crc8 = new Crc8();
			crc16 = new Crc16();
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
				eparams.flake_set_defaults(_compressionLevel);
			}
		}

		//[DllImport("kernel32.dll")]
		//static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
		//[DllImport("kernel32.dll")]
		//static extern IntPtr GetCurrentThread();

		void DoClose()
		{
			if (inited)
			{
				while (samplesInBuffer > 0)
				{
					eparams.block_size = samplesInBuffer;
					output_frame();
				}

				if (_IO.CanSeek)
				{
					if (md5 != null)
					{
						md5.TransformFinalBlock(frame_buffer, 0, 0);
						_IO.Position = 26;
						_IO.Write(md5.Hash, 0, md5.Hash.Length);
					}

					if (seek_table != null)
					{
						_IO.Position = seek_table_offset;
						int len = write_seekpoints(header, 0, 0);
						_IO.Write(header, 4, len - 4);
					}
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

		public int MaxPrecisionSearch
		{
			get { return eparams.lpc_precision_search; }
			set
			{
				if (value < 0 || value > 1)
					throw new Exception("unsupported MaxPrecisionSearch value");
				eparams.lpc_precision_search = value;
			}
		}

		public WindowFunction WindowFunction
		{
			get { return eparams.window_function; }
			set { eparams.window_function = value; }
		}

		public bool DoMD5
		{
			get { return eparams.do_md5; }
			set { eparams.do_md5 = value; }
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

		public int VBRMode
		{
			get { return eparams.variable_block_size; }
			set { eparams.variable_block_size = value; }
		}

		public int MinPredictionOrder
		{
			get
			{
				return PredictionType == PredictionType.Fixed ?
					MinFixedOrder : MinLPCOrder;
			}
			set
			{
				if (PredictionType == PredictionType.Fixed)
					MinFixedOrder = value;
				else
					MinLPCOrder = value;
			}
		}

		public int MaxPredictionOrder
		{
			get 
			{
				return PredictionType == PredictionType.Fixed ?
					MaxFixedOrder : MaxLPCOrder; 
			}
			set
			{
				if (PredictionType == PredictionType.Fixed)
					MaxFixedOrder = value;
				else
					MaxLPCOrder = value;
			}
		}

		public int MinLPCOrder
		{
			get
			{
				return eparams.min_prediction_order;
			}
			set
			{
				if (value < 1 || value > eparams.max_prediction_order)
					throw new Exception("invalid MinLPCOrder " + value.ToString());
				eparams.min_prediction_order = value;
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
				if (value > 32 || value < eparams.min_prediction_order)
					throw new Exception("invalid MaxLPCOrder " + value.ToString());
				eparams.max_prediction_order = value;
			}
		}

		public int MinFixedOrder
		{
			get
			{
				return eparams.min_fixed_order;
			}
			set
			{
				if (value < 0 || value > eparams.max_fixed_order)
					throw new Exception("invalid MinFixedOrder " + value.ToString());
				eparams.min_fixed_order = value;
			}
		}

		public int MaxFixedOrder
		{
			get
			{
				return eparams.max_fixed_order;
			}
			set
			{
				if (value > 4 || value < eparams.min_fixed_order)
					throw new Exception("invalid MaxFixedOrder " + value.ToString());
				eparams.max_fixed_order = value;
			}
		}

		public int MinPartitionOrder
		{
			get { return eparams.min_partition_order; }
			set
			{
				if (value < 0 || value > eparams.max_partition_order)
					throw new Exception("invalid MinPartitionOrder " + value.ToString());
				eparams.min_partition_order = value;
			}
		}

		public int MaxPartitionOrder
		{
			get { return eparams.max_partition_order; }
			set
			{
				if (value > 8 || value < eparams.min_partition_order)
					throw new Exception("invalid MaxPartitionOrder " + value.ToString());
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

		unsafe uint get_wasted_bits(int* signal, int samples)
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

			return (uint)shift;
		}

		unsafe void init_frame(FlacFrame * frame, int bs)
		{
			//if (channels == 2)
			//max_frame_size =			

			int i = 15;
			if (eparams.variable_block_size == 0)
			{
				for (i = 0; i < 15; i++)
				{
					if (bs == Flake.flac_blocksizes[i])
					{
						frame->blocksize = Flake.flac_blocksizes[i];
						frame->bs_code0 = i;
						frame->bs_code1 = -1;
						break;
					}
				}
			}
			if (i == 15)
			{
				frame->blocksize = bs;
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

		/// Copy channel-interleaved input samples into separate subframes
		unsafe void copy_samples(int[,] samples, int pos, int block)
		{
			fixed (int* fsamples = samplesBuffer, src = &samples[pos, 0])
			{
				if (channels == 2)
					Flake.deinterlace(fsamples + samplesInBuffer, fsamples + Flake.MAX_BLOCKSIZE + samplesInBuffer, src, block);
				else
					for (int ch = 0; ch < channels; ch++)
					{
						int* psamples = fsamples + ch * Flake.MAX_BLOCKSIZE + samplesInBuffer;
						for (int i = 0; i < block; i++)
							psamples[i] = src[i * channels + ch];
					}
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

		static unsafe int find_optimal_rice_param(uint sum, uint n, out uint nbits_best)
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
			return k_opt;
		}

		unsafe uint calc_decorr_score(FlacFrame* frame, int ch)
		{
			int* s = frame->subframes[ch].samples;
			int n = frame->blocksize;
			ulong sum = 0;
			for (int i = 2; i < n; i++)
				sum += (ulong)Math.Abs(s[i] - 2 * s[i - 1] + s[i - 2]);
			uint nbits;
			find_optimal_rice_param((uint)(2 * sum), (uint)n, out nbits);
			return nbits;
		}

		unsafe void initialize_subframe(FlacFrame* frame, int ch, int *s, int * r, uint bps, uint w)
		{
			if (w > bps)
				throw new Exception("internal error");
			frame->subframes[ch].samples = s;
			frame->subframes[ch].obits = bps - w;
			frame->subframes[ch].wbits = w;
			frame->subframes[ch].best.residual = r;
			frame->subframes[ch].best.type = SubframeType.Verbatim;
			frame->subframes[ch].best.size = UINT32_MAX;
			for (int iWindow = 0; iWindow < 2 * lpc.MAX_LPC_WINDOWS; iWindow++)
				frame->subframes[ch].done_lpcs[iWindow] = 0;
			frame->subframes[ch].done_fixed = 0;
			for (int iWindow = 0; iWindow < lpc.MAX_LPC_WINDOWS; iWindow++)
				frame->subframes[ch].lpcs_order[iWindow] = 0;
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
			// sums for highest level
			int parts = (1 << pmax);
			uint* res = data + pred_order;
			uint cnt = (n >> pmax) - pred_order;
			uint sum = 0;
			for (uint j = cnt; j > 0; j--)
				sum += *(res++);
			sums[pmax * Flake.MAX_PARTITIONS + 0] = sum;
			cnt = (n >> pmax);
			for (int i = 1; i < parts; i++)
			{
				sum = 0;
				for (uint j = cnt; j > 0; j--)
					sum += *(res++);
				sums[pmax * Flake.MAX_PARTITIONS + i] = sum;
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
			if (frame->current.size < frame->subframes[ch].best.size)
			{
				FlacSubframe tmp = frame->subframes[ch].best;
				frame->subframes[ch].best = frame->current;
				frame->current = tmp;
			}
		}

		unsafe void encode_residual_lpc_sub(FlacFrame* frame, double * lpcs, int iWindow, int order, int ch)
		{
			// select LPC precision based on block size
			uint lpc_precision;
			if (frame->blocksize <= 192) lpc_precision = 7U;
			else if (frame->blocksize <= 384) lpc_precision = 8U;
			else if (frame->blocksize <= 576) lpc_precision = 9U;
			else if (frame->blocksize <= 1152) lpc_precision = 10U;
			else if (frame->blocksize <= 2304) lpc_precision = 11U;
			else if (frame->blocksize <= 4608) lpc_precision = 12U;
			else if (frame->blocksize <= 8192) lpc_precision = 13U;
			else if (frame->blocksize <= 16384) lpc_precision = 14U;
			else lpc_precision = 15;

			for (uint i_precision = 0; i_precision <= eparams.lpc_precision_search && lpc_precision + i_precision < 16; i_precision++)
				// check if we already calculated with this order, window and precision
				if ((frame->subframes[ch].done_lpcs[iWindow + i_precision * lpc.MAX_LPC_WINDOWS] & (1U << (order - 1))) == 0) 
				{
					frame->subframes[ch].done_lpcs[iWindow + i_precision * lpc.MAX_LPC_WINDOWS] |= (1U << (order - 1));

					uint cbits = lpc_precision + i_precision;

					frame->current.type = SubframeType.LPC;
					frame->current.order = order;
					frame->current.window = iWindow;

					lpc.quantize_lpc_coefs(lpcs + (frame->current.order - 1) * lpc.MAX_LPC_ORDER,
						frame->current.order, cbits, frame->current.coefs, out frame->current.shift);

					if (frame->current.shift < 0 || frame->current.shift > 15)
						throw new Exception("negative shift");

					ulong csum = 0;
					for (int i = frame->current.order; i > 0; i--)
						csum += (ulong)Math.Abs(frame->current.coefs[i - 1]);

					if ((csum << (int)frame->subframes[ch].obits) >= 1UL << 32)
						lpc.encode_residual_long(frame->current.residual, frame->subframes[ch].samples, frame->blocksize, frame->current.order, frame->current.coefs, frame->current.shift);
					else
						lpc.encode_residual(frame->current.residual, frame->subframes[ch].samples, frame->blocksize, frame->current.order, frame->current.coefs, frame->current.shift);

					frame->current.size = calc_rice_params_lpc(&frame->current.rc, eparams.min_partition_order, eparams.max_partition_order,
						frame->current.residual, frame->blocksize, frame->current.order, frame->subframes[ch].obits, cbits);

					choose_best_subframe(frame, ch);
				}
		}

		unsafe void encode_residual_fixed_sub(FlacFrame* frame, int order, int ch)
		{
			if ((frame->subframes[ch].done_fixed & (1U << order)) != 0)
				return; // already calculated;

			frame->current.order = order;
			frame->current.type = SubframeType.Fixed;

			encode_residual_fixed(frame->current.residual, frame->subframes[ch].samples, frame->blocksize, frame->current.order);

			frame->current.size = calc_rice_params_fixed(&frame->current.rc, eparams.min_partition_order, eparams.max_partition_order,
				frame->current.residual, frame->blocksize, frame->current.order, frame->subframes[ch].obits);

			frame->subframes[ch].done_fixed |= (1U << order);

			choose_best_subframe(frame, ch);
		}

		unsafe void encode_residual(FlacFrame* frame, int ch, PredictionType predict, OrderMethod omethod)
		{
			int* smp = frame->subframes[ch].samples;
			int i, n = frame->blocksize;

			// CONSTANT
			for (i = 1; i < n; i++)
			{
				if (smp[i] != smp[0]) break;
			}
			if (i == n)
			{
				frame->subframes[ch].best.type = SubframeType.Constant;
				frame->subframes[ch].best.residual[0] = smp[0];
				frame->subframes[ch].best.size = frame->subframes[ch].obits;
				return;
			}

			// VERBATIM
			frame->current.type = SubframeType.Verbatim;
			frame->current.size = frame->subframes[ch].obits * (uint)frame->blocksize;
			choose_best_subframe(frame, ch);

			if (n < 5 || predict == PredictionType.None)
				return;

			// FIXED
			if (predict == PredictionType.Fixed ||
				predict == PredictionType.Search ||
				(predict == PredictionType.Estimated && frame->subframes[ch].best.type == SubframeType.Fixed) ||
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
				(predict == PredictionType.Estimated && frame->subframes[ch].best.type == SubframeType.LPC)))
			{
				//double* lpcs = stackalloc double[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
				int min_order = eparams.min_prediction_order;
				int max_order = eparams.max_prediction_order;

				for (int iWindow = 0; iWindow < _windowcount; iWindow++)
				{
					if (predict == PredictionType.Estimated && frame->subframes[ch].best.window != iWindow)
						continue;

					double* reff = frame->subframes[ch].lpcs_reff + iWindow * lpc.MAX_LPC_ORDER;
					if (frame->subframes[ch].lpcs_order[iWindow] != max_order)
					{
						double* autoc = stackalloc double[lpc.MAX_LPC_ORDER + 1];
						lpc.compute_autocorr(smp, (uint)n, (uint)max_order, autoc, frame->window_buffer + iWindow * Flake.MAX_BLOCKSIZE * 2);
						lpc.compute_schur_reflection(autoc, (uint)max_order, reff);
						frame->subframes[ch].lpcs_order[iWindow] = max_order;
					}

					int est_order = 1;
					int est_order2 = 1;
					if (omethod == OrderMethod.Estimate || omethod == OrderMethod.Estimate8 || omethod == OrderMethod.EstSearch)
					{
						// Estimate optimal order using reflection coefficients
						for (int r = max_order - 1; r >= 0; r--)
							if (Math.Abs(reff[r]) > 0.1)
							{
								est_order = r + 1;
								break;
							}
						for (int r = Math.Min(max_order, 8) - 1; r >= 0; r--)
							if (Math.Abs(reff[r]) > 0.1)
							{
								est_order2 = r + 1;
								break;
							}
					}
					else
						est_order = max_order;

					double* lpcs = stackalloc double[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
					lpc.compute_lpc_coefs(null, (uint)est_order, reff, lpcs);

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
						case OrderMethod.Estimate8:
							// estimated order
							encode_residual_lpc_sub(frame, lpcs, iWindow, est_order2, ch);
							break;
						//case OrderMethod.EstSearch:
						// brute-force search starting from estimate
						//encode_residual_lpc_sub(frame, lpcs, iWindow, est_order, ch);
						//encode_residual_lpc_sub(frame, lpcs, iWindow, est_order2, ch);
						//break;
						case OrderMethod.EstSearch:
							// brute-force search starting from estimate
							for (i = est_order; i >= min_order; i--)
								if (i == est_order || Math.Abs(reff[i - 1]) > 0.10)
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
							break;
						case OrderMethod.Search:
							// brute-force optimal order search
							for (i = max_order; i >= min_order; i--)
								encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
							break;
						case OrderMethod.LogFast:
							// Try max, est, 32,16,8,4,2,1
							encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
							//encode_residual_lpc_sub(frame, lpcs, est_order, ch);
							for (i = lpc.MAX_LPC_ORDER; i >= min_order; i >>= 1)
								if (i < max_order)
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
							break;
						case OrderMethod.LogSearch:
							// do LogFast first
							encode_residual_lpc_sub(frame, lpcs, iWindow, max_order, ch);
							//encode_residual_lpc_sub(frame, lpcs, est_order, ch);
							for (i = lpc.MAX_LPC_ORDER; i >= min_order; i >>= 1)
								if (i < max_order)
									encode_residual_lpc_sub(frame, lpcs, iWindow, i, ch);
							// if found a good order, try to search around it
							if (frame->subframes[ch].best.type == SubframeType.LPC)
							{
								// log search (written by Michael Niedermayer for FFmpeg)
								for (int step = lpc.MAX_LPC_ORDER; step > 0; step >>= 1)
								{
									int last = frame->subframes[ch].best.order;
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

		unsafe void output_residual(FlacFrame* frame, BitWriter bitwriter, FlacSubframeInfo* sub)
		{
			// rice-encoded block
			bitwriter.writebits(2, 0);

			// partition order
			int porder = sub->best.rc.porder;
			int psize = frame->blocksize >> porder;
			//assert(porder >= 0);
			bitwriter.writebits(4, porder);
			int res_cnt = psize - sub->best.order;

			// residual
			int j = sub->best.order;
			for (int p = 0; p < (1 << porder); p++)
			{
				int k = sub->best.rc.rparams[p];
				bitwriter.writebits(4, k);
				if (p == 1) res_cnt = psize;
				if (k == 0)
					for (int i = 0; i < res_cnt && j < frame->blocksize; i++, j++)
						bitwriter.write_unary_signed(sub->best.residual[j]);
				else
					for (int i = 0; i < res_cnt && j < frame->blocksize; i++, j++)
						bitwriter.write_rice_signed(k, sub->best.residual[j]);
			}
		}

		unsafe void 
		output_subframe_constant(FlacFrame* frame, BitWriter bitwriter, FlacSubframeInfo* sub)
		{
			bitwriter.writebits_signed(sub->obits, sub->best.residual[0]);
		}

		unsafe void
		output_subframe_verbatim(FlacFrame* frame, BitWriter bitwriter, FlacSubframeInfo* sub)
		{
			int n = frame->blocksize;
			for (int i = 0; i < n; i++)
				bitwriter.writebits_signed(sub->obits, sub->samples[i]); 
			// Don't use residual here, because we don't copy samples to residual for verbatim frames.
		}

		unsafe void
		output_subframe_fixed(FlacFrame* frame, BitWriter bitwriter, FlacSubframeInfo* sub)
		{
			// warm-up samples
			for (int i = 0; i < sub->best.order; i++)
				bitwriter.writebits_signed(sub->obits, sub->best.residual[i]);

			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void
		output_subframe_lpc(FlacFrame* frame, BitWriter bitwriter, FlacSubframeInfo* sub)
		{
			// warm-up samples
			for (int i = 0; i < sub->best.order; i++)
				bitwriter.writebits_signed(sub->obits, sub->best.residual[i]);

			// LPC coefficients
			int cbits = 1;
			for (int i = 0; i < sub->best.order; i++)
				while (cbits < 16 && sub->best.coefs[i] != (sub->best.coefs[i] << (32 - cbits)) >> (32 - cbits))
					cbits++;
			bitwriter.writebits(4, cbits - 1);
			bitwriter.writebits_signed(5, sub->best.shift);
			for (int i = 0; i < sub->best.order; i++)
				bitwriter.writebits_signed(cbits, sub->best.coefs[i]);
			
			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void output_subframes(FlacFrame* frame, BitWriter bitwriter)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				FlacSubframeInfo* sub = frame->subframes + ch;
				// subframe header
				int type_code = (int) sub->best.type;
				if (sub->best.type == SubframeType.Fixed)
					type_code |= sub->best.order;
				if (sub->best.type == SubframeType.LPC)
					type_code |= sub->best.order - 1;
				bitwriter.writebits(1, 0);
				bitwriter.writebits(6, type_code);
				bitwriter.writebits(1, sub->wbits != 0 ? 1 : 0);
				if (sub->wbits > 0)
					bitwriter.writebits((int)sub->wbits, 1);

				// subframe
				switch (sub->best.type)
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

		unsafe void estimate_frame(FlacFrame* frame, bool do_midside)
		{
			int subframes = do_midside ? channels * 2 : channels;

			switch (eparams.stereo_method)
			{
				case StereoMethod.Estimate:
					for (int ch = 0; ch < subframes; ch++)
						frame->subframes[ch].best.size = (uint)frame->blocksize * 32 + calc_decorr_score(frame, ch);
					break;
				case StereoMethod.Evaluate:
					{
						int max_prediction_order = eparams.max_prediction_order;
						int max_fixed_order = eparams.max_fixed_order;
						int min_fixed_order = eparams.min_fixed_order;
						int lpc_precision_search = eparams.lpc_precision_search;
						int max_partition_order = eparams.max_partition_order;
						OrderMethod omethod = OrderMethod.Estimate8;
						eparams.min_fixed_order = 2;
						eparams.max_fixed_order = 2;
						eparams.lpc_precision_search = 0;
						if (eparams.max_prediction_order > 12)
							eparams.max_prediction_order = 8;
						//if (eparams.max_partition_order > 4)
							//eparams.max_partition_order = 4;
						for (int ch = 0; ch < subframes; ch++)
							encode_residual(frame, ch, eparams.prediction_type, omethod);
						eparams.min_fixed_order = min_fixed_order;
						eparams.max_fixed_order = max_fixed_order;
						eparams.max_prediction_order = max_prediction_order;
						eparams.lpc_precision_search = lpc_precision_search;
						eparams.max_partition_order = max_partition_order;
						break;
					}
				case StereoMethod.Search:
					for (int ch = 0; ch < subframes; ch++)
						encode_residual(frame, ch, eparams.prediction_type, eparams.order_method);
					break;
			}
		}

		unsafe uint measure_frame_size(FlacFrame* frame, bool do_midside)
		{
			uint total = 48 + 16; // crude estimation of header/footer size;

			if (do_midside)
			{
				uint bitsBest = UINT32_MAX;
				ChannelMode modeBest = ChannelMode.LeftRight;

				if (bitsBest > frame->subframes[2].best.size + frame->subframes[3].best.size)
				{
					bitsBest = frame->subframes[2].best.size + frame->subframes[3].best.size;
					modeBest = ChannelMode.MidSide;
				}
				if (bitsBest > frame->subframes[3].best.size + frame->subframes[1].best.size)
				{
					bitsBest = frame->subframes[3].best.size + frame->subframes[1].best.size;
					modeBest = ChannelMode.RightSide;
				}
				if (bitsBest > frame->subframes[3].best.size + frame->subframes[0].best.size)
				{
					bitsBest = frame->subframes[3].best.size + frame->subframes[0].best.size;
					modeBest = ChannelMode.LeftSide;
				}
				if (bitsBest > frame->subframes[0].best.size + frame->subframes[1].best.size)
				{
					bitsBest = frame->subframes[0].best.size + frame->subframes[1].best.size;
					modeBest = ChannelMode.LeftRight;
				}
				frame->ch_mode = modeBest;
				return total + bitsBest;
			}

			for (int ch = 0; ch < channels; ch++)
				total += frame->subframes[ch].best.size;
			return total;
		}

		unsafe void encode_estimated_frame(FlacFrame* frame, bool do_midside)
		{
			if (do_midside)
				switch (frame->ch_mode)
				{
					case ChannelMode.MidSide:
						frame->subframes[0] = frame->subframes[2];
						frame->subframes[1] = frame->subframes[3];
						break;
					case ChannelMode.RightSide:
						frame->subframes[0] = frame->subframes[3];
						break;
					case ChannelMode.LeftSide:
						frame->subframes[1] = frame->subframes[3];
						break;
				}

			switch (eparams.stereo_method)
			{
				case StereoMethod.Estimate:
					for (int ch = 0; ch < channels; ch++)
					{
						frame->subframes[ch].best.size = UINT32_MAX;
						encode_residual(frame, ch, eparams.prediction_type, eparams.order_method);
					}
					break;
				case StereoMethod.Evaluate:
					for (int ch = 0; ch < channels; ch++)
						encode_residual(frame, ch, PredictionType.Estimated, eparams.order_method);
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
			double* pos = window + _windowcount * Flake.MAX_BLOCKSIZE * 2;
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

		unsafe int encode_frame(out int size)
		{
			FlacFrame frame;
			FlacFrame frame2, frame3;
			FlacSubframeInfo* sf = stackalloc FlacSubframeInfo[channels * 6];

			fixed (int* s = samplesBuffer, r = residualBuffer)
			fixed (double* window = windowBuffer)
			{
				frame.subframes = sf;

				init_frame(&frame, eparams.block_size);
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

				if (channels != 2 || frame.blocksize <= 32 || eparams.stereo_method == StereoMethod.Independent)
				{
					frame.window_buffer = window;
					frame.current.residual = r + channels * Flake.MAX_BLOCKSIZE;
					frame.ch_mode = channels != 2 ? ChannelMode.NotStereo : ChannelMode.LeftRight;
					for (int ch = 0; ch < channels; ch++)
						initialize_subframe(&frame, ch, s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE,
							bits_per_sample, get_wasted_bits(s + ch * Flake.MAX_BLOCKSIZE, frame.blocksize));

					for (int ch = 0; ch < channels; ch++)
						encode_residual(&frame, ch, eparams.prediction_type, eparams.order_method);
				}
				else
				{
					channel_decorrelation(s, s + Flake.MAX_BLOCKSIZE, s + 2 * Flake.MAX_BLOCKSIZE, s + 3 * Flake.MAX_BLOCKSIZE, frame.blocksize);
					frame.window_buffer = window;
					frame.current.residual = r + 4 * Flake.MAX_BLOCKSIZE;
					for (int ch = 0; ch < 4; ch++)
						initialize_subframe(&frame, ch, s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, 
							bits_per_sample + (ch == 3 ? 1U : 0U), get_wasted_bits(s + ch * Flake.MAX_BLOCKSIZE, frame.blocksize));
					estimate_frame(&frame, true);
					uint fs = measure_frame_size(&frame, true);

					if (0 != eparams.variable_block_size)
					{
						int tumbler = 1;
						while ((frame.blocksize & 1) == 0 && frame.blocksize >= 1024)
						{
							init_frame(&frame2, frame.blocksize / 2);
							frame2.window_buffer = frame.window_buffer + frame.blocksize;
							frame2.current.residual = r + tumbler * 5 * Flake.MAX_BLOCKSIZE;
							frame2.subframes = sf + tumbler * channels * 2;
							for (int ch = 0; ch < 4; ch++)
								initialize_subframe(&frame2, ch, frame.subframes[ch].samples, frame2.current.residual + (ch + 1) * frame2.blocksize,
									frame.subframes[ch].obits + frame.subframes[ch].wbits, frame.subframes[ch].wbits);
							estimate_frame(&frame2, true);
							uint fs2 = measure_frame_size(&frame2, true);
							uint fs3 = fs2;
							if (eparams.variable_block_size == 2 || eparams.variable_block_size == 4)
							{
								init_frame(&frame3, frame2.blocksize);
								frame3.window_buffer = frame2.window_buffer;
								frame3.current.residual = frame2.current.residual + 5 * frame2.blocksize;
								frame3.subframes = sf + channels * 4;
								for (int ch = 0; ch < 4; ch++)
									initialize_subframe(&frame3, ch, frame2.subframes[ch].samples + frame2.blocksize, frame3.current.residual + (ch + 1) * frame3.blocksize,
										frame.subframes[ch].obits + frame.subframes[ch].wbits, frame.subframes[ch].wbits);
								estimate_frame(&frame3, true);
								fs3 = measure_frame_size(&frame3, true);
							}
							if (fs2 + fs3 > fs)
								break;
							frame = frame2;
							fs = fs2;
							if (eparams.variable_block_size <= 2)
								break;
							tumbler = 1 - tumbler;
						}
					}

					encode_estimated_frame(&frame, true);
				}

				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);

				output_frame_header(&frame, bitwriter);
				output_subframes(&frame, bitwriter);
				output_frame_footer(bitwriter);

				if (frame_buffer != null)
				{
					if (eparams.variable_block_size > 0)
						frame_count += frame.blocksize;
					else
						frame_count++;
				}
				size = frame.blocksize;
				return bitwriter.Length;
			}
		}

		unsafe int output_frame()
		{
			if (verify != null)
			{
				fixed (int* s = verifyBuffer, r = samplesBuffer)
					for (int ch = 0; ch < channels; ch++)
						Flake.memcpy(s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, eparams.block_size);
			}

			int fs, bs;
			//if (0 != eparams.variable_block_size && 0 == (eparams.block_size & 7) && eparams.block_size >= 128)
			//    fs = encode_frame_vbs();
			//else
			fs = encode_frame(out bs);

			if (seek_table != null && _IO.CanSeek)
			{
				for (int sp = 0; sp < seek_table.Length; sp++)
				{
					if (seek_table[sp].framesize != 0)
						continue;
					if (seek_table[sp].number > (ulong)_position + (ulong)bs)
						break;
					if (seek_table[sp].number >= (ulong)_position)
					{
						seek_table[sp].number = (ulong)_position;
						seek_table[sp].offset = (ulong)(_IO.Position - first_frame_offset);
						seek_table[sp].framesize = (uint)bs;
					}
				}
			}

			_position += bs;
			_IO.Write(frame_buffer, 0, fs);
			_totalSize += fs;

			if (verify != null)
			{
				int decoded = verify.DecodeFrame(frame_buffer, 0, fs);
				if (decoded != fs || verify.Remaining != (ulong)bs)
					throw new Exception("validation failed!");
				fixed (int* s = verifyBuffer, r = verify.Samples)
				{
					for (int ch = 0; ch < channels; ch++)
						if (Flake.memcmp(s + ch * Flake.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, bs))
							throw new Exception("validation failed!");
				}
			}

			if (bs < eparams.block_size)
			{
				fixed (int* s = samplesBuffer)
					for (int ch = 0; ch < channels; ch++)
						Flake.memcpy(s + ch * Flake.MAX_BLOCKSIZE, s + bs + ch * Flake.MAX_BLOCKSIZE, eparams.block_size - bs);
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
				int header_size = flake_encode_init();
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

				if (md5 != null)
				{
					AudioSamples.FLACSamplesToBytes(buff, pos, frame_buffer, 0, block, channels, (int)bits_per_sample);
					md5.TransformBlock(frame_buffer, 0, block * channels * ((int)bits_per_sample >> 3), null, 0);
				}

				len -= block;
				pos += block;

				while (samplesInBuffer >= eparams.block_size)
					output_frame();
			}
		}

		public string Path { get { return _path; } }

		string vendor_string = "Flake#0.1";

		int select_blocksize(int samplerate, int time_ms)
		{
			int blocksize = Flake.flac_blocksizes[1];
			int target = (samplerate * time_ms) / 1000;
			if (eparams.variable_block_size > 0)
			{
				blocksize = 1024;
				while (target >= blocksize)
					blocksize <<= 1;
				return blocksize >> 1;
			}

			for (int i = 0; i < Flake.flac_blocksizes.Length; i++)
				if (target >= Flake.flac_blocksizes[i] && Flake.flac_blocksizes[i] > blocksize)
				{
					blocksize = Flake.flac_blocksizes[i];
				}
			return blocksize;
		}

		void write_streaminfo(byte[] header, int pos, int last)
		{
			Array.Clear(header, pos, 38);
			BitWriter bitwriter = new BitWriter(header, pos, 38);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, (int)MetadataType.FLAC__METADATA_TYPE_STREAMINFO);
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
			bitwriter.writebits(7, (int)MetadataType.FLAC__METADATA_TYPE_VORBIS_COMMENT);
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

		int write_seekpoints(byte[] header, int pos, int last)
		{
			seek_table_offset = pos + 4;

			BitWriter bitwriter = new BitWriter(header, pos, 4 + 18 * seek_table.Length);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, (int)MetadataType.FLAC__METADATA_TYPE_SEEKTABLE);
			bitwriter.writebits(24, 18 * seek_table.Length);
			for (int i = 0; i < seek_table.Length; i++)
			{
				bitwriter.writebits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN, seek_table[i].number);
				bitwriter.writebits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN, seek_table[i].offset);
				bitwriter.writebits(Flake.FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN, seek_table[i].framesize);
			}
			bitwriter.flush();
			return 4 + 18 * seek_table.Length;
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
			bitwriter.writebits(7, (int)MetadataType.FLAC__METADATA_TYPE_PADDING);
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

			// seek table
			if (_IO.CanSeek && seek_table != null)
				header_size += write_seekpoints(header, header_size, last);

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
				if (sample_rate == Flake.flac_samplerates[i])
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
				if (bits_per_sample == Flake.flac_bitdepths[i])
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

			if (_IO.CanSeek && eparams.do_seektable)
			{
				int seek_points_distance = sample_rate * 10;
				int num_seek_points = 1 + sample_count / seek_points_distance; // 1 seek point per 10 seconds
				if (sample_count % seek_points_distance == 0)
					num_seek_points--;
				seek_table = new SeekPoint[num_seek_points];
				for (int sp = 0; sp < num_seek_points; sp++)
				{
					seek_table[sp].framesize = 0;
					seek_table[sp].offset = 0;
					seek_table[sp].number = (ulong)(sp * seek_points_distance);
				}
			}

			// output header bytes
			header = new byte[eparams.padding_size + 1024 + (seek_table == null ? 0 : seek_table.Length * 18)];
			header_len = write_headers();

			// initialize CRC & MD5
			if (_IO.CanSeek && eparams.do_md5)
				md5 = new MD5CryptoServiceProvider();

			if (eparams.do_verify)
			{
				verify = new FlakeReader(channels, bits_per_sample);
				verifyBuffer = new int[Flake.MAX_BLOCKSIZE * channels];
			}

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

		// whether to try various lpc_precisions
		// 0 - use only one precision
		// 1 - try two precisions
		public int lpc_precision_search;

		public WindowFunction window_function;

		public bool do_md5;
		public bool do_verify;
		public bool do_seektable;

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
			stereo_method = StereoMethod.Evaluate;
			block_size = 0;
			block_time_ms = 105;			
			prediction_type = PredictionType.Search;
			min_prediction_order = 1;
			max_prediction_order = 8;
			min_fixed_order = 2;
			max_fixed_order = 2;
			min_partition_order = 0;
			max_partition_order = 6;
			variable_block_size = 0;
			lpc_precision_search = 0;
			do_md5 = true;
			do_verify = false;
			do_seektable = true; 

			// differences from level 5
			switch (lvl)
			{
				case 0:
					block_time_ms = 27;
					prediction_type = PredictionType.Fixed;
					max_partition_order = 4;
					break;
				case 1:
					prediction_type = PredictionType.Levinson;
					stereo_method = StereoMethod.Independent;
					window_function = WindowFunction.Welch;
					max_partition_order = 4;
					break;
				case 2:
					prediction_type = PredictionType.Search;
					stereo_method = StereoMethod.Independent;
					window_function = WindowFunction.Welch;
					max_prediction_order = 12;
					max_partition_order = 4;
					break;
				case 3:
					prediction_type = PredictionType.Levinson;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Welch;
					max_partition_order = 4;
					break;
				case 4:
					prediction_type = PredictionType.Levinson;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Welch;
					max_prediction_order = 12;
					max_partition_order = 4;
					break;
				case 5:
					prediction_type = PredictionType.Search;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Welch;
					max_prediction_order = 12;
					break;
				case 6:
					prediction_type = PredictionType.Levinson;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Flattop | WindowFunction.Tukey;
					max_prediction_order = 12;
					break;
				case 7:
					prediction_type = PredictionType.Search;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Flattop | WindowFunction.Tukey;
					max_prediction_order = 12;
					min_fixed_order = 0;
					max_fixed_order = 4;
					lpc_precision_search = 1;
					break;
				case 8:
					prediction_type = PredictionType.Search;
					stereo_method = StereoMethod.Evaluate;
					window_function = WindowFunction.Flattop | WindowFunction.Tukey;
					order_method = OrderMethod.EstSearch;
					max_prediction_order = 12;
					min_fixed_order = 0;
					max_fixed_order = 4;
					lpc_precision_search = 1;
					break;
				case 9:
					window_function = WindowFunction.Welch;
					max_prediction_order = 32;
					break;
				case 10:
					min_fixed_order = 0;
					max_fixed_order = 4;
					max_prediction_order = 32;
					lpc_precision_search = 0;
					break;
				case 11:
					order_method = OrderMethod.EstSearch;
					min_fixed_order = 0;
					max_fixed_order = 4;
					max_prediction_order = 32;
					//lpc_precision_search = 1;
					variable_block_size = 4;
					break;
			}

			return 0;
		}
	}
}
