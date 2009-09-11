/**
 * CUETools.FlaCuda: FLAC audio encoder using CUDA
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
using System.IO;
using System.Security.Cryptography;
using System.Text;
//using System.Runtime.InteropServices;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;
using GASS.CUDA;
using GASS.CUDA.Types;

namespace CUETools.Codecs.FlaCuda
{
	public class FlaCudaWriter : IAudioDest
	{
		Stream _IO = null;
		string _path;
		long _position;

		// number of audio channels
		// valid values are 1 to 8
		int channels, ch_code;

		// audio sample rate in Hz
		int sample_rate, sr_code0, sr_code1;

		// sample size in bits
		// only 16-bit is currently supported
		uint bits_per_sample;
		int bps_code;

		// total stream samples
		// if 0, stream length is unknown
		int sample_count;

		FlakeEncodeParams eparams;

		// maximum frame size in bytes
		// this can be used to allocate memory for output
		int max_frame_size;

		byte[] frame_buffer = null;

		int frame_count = 0;

		long first_frame_offset = 0;

		TimeSpan _userProcessorTime;

		// header bytes
		// allocated by flake_encode_init and freed by flake_encode_close
		byte[] header;

		int[] verifyBuffer;
		int[] residualBuffer;
		float[] windowBuffer;
		int samplesInBuffer = 0;

		int _compressionLevel = 5;
		int _blocksize = 0;
		int _totalSize = 0;
		int _windowsize = 0, _windowcount = 0;

		Crc8 crc8;
		Crc16 crc16;
		MD5 md5;

		FlacFrame frame;
		FlakeReader verify;

		SeekPoint[] seek_table;
		int seek_table_offset = -1;

		bool inited = false;

		CUDA cuda;
		CUfunction cudaComputeAutocor;
		CUfunction cudaComputeLPC;
		CUfunction cudaEstimateResidual;
		CUfunction cudaSumResidualChunks;
		CUfunction cudaSumResidual;
		CUfunction cudaEncodeResidual;
		CUdeviceptr cudaSamples;
		CUdeviceptr cudaWindow;
		CUdeviceptr cudaAutocorTasks;
		CUdeviceptr cudaAutocorOutput;
		CUdeviceptr cudaResidualTasks;
		CUdeviceptr cudaResidualOutput;
		CUdeviceptr cudaResidualSums;
		IntPtr samplesBufferPtr = IntPtr.Zero;
		IntPtr autocorTasksPtr = IntPtr.Zero;
		IntPtr residualTasksPtr = IntPtr.Zero;
		CUstream cudaStream;
		CUstream cudaStream1;

		int nResidualTasks = 0;
		int nAutocorTasks = 0;

		const int MAX_BLOCKSIZE = 8192;
		const int maxResidualParts = 64;
		const int maxAutocorParts = MAX_BLOCKSIZE / (256 - 32);

		public FlaCudaWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO)
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

			residualBuffer = new int[FlaCudaWriter.MAX_BLOCKSIZE * (channels == 2 ? 10 : channels + 1)];
			windowBuffer = new float[FlaCudaWriter.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS];

			eparams.flake_set_defaults(_compressionLevel);
			eparams.padding_size = 8192;

			crc8 = new Crc8();
			crc16 = new Crc16();
			frame = new FlacFrame(channels * 2);
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

				cuda.Free(cudaWindow);
				cuda.Free(cudaSamples);
				cuda.Free(cudaAutocorTasks);
				cuda.Free(cudaAutocorOutput);
				cuda.Free(cudaResidualTasks);
				cuda.Free(cudaResidualOutput);
				cuda.Free(cudaResidualSums);
				CUDADriver.cuMemFreeHost(samplesBufferPtr);
				CUDADriver.cuMemFreeHost(residualTasksPtr);
				CUDADriver.cuMemFreeHost(autocorTasksPtr);
				cuda.DestroyStream(cudaStream);
				cuda.DestroyStream(cudaStream1);
				cuda.Dispose();
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
				cuda.Free(cudaWindow);
				cuda.Free(cudaSamples);
				cuda.Free(cudaAutocorTasks);
				cuda.Free(cudaAutocorOutput);
				cuda.Free(cudaResidualTasks);
				cuda.Free(cudaResidualOutput);
				cuda.Free(cudaResidualSums);
				CUDADriver.cuMemFreeHost(samplesBufferPtr);
				CUDADriver.cuMemFreeHost(residualTasksPtr);
				CUDADriver.cuMemFreeHost(autocorTasksPtr);
				cuda.DestroyStream(cudaStream);
				cuda.DestroyStream(cudaStream1);
				cuda.Dispose();
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
			set {
				if (value < 256 || value > MAX_BLOCKSIZE )
					throw new Exception("unsupported BlockSize value");
				_blocksize = (int)value; 
			}
			get { return _blocksize == 0 ? eparams.block_size : _blocksize; }
		}

		public StereoMethod StereoMethod
		{
			get { return eparams.do_midside ? StereoMethod.Search : StereoMethod.Independent; }
			set { eparams.do_midside = value != StereoMethod.Independent; }
		}

		public int MinPrecisionSearch
		{
			get { return eparams.lpc_min_precision_search; }
			set
			{
				if (value < 0 || value > eparams.lpc_max_precision_search)
					throw new Exception("unsupported MinPrecisionSearch value");
				eparams.lpc_min_precision_search = value;
			}
		}

		public int MaxPrecisionSearch
		{
			get { return eparams.lpc_max_precision_search; }
			set
			{
				if (value < eparams.lpc_min_precision_search || value >= lpc.MAX_LPC_PRECISIONS)
					throw new Exception("unsupported MaxPrecisionSearch value");
				eparams.lpc_max_precision_search = value;
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
				if (value > lpc.MAX_LPC_ORDER || value < eparams.min_prediction_order)
					throw new Exception("invalid MaxLPCOrder " + value.ToString());
				eparams.max_prediction_order = value;
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

		/// <summary>
		/// Copy channel-interleaved input samples into separate subframes
		/// </summary>
		/// <param name="samples"></param>
		/// <param name="pos"></param>
		/// <param name="block"></param>
 		unsafe void copy_samples(int[,] samples, int pos, int block)
		{
			int* fsamples = (int*)samplesBufferPtr;
			fixed (int *src = &samples[pos, 0])
			{
				if (channels == 2)
					AudioSamples.Deinterlace(fsamples + samplesInBuffer, fsamples + FlaCudaWriter.MAX_BLOCKSIZE + samplesInBuffer, src, block);
				else
					for (int ch = 0; ch < channels; ch++)
					{
						int* psamples = fsamples + ch * FlaCudaWriter.MAX_BLOCKSIZE + samplesInBuffer;
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
			AudioSamples.MemCpy(res, smp, (int) n);
		}

		unsafe void encode_residual_fixed(int* res, int* smp, int n, int order)
		{
			int i;
			int s0, s1, s2;
			switch (order)
			{
				case 0:
					AudioSamples.MemCpy(res, smp, n);
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

		static unsafe uint calc_optimal_rice_params(ref RiceContext rc, int porder, uint* sums, uint n, uint pred_order)
		{
			uint part = (1U << porder);
			uint all_bits = 0;			
			rc.rparams[0] = find_optimal_rice_param(sums[0], (n >> porder) - pred_order, out all_bits);
			uint cnt = (n >> porder);
			for (uint i = 1; i < part; i++)
			{
				uint nbits;
				rc.rparams[i] = find_optimal_rice_param(sums[i], cnt, out nbits);
				all_bits += nbits;
			}
			all_bits += (4 * part);
			rc.porder = porder;
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

		static unsafe uint calc_rice_params(ref RiceContext rc, int pmin, int pmax, int* data, uint n, uint pred_order)
		{
			RiceContext tmp_rc = new RiceContext(), tmp_rc2;
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
			uint opt_bits = AudioSamples.UINT32_MAX;
			for (int i = pmin; i <= pmax; i++)
			{
				uint bits = calc_optimal_rice_params(ref tmp_rc, i, sums + i * Flake.MAX_PARTITIONS, n, pred_order);
				if (bits <= opt_bits)
				{
					opt_porder = i;
					opt_bits = bits;
					tmp_rc2 = rc;
					rc = tmp_rc;
					tmp_rc = tmp_rc2;
				}
			}

			return opt_bits;
		}

		static int get_max_p_order(int max_porder, int n, int order)
		{
			int porder = Math.Min(max_porder, BitReader.log2i(n ^ (n - 1)));
			if (order > 0)
				porder = Math.Min(porder, BitReader.log2i(n / order));
			return porder;
		}

		static unsafe uint calc_rice_params_fixed(ref RiceContext rc, int pmin, int pmax,
			int* data, int n, int pred_order, uint bps)
		{
			pmin = get_max_p_order(pmin, n, pred_order);
			pmax = get_max_p_order(pmax, n, pred_order);
			uint bits = (uint)pred_order * bps + 6;
			bits += calc_rice_params(ref rc, pmin, pmax, data, (uint)n, (uint)pred_order);
			return bits;
		}

		static unsafe uint calc_rice_params_lpc(ref RiceContext rc, int pmin, int pmax,
			int* data, int n, int pred_order, uint bps, uint precision)
		{
			pmin = get_max_p_order(pmin, n, pred_order);
			pmax = get_max_p_order(pmax, n, pred_order);
			uint bits = (uint)pred_order * bps + 4 + 5 + (uint)pred_order * precision + 6;
			bits += calc_rice_params(ref rc, pmin, pmax, data, (uint)n, (uint)pred_order);
			return bits;
		}

		// select LPC precision based on block size
		static uint get_precision(int blocksize)
		{
			uint lpc_precision;
			if (blocksize <= 192) lpc_precision = 7U;
			else if (blocksize <= 384) lpc_precision = 8U;
			else if (blocksize <= 576) lpc_precision = 9U;
			else if (blocksize <= 1152) lpc_precision = 10U;
			else if (blocksize <= 2304) lpc_precision = 11U;
			else if (blocksize <= 4608) lpc_precision = 12U;
			else if (blocksize <= 8192) lpc_precision = 13U;
			else if (blocksize <= 16384) lpc_precision = 14U;
			else lpc_precision = 15;
			return lpc_precision;
		}

		unsafe void output_frame_header(FlacFrame frame, BitWriter bitwriter)
		{
			bitwriter.writebits(15, 0x7FFC);
			bitwriter.writebits(1, eparams.variable_block_size > 0 ? 1 : 0);
			bitwriter.writebits(4, frame.bs_code0);
			bitwriter.writebits(4, sr_code0);
			if (frame.ch_mode == ChannelMode.NotStereo)
				bitwriter.writebits(4, ch_code);
			else
				bitwriter.writebits(4, (int) frame.ch_mode);
			bitwriter.writebits(3, bps_code);
			bitwriter.writebits(1, 0);
			bitwriter.write_utf8(frame_count);

			// custom block size
			if (frame.bs_code1 >= 0)
			{
				if (frame.bs_code1 < 256)
					bitwriter.writebits(8, frame.bs_code1);
				else
					bitwriter.writebits(16, frame.bs_code1);
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

		unsafe void output_residual(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			// rice-encoded block
			bitwriter.writebits(2, 0);

			// partition order
			int porder = sub.best.rc.porder;
			int psize = frame.blocksize >> porder;
			//assert(porder >= 0);
			bitwriter.writebits(4, porder);
			int res_cnt = psize - sub.best.order;

			// residual
			int j = sub.best.order;
			for (int p = 0; p < (1 << porder); p++)
			{
				int k = sub.best.rc.rparams[p];
				bitwriter.writebits(4, k);
				if (p == 1) res_cnt = psize;
				int cnt = Math.Min(res_cnt, frame.blocksize - j);
				bitwriter.write_rice_block_signed(k, sub.best.residual + j, cnt);
				j += cnt;
			}
		}

		unsafe void 
		output_subframe_constant(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			bitwriter.writebits_signed(sub.obits, sub.samples[0]);
		}

		unsafe void
		output_subframe_verbatim(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			int n = frame.blocksize;
			for (int i = 0; i < n; i++)
				bitwriter.writebits_signed(sub.obits, sub.samples[i]); 
			// Don't use residual here, because we don't copy samples to residual for verbatim frames.
		}

		unsafe void
		output_subframe_fixed(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			// warm-up samples
			for (int i = 0; i < sub.best.order; i++)
				bitwriter.writebits_signed(sub.obits, sub.samples[i]);

			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void
		output_subframe_lpc(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			// warm-up samples
			for (int i = 0; i < sub.best.order; i++)
				bitwriter.writebits_signed(sub.obits, sub.samples[i]);

			// LPC coefficients
			int cbits = 1;
			for (int i = 0; i < sub.best.order; i++)
				while (cbits < 16 && sub.best.coefs[i] != (sub.best.coefs[i] << (32 - cbits)) >> (32 - cbits))
					cbits++;
			bitwriter.writebits(4, cbits - 1);
			bitwriter.writebits_signed(5, sub.best.shift);
			for (int i = 0; i < sub.best.order; i++)
				bitwriter.writebits_signed(cbits, sub.best.coefs[i]);
			
			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void output_subframes(FlacFrame frame, BitWriter bitwriter)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				FlacSubframeInfo sub = frame.subframes[ch];
				// subframe header
				int type_code = (int) sub.best.type;
				if (sub.best.type == SubframeType.Fixed)
					type_code |= sub.best.order;
				if (sub.best.type == SubframeType.LPC)
					type_code |= sub.best.order - 1;
				bitwriter.writebits(1, 0);
				bitwriter.writebits(6, type_code);
				bitwriter.writebits(1, sub.wbits != 0 ? 1 : 0);
				if (sub.wbits > 0)
					bitwriter.writebits((int)sub.wbits, 1);

				// subframe
				switch (sub.best.type)
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

		unsafe uint measure_frame_size(FlacFrame frame, bool do_midside)
		{
			// crude estimation of header/footer size
			uint total = (uint)(32 + ((BitReader.log2i(frame_count) + 4) / 5) * 8 + (eparams.variable_block_size != 0 ? 16 : 0) + 16);

			if (do_midside)
			{
				uint bitsBest = AudioSamples.UINT32_MAX;
				ChannelMode modeBest = ChannelMode.LeftRight;

				if (bitsBest > frame.subframes[2].best.size + frame.subframes[3].best.size)
				{
					bitsBest = frame.subframes[2].best.size + frame.subframes[3].best.size;
					modeBest = ChannelMode.MidSide;
				}
				if (bitsBest > frame.subframes[3].best.size + frame.subframes[1].best.size)
				{
					bitsBest = frame.subframes[3].best.size + frame.subframes[1].best.size;
					modeBest = ChannelMode.RightSide;
				}
				if (bitsBest > frame.subframes[3].best.size + frame.subframes[0].best.size)
				{
					bitsBest = frame.subframes[3].best.size + frame.subframes[0].best.size;
					modeBest = ChannelMode.LeftSide;
				}
				if (bitsBest > frame.subframes[0].best.size + frame.subframes[1].best.size)
				{
					bitsBest = frame.subframes[0].best.size + frame.subframes[1].best.size;
					modeBest = ChannelMode.LeftRight;
				}
				frame.ch_mode = modeBest;
				return total + bitsBest;
			}

			for (int ch = 0; ch < channels; ch++)
				total += frame.subframes[ch].best.size;
			return total;
		}

		unsafe delegate void window_function(float* window, int size);

		unsafe void calculate_window(float* window, window_function func, WindowFunction flag)
		{
			if ((eparams.window_function & flag) == 0 || _windowcount == lpc.MAX_LPC_WINDOWS)
				return;
			int sz = _windowsize;
			float* pos = window + _windowcount * FlaCudaWriter.MAX_BLOCKSIZE * 2;
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

		unsafe void initialize_autocorTasks(int channelsCount, int max_order)
		{
			computeAutocorTaskStruct* autocorTasks = (computeAutocorTaskStruct*)autocorTasksPtr;
			encodeResidualTaskStruct* residualTasks = (encodeResidualTaskStruct*)residualTasksPtr;
			nAutocorTasks = 0;
			nResidualTasks = 0;
			for (int ch = 0; ch < channelsCount; ch++)
				for (int iWindow = 0; iWindow < _windowcount; iWindow++)
				{
					// Autocorelation task
					autocorTasks[nAutocorTasks].samplesOffs = ch * FlaCudaWriter.MAX_BLOCKSIZE;
					autocorTasks[nAutocorTasks].windowOffs = iWindow * 2 * FlaCudaWriter.MAX_BLOCKSIZE;
					nAutocorTasks++;
					// LPC tasks
					for (int order = 1; order <= ((max_order + 7) & ~7); order++)
					{
						residualTasks[nResidualTasks].residualOrder = order <= max_order ? order : 0;
						residualTasks[nResidualTasks].samplesOffs = ch * FlaCudaWriter.MAX_BLOCKSIZE;
						nResidualTasks++;
					}
				}
			// Fixed prediction
			for (int ch = 0; ch < channelsCount; ch++)
			{
				for (int order = 1; order <= 8; order++)
				{
					residualTasks[nResidualTasks].residualOrder = order <= 4 ? order : 0;
					residualTasks[nResidualTasks].samplesOffs = ch * FlaCudaWriter.MAX_BLOCKSIZE;
					residualTasks[nResidualTasks].shift = 0;
					switch (order)
					{
						case 1:
							residualTasks[nResidualTasks].coefs[0] = 1;
							break;
						case 2:
							residualTasks[nResidualTasks].coefs[0] = 2;
							residualTasks[nResidualTasks].coefs[1] = -1;
							break;
						case 3:
							residualTasks[nResidualTasks].coefs[0] = 3;
							residualTasks[nResidualTasks].coefs[1] = -3;
							residualTasks[nResidualTasks].coefs[2] = 1;
							break;
						case 4:
							residualTasks[nResidualTasks].coefs[0] = 4;
							residualTasks[nResidualTasks].coefs[1] = -6;
							residualTasks[nResidualTasks].coefs[2] = 4;
							residualTasks[nResidualTasks].coefs[3] = -1;
							break;
					}
					nResidualTasks++;
				}
			}

			cuda.CopyHostToDeviceAsync(cudaAutocorTasks, autocorTasksPtr, (uint)(sizeof(computeAutocorTaskStruct) * nAutocorTasks), cudaStream);
			cuda.CopyHostToDeviceAsync(cudaResidualTasks, residualTasksPtr, (uint)(sizeof(encodeResidualTaskStruct) * nResidualTasks), cudaStream);
			cuda.SynchronizeStream(cudaStream);
		}

		unsafe void encode_residual(FlacFrame frame)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				switch (frame.subframes[ch].best.type)
				{
					case SubframeType.Constant:
						break;
					case SubframeType.Verbatim:
						break;
					case SubframeType.Fixed:
						encode_residual_fixed(frame.subframes[ch].best.residual, frame.subframes[ch].samples,
							frame.blocksize, frame.subframes[ch].best.order);
						frame.subframes[ch].best.size = calc_rice_params_fixed(
							ref frame.subframes[ch].best.rc, eparams.min_partition_order, eparams.max_partition_order,
							frame.subframes[ch].best.residual, frame.blocksize, frame.subframes[ch].best.order, frame.subframes[ch].obits);
						break;
					case SubframeType.LPC:
						fixed (int* coefs = frame.subframes[ch].best.coefs)
						{
							ulong csum = 0;
							for (int i = frame.subframes[ch].best.order; i > 0; i--)
								csum += (ulong)Math.Abs(coefs[i - 1]);
							if ((csum << (int)frame.subframes[ch].obits) >= 1UL << 32)
								lpc.encode_residual_long(frame.subframes[ch].best.residual, frame.subframes[ch].samples, frame.blocksize, frame.subframes[ch].best.order, coefs, frame.subframes[ch].best.shift);
							else
								lpc.encode_residual(frame.subframes[ch].best.residual, frame.subframes[ch].samples, frame.blocksize, frame.subframes[ch].best.order, coefs, frame.subframes[ch].best.shift);
							frame.subframes[ch].best.size = calc_rice_params_lpc(
								ref frame.subframes[ch].best.rc, eparams.min_partition_order, eparams.max_partition_order,
								frame.subframes[ch].best.residual, frame.blocksize, frame.subframes[ch].best.order, frame.subframes[ch].obits, (uint)frame.subframes[ch].best.cbits);
						}
						break;
				}
			}
		}

		unsafe void select_best_methods(FlacFrame frame, int channelsCount, int max_order, int partCount)
		{
			encodeResidualTaskStruct* residualTasks = (encodeResidualTaskStruct*)residualTasksPtr;
			for (int ch = 0; ch < channelsCount; ch++)
			{
				int i;
				for (i = 1; i < frame.blocksize; i++)
					if (frame.subframes[ch].samples[i] != frame.subframes[ch].samples[0])
						break;
				// CONSTANT
				if (i == frame.blocksize)
				{
					frame.subframes[ch].best.type = SubframeType.Constant;
					frame.subframes[ch].best.size = frame.subframes[ch].obits;
				}
				// VERBATIM
				else
				{
					frame.subframes[ch].best.type = SubframeType.Verbatim;
					frame.subframes[ch].best.size = frame.subframes[ch].obits * (uint)frame.blocksize;
				}
			}

			if (frame.blocksize <= 4)
				return;

			// LPC
			for (int ch = 0; ch < channelsCount; ch++)
			{
				for (int iWindow = 0; iWindow < _windowcount; iWindow++)
				{
					for (int order = 1; order <= max_order && order < frame.blocksize; order++)
					{
						int index = (order - 1) + ((max_order + 7) & ~7) * (iWindow + _windowcount * ch);
						int cbits = residualTasks[index].cbits;
						int nbits = order * (int)frame.subframes[ch].obits + 4 + 5 + order * cbits + 6 + residualTasks[index].size;
						if (residualTasks[index].residualOrder != order)
							throw new Exception("oops");
						if (frame.subframes[ch].best.size > nbits)
						{
							frame.subframes[ch].best.type = SubframeType.LPC;
							frame.subframes[ch].best.size = (uint)nbits;
							frame.subframes[ch].best.order = order;
							frame.subframes[ch].best.window = iWindow;
							frame.subframes[ch].best.cbits = cbits;
							frame.subframes[ch].best.shift = residualTasks[index].shift;
							for (int i = 0; i < order; i++)
								frame.subframes[ch].best.coefs[i] = residualTasks[index].coefs[i];//order - 1 - i];
						}
					}
				}
			}

			// FIXED
			for (int ch = 0; ch < channelsCount; ch++)
			{
				for (int order = 1; order <= 4 && order < frame.blocksize; order++)
				{
					int index = (order - 1) + 8 * ch + ((max_order + 7) & ~7) * _windowcount * channelsCount;
					int nbits = order * (int)frame.subframes[ch].obits + 6 + residualTasks[index].size;
					if (residualTasks[index].residualOrder != order)
						throw new Exception("oops");
					if (frame.subframes[ch].best.size > nbits)
					{
						frame.subframes[ch].best.type = SubframeType.Fixed;
						frame.subframes[ch].best.size = (uint)nbits;
						frame.subframes[ch].best.order = order;
					}
				}
			}
		}

		unsafe void estimate_residual(FlacFrame frame, int channelsCount, int max_order, int autocorPartCount, out int partCount)
		{
			if (frame.blocksize <= 4)
			{
				partCount = 0;
				return;
			}

			uint cbits = get_precision(frame.blocksize) + 1;
			int partSize = 256 - 32;

			partCount = (frame.blocksize + partSize - 1) / partSize;

			if (partCount > maxResidualParts)
				throw new Exception("invalid combination of block size and LPC order");

			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 0, (uint)cudaResidualOutput.Pointer);
			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 1, (uint)cudaSamples.Pointer);
			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 2, (uint)cudaResidualTasks.Pointer);
			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 3, (uint)max_order);
			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 4, (uint)frame.blocksize);
			cuda.SetParameter(cudaEstimateResidual, sizeof(uint) * 5, (uint)partSize);
			cuda.SetParameterSize(cudaEstimateResidual, sizeof(uint) * 6);
			cuda.SetFunctionBlockShape(cudaEstimateResidual, 64, 4, 1);

			//cuda.SetParameter(cudaSumResidualChunks, 0, (uint)cudaResidualSums.Pointer);
			//cuda.SetParameter(cudaSumResidualChunks, sizeof(uint), (uint)cudaResidualTasks.Pointer);
			//cuda.SetParameter(cudaSumResidualChunks, sizeof(uint) * 2, (uint)cudaResidualOutput.Pointer);
			//cuda.SetParameter(cudaSumResidualChunks, sizeof(uint) * 3, (uint)frame.blocksize);
			//cuda.SetParameter(cudaSumResidualChunks, sizeof(uint) * 4, (uint)partSize);
			//cuda.SetParameterSize(cudaSumResidualChunks, sizeof(uint) * 5U);
			//cuda.SetFunctionBlockShape(cudaSumResidualChunks, residualThreads, 1, 1);

			cuda.SetParameter(cudaSumResidual, 0, (uint)cudaResidualTasks.Pointer);
			cuda.SetParameter(cudaSumResidual, sizeof(uint), (uint)cudaResidualOutput.Pointer);
			cuda.SetParameter(cudaSumResidual, sizeof(uint) * 2, (uint)partSize);
			cuda.SetParameter(cudaSumResidual, sizeof(uint) * 3, (uint)partCount);
			cuda.SetParameterSize(cudaSumResidual, sizeof(uint) * 4U);
			cuda.SetFunctionBlockShape(cudaSumResidual, 64, 1, 1);

			// issue work to the GPU
			cuda.LaunchAsync(cudaEstimateResidual, partCount, nResidualTasks / 4, cudaStream);
			//cuda.LaunchAsync(cudaSumResidualChunks, partCount, nResidualTasks, cudaStream);
			cuda.LaunchAsync(cudaSumResidual, 1, nResidualTasks, cudaStream);
			cuda.CopyDeviceToHostAsync(cudaResidualTasks, residualTasksPtr, (uint)(sizeof(encodeResidualTaskStruct) * nResidualTasks), cudaStream);
			cuda.SynchronizeStream(cudaStream);
		}

		unsafe void compute_autocorellation(FlacFrame frame, int channelsCount, int max_order, out int partCount)
		{
			int autocorThreads = 256;
			int partSize = 2 * autocorThreads - max_order;
			partSize &= 0xffffff0;

			partCount = (frame.blocksize + partSize - 1) / partSize;
			if (partCount > maxAutocorParts)
				throw new Exception("internal error");

			if (frame.blocksize <= 4)
				return;

			cuda.SetParameter(cudaComputeAutocor, 0, (uint)cudaAutocorOutput.Pointer);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint), (uint)cudaSamples.Pointer);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint) * 2, (uint)cudaWindow.Pointer);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint) * 3, (uint)cudaAutocorTasks.Pointer);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint) * 4, (uint)max_order);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint) * 4 + sizeof(uint), (uint)frame.blocksize);
			cuda.SetParameter(cudaComputeAutocor, sizeof(uint) * 4 + sizeof(uint) * 2, (uint)partSize);
			cuda.SetParameterSize(cudaComputeAutocor, (uint)(sizeof(uint) * 4) + sizeof(uint) * 3);
			cuda.SetFunctionBlockShape(cudaComputeAutocor, autocorThreads, 1, 1);

			cuda.SetParameter(cudaComputeLPC, 0, (uint)cudaResidualTasks.Pointer);
			cuda.SetParameter(cudaComputeLPC, sizeof(uint), (uint)cudaAutocorOutput.Pointer);
			cuda.SetParameter(cudaComputeLPC, sizeof(uint) * 2, (uint)cudaAutocorTasks.Pointer);
			cuda.SetParameter(cudaComputeLPC, sizeof(uint) * 3, (uint)max_order);
			cuda.SetParameter(cudaComputeLPC, sizeof(uint) * 3 + sizeof(uint), (uint)partCount);
			cuda.SetParameterSize(cudaComputeLPC, (uint)(sizeof(uint) * 3) + sizeof(uint) * 2);
			cuda.SetFunctionBlockShape(cudaComputeLPC, 64, 1, 1);

			// issue work to the GPU
			cuda.CopyHostToDeviceAsync(cudaSamples, samplesBufferPtr, (uint)(sizeof(int) * FlaCudaWriter.MAX_BLOCKSIZE * channelsCount), cudaStream);
			cuda.LaunchAsync(cudaComputeAutocor, partCount, nAutocorTasks, cudaStream);
			cuda.LaunchAsync(cudaComputeLPC, 1, nAutocorTasks, cudaStream);
			cuda.SynchronizeStream(cudaStream);
			//cuda.CopyDeviceToHostAsync(cudaResidualTasks, residualTasksPtr, (uint)(sizeof(encodeResidualTaskStruct) * nResidualTasks), cudaStream1);
		}
	
		unsafe int encode_frame(out int size)
		{
			int* s = (int*)samplesBufferPtr;
			fixed (int* r = residualBuffer)
			fixed (float* window = windowBuffer)
			{
				frame.InitSize(eparams.block_size, eparams.variable_block_size != 0);

				bool doMidside = channels == 2 && eparams.do_midside;
				int channelCount = doMidside ? 2 * channels : channels;

				if (frame.blocksize != _windowsize && frame.blocksize > 4)
				{
					_windowsize = frame.blocksize;
					_windowcount = 0;
					calculate_window(window, lpc.window_welch, WindowFunction.Welch);
					calculate_window(window, lpc.window_tukey, WindowFunction.Tukey);
					calculate_window(window, lpc.window_hann, WindowFunction.Hann);
					calculate_window(window, lpc.window_flattop, WindowFunction.Flattop);
					calculate_window(window, lpc.window_bartlett, WindowFunction.Bartlett);
					if (_windowcount == 0)
						throw new Exception("invalid windowfunction");
					cuda.CopyHostToDevice<float>(cudaWindow, windowBuffer);
					initialize_autocorTasks(channelCount, eparams.max_prediction_order);
				}

				if (doMidside)
					channel_decorrelation(s, s + FlaCudaWriter.MAX_BLOCKSIZE, s + 2 * FlaCudaWriter.MAX_BLOCKSIZE, s + 3 * FlaCudaWriter.MAX_BLOCKSIZE, frame.blocksize);

				frame.window_buffer = window;
				for (int ch = 0; ch < channelCount; ch++)
					frame.subframes[ch].Init(s + ch * FlaCudaWriter.MAX_BLOCKSIZE, r + ch * FlaCudaWriter.MAX_BLOCKSIZE,
						bits_per_sample + (doMidside && ch == 3 ? 1U : 0U), get_wasted_bits(s + ch * FlaCudaWriter.MAX_BLOCKSIZE, frame.blocksize));

				int autocorPartCount, residualPartCount;
				compute_autocorellation(frame, channelCount, eparams.max_prediction_order, out autocorPartCount);
				estimate_residual(frame, channelCount, eparams.max_prediction_order, autocorPartCount, out residualPartCount);
				select_best_methods(frame, channelCount, eparams.max_prediction_order, residualPartCount);

				if (doMidside)
				{
					measure_frame_size(frame, true);
					frame.ChooseSubframes();
				}
				else
					frame.ch_mode = channels != 2 ? ChannelMode.NotStereo : ChannelMode.LeftRight;

				encode_residual(frame);

				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);

				output_frame_header(frame, bitwriter);
				output_subframes(frame, bitwriter);
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
				int* r = (int*)samplesBufferPtr;
				fixed (int* s = verifyBuffer)
					for (int ch = 0; ch < channels; ch++)
						AudioSamples.MemCpy(s + ch * FlaCudaWriter.MAX_BLOCKSIZE, r + ch * FlaCudaWriter.MAX_BLOCKSIZE, eparams.block_size);
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
						if (AudioSamples.MemCmp(s + ch * FlaCudaWriter.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, bs))
							throw new Exception("validation failed!");
				}
			}

			if (bs < eparams.block_size)
			{
				int* s = (int*)samplesBufferPtr;
				for (int ch = 0; ch < channels; ch++)
					AudioSamples.MemCpy(s + ch * FlaCudaWriter.MAX_BLOCKSIZE, s + bs + ch * FlaCudaWriter.MAX_BLOCKSIZE, eparams.block_size - bs);
			}

			samplesInBuffer -= bs;

			return bs;
		}

		public unsafe void Write(int[,] buff, int pos, int sampleCount)
		{
			if (!inited)
			{
				cuda = new CUDA(true, InitializationFlags.None);
				cuda.CreateContext(0, CUCtxFlags.BlockingSync);
				using (Stream cubin = GetType().Assembly.GetManifestResourceStream(GetType(), "flacuda.cubin"))
				using (StreamReader sr = new StreamReader(cubin))
					cuda.LoadModule(new ASCIIEncoding().GetBytes(sr.ReadToEnd()));
				//cuda.LoadModule(System.IO.Path.Combine(Environment.CurrentDirectory, "flacuda.cubin"));
				cudaComputeAutocor = cuda.GetModuleFunction("cudaComputeAutocor");
				cudaComputeLPC = cuda.GetModuleFunction("cudaComputeLPC");
				cudaEstimateResidual = cuda.GetModuleFunction("cudaEstimateResidual");
				cudaSumResidual = cuda.GetModuleFunction("cudaSumResidual");
				cudaSumResidualChunks = cuda.GetModuleFunction("cudaSumResidualChunks");
				cudaEncodeResidual = cuda.GetModuleFunction("cudaEncodeResidual");
				cudaSamples = cuda.Allocate((uint)(sizeof(int) * FlaCudaWriter.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels)));
				cudaWindow = cuda.Allocate((uint)sizeof(float) * FlaCudaWriter.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS);
				cudaAutocorTasks = cuda.Allocate((uint)(sizeof(computeAutocorTaskStruct) * (channels == 2 ? 4 : channels) * lpc.MAX_LPC_WINDOWS));
				cudaAutocorOutput = cuda.Allocate((uint)(sizeof(float) * (lpc.MAX_LPC_ORDER + 1) * (channels == 2 ? 4 : channels) * lpc.MAX_LPC_WINDOWS) * maxAutocorParts);
				cudaResidualTasks = cuda.Allocate((uint)(sizeof(encodeResidualTaskStruct) * (channels == 2 ? 4 : channels) * (lpc.MAX_LPC_ORDER * lpc.MAX_LPC_WINDOWS + 4)));
				cudaResidualOutput = cuda.Allocate((uint)(sizeof(int) * FlaCudaWriter.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels) * (lpc.MAX_LPC_ORDER * lpc.MAX_LPC_WINDOWS + 4)));
				cudaResidualSums = cuda.Allocate((uint)(sizeof(int) * (channels == 2 ? 4 : channels) * (lpc.MAX_LPC_ORDER * lpc.MAX_LPC_WINDOWS + 4) * maxResidualParts));
				//cudaResidualOutput = cuda.Allocate((uint)(sizeof(int) * (channels == 2 ? 4 : channels) * (lpc.MAX_LPC_ORDER * lpc.MAX_LPC_WINDOWS + 4) * maxResidualParts));
				CUResult cuErr = CUDADriver.cuMemAllocHost(ref samplesBufferPtr, (uint)(sizeof(int) * (channels == 2 ? 4 : channels) * FlaCudaWriter.MAX_BLOCKSIZE));
				if (cuErr == CUResult.Success)
					cuErr = CUDADriver.cuMemAllocHost(ref autocorTasksPtr, (uint)(sizeof(computeAutocorTaskStruct) * (channels == 2 ? 4 : channels) * lpc.MAX_LPC_WINDOWS));
				if (cuErr == CUResult.Success)
					cuErr = CUDADriver.cuMemAllocHost(ref residualTasksPtr, (uint)(sizeof(encodeResidualTaskStruct) * (channels == 2 ? 4 : channels) * (lpc.MAX_LPC_WINDOWS * lpc.MAX_LPC_ORDER + 8)));
				if (cuErr != CUResult.Success)
				{
					if (samplesBufferPtr != IntPtr.Zero) CUDADriver.cuMemFreeHost(samplesBufferPtr); samplesBufferPtr = IntPtr.Zero;
					if (autocorTasksPtr != IntPtr.Zero) CUDADriver.cuMemFreeHost(autocorTasksPtr); autocorTasksPtr = IntPtr.Zero;
					if (residualTasksPtr != IntPtr.Zero) CUDADriver.cuMemFreeHost(residualTasksPtr); residualTasksPtr = IntPtr.Zero;
					throw new CUDAException(cuErr);
				}				
				cudaStream = cuda.CreateStream();
				cudaStream1 = cuda.CreateStream();
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

		string vendor_string = "FlaCuda#0.1";

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
			bitwriter.writebits(7, (int)MetadataType.StreamInfo);
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
			bitwriter.writebits(7, (int)MetadataType.VorbisComment);
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
			bitwriter.writebits(7, (int)MetadataType.Seektable);
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
			bitwriter.writebits(7, (int)MetadataType.Padding);
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
				verifyBuffer = new int[FlaCudaWriter.MAX_BLOCKSIZE * channels];
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

		// stereo decorrelation method
		// set by user prior to calling flake_encode_init
		// if set to less than 0, it is chosen based on compression.
		// valid values are 0 to 2
		// 0 = independent L+R channels
		// 1 = mid-side encoding
		public bool do_midside;

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
		public int lpc_max_precision_search;

		public int lpc_min_precision_search;

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
			do_midside = true;
			block_size = 0;
			block_time_ms = 105;			
			min_prediction_order = 1;
			max_prediction_order = 12;
			min_partition_order = 0;
			max_partition_order = 6;
			variable_block_size = 0;
			lpc_min_precision_search = 1;
			lpc_max_precision_search = 1;
			do_md5 = true;
			do_verify = false;
			do_seektable = true; 

			// differences from level 7
			switch (lvl)
			{
				case 0:
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					max_prediction_order = 8;
					max_partition_order = 4;
					break;
				case 1:
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					max_prediction_order = 8;
					max_partition_order = 4;
					break;
				case 2:
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					max_partition_order = 4;
					break;
				case 3:
					window_function = WindowFunction.Bartlett;
					max_prediction_order = 8;
					break;
				case 4:
					window_function = WindowFunction.Bartlett;
					max_prediction_order = 8;
					break;
				case 5:
					window_function = WindowFunction.Bartlett;
					break;
				case 6:
					//max_prediction_order = 10;
					break;
				case 7:
					break;
				case 8:
					lpc_max_precision_search = 2;
					break;
				case 9:
					window_function = WindowFunction.Bartlett;
					max_prediction_order = 32;
					break;
				case 10:
					max_prediction_order = 32;
					//lpc_max_precision_search = 2;
					break;
				case 11:
					max_prediction_order = 32;
					//lpc_max_precision_search = 2;
					variable_block_size = 4;
					break;
			}

			return 0;
		}
	}

	unsafe struct computeAutocorTaskStruct
	{
		public int samplesOffs;
		public int windowOffs;
	};
	
	unsafe struct encodeResidualTaskStruct
	{
		public int residualOrder;
		public int samplesOffs;
		public int shift;
		public int cbits;
		public int size;
		public fixed int reserved[11];
		public fixed int coefs[32];
	};
}
