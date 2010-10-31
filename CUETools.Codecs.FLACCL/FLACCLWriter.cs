/**
 * CUETools.FLACCL: FLAC audio encoder using CUDA
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
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Text;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;
using OpenCLNet;

namespace CUETools.Codecs.FLACCL
{
	public class FLACCLWriterSettings
	{
		public FLACCLWriterSettings() 
		{ 
			this.DoVerify = false; 
			this.GPUOnly = true;
            this.MappedMemory = false;
			this.DoMD5 = true; 
			this.GroupSize = 128;
			this.DeviceType = OpenCLDeviceType.GPU;
		}

		[DefaultValue(false)]
		[DisplayName("Verify")]
		[SRDescription(typeof(Properties.Resources), "DoVerifyDescription")]
		public bool DoVerify { get; set; }

		[DefaultValue(true)]
		[DisplayName("MD5")]
		[SRDescription(typeof(Properties.Resources), "DoMD5Description")]
		public bool DoMD5 { get; set; }

		[DefaultValue(true)]
		[SRDescription(typeof(Properties.Resources), "DescriptionGPUOnly")]
		public bool GPUOnly { get; set; }

        [DefaultValue(false)]
        [SRDescription(typeof(Properties.Resources), "DescriptionMappedMemory")]
        public bool MappedMemory { get; set; }

		[DefaultValue(128)]
		[SRDescription(typeof(Properties.Resources), "DescriptionGroupSize")]
		public int GroupSize { get; set; }

		[SRDescription(typeof(Properties.Resources), "DescriptionDefines")]
		public string Defines { get; set; }

		[SRDescription(typeof(Properties.Resources), "DescriptionPlatform")]
		public string Platform { get; set; }

		[DefaultValue(OpenCLDeviceType.GPU)]
		[SRDescription(typeof(Properties.Resources), "DescriptionDeviceType")]
		public OpenCLDeviceType DeviceType { get; set; }

		int cpu_threads = 1;
		[DefaultValue(1)]
		[SRDescription(typeof(Properties.Resources), "DescriptionCPUThreads")]
		public int CPUThreads
		{
			get
			{
				return cpu_threads;
			}
			set
			{
				if (value < 0 || value > 16)
					throw new Exception("CPUThreads must be between 0..16");
				cpu_threads = value;
			}
		}
	}

	public enum OpenCLDeviceType : ulong
	{
		CPU = DeviceType.CPU,
		GPU = DeviceType.GPU
	}

	[AudioEncoderClass("FLACCL", "flac", true, "0 1 2 3 4 5 6 7 8 9 10 11", "8", 2, typeof(FLACCLWriterSettings))]
	//[AudioEncoderClass("FLACCL nonsub", "flac", true, "9 10 11", "9", 1, typeof(FLACCLWriterSettings))]
	public class FLACCLWriter : IAudioDest
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
		int sample_count = -1;

		internal FlakeEncodeParams eparams;

		// maximum frame size in bytes
		// this can be used to allocate memory for output
		int max_frame_size;

		int frame_count = 0;
		int frame_pos = 0;

		long first_frame_offset = 0;

		TimeSpan _userProcessorTime;

		// header bytes
		// allocated by flake_encode_init and freed by flake_encode_close
		byte[] header;

		int samplesInBuffer = 0;
		int max_frames = 0;

		int _compressionLevel = 7;
		int _blocksize = 0;
		int _totalSize = 0;

		Crc8 crc8;
		Crc16 crc16;
		MD5 md5;

		SeekPoint[] seek_table;
		int seek_table_offset = -1;

		bool inited = false;

		OpenCLManager OCLMan;
		Context openCLContext;
		Program openCLProgram;

		FLACCLTask task1;
		FLACCLTask task2;
		FLACCLTask[] cpu_tasks;
		int oldest_cpu_task = 0;

		AudioPCMConfig _pcm;

		public const int MAX_BLOCKSIZE = 4096 * 16;
		internal const int maxFrames = 128;

		public FLACCLWriter(string path, Stream IO, AudioPCMConfig pcm)
		{
			_pcm = pcm;

			if (pcm.BitsPerSample != 16)
				throw new Exception("Bits per sample must be 16.");
			if (pcm.ChannelCount != 2)
				throw new Exception("ChannelCount must be 2.");

			channels = pcm.ChannelCount;
			sample_rate = pcm.SampleRate;
			bits_per_sample = (uint) pcm.BitsPerSample;

			// flake_validate_params

			_path = path;
			_IO = IO;

			eparams.flake_set_defaults(_compressionLevel, !_settings.GPUOnly);
			eparams.padding_size = 8192;

			crc8 = new Crc8();
			crc16 = new Crc16();
		}

		public FLACCLWriter(string path, AudioPCMConfig pcm)
			: this(path, null, pcm)
		{
		}

		public int TotalSize
		{
			get
			{
				return _totalSize;
			}
		}

		public long Padding
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
				eparams.flake_set_defaults(_compressionLevel, !_settings.GPUOnly);
			}
		}

		internal FLACCLWriterSettings _settings = new FLACCLWriterSettings();

		public object Settings
		{
			get
			{
				return _settings;
			}
			set
			{
				if (value as FLACCLWriterSettings == null)
					throw new Exception("Unsupported options " + value);
				_settings = value as FLACCLWriterSettings;
				if (_settings.DeviceType == OpenCLDeviceType.CPU)
				{
					_settings.GroupSize = 1;
					//_settings.GPUOnly = true;
                    _settings.MappedMemory = true;
				}
				eparams.flake_set_defaults(_compressionLevel, !_settings.GPUOnly);
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
				int nFrames = samplesInBuffer / eparams.block_size;
				if (nFrames > 0)
					do_output_frames(nFrames);
				if (samplesInBuffer > 0)
				{
					eparams.block_size = samplesInBuffer;
					do_output_frames(1);
				}
				if (task2.frameCount > 0)
				{
					if (cpu_tasks != null)
					{
						for (int i = 0; i < cpu_tasks.Length; i++)
						{
							wait_for_cpu_task();
							FLACCLTask task = cpu_tasks[oldest_cpu_task];
							oldest_cpu_task = (oldest_cpu_task + 1) % cpu_tasks.Length;
							if (task.frameCount > 0)
							{
								write_result(task);
								task.frameCount = 0;
							}
						}
					}
					task2.openCLCQ.Finish(); // cuda.SynchronizeStream(task2.stream);
					process_result(task2);
					write_result(task2);
					task2.frameCount = 0;
				}

				if (_IO.CanSeek)
				{
					if (sample_count <= 0 && _position != 0)
					{
						BitWriter bitwriter = new BitWriter(header, 0, 4);
						bitwriter.writebits(32, (int)_position);
						bitwriter.flush();
						_IO.Position = 22;
						_IO.Write(header, 0, 4);
					}

					if (md5 != null)
					{
						md5.TransformFinalBlock(new byte[] { 0 }, 0, 0);
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

				task1.Dispose();
				task2.Dispose();
				if (cpu_tasks != null)
					foreach (FLACCLTask task in cpu_tasks)
						task.Dispose();
				openCLProgram.Dispose();
				OCLMan.Dispose();
				inited = false;
			}
		}

		public void Close()
		{
			DoClose();
			if (sample_count > 0 && _position != sample_count)
				throw new Exception(string.Format("Samples written differs from the expected sample count. Expected {0}, got {1}.", sample_count, _position));
		}

		public void Delete()
		{
			if (inited)
			{
				_IO.Close();
				task1.Dispose();
				task2.Dispose();
				if (cpu_tasks != null)
					foreach (FLACCLTask task in cpu_tasks)
						task.Dispose();
				openCLProgram.Dispose();
				OCLMan.Dispose();
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

		public int OrdersPerWindow
		{
			get
			{
				return eparams.orders_per_window;
			}
			set
			{
				if (value < 0 || value > 32)
					throw new Exception("invalid OrdersPerWindow " + value.ToString());
				eparams.orders_per_window = value;
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
				if (value > lpc.MAX_LPC_ORDER || value < eparams.min_prediction_order)
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
				if (value < 0 || value > 4)
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
				if (value > 4 || value < 0)
					throw new Exception("invalid MaxFixedOrder " + value.ToString());
				eparams.max_fixed_order = value;
			}
		}

		public bool DoConstant
		{
			get { return eparams.do_constant; }
			set { eparams.do_constant = value; }
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

		public AudioPCMConfig PCM
		{
			get { return _pcm; }
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

		static unsafe uint calc_optimal_rice_params(int porder, int* parm, uint* sums, uint n, uint pred_order)
		{
			uint part = (1U << porder);
			uint cnt = (n >> porder) - pred_order;
			int k = cnt > 0 ? Math.Min(Flake.MAX_RICE_PARAM, BitReader.log2i(sums[0] / cnt)) : 0;
			uint all_bits = cnt * ((uint)k + 1U) + (sums[0] >> k);
			parm[0] = k;
			cnt = (n >> porder);
			for (uint i = 1; i < part; i++)
			{
				k = Math.Min(Flake.MAX_RICE_PARAM, BitReader.log2i(sums[i] / cnt));
				all_bits += cnt * ((uint)k + 1U) + (sums[i] >> k);
				parm[i] = k;
			}
			return all_bits + (4 * part);
		}

		static unsafe void calc_lower_sums(int pmin, int pmax, uint* sums)
		{
			for (int i = pmax - 1; i >= pmin; i--)
			{
				for (int j = 0; j < (1 << i); j++)
				{
					sums[i * Flake.MAX_PARTITIONS + j] =
						sums[(i + 1) * Flake.MAX_PARTITIONS + 2 * j] +
						sums[(i + 1) * Flake.MAX_PARTITIONS + 2 * j + 1];
				}
			}
		}

		static unsafe void calc_sums(int pmin, int pmax, uint* data, uint n, uint pred_order, uint* sums)
		{
			int parts = (1 << pmax);
			uint* res = data + pred_order;
			uint cnt = (n >> pmax) - pred_order;
			uint sum = 0;
			for (uint j = cnt; j > 0; j--)
				sum += *(res++);
			sums[0] = sum;
			cnt = (n >> pmax);
			for (int i = 1; i < parts; i++)
			{
				sum = 0;
				for (uint j = cnt; j > 0; j--)
					sum += *(res++);
				sums[i] = sum;
			}
		}

		/// <summary>
		/// Special case when (n >> pmax) == 18
		/// </summary>
		/// <param name="pmin"></param>
		/// <param name="pmax"></param>
		/// <param name="data"></param>
		/// <param name="n"></param>
		/// <param name="pred_order"></param>
		/// <param name="sums"></param>
		static unsafe void calc_sums18(int pmin, int pmax, uint* data, uint n, uint pred_order, uint* sums)
		{
			int parts = (1 << pmax);
			uint* res = data + pred_order;
			uint cnt = 18 - pred_order;
			uint sum = 0;
			for (uint j = cnt; j > 0; j--)
				sum += *(res++);
			sums[0] = sum;
			for (int i = 1; i < parts; i++)
			{
				sums[i] =
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++);
			}
		}

		/// <summary>
		/// Special case when (n >> pmax) == 18
		/// </summary>
		/// <param name="pmin"></param>
		/// <param name="pmax"></param>
		/// <param name="data"></param>
		/// <param name="n"></param>
		/// <param name="pred_order"></param>
		/// <param name="sums"></param>
		static unsafe void calc_sums16(int pmin, int pmax, uint* data, uint n, uint pred_order, uint* sums)
		{
			int parts = (1 << pmax);
			uint* res = data + pred_order;
			uint cnt = 16 - pred_order;
			uint sum = 0;
			for (uint j = cnt; j > 0; j--)
				sum += *(res++);
			sums[0] = sum;
			for (int i = 1; i < parts; i++)
			{
				sums[i] =
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++) +
					*(res++) + *(res++) + *(res++) + *(res++);
			}
		}

		static unsafe uint calc_rice_params(RiceContext rc, int pmin, int pmax, int* data, uint n, uint pred_order)
		{
			uint* udata = stackalloc uint[(int)n];
			uint* sums = stackalloc uint[(pmax + 1) * Flake.MAX_PARTITIONS];
			int* parm = stackalloc int[(pmax + 1) * Flake.MAX_PARTITIONS];
			//uint* bits = stackalloc uint[Flake.MAX_PARTITION_ORDER];

			//assert(pmin >= 0 && pmin <= Flake.MAX_PARTITION_ORDER);
			//assert(pmax >= 0 && pmax <= Flake.MAX_PARTITION_ORDER);
			//assert(pmin <= pmax);

			for (uint i = 0; i < n; i++)
				udata[i] = (uint)((data[i] << 1) ^ (data[i] >> 31));

			// sums for highest level
			if ((n >> pmax) == 18)
				calc_sums18(pmin, pmax, udata, n, pred_order, sums + pmax * Flake.MAX_PARTITIONS);
			else if ((n >> pmax) == 16)
				calc_sums16(pmin, pmax, udata, n, pred_order, sums + pmax * Flake.MAX_PARTITIONS);
			else
				calc_sums(pmin, pmax, udata, n, pred_order, sums + pmax * Flake.MAX_PARTITIONS);
			// sums for lower levels
			calc_lower_sums(pmin, pmax, sums);

			uint opt_bits = AudioSamples.UINT32_MAX;
			int opt_porder = pmin;
			for (int i = pmin; i <= pmax; i++)
			{
				uint bits = calc_optimal_rice_params(i, parm + i * Flake.MAX_PARTITIONS, sums + i * Flake.MAX_PARTITIONS, n, pred_order);
				if (bits <= opt_bits)
				{
					opt_bits = bits;
					opt_porder = i;
				}
			}

			rc.porder = opt_porder;
			fixed (int* rparms = rc.rparams)
				AudioSamples.MemCpy(rparms, parm + opt_porder * Flake.MAX_PARTITIONS, (1 << opt_porder));

			return opt_bits;
		}

		internal static int get_max_p_order(int max_porder, int n, int order)
		{
			int porder = Math.Min(max_porder, BitReader.log2i(n ^ (n - 1)));
			if (order > 0)
				porder = Math.Min(porder, BitReader.log2i(n / order));
			return porder;
		}

		unsafe void output_frame_header(FlacFrame frame)
		{
			frame.writer.writebits(15, 0x7FFC);
			frame.writer.writebits(1, eparams.variable_block_size > 0 ? 1 : 0);
			frame.writer.writebits(4, frame.bs_code0);
			frame.writer.writebits(4, sr_code0);
			if (frame.ch_mode == ChannelMode.NotStereo)
				frame.writer.writebits(4, ch_code);
			else
				frame.writer.writebits(4, (int)frame.ch_mode);
			frame.writer.writebits(3, bps_code);
			frame.writer.writebits(1, 0);
			frame.writer.write_utf8(frame.frame_number);

			// custom block size
			if (frame.bs_code1 >= 0)
			{
				if (frame.bs_code1 < 256)
					frame.writer.writebits(8, frame.bs_code1);
				else
					frame.writer.writebits(16, frame.bs_code1);
			}

			// custom sample rate
			if (sr_code1 > 0)
			{
				if (sr_code1 < 256)
					frame.writer.writebits(8, sr_code1);
				else
					frame.writer.writebits(16, sr_code1);
			}

			// CRC-8 of frame header
			frame.writer.flush();
			byte crc = crc8.ComputeChecksum(frame.writer.Buffer, frame.writer_offset, frame.writer.Length - frame.writer_offset);
			frame.writer.writebits(8, crc);
		}

		unsafe int measure_residual(FlacFrame frame, FlacSubframeInfo sub, int pos, int cnt, int k)
		{
			int q = 0;
			for (int i = pos; i < pos + cnt; i++)
			{
				int v = sub.best.residual[i];
				v = (v << 1) ^ (v >> 31);
				q += (v >> k);
			}
			return (k + 1) * cnt + q;
		}

		unsafe int measure_residual(FlacFrame frame, FlacSubframeInfo sub)
		{
			// partition order
			int porder = sub.best.rc.porder;
			int psize = frame.blocksize >> porder;
			//assert(porder >= 0);
			int size = 6 + (4 << porder);
			size += measure_residual(frame, sub, sub.best.order, psize - sub.best.order, sub.best.rc.rparams[0]);
			// residual
			for (int p = 1; p < (1 << porder); p++)
				size += measure_residual(frame, sub, p * psize, psize, sub.best.rc.rparams[p]);
			return size;
		}

		unsafe void output_residual(FlacFrame frame, FlacSubframeInfo sub)
		{
			// rice-encoded block
			frame.writer.writebits(2, 0);

			// partition order
			int porder = sub.best.rc.porder;
			int psize = frame.blocksize >> porder;
			//assert(porder >= 0);
			frame.writer.writebits(4, porder);
			int res_cnt = psize - sub.best.order;

			// residual
			int j = sub.best.order;
			fixed (byte* fixbuf = frame.writer.Buffer)
			for (int p = 0; p < (1 << porder); p++)
			{
				int k = sub.best.rc.rparams[p];
				frame.writer.writebits(4, k);
				if (p == 1) res_cnt = psize;
				int cnt = Math.Min(res_cnt, frame.blocksize - j);
				frame.writer.write_rice_block_signed(fixbuf, k, sub.best.residual + j, cnt);
				j += cnt;
			}
		}

		unsafe void 
		output_subframe_constant(FlacFrame frame, FlacSubframeInfo sub)
		{
			frame.writer.writebits_signed(sub.obits, sub.samples[0]);
		}

		unsafe void
		output_subframe_verbatim(FlacFrame frame, FlacSubframeInfo sub)
		{
			int n = frame.blocksize;
			for (int i = 0; i < n; i++)
				frame.writer.writebits_signed(sub.obits, sub.samples[i]);
			// Don't use residual here, because we don't copy samples to residual for verbatim frames.
		}

		unsafe void
		output_subframe_fixed(FlacFrame frame, FlacSubframeInfo sub)
		{
			// warm-up samples
			for (int i = 0; i < sub.best.order; i++)
				frame.writer.writebits_signed(sub.obits, sub.samples[i]);

			// residual
			output_residual(frame, sub);
		}

		unsafe uint
		measure_subframe_lpc(FlacFrame frame, FlacSubframeInfo sub)
		{
			return (uint)(sub.best.order * sub.obits + 9 + sub.best.order * sub.best.cbits + measure_residual(frame, sub));
		}

		unsafe uint
		measure_subframe_fixed(FlacFrame frame, FlacSubframeInfo sub)
		{
			return (uint)(sub.best.order * sub.obits + measure_residual(frame, sub));
		}

		unsafe uint
		measure_subframe(FlacFrame frame, FlacSubframeInfo sub)
		{
			switch (sub.best.type)
			{
				case SubframeType.Fixed:
					return measure_subframe_fixed(frame, sub);
				case SubframeType.LPC:
					return measure_subframe_lpc(frame, sub);
			}
			throw new Exception("not supported subframe type");
		}

		unsafe void
		output_subframe_lpc(FlacFrame frame, FlacSubframeInfo sub)
		{
			// warm-up samples
			for (int i = 0; i < sub.best.order; i++)
				frame.writer.writebits_signed(sub.obits, sub.samples[i]);

			// LPC coefficients
			frame.writer.writebits(4, sub.best.cbits - 1);
			frame.writer.writebits_signed(5, sub.best.shift);
			for (int i = 0; i < sub.best.order; i++)
				frame.writer.writebits_signed(sub.best.cbits, sub.best.coefs[i]);
			
			// residual
			output_residual(frame, sub);
		}

		unsafe void output_subframes(FlacFrame frame)
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
				frame.writer.writebits(1, 0);
				frame.writer.writebits(6, type_code);
				frame.writer.writebits(1, sub.wbits != 0 ? 1 : 0);
				if (sub.wbits > 0)
					frame.writer.writebits((int)sub.wbits, 1);

				//if (frame_writer.Length >= frame_buffer.Length)
				//    throw new Exception("buffer overflow");

				// subframe
				switch (sub.best.type)
				{
					case SubframeType.Constant:
						output_subframe_constant(frame, sub);
						break;
					case SubframeType.Verbatim:
						output_subframe_verbatim(frame, sub);
						break;
					case SubframeType.Fixed:
						output_subframe_fixed(frame, sub);
						break;
					case SubframeType.LPC:
						output_subframe_lpc(frame, sub);
						break;
				}
				//if (frame_writer.Length >= frame_buffer.Length)
				//    throw new Exception("buffer overflow");
			}
		}

		void output_frame_footer(FlacFrame frame)
		{
			frame.writer.flush();
			ushort crc = crc16.ComputeChecksum(frame.writer.Buffer, frame.writer_offset, frame.writer.Length - frame.writer_offset);
			frame.writer.writebits(16, crc);
			frame.writer.flush();
		}

		unsafe delegate void window_function(float* window, int size);

		unsafe void calculate_window(FLACCLTask task, window_function func, WindowFunction flag)
		{
			if ((eparams.window_function & flag) == 0 || task.nWindowFunctions == lpc.MAX_LPC_WINDOWS)
				return;

			func(((float*)task.clWindowFunctionsPtr) + task.nWindowFunctions * task.frameSize, task.frameSize);
			//int sz = _windowsize;
			//float* pos = window + _windowcount * FLACCLWriter.MAX_BLOCKSIZE * 2;
			//do
			//{
			//    func(pos, sz);
			//    if ((sz & 1) != 0)
			//        break;
			//    pos += sz;
			//    sz >>= 1;
			//} while (sz >= 32);
			task.nWindowFunctions++;
		}

		unsafe void initializeSubframeTasks(int blocksize, int channelsCount, int nFrames, FLACCLTask task)
		{
			task.frameSize = blocksize;
			task.nWindowFunctions = 0;
			if (task.frameSize > 4)
			{
				calculate_window(task, lpc.window_welch, WindowFunction.Welch);
				calculate_window(task, lpc.window_flattop, WindowFunction.Flattop);
				calculate_window(task, lpc.window_tukey, WindowFunction.Tukey);
				calculate_window(task, lpc.window_hann, WindowFunction.Hann);
				calculate_window(task, lpc.window_bartlett, WindowFunction.Bartlett);
				if (task.nWindowFunctions == 0)
					throw new Exception("invalid windowfunction");
                if (!_settings.MappedMemory)
				    task.openCLCQ.EnqueueWriteBuffer(task.clWindowFunctions, true, 0, sizeof(float) * task.nWindowFunctions * task.frameSize, task.clWindowFunctionsPtr);
			}

			task.nResidualTasks = 0;
			task.nTasksPerWindow = Math.Min(32, eparams.orders_per_window);
			task.nResidualTasksPerChannel = task.nWindowFunctions * task.nTasksPerWindow + (eparams.do_constant ? 1 : 0) + Math.Max(0, 1 + eparams.max_fixed_order - eparams.min_fixed_order);
            if (task.nResidualTasksPerChannel > 32)
                throw new Exception("too many tasks");

			//if (task.nResidualTasksPerChannel >= 4)
			//    task.nResidualTasksPerChannel = (task.nResidualTasksPerChannel + 7) & ~7;
			for (int iFrame = 0; iFrame < nFrames; iFrame++)
			{
				for (int ch = 0; ch < channelsCount; ch++)
				{
					for (int iWindow = 0; iWindow < task.nWindowFunctions; iWindow++)
					{
						// LPC tasks
						for (int order = 0; order < task.nTasksPerWindow; order++)
						{
							task.ResidualTasks[task.nResidualTasks].type = (int)SubframeType.LPC;
							task.ResidualTasks[task.nResidualTasks].channel = ch;
							task.ResidualTasks[task.nResidualTasks].obits = (int)bits_per_sample + (channels == 2 && ch == 3 ? 1 : 0);
							task.ResidualTasks[task.nResidualTasks].abits = task.ResidualTasks[task.nResidualTasks].obits;
							task.ResidualTasks[task.nResidualTasks].blocksize = blocksize;
							task.ResidualTasks[task.nResidualTasks].residualOrder = order + 1;
							task.ResidualTasks[task.nResidualTasks].samplesOffs = ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * blocksize;
							task.ResidualTasks[task.nResidualTasks].residualOffs = task.ResidualTasks[task.nResidualTasks].samplesOffs;
							task.nResidualTasks++;
						}
					}
					// Constant frames
					if (eparams.do_constant)
					{
						task.ResidualTasks[task.nResidualTasks].type = (int)SubframeType.Constant;
						task.ResidualTasks[task.nResidualTasks].channel = ch;
						task.ResidualTasks[task.nResidualTasks].obits = (int)bits_per_sample + (channels == 2 && ch == 3 ? 1 : 0);
						task.ResidualTasks[task.nResidualTasks].abits = task.ResidualTasks[task.nResidualTasks].obits;
						task.ResidualTasks[task.nResidualTasks].blocksize = blocksize;
						task.ResidualTasks[task.nResidualTasks].samplesOffs = ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * blocksize;
						task.ResidualTasks[task.nResidualTasks].residualOffs = task.ResidualTasks[task.nResidualTasks].samplesOffs;
						task.ResidualTasks[task.nResidualTasks].residualOrder = 1;
						task.ResidualTasks[task.nResidualTasks].shift = 0;
						task.ResidualTasks[task.nResidualTasks].coefs[0] = 1;
						task.nResidualTasks++;
					}
					// Fixed prediction
					for (int order = eparams.min_fixed_order; order <= eparams.max_fixed_order; order++)
					{
						task.ResidualTasks[task.nResidualTasks].type = (int)SubframeType.Fixed;
						task.ResidualTasks[task.nResidualTasks].channel = ch;
						task.ResidualTasks[task.nResidualTasks].obits = (int)bits_per_sample + (channels == 2 && ch == 3 ? 1 : 0);
						task.ResidualTasks[task.nResidualTasks].abits = task.ResidualTasks[task.nResidualTasks].obits;
						task.ResidualTasks[task.nResidualTasks].blocksize = blocksize;
						task.ResidualTasks[task.nResidualTasks].residualOrder = order;
						task.ResidualTasks[task.nResidualTasks].samplesOffs = ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * blocksize;
						task.ResidualTasks[task.nResidualTasks].residualOffs = task.ResidualTasks[task.nResidualTasks].samplesOffs;
						task.ResidualTasks[task.nResidualTasks].shift = 0;
						switch (order)
						{
							case 0:
								break;
							case 1:
								task.ResidualTasks[task.nResidualTasks].coefs[0] = 1;
								break;
							case 2:
								task.ResidualTasks[task.nResidualTasks].coefs[1] = 2;
								task.ResidualTasks[task.nResidualTasks].coefs[0] = -1;
								break;
							case 3:
								task.ResidualTasks[task.nResidualTasks].coefs[2] = 3;
								task.ResidualTasks[task.nResidualTasks].coefs[1] = -3;
								task.ResidualTasks[task.nResidualTasks].coefs[0] = 1;
								break;
							case 4:
								task.ResidualTasks[task.nResidualTasks].coefs[3] = 4;
								task.ResidualTasks[task.nResidualTasks].coefs[2] = -6;
								task.ResidualTasks[task.nResidualTasks].coefs[1] = 4;
								task.ResidualTasks[task.nResidualTasks].coefs[0] = -1;
								break;
						}
						task.nResidualTasks++;
					}
					//// Filler
					//while ((task.nResidualTasks % task.nResidualTasksPerChannel) != 0)
					//{
					//    task.ResidualTasks[task.nResidualTasks].type = (int)SubframeType.Verbatim;
					//    task.ResidualTasks[task.nResidualTasks].channel = ch;
					//    task.ResidualTasks[task.nResidualTasks].obits = (int)bits_per_sample + (channels == 2 && ch == 3 ? 1 : 0);
					//    task.ResidualTasks[task.nResidualTasks].abits = task.ResidualTasks[task.nResidualTasks].obits;
					//    task.ResidualTasks[task.nResidualTasks].blocksize = blocksize;
					//    task.ResidualTasks[task.nResidualTasks].residualOrder = 0;
					//    task.ResidualTasks[task.nResidualTasks].samplesOffs = ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * blocksize;
					//    task.ResidualTasks[task.nResidualTasks].residualOffs = task.ResidualTasks[task.nResidualTasks].samplesOffs;
					//    task.ResidualTasks[task.nResidualTasks].shift = 0;
					//    task.nResidualTasks++;
					//}
				}
			}
			if (sizeof(FLACCLSubframeTask) * task.nResidualTasks > task.residualTasksLen)
				throw new Exception("oops");

            if (!_settings.MappedMemory)
			    task.openCLCQ.EnqueueWriteBuffer(task.clResidualTasks, true, 0, sizeof(FLACCLSubframeTask) * task.nResidualTasks, task.clResidualTasksPtr);
		}

		unsafe void encode_residual(FLACCLTask task)
		{
			bool unpacked = false;
			unpack_samples(task, Math.Min(32, task.frameSize));
			for (int ch = 0; ch < channels; ch++)
			{
				switch (task.frame.subframes[ch].best.type)
				{
					case SubframeType.Constant:
						break;
					case SubframeType.Verbatim:
						if (!unpacked) unpack_samples(task, task.frameSize); unpacked = true;
						break;
					case SubframeType.Fixed:
						if (!_settings.GPUOnly)
						{
							if (!unpacked) unpack_samples(task, task.frameSize); unpacked = true;
							encode_residual_fixed(task.frame.subframes[ch].best.residual, task.frame.subframes[ch].samples,
								task.frame.blocksize, task.frame.subframes[ch].best.order);

							int pmin = get_max_p_order(eparams.min_partition_order, task.frame.blocksize, task.frame.subframes[ch].best.order);
							int pmax = get_max_p_order(eparams.max_partition_order, task.frame.blocksize, task.frame.subframes[ch].best.order);
							uint bits = (uint)(task.frame.subframes[ch].best.order * task.frame.subframes[ch].obits) + 6;
							task.frame.subframes[ch].best.size = bits + calc_rice_params(task.frame.subframes[ch].best.rc, pmin, pmax, task.frame.subframes[ch].best.residual, (uint)task.frame.blocksize, (uint)task.frame.subframes[ch].best.order);
						}
						break;
					case SubframeType.LPC:
						fixed (int* coefs = task.frame.subframes[ch].best.coefs)
						{
							ulong csum = 0;
							for (int i = task.frame.subframes[ch].best.order; i > 0; i--)
								csum += (ulong)Math.Abs(coefs[i - 1]);

#if DEBUG
							// check size
							if (_settings.GPUOnly)
							{
								uint real_size = measure_subframe(task.frame, task.frame.subframes[ch]);
								if (real_size != task.frame.subframes[ch].best.size)
									throw new Exception("size reported incorrectly");
							}
#endif

							if ((csum << task.frame.subframes[ch].obits) >= 1UL << 32 || !_settings.GPUOnly)
							{
								if (!unpacked) unpack_samples(task, task.frameSize); unpacked = true;
								if ((csum << task.frame.subframes[ch].obits) >= 1UL << 32)
									lpc.encode_residual_long(task.frame.subframes[ch].best.residual, task.frame.subframes[ch].samples, task.frame.blocksize, task.frame.subframes[ch].best.order, coefs, task.frame.subframes[ch].best.shift);
								else
									lpc.encode_residual(task.frame.subframes[ch].best.residual, task.frame.subframes[ch].samples, task.frame.blocksize, task.frame.subframes[ch].best.order, coefs, task.frame.subframes[ch].best.shift);
								int pmin = get_max_p_order(eparams.min_partition_order, task.frame.blocksize, task.frame.subframes[ch].best.order);
								int pmax = get_max_p_order(eparams.max_partition_order, task.frame.blocksize, task.frame.subframes[ch].best.order);
								uint bits = (uint)(task.frame.subframes[ch].best.order * task.frame.subframes[ch].obits) + 4 + 5 + (uint)task.frame.subframes[ch].best.order * (uint)task.frame.subframes[ch].best.cbits + 6;
#if KLJLKJLKJL
								uint oldsize = task.frame.subframes[ch].best.size;
								RiceContext rc1 = task.frame.subframes[ch].best.rc;
								task.frame.subframes[ch].best.rc = new RiceContext();
#endif
								task.frame.subframes[ch].best.size = bits + calc_rice_params(task.frame.subframes[ch].best.rc, pmin, pmax, task.frame.subframes[ch].best.residual, (uint)task.frame.blocksize, (uint)task.frame.subframes[ch].best.order);								
								task.frame.subframes[ch].best.size = measure_subframe(task.frame, task.frame.subframes[ch]);
#if KJHKJH
								// check size
								if (_settings.GPUOnly && oldsize > task.frame.subframes[ch].best.size)
									throw new Exception("unoptimal size reported");
#endif
								//if (task.frame.subframes[ch].best.size > task.frame.subframes[ch].obits * (uint)task.frame.blocksize &&
								//    oldsize <= task.frame.subframes[ch].obits * (uint)task.frame.blocksize)
								//    throw new Exception("oops");
							}
						}
						break;
				}
				if (task.frame.subframes[ch].best.size > task.frame.subframes[ch].obits * task.frame.blocksize)
				{
#if DEBUG
					throw new Exception("larger than verbatim");
#endif
					task.frame.subframes[ch].best.type = SubframeType.Verbatim;
					task.frame.subframes[ch].best.size = (uint)(task.frame.subframes[ch].obits * task.frame.blocksize);
					if (!unpacked) unpack_samples(task, task.frameSize); unpacked = true;
				}
			}
		}

		unsafe void select_best_methods(FlacFrame frame, int channelsCount, int iFrame, FLACCLTask task)
		{
			if (channelsCount == 4 && channels == 2)
			{
				if (task.BestResidualTasks[iFrame * 2].channel == 0 && task.BestResidualTasks[iFrame * 2 + 1].channel == 1)
					frame.ch_mode = ChannelMode.LeftRight;
				else if (task.BestResidualTasks[iFrame * 2].channel == 0 && task.BestResidualTasks[iFrame * 2 + 1].channel == 3)
					frame.ch_mode = ChannelMode.LeftSide;
				else if (task.BestResidualTasks[iFrame * 2].channel == 3 && task.BestResidualTasks[iFrame * 2 + 1].channel == 1)
					frame.ch_mode = ChannelMode.RightSide;
				else if (task.BestResidualTasks[iFrame * 2].channel == 2 && task.BestResidualTasks[iFrame * 2 + 1].channel == 3)
					frame.ch_mode = ChannelMode.MidSide;
				else
					throw new Exception("internal error: invalid stereo mode");
				frame.SwapSubframes(0, task.BestResidualTasks[iFrame * 2].channel);
				frame.SwapSubframes(1, task.BestResidualTasks[iFrame * 2 + 1].channel);
			}
			else
				frame.ch_mode = channels != 2 ? ChannelMode.NotStereo : ChannelMode.LeftRight;

			for (int ch = 0; ch < channels; ch++)
			{
				int index = ch + iFrame * channels;
				frame.subframes[ch].best.residual = ((int*)task.clResidualPtr) + task.BestResidualTasks[index].residualOffs;
				frame.subframes[ch].best.type = SubframeType.Verbatim;
				frame.subframes[ch].best.size = (uint)(frame.subframes[ch].obits * frame.blocksize);
				frame.subframes[ch].wbits = 0;

				if (task.BestResidualTasks[index].size < 0)
					throw new Exception("internal error");
				if (frame.blocksize > Math.Max(4, eparams.max_prediction_order) && frame.subframes[ch].best.size > task.BestResidualTasks[index].size)
				{
					frame.subframes[ch].best.type = (SubframeType)task.BestResidualTasks[index].type;
					frame.subframes[ch].best.size = (uint)task.BestResidualTasks[index].size;
					frame.subframes[ch].best.order = task.BestResidualTasks[index].residualOrder;
					frame.subframes[ch].best.cbits = task.BestResidualTasks[index].cbits;
					frame.subframes[ch].best.shift = task.BestResidualTasks[index].shift;
					frame.subframes[ch].obits -= task.BestResidualTasks[index].wbits;
					frame.subframes[ch].wbits = task.BestResidualTasks[index].wbits;
					frame.subframes[ch].best.rc.porder = task.BestResidualTasks[index].porder;
					for (int i = 0; i < task.BestResidualTasks[index].residualOrder; i++)
						frame.subframes[ch].best.coefs[i] = task.BestResidualTasks[index].coefs[task.BestResidualTasks[index].residualOrder - 1 - i];
					if (_settings.GPUOnly && (frame.subframes[ch].best.type == SubframeType.Fixed || frame.subframes[ch].best.type == SubframeType.LPC))
					{
						int* riceParams = ((int*)task.clBestRiceParamsPtr) + (index << task.max_porder);
						fixed (int* dstParams = frame.subframes[ch].best.rc.rparams)
							AudioSamples.MemCpy(dstParams, riceParams, (1 << frame.subframes[ch].best.rc.porder));
						//for (int i = 0; i < (1 << frame.subframes[ch].best.rc.porder); i++)
						//    frame.subframes[ch].best.rc.rparams[i] = riceParams[i];
						uint real_size = measure_subframe(frame, frame.subframes[ch]);
						if (real_size != task.frame.subframes[ch].best.size)
							throw new Exception("size reported incorrectly");
					}
				}
			}
		}

		unsafe void estimate_residual(FLACCLTask task, int channelsCount)
		{
			if (task.frameSize >= 4)
				task.EnqueueKernels();
		}

		/// <summary>
		/// Copy channel-interleaved input samples into separate subframes
		/// </summary>
		/// <param name="task"></param>
		/// <param name="doMidside"></param>
		unsafe void unpack_samples(FLACCLTask task, int count)
		{
			int iFrame = task.frame.frame_number;
			short* src = ((short*)task.clSamplesBytesPtr) + iFrame * channels * task.frameSize;

			switch (task.frame.ch_mode)
			{
				case ChannelMode.NotStereo:
					for (int ch = 0; ch < channels; ch++)
					{
						int* s = task.frame.subframes[ch].samples;
						int wbits = (int)task.frame.subframes[ch].wbits;
						for (int i = 0; i < count; i++)
							s[i] = src[i * channels + ch] >>= wbits;
					}
					break;
				case ChannelMode.LeftRight:
					{
						int* left = task.frame.subframes[0].samples;
						int* right = task.frame.subframes[1].samples;
						int lwbits = (int)task.frame.subframes[0].wbits;
						int rwbits = (int)task.frame.subframes[1].wbits;
						for (int i = 0; i < count; i++)
						{
							int l = *(src++);
							int r = *(src++);
							left[i] = l >> lwbits;
							right[i] = r >> rwbits;
						}
						break;
					}
				case ChannelMode.LeftSide:
					{
						int* left = task.frame.subframes[0].samples;
						int* right = task.frame.subframes[1].samples;
						int lwbits = (int)task.frame.subframes[0].wbits;
						int rwbits = (int)task.frame.subframes[1].wbits;
						for (int i = 0; i < count; i++)
						{
							int l = *(src++);
							int r = *(src++);
							left[i] = l >> lwbits;
							right[i] = (l - r) >> rwbits;
						}
						break;
					}
				case ChannelMode.RightSide:
					{
						int* left = task.frame.subframes[0].samples;
						int* right = task.frame.subframes[1].samples;
						int lwbits = (int)task.frame.subframes[0].wbits;
						int rwbits = (int)task.frame.subframes[1].wbits;
						for (int i = 0; i < count; i++)
						{
							int l = *(src++);
							int r = *(src++);
							left[i] = (l - r) >> lwbits;
							right[i] = r >> rwbits;
						}
						break;
					}
				case ChannelMode.MidSide:
					{
						int* left = task.frame.subframes[0].samples;
						int* right = task.frame.subframes[1].samples;
						int lwbits = (int)task.frame.subframes[0].wbits;
						int rwbits = (int)task.frame.subframes[1].wbits;
						for (int i = 0; i < count; i++)
						{
							int l = *(src++);
							int r = *(src++);
							left[i] = (l + r) >> (1 + lwbits);
							right[i] = (l - r) >> rwbits;
						}
						break;
					}
			}
		}

		unsafe int encode_frame(bool doMidside, int channelCount, int iFrame, FLACCLTask task, int current_frame_number)
		{
			task.frame.InitSize(task.frameSize, eparams.variable_block_size != 0);
			task.frame.frame_number = iFrame;
			task.frame.ch_mode = ChannelMode.NotStereo;

			fixed (int* smp = task.samplesBuffer)
			{
				for (int ch = 0; ch < channelCount; ch++)
					task.frame.subframes[ch].Init(
						smp + ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * task.frameSize,
						((int*)task.clResidualPtr) + ch * FLACCLWriter.MAX_BLOCKSIZE + iFrame * task.frameSize,
						_pcm.BitsPerSample + (doMidside && ch == 3 ? 1 : 0), 0);

				select_best_methods(task.frame, channelCount, iFrame, task);
				//unpack_samples(task);
				encode_residual(task);

				//task.frame.writer.Reset();
				task.frame.frame_number = current_frame_number;
				task.frame.writer_offset = task.frame.writer.Length;

				output_frame_header(task.frame);
				output_subframes(task.frame);
				output_frame_footer(task.frame);
				if (task.frame.writer.Length - task.frame.writer_offset >= max_frame_size)
					throw new Exception("buffer overflow");

				return task.frame.writer.Length - task.frame.writer_offset;
			}
		}

		unsafe void send_to_GPU(FLACCLTask task, int nFrames, int blocksize)
		{
			bool doMidside = channels == 2 && eparams.do_midside;
			int channelsCount = doMidside ? 2 * channels : channels;
			if (blocksize != task.frameSize)
				task.nResidualTasks = 0;
			task.frameCount = nFrames;
			task.frameSize = blocksize;
			task.frameNumber = eparams.variable_block_size > 0 ? frame_pos : frame_count;
			task.framePos = frame_pos;
			frame_count += nFrames;
			frame_pos += nFrames * blocksize;
            if (!_settings.MappedMemory)
			    task.openCLCQ.EnqueueWriteBuffer(task.clSamplesBytes, false, 0, sizeof(short) * channels * blocksize * nFrames, task.clSamplesBytesPtr);
			//task.openCLCQ.EnqueueUnmapMemObject(task.clSamplesBytes, task.clSamplesBytes.HostPtr);
			//task.openCLCQ.EnqueueMapBuffer(task.clSamplesBytes, true, MapFlags.WRITE, 0, task.samplesBufferLen / 2);
		}

		unsafe void run_GPU_task(FLACCLTask task)
		{
			bool doMidside = channels == 2 && eparams.do_midside;
			int channelsCount = doMidside ? 2 * channels : channels;

			if (task.nResidualTasks == 0)
				initializeSubframeTasks(task.frameSize, channelsCount, max_frames, task);

			estimate_residual(task, channelsCount);
		}

		unsafe void process_result(FLACCLTask task)
		{
			bool doMidside = channels == 2 && eparams.do_midside;
			int channelCount = doMidside ? 2 * channels : channels;

			long iSample = 0;
			long iByte = 0;
			task.frame.writer.Reset();
			task.frame.writer_offset = 0;
			for (int iFrame = 0; iFrame < task.frameCount; iFrame++)
			{
				//if (0 != eparams.variable_block_size && 0 == (task.blocksize & 7) && task.blocksize >= 128)
				//    fs = encode_frame_vbs();
				//else
				int fn = task.frameNumber + (eparams.variable_block_size > 0 ? (int)iSample : iFrame);
				int fs = encode_frame(doMidside, channelCount, iFrame, task, fn);

				if (task.verify != null)
				{					
					int decoded = task.verify.DecodeFrame(task.frame.writer.Buffer, task.frame.writer_offset, fs);
					if (decoded != fs || task.verify.Remaining != task.frameSize)
						throw new Exception("validation failed! frame size mismatch");
					fixed (int* r = task.verify.Samples)
					{
						for (int ch = 0; ch < channels; ch++)
						{
							short* res = ((short*)task.clSamplesBytesPtr) + iFrame * channels * task.frameSize + ch;
							int* smp = r + ch * Flake.MAX_BLOCKSIZE;
							for (int i = task.frameSize; i > 0; i--)
							{
								//if (AudioSamples.MemCmp(s + iFrame * task.frameSize + ch * FLACCLWriter.MAX_BLOCKSIZE, r + ch * Flake.MAX_BLOCKSIZE, task.frameSize))
								if (*res != *(smp++))
									throw new Exception(string.Format("validation failed! iFrame={0}, ch={1}", iFrame, ch));
								res += channels;
							}
						}
					}
				}

				if (seek_table != null && _IO.CanSeek)
				{
					for (int sp = 0; sp < seek_table.Length; sp++)
					{
						if (seek_table[sp].framesize != 0)
							continue;
						if (seek_table[sp].number >= task.framePos + iSample + task.frameSize)
							break;
						if (seek_table[sp].number >= task.framePos + iSample)
						{
							seek_table[sp].number = task.framePos + iSample;
							seek_table[sp].offset = iByte;
							seek_table[sp].framesize = task.frameSize;
						}
					}
				}

				//Array.Copy(task.frame.buffer, 0, task.outputBuffer, iByte, fs);

				iSample += task.frameSize;
				iByte += fs;
			}
			task.outputSize = (int)iByte;
			if (iByte != task.frame.writer.Length)
				throw new Exception("invalid length");
		}

		unsafe void write_result(FLACCLTask task)
		{
			int iSample = task.frameSize * task.frameCount;

			if (seek_table != null && _IO.CanSeek)
				for (int sp = 0; sp < seek_table.Length; sp++)
				{
					if (seek_table[sp].number >= task.framePos + iSample)
						break;
					if (seek_table[sp].number >= task.framePos)
						seek_table[sp].offset += _IO.Position - first_frame_offset;
				}
			_IO.Write(task.outputBuffer, 0, task.outputSize);
			_position += iSample;
			_totalSize += task.outputSize;
		}

		public unsafe void InitTasks()
		{
			bool doMidside = channels == 2 && eparams.do_midside;
			int channelCount = doMidside ? 2 * channels : channels;

			if (!inited)
			{
				if (OpenCL.NumberOfPlatforms < 1)
					throw new Exception("no opencl platforms found");

				int groupSize = _settings.DeviceType == OpenCLDeviceType.CPU ? 1 : _settings.GroupSize;
				OCLMan = new OpenCLManager();
				// Attempt to save binaries after compilation, as well as load precompiled binaries
				// to avoid compilation. Usually you'll want this to be true. 
				OCLMan.AttemptUseBinaries = true; // true;
				// Attempt to compile sources. This should probably be true for almost all projects.
				// Setting it to false means that when you attempt to compile "mysource.cl", it will
				// only scan the precompiled binary directory for a binary corresponding to a source
				// with that name. There's a further restriction that the compiled binary also has to
				// use the same Defines and BuildOptions
				OCLMan.AttemptUseSource = true;
				// Binary and source paths
				// This is where we store our sources and where compiled binaries are placed
				//OCLMan.BinaryPath = @"OpenCL\bin";
				//OCLMan.SourcePath = @"OpenCL\src";
				// If true, RequireImageSupport will filter out any devices without image support
				// In this project we don't need image support though, so we set it to false
				OCLMan.RequireImageSupport = false;
				// The Defines string gets prepended to any and all sources that are compiled
				// and serve as a convenient way to pass configuration information to the compilation process
				OCLMan.Defines =
					"#define MAX_ORDER " + eparams.max_prediction_order.ToString() + "\n" +
					"#define GROUP_SIZE " + groupSize.ToString() + "\n" +
					"#define FLACCL_VERSION \"" + vendor_string + "\"\n" +
#if DEBUG
					"#define DEBUG\n" +
#endif
					_settings.Defines + "\n";
				// The BuildOptions string is passed directly to clBuild and can be used to do debug builds etc
				OCLMan.BuildOptions = "";
				OCLMan.SourcePath = System.IO.Path.GetDirectoryName(GetType().Assembly.Location);
				OCLMan.BinaryPath = System.IO.Path.Combine(System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CUE Tools"), "OpenCL");
				int platformId = 0;
				if (_settings.Platform != null)
				{
					platformId = -1;
					string platforms = "";
					for (int i = 0; i < OpenCL.NumberOfPlatforms; i++)
					{
						var platform = OpenCL.GetPlatform(i);
						platforms += " \"" + platform.Name + "\"";
						if (platform.Name.Equals(_settings.Platform, StringComparison.InvariantCultureIgnoreCase))
						{
							platformId = i;
							break;
						}
					}
					if (platformId < 0)
						throw new Exception("unknown platform \"" + _settings.Platform + "\". Platforms available:" + platforms);
				}
				OCLMan.CreateDefaultContext(platformId, (DeviceType)_settings.DeviceType);

				openCLContext = OCLMan.Context;
				try
				{
					openCLProgram = OCLMan.CompileFile(_settings.DeviceType == OpenCLDeviceType.CPU ? "flaccpu.cl" : "flac.cl");
				}
				catch (OpenCLBuildException ex)
				{
					string buildLog = ex.BuildLogs[0];
					throw ex;
				}
				//using (Stream kernel = GetType().Assembly.GetManifestResourceStream(GetType(), "flac.cl"))
				//using (StreamReader sr = new StreamReader(kernel))
				//{
				//    try
				//    {
				//        openCLProgram = OCLMan.CompileSource(sr.ReadToEnd()); ;
				//    }
				//    catch (OpenCLBuildException ex)
				//    {
				//        string buildLog = ex.BuildLogs[0];
				//        throw ex;
				//    }
				//}
#if TTTTKJHSKJH
				var openCLPlatform = OpenCL.GetPlatform(0);
				openCLContext = openCLPlatform.CreateDefaultContext();
				using (Stream kernel = GetType().Assembly.GetManifestResourceStream(GetType(), "flac.cl"))
				using (StreamReader sr = new StreamReader(kernel))
					openCLProgram = openCLContext.CreateProgramWithSource(sr.ReadToEnd());
				try
				{
					openCLProgram.Build();
				}
				catch (OpenCLException)
				{
					string buildLog = openCLProgram.GetBuildLog(openCLProgram.Devices[0]);
					throw;
				}
#endif

				if (_IO == null)
					_IO = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read);
				int header_size = flake_encode_init();
				_IO.Write(header, 0, header_size);
				if (_IO.CanSeek)
					first_frame_offset = _IO.Position;

				task1 = new FLACCLTask(openCLProgram, channelCount, channels, bits_per_sample, max_frame_size, this, groupSize);
				task2 = new FLACCLTask(openCLProgram, channelCount, channels, bits_per_sample, max_frame_size, this, groupSize);
				if (_settings.CPUThreads > 0)
				{
					cpu_tasks = new FLACCLTask[_settings.CPUThreads];
					for (int i = 0; i < cpu_tasks.Length; i++)
						cpu_tasks[i] = new FLACCLTask(openCLProgram, channelCount, channels, bits_per_sample, max_frame_size, this, groupSize);
				}
				inited = true;
			}
		}

		public unsafe void Write(AudioBuffer buff)
		{
			InitTasks();
			buff.Prepare(this);
			int pos = 0;
			while (pos < buff.Length)
			{
				int block = Math.Min(buff.Length - pos, eparams.block_size * max_frames - samplesInBuffer);

				fixed (byte* buf = buff.Bytes)
					AudioSamples.MemCpy(((byte*)task1.clSamplesBytesPtr) + samplesInBuffer * _pcm.BlockAlign, buf + pos * _pcm.BlockAlign, block * _pcm.BlockAlign);
				
				samplesInBuffer += block;
				pos += block;

				int nFrames = samplesInBuffer / eparams.block_size;
				if (nFrames >= max_frames)
					do_output_frames(nFrames);
			}
			if (md5 != null)
				md5.TransformBlock(buff.Bytes, 0, buff.ByteLength, null, 0);
		}

		public void wait_for_cpu_task()
		{
			FLACCLTask task = cpu_tasks[oldest_cpu_task];
			if (task.workThread == null)
				return;
			lock (task)
			{
				while (!task.done && task.exception == null)
					Monitor.Wait(task);
				if (task.exception != null)
					throw task.exception;
			}
		}

		public void cpu_task_thread(object param)
		{
			FLACCLTask task = param as FLACCLTask;
			try
			{
				while (true)
				{
					lock (task)
					{
						while (task.done && !task.exit)
							Monitor.Wait(task);
						if (task.exit)
							return;
					}
					process_result(task);
					lock (task)
					{
						task.done = true;
						Monitor.Pulse(task);
					}
				}
			}
			catch (Exception ex)
			{
				lock (task)
				{
					task.exception = ex;
					Monitor.Pulse(task);
				}
			}
		}

		public void start_cpu_task()
		{
			FLACCLTask task = cpu_tasks[oldest_cpu_task];
			if (task.workThread == null)
			{
				task.done = false;
				task.exit = false;
				task.workThread = new Thread(cpu_task_thread);
				task.workThread.IsBackground = true;
				//task.workThread.Priority = ThreadPriority.BelowNormal;
				task.workThread.Start(task);
			}
			else
			{
				lock (task)
				{
					task.done = false;
					Monitor.Pulse(task);
				}
			}
		}

		public unsafe void do_output_frames(int nFrames)
		{
			send_to_GPU(task1, nFrames, eparams.block_size);
			if (task2.frameCount > 0)
				task2.openCLCQ.Finish();
			run_GPU_task(task1);
			if (task2.frameCount > 0)
			{
				if (cpu_tasks != null)
				{
					wait_for_cpu_task();
					
					FLACCLTask ttmp = cpu_tasks[oldest_cpu_task];
					cpu_tasks[oldest_cpu_task] = task2;
					task2 = ttmp;

					start_cpu_task();					

					oldest_cpu_task = (oldest_cpu_task + 1) % cpu_tasks.Length;
					
					if (task2.frameCount > 0)
						write_result(task2);
				}
				else
				{
					process_result(task2);
					write_result(task2);
				}
			}
			int bs = eparams.block_size * nFrames;
			samplesInBuffer -= bs;
			if (samplesInBuffer > 0)
				AudioSamples.MemCpy(
					((byte*)task2.clSamplesBytesPtr),
					((byte*)task1.clSamplesBytesPtr) + bs * _pcm.BlockAlign, 
					samplesInBuffer * _pcm.BlockAlign);
			FLACCLTask tmp = task1;
			task1 = task2;
			task2 = tmp;
			task1.frameCount = 0;
		}

		public string Path { get { return _path; } }

		public static readonly string vendor_string = "FLACCL#0.2";

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
				bitwriter.writebits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN, (ulong)seek_table[i].number);
				bitwriter.writebits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN, (ulong)seek_table[i].offset);
				bitwriter.writebits(Flake.FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN, seek_table[i].framesize);
			}
			bitwriter.flush();
			return 4 + 18 * seek_table.Length;
		}

		/**
		 * Write padding metadata block to byte array.
		 */
		int
		write_padding(byte[] padding, int pos, int last, long padlen)
		{
			BitWriter bitwriter = new BitWriter(padding, pos, 4);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, (int)MetadataType.Padding);
			bitwriter.writebits(24, (int)padlen);

			return (int)padlen + 4;
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

			max_frames = Math.Min(maxFrames, FLACCLWriter.MAX_BLOCKSIZE / eparams.block_size);

			// set maximum encoded frame size (if larger, re-encodes in verbatim mode)
			if (channels == 2)
				max_frame_size = 16 + ((eparams.block_size * (int)(bits_per_sample + bits_per_sample + 1) + 7) >> 3);
			else
				max_frame_size = 16 + ((eparams.block_size * channels * (int)bits_per_sample + 7) >> 3);

			if (_IO.CanSeek && eparams.do_seektable && sample_count > 0)
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
					seek_table[sp].number = sp * seek_points_distance;
				}
			}

			// output header bytes
			header = new byte[eparams.padding_size + 1024 + (seek_table == null ? 0 : seek_table.Length * 18)];
			header_len = write_headers();

			// initialize CRC & MD5
			if (_IO.CanSeek && _settings.DoMD5)
				md5 = new MD5CryptoServiceProvider();

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
		public long padding_size;

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

		public int orders_per_window;

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

		public bool do_wasted;

		public bool do_constant;

		public WindowFunction window_function;

		public bool do_seektable;

		public int flake_set_defaults(int lvl, bool encode_on_cpu)
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
			block_time_ms = 100;
			min_fixed_order = 0;
			max_fixed_order = 4;
			min_prediction_order = 1;
			max_prediction_order = 12;
			min_partition_order = 0;
			max_partition_order = 6;
			variable_block_size = 0;
			lpc_min_precision_search = 0;
			lpc_max_precision_search = 0;
			do_seektable = true;
			do_wasted = true;
			do_constant = true;

			// differences from level 7
			switch (lvl)
			{
				case 0:
					do_constant = false;
					do_wasted = false;
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					orders_per_window = 1;
					max_partition_order = 4;
					max_prediction_order = 7;
					min_fixed_order = 3;
					max_fixed_order = 2;
					break;
				case 1:
					do_constant = false;
					do_wasted = false;
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					orders_per_window = 1;
					min_fixed_order = 2;
					max_fixed_order = 2;
					max_prediction_order = 7;
					max_partition_order = 4;
					break;
				case 2:
					do_constant = false;
					do_midside = false;
					window_function = WindowFunction.Bartlett;
					orders_per_window = 1;
					min_fixed_order = 2;
					max_fixed_order = 2;
					max_prediction_order = 8;
					max_partition_order = 4;
					break;
				case 3:
					window_function = WindowFunction.Bartlett;
					do_constant = false;
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 1;
					max_prediction_order = 8;
					max_partition_order = 4;
					break;
				case 4:
					do_constant = false;
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 1;
					max_prediction_order = 8;
					max_partition_order = 4;
					break;
				case 5:
					do_constant = false;
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 2;
					max_prediction_order = 8;
					break;
				case 6:
					do_constant = false;
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 1;
					break;
				case 7:
					do_constant = false;
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 3;
					break;
				case 8:
					orders_per_window = 12;
					break;
				case 9:
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 3;
					max_prediction_order = 32;
					break;
				case 10:
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 7;
					max_prediction_order = 32;
					break;
				case 11:
					min_fixed_order = 2;
					max_fixed_order = 2;
					orders_per_window = 11;
					max_prediction_order = 32;
					break;
			}

			if (!encode_on_cpu)
				max_partition_order = 8;

			return 0;
		}
	}

	unsafe struct FLACCLSubframeTask
	{
		public int residualOrder;
		public int samplesOffs;
		public int shift;
		public int cbits;
		public int size;
		public int type;
		public int obits;
		public int blocksize;
		public int best_index;
		public int channel;
		public int residualOffs;
		public int wbits;
		public int abits;
		public int porder;
		public fixed int reserved[2];
		public fixed int coefs[32];
	};

	internal class FLACCLTask
	{
		Program openCLProgram;
		public CommandQueue openCLCQ;
		public Kernel clStereoDecorr;
		//public Kernel cudaChannelDecorr;
		public Kernel clChannelDecorr2;
		public Kernel clFindWastedBits;
		public Kernel clComputeAutocor;
		public Kernel clComputeLPC;
		//public Kernel cudaComputeLPCLattice;
		public Kernel clQuantizeLPC;
		public Kernel clEstimateResidual;
		public Kernel clChooseBestMethod;
		public Kernel clCopyBestMethod;
		public Kernel clCopyBestMethodStereo;
		public Kernel clEncodeResidual;
		public Kernel clCalcPartition;
		public Kernel clCalcPartition16;
		public Kernel clSumPartition;
		public Kernel clFindRiceParameter;
		public Kernel clFindPartitionOrder;
		public Mem clSamplesBytes;
		public Mem clSamples;
		public Mem clLPCData;
		public Mem clResidual;
		public Mem clPartitions;
		public Mem clRiceParams;
		public Mem clBestRiceParams;
		public Mem clAutocorOutput;
		public Mem clResidualTasks;
		public Mem clBestResidualTasks;
		public Mem clWindowFunctions;

        public Mem clSamplesBytesPinned;
        public Mem clResidualPinned;
        public Mem clBestRiceParamsPinned;
        public Mem clResidualTasksPinned;
        public Mem clBestResidualTasksPinned;
        public Mem clWindowFunctionsPinned;

        public IntPtr clSamplesBytesPtr;
        public IntPtr clResidualPtr;
        public IntPtr clBestRiceParamsPtr;
        public IntPtr clResidualTasksPtr;
        public IntPtr clBestResidualTasksPtr;
        public IntPtr clWindowFunctionsPtr;

        public int[] samplesBuffer;
		public byte[] outputBuffer;
		public int outputSize = 0;
		public int frameSize = 0;
		public int frameCount = 0;
		public int frameNumber = 0;
		public int framePos = 0;
		public FlacFrame frame;
		public int residualTasksLen;
		public int bestResidualTasksLen;
		public int samplesBufferLen;
		public int nResidualTasks = 0;
		public int nResidualTasksPerChannel = 0;
		public int nTasksPerWindow = 0;
		public int nWindowFunctions = 0;
		public int max_porder = 0;

		public FlakeReader verify;

		public Thread workThread = null;
		public Exception exception = null;
		public bool done = false;
		public bool exit = false;

		public int groupSize = 128;
		public int channels, channelsCount;
		public FLACCLWriter writer;

		unsafe public FLACCLTask(Program _openCLProgram, int channelsCount, int channels, uint bits_per_sample, int max_frame_size, FLACCLWriter writer, int groupSize)
		{
			this.groupSize = groupSize;
			this.channels = channels;
			this.channelsCount = channelsCount;
			this.writer = writer;
			openCLProgram = _openCLProgram;
#if DEBUG
			var prop = CommandQueueProperties.PROFILING_ENABLE;
#else
			var prop = CommandQueueProperties.NONE;
#endif
			openCLCQ = openCLProgram.Context.CreateCommandQueue(openCLProgram.Context.Devices[0], prop);

            int MAX_ORDER = this.writer.eparams.max_prediction_order;
			residualTasksLen = sizeof(FLACCLSubframeTask) * 32 * channelsCount * FLACCLWriter.maxFrames;
			bestResidualTasksLen = sizeof(FLACCLSubframeTask) * channels * FLACCLWriter.maxFrames;
			samplesBufferLen = sizeof(int) * FLACCLWriter.MAX_BLOCKSIZE * channelsCount;
			int partitionsLen = sizeof(int) * (30 << 8) * channels * FLACCLWriter.maxFrames;
			int riceParamsLen = sizeof(int) * (4 << 8) * channels * FLACCLWriter.maxFrames;
            int autocorLen = sizeof(float) * (MAX_ORDER + 1) * lpc.MAX_LPC_WINDOWS * channelsCount * FLACCLWriter.maxFrames;
            int lpcDataLen = autocorLen * 32;
            int resOutLen = sizeof(int) * channelsCount * (lpc.MAX_LPC_WINDOWS * lpc.MAX_LPC_ORDER + 8) * FLACCLWriter.maxFrames;
            int wndLen = sizeof(float) * FLACCLWriter.MAX_BLOCKSIZE /** 2*/ * lpc.MAX_LPC_WINDOWS;

            if (!writer._settings.MappedMemory)
            {
                clSamplesBytes = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, samplesBufferLen / 2);
                clResidual = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, samplesBufferLen);
                clBestRiceParams = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, riceParamsLen / 4);
                clResidualTasks = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, residualTasksLen);
                clBestResidualTasks = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, bestResidualTasksLen);
                clWindowFunctions = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, wndLen);

                clSamplesBytesPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, samplesBufferLen / 2);
                clResidualPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, samplesBufferLen);
                clBestRiceParamsPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, riceParamsLen / 4);
                clResidualTasksPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, residualTasksLen);
                clBestResidualTasksPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, bestResidualTasksLen);
                clWindowFunctionsPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, wndLen);

                clSamplesBytesPtr = openCLCQ.EnqueueMapBuffer(clSamplesBytesPinned, true, MapFlags.WRITE, 0, samplesBufferLen / 2);
                clResidualPtr = openCLCQ.EnqueueMapBuffer(clResidualPinned, true, MapFlags.WRITE, 0, samplesBufferLen);
                clBestRiceParamsPtr = openCLCQ.EnqueueMapBuffer(clBestRiceParamsPinned, true, MapFlags.WRITE, 0, riceParamsLen / 4);
                clResidualTasksPtr = openCLCQ.EnqueueMapBuffer(clResidualTasksPinned, true, MapFlags.WRITE, 0, residualTasksLen);
                clBestResidualTasksPtr = openCLCQ.EnqueueMapBuffer(clBestResidualTasksPinned, true, MapFlags.WRITE, 0, bestResidualTasksLen);
                clWindowFunctionsPtr = openCLCQ.EnqueueMapBuffer(clWindowFunctionsPinned, true, MapFlags.WRITE, 0, wndLen);
            }
            else
            {
                clSamplesBytes = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, (uint)samplesBufferLen / 2);
                clResidual = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, samplesBufferLen);
                clBestRiceParams = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, riceParamsLen / 4);
                clResidualTasks = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, residualTasksLen);
                clBestResidualTasks = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, bestResidualTasksLen);
                clWindowFunctions = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, wndLen);

                clSamplesBytesPtr = openCLCQ.EnqueueMapBuffer(clSamplesBytes, true, MapFlags.WRITE, 0, samplesBufferLen / 2);
                clResidualPtr = openCLCQ.EnqueueMapBuffer(clResidual, true, MapFlags.WRITE, 0, samplesBufferLen);
                clBestRiceParamsPtr = openCLCQ.EnqueueMapBuffer(clBestRiceParams, true, MapFlags.WRITE, 0, riceParamsLen / 4);
                clResidualTasksPtr = openCLCQ.EnqueueMapBuffer(clResidualTasks, true, MapFlags.WRITE, 0, residualTasksLen);
                clBestResidualTasksPtr = openCLCQ.EnqueueMapBuffer(clBestResidualTasks, true, MapFlags.WRITE, 0, bestResidualTasksLen);
                clWindowFunctionsPtr = openCLCQ.EnqueueMapBuffer(clWindowFunctions, true, MapFlags.WRITE, 0, wndLen);

                //clSamplesBytesPtr = clSamplesBytes.HostPtr;
                //clResidualPtr = clResidual.HostPtr;
                //clBestRiceParamsPtr = clBestRiceParams.HostPtr;
                //clResidualTasksPtr = clResidualTasks.HostPtr;
                //clBestResidualTasksPtr = clBestResidualTasks.HostPtr;
                //clWindowFunctionsPtr = clWindowFunctions.HostPtr;
            }

            clSamples = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, samplesBufferLen);
            clLPCData = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, lpcDataLen);
            clAutocorOutput = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, autocorLen);
            if (writer._settings.GPUOnly)
            {
                clPartitions = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, partitionsLen);
                clRiceParams = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, riceParamsLen);
            }

            //openCLCQ.EnqueueMapBuffer(clSamplesBytes, true, MapFlags.WRITE, 0, samplesBufferLen / 2);

            clComputeAutocor = openCLProgram.CreateKernel("clComputeAutocor");
			clStereoDecorr = openCLProgram.CreateKernel("clStereoDecorr");
			//cudaChannelDecorr = openCLProgram.CreateKernel("clChannelDecorr");
			clChannelDecorr2 = openCLProgram.CreateKernel("clChannelDecorr2");
			clFindWastedBits = openCLProgram.CreateKernel("clFindWastedBits");
			clComputeLPC = openCLProgram.CreateKernel("clComputeLPC");
			clQuantizeLPC = openCLProgram.CreateKernel("clQuantizeLPC");
			//cudaComputeLPCLattice = openCLProgram.CreateKernel("clComputeLPCLattice");
			clEstimateResidual = openCLProgram.CreateKernel("clEstimateResidual");
			clChooseBestMethod = openCLProgram.CreateKernel("clChooseBestMethod");
			clCopyBestMethod = openCLProgram.CreateKernel("clCopyBestMethod");
			clCopyBestMethodStereo = openCLProgram.CreateKernel("clCopyBestMethodStereo");
			if (writer._settings.GPUOnly)
			{
				clEncodeResidual = openCLProgram.CreateKernel("clEncodeResidual");
				clCalcPartition = openCLProgram.CreateKernel("clCalcPartition");
				clCalcPartition16 = openCLProgram.CreateKernel("clCalcPartition16");
				clSumPartition = openCLProgram.CreateKernel("clSumPartition");
				clFindRiceParameter = openCLProgram.CreateKernel("clFindRiceParameter");
				clFindPartitionOrder = openCLProgram.CreateKernel("clFindPartitionOrder");
			}

			samplesBuffer = new int[FLACCLWriter.MAX_BLOCKSIZE * channelsCount];
			outputBuffer = new byte[max_frame_size * FLACCLWriter.maxFrames + 1];
			frame = new FlacFrame(channelsCount);
			frame.writer = new BitWriter(outputBuffer, 0, outputBuffer.Length);

			if (writer._settings.DoVerify)
			{
				verify = new FlakeReader(new AudioPCMConfig((int)bits_per_sample, channels, 44100));
				verify.DoCRC = false;
			}
		}

		public void Dispose()
		{
			if (workThread != null)
			{
				lock (this)
				{
					exit = true;
					Monitor.Pulse(this);
				}
				workThread.Join();
				workThread = null;
			}

            openCLCQ.Finish();

			clComputeAutocor.Dispose();
			clStereoDecorr.Dispose();
			//cudaChannelDecorr.Dispose();
			clChannelDecorr2.Dispose();
			clFindWastedBits.Dispose();
			clComputeLPC.Dispose();
			clQuantizeLPC.Dispose();
			//cudaComputeLPCLattice.Dispose();
			clEstimateResidual.Dispose();
			clChooseBestMethod.Dispose();
			clCopyBestMethod.Dispose();
			clCopyBestMethodStereo.Dispose();
			if (writer._settings.GPUOnly)
			{
				clEncodeResidual.Dispose();
				clCalcPartition.Dispose();
				clCalcPartition16.Dispose();
				clSumPartition.Dispose();
				clFindRiceParameter.Dispose();
				clFindPartitionOrder.Dispose();

                clPartitions.Dispose();
                clRiceParams.Dispose();
            }

            if (!writer._settings.MappedMemory)
            {
                if (clSamplesBytesPtr != IntPtr.Zero) 
                    openCLCQ.EnqueueUnmapMemObject(clSamplesBytesPinned, clSamplesBytesPtr);
                clSamplesBytesPtr = IntPtr.Zero;
                if (clResidualPtr != IntPtr.Zero) 
                    openCLCQ.EnqueueUnmapMemObject(clResidualPinned, clResidualPtr);
                clResidualPtr = IntPtr.Zero;
                if (clBestRiceParamsPtr != IntPtr.Zero) 
                    openCLCQ.EnqueueUnmapMemObject(clBestRiceParamsPinned, clBestRiceParamsPtr);
                clBestRiceParamsPtr = IntPtr.Zero;
                if (clResidualTasksPtr != IntPtr.Zero)
                    openCLCQ.EnqueueUnmapMemObject(clResidualTasksPinned, clResidualTasksPtr);
                clResidualTasksPtr = IntPtr.Zero;
                if (clBestResidualTasksPtr != IntPtr.Zero)
                    openCLCQ.EnqueueUnmapMemObject(clBestResidualTasksPinned, clBestResidualTasksPtr);
                clBestResidualTasksPtr = IntPtr.Zero;
                if (clWindowFunctionsPtr != IntPtr.Zero)
                    openCLCQ.EnqueueUnmapMemObject(clWindowFunctionsPinned, clWindowFunctionsPtr);
                clWindowFunctionsPtr = IntPtr.Zero;

                clSamplesBytesPinned.Dispose();
                clResidualPinned.Dispose();
                clBestRiceParamsPinned.Dispose();
                clResidualTasksPinned.Dispose();
                clBestResidualTasksPinned.Dispose();
                clWindowFunctionsPinned.Dispose();
            }
            else
            {
                openCLCQ.EnqueueUnmapMemObject(clSamplesBytes, clSamplesBytesPtr);
                openCLCQ.EnqueueUnmapMemObject(clResidual, clResidualPtr);
                openCLCQ.EnqueueUnmapMemObject(clBestRiceParams, clBestRiceParamsPtr);
                openCLCQ.EnqueueUnmapMemObject(clResidualTasks, clResidualTasksPtr);
                openCLCQ.EnqueueUnmapMemObject(clBestResidualTasks, clBestResidualTasksPtr);
                openCLCQ.EnqueueUnmapMemObject(clWindowFunctions, clWindowFunctionsPtr);
            }

			clSamples.Dispose();
			clSamplesBytes.Dispose();
			clLPCData.Dispose();
			clResidual.Dispose();
			clAutocorOutput.Dispose();
			clResidualTasks.Dispose();
			clBestResidualTasks.Dispose();
			clWindowFunctions.Dispose();

			openCLCQ.Dispose();

            GC.SuppressFinalize(this);
        }

		public unsafe FLACCLSubframeTask* ResidualTasks
		{
			get
			{
				return (FLACCLSubframeTask*)clResidualTasksPtr;
			}
		}

		public unsafe FLACCLSubframeTask* BestResidualTasks
		{
			get
			{
				return (FLACCLSubframeTask*)clBestResidualTasksPtr;
			}
		}

		internal unsafe void EnqueueKernels()
		{
			FlakeEncodeParams eparams = writer.eparams;

			this.max_porder = FLACCLWriter.get_max_p_order(eparams.max_partition_order, frameSize, eparams.max_prediction_order);
			while ((frameSize >> max_porder) < 16 && max_porder > 0)
				this.max_porder--;

			if (channels != 2) throw new Exception("channels != 2"); // need to Enqueue cudaChannelDecorr for each channel
			Kernel clChannelDecorr = channels == 2 ? (channelsCount == 4 ? clStereoDecorr : clChannelDecorr2) : null;// cudaChannelDecorr;

			// openCLCQ.EnqueueMapBuffer(cudaSamplesBytes
			//openCLCQ.EnqueueUnmapMemObject(cudaSamplesBytes, cudaSamplesBytes.HostPtr);

			// issue work to the GPU
			clChannelDecorr.SetArgs(
				clSamples,
				clSamplesBytes,
				FLACCLWriter.MAX_BLOCKSIZE);

            openCLCQ.EnqueueNDRangeKernel(clChannelDecorr, 0, frameSize * frameCount);

            if (eparams.do_wasted)
			{
				clFindWastedBits.SetArgs(
					clResidualTasks,
					clSamples,
					nResidualTasksPerChannel);

				openCLCQ.EnqueueNDRangeKernel(
					clFindWastedBits,
					groupSize,
					frameCount * channelsCount);
			}

            clComputeAutocor.SetArgs(
                clAutocorOutput,
				clSamples,
				clWindowFunctions,
				clResidualTasks,
				nResidualTasksPerChannel);

			openCLCQ.EnqueueNDRangeKernel(
				clComputeAutocor,
				groupSize, 1,
				channelsCount * frameCount,
				nWindowFunctions);

            clComputeLPC.SetArgs(
				clAutocorOutput,
				clLPCData,
				nWindowFunctions);

            //openCLCQ.EnqueueNDRangeKernel(
            //    clComputeLPC,
            //    64, 1,
            //    nWindowFunctions,
            //    frameCount);
            openCLCQ.EnqueueNDRangeKernel(
                clComputeLPC,
                Math.Min(groupSize, 32), 1,
                nWindowFunctions,
                channelsCount * frameCount);

            clQuantizeLPC.SetArgs(
				clResidualTasks,
				clLPCData,
				nResidualTasksPerChannel,
				nTasksPerWindow,
				eparams.lpc_min_precision_search,
				eparams.lpc_max_precision_search - eparams.lpc_min_precision_search);

			openCLCQ.EnqueueNDRangeKernel(
				clQuantizeLPC,
				Math.Min(groupSize, 32), 1,
				nWindowFunctions,
				channelsCount * frameCount);

            clEstimateResidual.SetArgs(
				clSamples,
				clResidualTasks);

			openCLCQ.EnqueueNDRangeKernel(
				clEstimateResidual,
				groupSize,
				nResidualTasksPerChannel * channelsCount * frameCount);

            clChooseBestMethod.SetArgs(
				clResidualTasks,
				nResidualTasksPerChannel);

			openCLCQ.EnqueueNDRangeKernel(
				clChooseBestMethod,
				Math.Min(groupSize, 32), channelsCount * frameCount);

            if (channels == 2 && channelsCount == 4)
			{
				clCopyBestMethodStereo.SetArgs(
					clBestResidualTasks,
					clResidualTasks,
					nResidualTasksPerChannel);

				openCLCQ.EnqueueNDRangeKernel(
					clCopyBestMethodStereo,
					Math.Min(groupSize, 64), frameCount);
			}
			else
			{
				clCopyBestMethod.SetArgs(
					clBestResidualTasks,
					clResidualTasks,
					nResidualTasksPerChannel);

				openCLCQ.EnqueueNDRangeKernel(
					clCopyBestMethod,
					Math.Min(groupSize, 64), channels * frameCount);
			}

			if (writer._settings.GPUOnly)
			{
				if (frameSize >> max_porder == 16)
				{
					clCalcPartition16.SetArgs(
						clPartitions,
						clResidual,
						clSamples,
						clBestResidualTasks,
						max_porder);

					openCLCQ.EnqueueNDRangeKernel(
						clCalcPartition16,
						groupSize, channels * frameCount);
				}
				else
				{
					clEncodeResidual.SetArgs(
						clResidual,
						clSamples,
						clBestResidualTasks);

					openCLCQ.EnqueueNDRangeKernel(
						clEncodeResidual,
						groupSize, channels * frameCount);

					clCalcPartition.SetArgs(
						clPartitions,
						clResidual,
						clBestResidualTasks,
						max_porder,
						frameSize >> max_porder);

					if (openCLCQ.Device.DeviceType == DeviceType.CPU)
						openCLCQ.EnqueueNDRangeKernel(
							clCalcPartition,
							groupSize, 1,
							1,
							channels * frameCount);
					else
						openCLCQ.EnqueueNDRangeKernel(
							clCalcPartition,
							groupSize, 1,
							1 + ((1 << max_porder) - 1) / (groupSize / 16),
							channels * frameCount);
				}

				if (max_porder > 0)
				{
					clSumPartition.SetArgs(
						clPartitions,
						max_porder);

					if (openCLCQ.Device.DeviceType == DeviceType.CPU)
						openCLCQ.EnqueueNDRangeKernel(
							clSumPartition,
							1, 1, 1,
							channels * frameCount);
					else
						openCLCQ.EnqueueNDRangeKernel(
							clSumPartition,
							128, 1,
							(Flake.MAX_RICE_PARAM + 1),
							channels * frameCount);
				}

				clFindRiceParameter.SetArgs(
					clBestResidualTasks,
					clRiceParams,
					clPartitions,
					max_porder);

				openCLCQ.EnqueueNDRangeKernel(
					clFindRiceParameter,
					groupSize, channels * frameCount);

				//if (max_porder > 0) // need to run even if max_porder==0 just to calculate the final frame size
				clFindPartitionOrder.SetArgs(
					clResidual,
					clBestRiceParams,
					clBestResidualTasks,
					clRiceParams,
					max_porder);

				openCLCQ.EnqueueNDRangeKernel(
					clFindPartitionOrder,
					groupSize,
					channels * frameCount);

                if (!writer._settings.MappedMemory)
				    openCLCQ.EnqueueReadBuffer(clBestRiceParams, false, 0, sizeof(int) * (1 << max_porder) * channels * frameCount, clBestRiceParamsPtr);
                if (!writer._settings.MappedMemory)
				    openCLCQ.EnqueueReadBuffer(clResidual, false, 0, sizeof(int) * FLACCLWriter.MAX_BLOCKSIZE * channels, clResidualPtr);
			}
            if (!writer._settings.MappedMemory)
			    openCLCQ.EnqueueReadBuffer(clBestResidualTasks, false, 0, sizeof(FLACCLSubframeTask) * channels * frameCount, clBestResidualTasksPtr);
        }
	}

#if HJHKHJ
    public static class OpenCLExtensions
    {
        public static void SetArgs(this Kernel kernel, params object[] args)
        {
            int i = 0;
            foreach (object arg in args)
            {
                if (arg is int)
                    kernel.SetArg(i, (int)arg);
                else if (arg is Mem)
                    kernel.SetArg(i, (Mem)arg);
                else
                    throw new ArgumentException("Invalid argument type", arg.GetType().ToString());
                i++;
            }
        }

        public static void EnqueueNDRangeKernel(this CommandQueue queue, Kernel kernel, long localSize, long globalSize)
        {
            if (localSize == 0)
                queue.EnqueueNDRangeKernel(kernel, 1, null, new long[] { globalSize }, null);
            else
                queue.EnqueueNDRangeKernel(kernel, 1, null, new long[] { localSize * globalSize }, new long[] { localSize });
        }

        public static void EnqueueNDRangeKernel(this CommandQueue queue, Kernel kernel, long localSizeX, long localSizeY, long globalSizeX, long globalSizeY)
        {
            queue.EnqueueNDRangeKernel(kernel, 2, null, new long[] { localSizeX * globalSizeX, localSizeY * globalSizeY }, new long[] { localSizeX, localSizeY });
        }
    }
#endif
}
