/**
 * CUETools.Flake: pure managed FLAC audio encoder
 * Copyright (c) 2009-2020 Grigory Chudov
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

#define NOINTEROP
#define VARIANT1

using System;
using System.ComponentModel;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Security.Cryptography;
#if INTEROP
using System.Runtime.InteropServices;
#endif
using CUETools.Codecs;
using Newtonsoft.Json;

namespace CUETools.Codecs.Flake
{
	public class AudioEncoder : IAudioDest
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
		int sr_code0, sr_code1;

		// sample size in bits
		// set by user prior to calling flake_encode_init
		// only 16-bit is currently supported
		int bps_code;

		// total stream samples
		// set by user prior to calling flake_encode_init
		// if 0, stream length is unknown
		int sample_count = -1;

		FlakeEncodeParams eparams;

		// maximum frame size in bytes
		// set by flake_encode_init
		// this can be used to allocate memory for output
		int max_frame_size;

		byte[] frame_buffer = null;

		int frame_count = 0;

		long first_frame_offset = 0;

#if INTEROP
		TimeSpan _userProcessorTime;
#endif

		// header bytes
		// allocated by flake_encode_init and freed by flake_encode_close
		byte[] header;

		int[] samplesBuffer;
		int[] verifyBuffer;
		int[] residualBuffer;
		float[] windowBuffer;
		double[] windowScale;
        LpcWindowSection[, ,] windowSections;
        
        WindowFunction[] windowType;
		int samplesInBuffer = 0;

		int m_blockSize = 0;
		int _totalSize = 0;
		int _windowsize = 0, _windowcount = 0;

		Crc8 crc8;
		MD5 md5;

		FlacFrame frame;
		AudioDecoder verify;

		SeekPoint[] seek_table;
		int seek_table_offset = -1;

		bool inited = false;

		public AudioEncoder(EncoderSettings settings, string path, Stream IO = null)
		{
            m_settings = settings.Clone() as EncoderSettings;
            m_settings.Validate();

			//if (Settings.PCM.BitsPerSample != 16)
			//    throw new Exception("Bits per sample must be 16.");
            //if (Settings.PCM.ChannelCount != 2)
            //    throw new Exception("ChannelCount must be 2.");

            channels = Settings.PCM.ChannelCount;

			// flake_validate_params

			_path = path;
			_IO = IO;

			samplesBuffer = new int[FlakeConstants.MAX_BLOCKSIZE * (channels == 2 ? 4 : channels)];
			residualBuffer = new int[FlakeConstants.MAX_BLOCKSIZE * (channels == 2 ? 10 : channels + 1)];
			windowBuffer = new float[FlakeConstants.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS];
            windowScale = new double[lpc.MAX_LPC_WINDOWS];
            windowType = new WindowFunction[lpc.MAX_LPC_WINDOWS];
            windowSections = new LpcWindowSection[12, lpc.MAX_LPC_WINDOWS, lpc.MAX_LPC_SECTIONS];

            eparams.flake_set_defaults(m_settings);

			crc8 = new Crc8();
			frame = new FlacFrame(channels * 2);
		}

		public int TotalSize
		{
			get
			{
				return _totalSize;
			}
		}

		EncoderSettings m_settings;

        public IAudioEncoderSettings Settings => m_settings;

#if INTEROP
		[DllImport("kernel32.dll")]
		static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
		[DllImport("kernel32.dll")]
		static extern IntPtr GetCurrentThread();
#endif

		void DoClose()
		{
			if (inited)
			{
				while (samplesInBuffer > 0)
				{
                    m_blockSize = samplesInBuffer;
					output_frame();
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

#if INTEROP
			long fake, KernelStart, UserStart;
			GetThreadTimes(GetCurrentThread(), out fake, out fake, out KernelStart, out UserStart);
			_userProcessorTime = new TimeSpan(UserStart);
#endif
		}

		public void Close()
		{
			DoClose();
			if (sample_count > 0 && _position != sample_count)
				throw new Exception(Properties.Resources.ExceptionSampleCount);
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

		public OrderMethod OrderMethod
		{
			get { return eparams.order_method; }
			set { eparams.order_method = value; }
		}

        public int DevelopmentMode
        {
            get { return eparams.development_mode; }
            set { eparams.development_mode = value; }
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

		public TimeSpan UserProcessorTime
		{
			get 
			{
#if INTEROP 
				return _userProcessorTime; 
#else
				return new TimeSpan(0);
#endif
			}
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
				{
					if (m_settings.StereoMethod == StereoMethod.Independent)
						AudioSamples.Deinterlace(fsamples + samplesInBuffer, fsamples + FlakeConstants.MAX_BLOCKSIZE + samplesInBuffer, src, block);
					else
					{
						int* left = fsamples + samplesInBuffer;
						int* right = left + FlakeConstants.MAX_BLOCKSIZE;
						int* leftM = right + FlakeConstants.MAX_BLOCKSIZE;
						int* rightM = leftM + FlakeConstants.MAX_BLOCKSIZE;
						for (int i = 0; i < block; i++)
						{
							int l = src[2 * i];
							int r = src[2 * i + 1];
							left[i] = l;
							right[i] = r;
							leftM[i] = (l + r) >> 1;
							rightM[i] = l - r;
						}
					}
				}
				else
					for (int ch = 0; ch < channels; ch++)
					{
						int* psamples = fsamples + ch * FlakeConstants.MAX_BLOCKSIZE + samplesInBuffer;
						for (int i = 0; i < block; i++)
							psamples[i] = src[i * channels + ch];
					}
			}
			samplesInBuffer += block;
		}

		//unsafe static void channel_decorrelation(int* leftS, int* rightS, int *leftM, int *rightM, int blocksize)
		//{
		//    for (int i = 0; i < blocksize; i++)
		//    {
		//        leftM[i] = (leftS[i] + rightS[i]) >> 1;
		//        rightM[i] = leftS[i] - rightS[i];
		//    }
		//}

		unsafe void encode_residual_verbatim(int* res, int* smp, uint n)
		{
			AudioSamples.MemCpy(res, smp, (int) n);
		}

        unsafe static ulong encode_residual_fixed_partition(int* res, int* smp, int* end, int order, int* last_errors)
        {
            ulong sum = 0UL;
            switch (order)
            {
                case 0:
                    {
                        while (smp < end)
                        {
                            int error = *(res++) = *(smp++);
                            sum += (uint)((error << 1) ^ (error >> 31));
                        }
                        break;
                    }
                case 1:
                    {
                        int last_error_0 = last_errors[0];
                        while (smp < end)
                        {
                            int error, save;
                            error = *(smp++); save = error;
                            error -= last_error_0; *(res++) = error; last_error_0 = save;
                            sum += (uint)((error << 1) ^ (error >> 31));
                        }
                        last_errors[0] = last_error_0;
                        break;
                    }
                case 2:
                    {
                        int last_error_0 = last_errors[0], last_error_1 = last_errors[1];
                        while (smp < end)
                        {
                            int error, save;
                            error = *(smp++); save = error;
                            error -= last_error_0; last_error_0 = save; save = error;
                            error -= last_error_1; *(res++) = error; last_error_1 = save;
                            sum += (uint)((error << 1) ^ (error >> 31));
                        }
                        last_errors[0] = last_error_0; last_errors[1] = last_error_1; ;
                        break;
                    }
                case 3:
                    {
                        int last_error_0 = last_errors[0], last_error_1 = last_errors[1], last_error_2 = last_errors[2];
                        while (smp < end)
                        {
                            int error, save;
                            error = *(smp++); save = error;
                            error -= last_error_0; last_error_0 = save; save = error;
                            error -= last_error_1; last_error_1 = save; save = error;
                            error -= last_error_2; *(res++) = error; last_error_2 = save;
                            sum += (uint)((error << 1) ^ (error >> 31));
                        }
                        last_errors[0] = last_error_0; last_errors[1] = last_error_1; last_errors[2] = last_error_2;
                        break;
                    }
                case 4:
                    {
                        int last_error_0 = last_errors[0], last_error_1 = last_errors[1], last_error_2 = last_errors[2], last_error_3 = last_errors[3];
                        while (smp < end)
                        {
                            int error, save;
                            error = *(smp++); save = error;
                            error -= last_error_0; last_error_0 = save; save = error;
                            error -= last_error_1; last_error_1 = save; save = error;
                            error -= last_error_2; last_error_2 = save; save = error;
                            error -= last_error_3; *(res++) = error; last_error_3 = save;
                            sum += (uint)((error << 1) ^ (error >> 31));
                        }
                        last_errors[0] = last_error_0; last_errors[1] = last_error_1; last_errors[2] = last_error_2; last_errors[3] = last_error_3;
                        break;
                    }
                default:
                    throw new ArgumentOutOfRangeException();
            }
            return sum;
        }

        unsafe static void encode_residual_fixed(int* res, int* smp, int n, int order, ulong* sums, int pmax)
        {
            int* last_errors = stackalloc int[4];
            int* end = smp + n;
            int* seg_end = smp + (n >> pmax);

            if (order > 4)
                throw new ArgumentOutOfRangeException();

            for (int i = 0; i < order; i++)
            {
                int* next_errors = stackalloc int[4];
                next_errors[0] = *(res++) = *(smp++);
                for (int j = 0; j < i; j++)
                    next_errors[j + 1] = next_errors[j] - last_errors[j];
                for (int j = 0; j <= i; j++)
                    last_errors[j] = next_errors[j];
            }

            while (smp < end)
            {
                *(sums++) = encode_residual_fixed_partition(res, smp, seg_end, order, last_errors);
                res += seg_end - smp;
                smp = seg_end;
                seg_end += n >> pmax;
            }
        }

#if XXX
        unsafe static int encode_residual_fixed_estimate_best_order(int* res, int* smp, int n, int order)
        {
            int next_error_0, next_error_1, next_error_2, next_error_3, next_error_4;
            int last_error_0, last_error_1, last_error_2, last_error_3;
            int* end = smp + n;
            ulong total_error_0 = 0, total_error_1 = 0, total_error_2 = 0, total_error_3 = 0, total_error_4 = 0;

            if (order == 0)
            {
                AudioSamples.MemCpy(res, smp, n);
                return 0;
            }

            next_error_0 = *(res++) = *(smp++);
            last_error_0 = next_error_0;

            if (order == 1)
            {
                while (smp < end)
                {
                    next_error_0 = *(smp++);
                    next_error_1 = next_error_0 - last_error_0;

                    last_error_0 = next_error_0;

                    total_error_0 += (ulong)((next_error_0 << 1) ^ (next_error_0 >> 31));
                    total_error_1 += (ulong)((next_error_1 << 1) ^ (next_error_1 >> 31));

                    *(res++) = (int)next_error_1;
                }

                if ((total_error_0 < total_error_1))
                    return 0;
                return 1;
            }

            next_error_0 = *(res++) = *(smp++);
            next_error_1 = next_error_0 - last_error_0;
            last_error_0 = next_error_0;
            last_error_1 = next_error_1;

            if (order == 2)
            {
                while (smp < end)
                {
                    next_error_0 = *(smp++);
                    next_error_1 = next_error_0 - last_error_0;
                    next_error_2 = next_error_1 - last_error_1;

                    last_error_0 = next_error_0;
                    last_error_1 = next_error_1;

                    total_error_0 += (ulong)((next_error_0 << 1) ^ (next_error_0 >> 31));
                    total_error_1 += (ulong)((next_error_1 << 1) ^ (next_error_1 >> 31));
                    total_error_2 += (ulong)((next_error_2 << 1) ^ (next_error_2 >> 31));

                    *(res++) = (int)next_error_2;
                }

                if ((total_error_0 < total_error_1) & (total_error_0 < total_error_2))
                    return 0;
                else if ((total_error_1 < total_error_2))
                    return 1;
                return 2;
            }

            next_error_0 = *(res++) = *(smp++);
            next_error_1 = next_error_0 - last_error_0;
            next_error_2 = next_error_1 - last_error_1;
            last_error_0 = next_error_0;
            last_error_1 = next_error_1;
            last_error_2 = next_error_2;

            if (order == 3)
            {
                while (smp < end)
                {
                    next_error_0 = *(smp++);
                    next_error_1 = next_error_0 - last_error_0;
                    next_error_2 = next_error_1 - last_error_1;
                    next_error_3 = next_error_2 - last_error_2;

                    last_error_0 = next_error_0;
                    last_error_1 = next_error_1;
                    last_error_2 = next_error_2;

                    total_error_0 += (ulong)((next_error_0 << 1) ^ (next_error_0 >> 31));
                    total_error_1 += (ulong)((next_error_1 << 1) ^ (next_error_1 >> 31));
                    total_error_2 += (ulong)((next_error_2 << 1) ^ (next_error_2 >> 31));
                    total_error_3 += (ulong)((next_error_3 << 1) ^ (next_error_3 >> 31));

                    *(res++) = (int)next_error_3;
                }

                if ((total_error_0 < total_error_1) & (total_error_0 < total_error_2) & (total_error_0 < total_error_3))
                    return 0;
                else if ((total_error_1 < total_error_2) & (total_error_1 < total_error_3))
                    return 1;
                else if ((total_error_2 < total_error_3))
                    return 2;
                return 3;
            }

            next_error_0 = *(res++) = *(smp++);
            next_error_1 = next_error_0 - last_error_0;
            next_error_2 = next_error_1 - last_error_1;
            next_error_3 = next_error_2 - last_error_2;
            last_error_0 = next_error_0;
            last_error_1 = next_error_1;
            last_error_2 = next_error_2;
            last_error_3 = next_error_3;

            if (order == 4)
            {
                while (smp < end)
                {
                    next_error_0 = *(smp++);
                    next_error_1 = next_error_0 - last_error_0;
                    next_error_2 = next_error_1 - last_error_1;
                    next_error_3 = next_error_2 - last_error_2;
                    next_error_4 = next_error_3 - last_error_3;

                    last_error_0 = next_error_0;
                    last_error_1 = next_error_1;
                    last_error_2 = next_error_2;
                    last_error_3 = next_error_3;

                    total_error_0 += (ulong)((next_error_0 << 1) ^ (next_error_0 >> 31));
                    total_error_1 += (ulong)((next_error_1 << 1) ^ (next_error_1 >> 31));
                    total_error_2 += (ulong)((next_error_2 << 1) ^ (next_error_2 >> 31));
                    total_error_3 += (ulong)((next_error_3 << 1) ^ (next_error_3 >> 31));
                    total_error_4 += (ulong)((next_error_4 << 1) ^ (next_error_4 >> 31));

                    *(res++) = (int)next_error_4;
                }

                if ((total_error_0 < total_error_1) & (total_error_0 < total_error_2) & (total_error_0 < total_error_3) & (total_error_0 < total_error_4))
                    return 0;
                else if ((total_error_1 < total_error_2) & (total_error_1 < total_error_3) & (total_error_1 < total_error_4))
                    return 1;
                else if ((total_error_2 < total_error_3) & (total_error_2 < total_error_4))
                    return 2;
                else if (total_error_3 < total_error_4)
                    return 3;
                return 4;
            }

            throw new ArgumentOutOfRangeException();
        }
#endif
		static unsafe uint calc_optimal_rice_params(int porder, int* parm, ulong* sums, uint n, uint pred_order, ref int method)
		{
			uint part = (1U << porder);
			uint cnt = (n >> porder) - pred_order;
			int maxK = method > 0 ? 30 : FlakeConstants.MAX_RICE_PARAM;
			int k = cnt > 0 ? Math.Min(maxK, BitReader.log2i(sums[0] / cnt)) : 0;
			int realMaxK0 = k;
			ulong all_bits = cnt * ((uint)k + 1U) + (sums[0] >> k);
			parm[0] = k;
			cnt = (n >> porder);
            int logcnt = BitReader.log2i(cnt);
            if (cnt == 1 << logcnt)
            {
                for (uint i = 1; i < part; i++)
                {
                    ulong s = sums[i];
                    ulong u = s >> logcnt;
                    k = u >> maxK != 0 ? maxK : BitReader.log2i((uint)u);
                    realMaxK0 = Math.Max(realMaxK0, k);
                    all_bits += ((uint)k << logcnt) + (s >> k);
                    parm[i] = k;
                }
            }
            else
            {
                for (uint i = 1; i < part; i++)
                {
                    ulong s = sums[i];
                    ulong u = s / cnt;
                    k = u >> maxK != 0 ? maxK : BitReader.log2i((uint)u);
                    realMaxK0 = Math.Max(realMaxK0, k);
                    all_bits += cnt * (uint)k + (s >> k);
                    parm[i] = k;
                }
            }
            all_bits += cnt * (part - 1U);
            method = realMaxK0 > FlakeConstants.MAX_RICE_PARAM ? 1 : 0;
			return (uint)all_bits + ((4U + (uint)method) * part);
		}

		static unsafe void calc_lower_sums(int pmin, int pmax, ulong* sums)
		{
			for (int i = pmax - 1; i >= pmin; i--)
			{
				for (int j = 0; j < (1 << i); j++)
				{
					sums[i * FlakeConstants.MAX_PARTITIONS + j] =
						sums[(i + 1) * FlakeConstants.MAX_PARTITIONS + 2 * j] +
						sums[(i + 1) * FlakeConstants.MAX_PARTITIONS + 2 * j + 1];
				}
			}
		}

        static unsafe uint calc_rice_params_sums(RiceContext rc, int pmin, int pmax, ulong* sums, uint n, uint pred_order, int bps)
        {
            int* parm = stackalloc int[(pmax + 1) * FlakeConstants.MAX_PARTITIONS];
            //uint* bits = stackalloc uint[FlakeConstants.MAX_PARTITION_ORDER];

            //assert(pmin >= 0 && pmin <= FlakeConstants.MAX_PARTITION_ORDER);
            //assert(pmax >= 0 && pmax <= FlakeConstants.MAX_PARTITION_ORDER);
            //assert(pmin <= pmax);

            // sums for lower levels
            calc_lower_sums(pmin, pmax, sums);

            uint opt_bits = AudioSamples.UINT32_MAX;
            int opt_porder = pmin;
            int opt_method = 0;
            for (int i = pmin; i <= pmax; i++)
            {
                int method = bps > 16 ? 1 : 0;
                uint bits = calc_optimal_rice_params(i, parm + i * FlakeConstants.MAX_PARTITIONS, sums + i * FlakeConstants.MAX_PARTITIONS, n, pred_order, ref method);
                if (bits <= opt_bits)
                {
                    opt_bits = bits;
                    opt_porder = i;
                    opt_method = method;
                }
            }

            rc.porder = opt_porder;
            rc.coding_method = opt_method;
            fixed (int* rparms = rc.rparams)
                AudioSamples.MemCpy(rparms, parm + opt_porder * FlakeConstants.MAX_PARTITIONS, (1 << opt_porder));

            return opt_bits;
        }

		static int get_max_p_order(int max_porder, int n, int order)
		{
			int porder = Math.Min(max_porder, BitReader.log2i(n ^ (n - 1)));
			if (order > 0)
				porder = Math.Min(porder, BitReader.log2i(n / order));
			return porder;
		}

//        private static int[,] best_x = new int[14,8193];
        private static int[][] good_x = new int[][] {
new int[] {}, // 0
new int[] { // 1
0x03,0x01,0x00,0x02
}, 
new int[] {// 2
0x01,0x07,0x06,0x02, 0x03,0x04,0x00,0x05
},
new int[] { // 3
0x0b,0x0f,0x0e,0x0d, 0x03,0x01,0x05,0x02
}, 
new int[] { //4
0x17,0x09,0x03,0x0a, 0x06,0x1d,0x1f,0x05, 0x1c,0x0d,0x07,0x0c,
},
new int[] { // 5
0x2b,0x3d,0x37,0x07, 0x11,0x15,0x36,0x3f,
}, 
new int[] { // 6
0x6b,0x15,0x7e,0x31, 0x07,0x1a,0x29,0x26, 0x5d,0x23,0x6f,0x19, 0x56,0x75
},
new int[] { // 7
0xdb,0xef,0xb5,0x47, 0xee,0x63,0x0b,0xfd, 0x31,0xbe,0xed,0x33, 0xff,0xfb,0xd6,0xbb
},
new int[] { // 8
0x1bb,0x1c7,0x069,0x087, 0x1fd,0x16e,0x095,0x1de, 0x066,0x071,0x055,0x09a,
},
new int[] { // 9
0x36b,0x3bd,0x097,0x0c3, 0x0e3,0x0b1,0x107,0x2de, 0x3ef,0x2fb,0x3d5,0x139
},
new int[] { // 10
//0x0e3,0x199,0x383,0x307, 0x1e3,0x01f,0x269,0x0f1, 0x266,0x03f,0x2cd,0x1c3, 0x19a,0x387,0x339,0x259,
0x6eb,0x187,0x77d,0x271, 0x195,0x259,0x5ae,0x169,
},
new int[] { // 11
0xddb,0xf77,0xb6d,0x587, 0x2c3,0x03b,0xef5,0x1e3, 0xdbe,
},
new int[] { // 12
0x1aeb,0x0587,0x0a71,0x1dbd, 0x0559,0x0aa5,0x0a2e,0x0d43, 0x05aa,0x00f3,0x0696,0x03c6,
},
new int[] { // 13
0x35d7,0x2f6f,0x0aa3,0x1569, 0x150f,0x3d79,0x0dc3,0x309f/*?*/,
},
new int[] { // 14
0x75d7,0x5f7b,0x6a8f,0x29a3,
},
new int[] { // 15
0xddd7,0xaaaf,0x55c3,0xf77b,
},
new int[] { // 16
0x1baeb,0x1efaf,0x1d5bf,0x1cff3,
},
new int[] { // 17
0x36dd7,0x3bb7b,0x3df6f,0x2d547,
},
new int[] { // 18
0x75dd7,0x6f77b,0x7aaaf,0x5ddd3,
},
new int[] { // 19
0xdddd7,0xf777b,0xd5547,0xb6ddb,
},
new int[] { // 20
0x1baeeb,0x1efbaf,0x1aaabf,0x17bbeb,
},
new int[] { // 21
0x376dd7,0x3ddf7b,0x2d550f,0x0aaaa3,
},
new int[] { // 22
0x6eddd7,0x77777b,0x5dcd4f,0x5d76f9,
},
new int[] { // 23
0xdeddd7,0xb5b6eb,0x55552b,0x2aaac3,
},
new int[] { // 24
0x1dddbb7,0x1b76eeb,0x17bbf5f,0x1eeaa9f,
},
new int[] { // 25
},
new int[] { // 26
},
new int[] { // 27
},
new int[] { // 28
},
new int[] { // 29
},
new int[] { // 30
},
        };

        unsafe void postprocess_coefs(FlacFrame frame, FlacSubframe sf, int ch)
        {
            if (eparams.development_mode < 0)
                return;
            if (sf.type != SubframeType.LPC || sf.order > 30)
                return;
            int orig_window = sf.window;
            int orig_order = sf.order;
            int orig_shift = sf.shift;
            int orig_cbits = sf.cbits;
            uint orig_size = sf.size;
            var orig_coefs = stackalloc int[orig_order];
            for (int i = 0; i < orig_order; i++) orig_coefs[i] = sf.coefs[i];
            int orig_xx = -1;
            int orig_seq = 0;
            int maxxx = Math.Min(good_x[orig_order].Length, eparams.development_mode);
            var pmax = get_max_p_order(m_settings.MaxPartitionOrder, frame.blocksize, orig_order);
            var pmin = Math.Min(m_settings.MinPartitionOrder, pmax);
            ulong* sums = stackalloc ulong[(pmax + 1) * FlakeConstants.MAX_PARTITIONS];

            while (true)
            {
                var best_coefs = stackalloc int[orig_order];
                int best_shift = orig_shift;
                int best_cbits = orig_cbits;
                uint best_size = orig_size;
                int best_xx = -1;
                for (int xx = -1; xx < maxxx; xx++)
                {
                    int x = xx;
                    if (xx < 0)
                    {
                        if (orig_xx < 0 || maxxx < 1/*3*/)// || (orig_xx >> orig_order) != 0)
                            continue;
                        x = orig_xx;
                        orig_seq++;
                    }
                    else
                    {
                        orig_seq = 0;
                        if (orig_order < good_x.Length && good_x[orig_order] != null)
                            x = good_x[orig_order][xx];
                    }

                    frame.current.type = SubframeType.LPC;
                    frame.current.order = orig_order;
                    frame.current.window = orig_window;
                    frame.current.shift = orig_shift;
                    frame.current.cbits = orig_cbits;

                    if (((x >> orig_order) & 1) != 0)
                    {
                        frame.current.shift--;
                        frame.current.cbits--;
                        if (frame.current.shift < 0 || frame.current.cbits < 2)
                            continue;
                    }

                    ulong csum = 0;
                    int qmax = (1 << (frame.current.cbits - 1)) - 1;
                    for (int i = 0; i < frame.current.order; i++)
                    {
                        int shift = (x >> orig_order) & 1;
                        int increment = (x == 1 << orig_order) ? 0 : (((x >> i) & 1) << 1) - 1;
                        frame.current.coefs[i] = (orig_coefs[i] + (increment << orig_seq)) >> shift;
                        if (frame.current.coefs[i] < -(qmax + 1)) frame.current.coefs[i] = -(qmax + 1);
                        if (frame.current.coefs[i] > qmax) frame.current.coefs[i] = qmax;
                        csum += (ulong)Math.Abs(frame.current.coefs[i]);
                    }

                    fixed (int* coefs = frame.current.coefs)
                    {
                        if ((csum << frame.subframes[ch].obits) >= 1UL << 32)
                            lpc.encode_residual_long(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order, coefs, frame.current.shift, sums + pmax * FlakeConstants.MAX_PARTITIONS, pmax);
                        else
                            lpc.encode_residual(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order, coefs, frame.current.shift, sums + pmax * FlakeConstants.MAX_PARTITIONS, pmax);
                    }

                    var cur_size = calc_rice_params_sums(frame.current.rc, pmin, pmax, sums, (uint)frame.blocksize, (uint)frame.current.order, Settings.PCM.BitsPerSample);
                    frame.current.size = (uint)(frame.current.order * frame.subframes[ch].obits + 4 + 5 + frame.current.order * frame.current.cbits + 6 + (int)cur_size);

                    if (frame.current.size < best_size)
                    {
                        //var dif = best_size - frame.current.size;
                        for (int i = 0; i < frame.current.order; i++) best_coefs[i] = frame.current.coefs[i];
                        best_shift = frame.current.shift;
                        best_cbits = frame.current.cbits;
                        best_size = frame.current.size;
                        best_xx = x;
                        frame.ChooseBestSubframe(ch);
                        //if (dif > orig_order * 5)
                        //    break;
                    }

                    if (xx < 0 && best_size < orig_size)
                        break;
                }

                if (best_size < orig_size)
                {
                    //if (best_xx >= 0) best_x[order, best_xx]++;
                    //if (orig_size != 0x7FFFFFFF)
                    //    System.Console.Write(string.Format(" {0}[{1:x}]", orig_size - best_size, best_xx));
                    for (int i = 0; i < orig_order; i++) orig_coefs[i] = best_coefs[i];
                    orig_shift = best_shift;
                    orig_cbits = best_cbits;
                    orig_size = best_size;
                    orig_xx = best_xx;
                }
                else
                {
                    break;
                }
            }

            //if (orig_size != 0x7FFFFFFF)
            //    System.Console.WriteLine();

            //if (frame_count % 0x400 == 0)
            //{
            //    for (int o = 0; o < best_x.GetLength(0); o++)
            //    {
            //        //for (int x = 0; x <= (1 << o); x++)
            //        //    if (best_x[o, x] != 0)
            //        //        System.Console.WriteLine(string.Format("{0:x2}\t{1:x4}\t{2}", o, x, best_x[o, x]));
            //        var s = new List<KeyValuePair<int, int>>();
            //        for (int x = 0; x < (1 << o); x++)
            //            if (best_x[o, x] != 0)
            //                s.Add(new KeyValuePair<int, int>(x, best_x[o, x]));
            //        s.Sort((x, y) => y.Value.CompareTo(x.Value));
            //        foreach (var x in s)
            //            System.Console.WriteLine(string.Format("{0:x2}\t{1:x4}\t{2}", o, x.Key, x.Value));
            //        int i = 0;
            //        foreach (var x in s)
            //        {
            //            System.Console.Write(string.Format(o <= 8 ? "0x{0:x2}," : "0x{0:x3},", x.Key));
            //            if ((++i) % 16 == 0)
            //                System.Console.WriteLine();
            //        }
            //        System.Console.WriteLine();
            //    }
            //}
        }

        public static void SetCoefs(int order, int[] coefs)
        {
            good_x[order] = new int[coefs.Length];
            for (int i = 0; i < coefs.Length; i++)
                good_x[order][i] = coefs[i];
        }

        unsafe void encode_residual_lpc_sub(FlacFrame frame, float* lpcs, int iWindow, int order, int ch)
        {
            // select LPC precision based on block size
            uint lpc_precision;
            if (frame.blocksize <= 192) lpc_precision = 7U;
            else if (frame.blocksize <= 384) lpc_precision = 8U;
            else if (frame.blocksize <= 576) lpc_precision = 9U;
            else if (frame.blocksize <= 1152) lpc_precision = 10U;
            else if (frame.blocksize <= 2304) lpc_precision = 11U;
            else if (frame.blocksize <= 4608) lpc_precision = 12U;
            else if (frame.blocksize <= 8192) lpc_precision = 13U;
            else if (frame.blocksize <= 16384) lpc_precision = 14U;
            else lpc_precision = 15;

            for (int i_precision = m_settings.MinPrecisionSearch; i_precision <= m_settings.MaxPrecisionSearch && lpc_precision + i_precision < 16; i_precision++)
                // check if we already calculated with this order, window and precision
                if ((frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[i_precision] & (1U << (order - 1))) == 0)
                {
                    frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[i_precision] |= (1U << (order - 1));

                    uint cbits = lpc_precision + (uint)i_precision;

                    frame.current.type = SubframeType.LPC;
                    frame.current.order = order;
                    frame.current.window = iWindow;
                    frame.current.cbits = (int)cbits;

                    int pmax = get_max_p_order(m_settings.MaxPartitionOrder, frame.blocksize, frame.current.order);
                    int pmin = Math.Min(m_settings.MinPartitionOrder, pmax);
                    ulong* sums = stackalloc ulong[(pmax + 1) * FlakeConstants.MAX_PARTITIONS];
                    ulong csum = 0;
                    fixed (int* coefs = frame.current.coefs)
                    {
                        lpc.quantize_lpc_coefs(lpcs + (frame.current.order - 1) * lpc.MAX_LPC_ORDER,
                            frame.current.order, cbits, coefs, out frame.current.shift, 15, 0);

                        if (frame.current.shift < 0 || frame.current.shift > 15)
                            throw new Exception("negative shift");

                        for (int i = frame.current.order; i > 0; i--)
                            csum += (ulong)Math.Abs(coefs[i - 1]);

                        if ((csum << frame.subframes[ch].obits) >= 1UL << 32)
                            lpc.encode_residual_long(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order, coefs, frame.current.shift, sums + pmax * FlakeConstants.MAX_PARTITIONS, pmax);
                        else
                            lpc.encode_residual(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order, coefs, frame.current.shift, sums + pmax * FlakeConstants.MAX_PARTITIONS, pmax);

                    }
                    uint best_size = calc_rice_params_sums(frame.current.rc, pmin, pmax, sums, (uint)frame.blocksize, (uint)frame.current.order, Settings.PCM.BitsPerSample);
                    frame.current.size = (uint)(frame.current.order * frame.subframes[ch].obits + 4 + 5 + frame.current.order * (int)cbits + 6 + (int)best_size);
                    frame.ChooseBestSubframe(ch);
                    //if (frame.current.size >= frame.subframes[ch].best.size)
                    //    postprocess_coefs(frame, frame.current, ch);
                    //else
                    //{
                    //    frame.ChooseBestSubframe(ch);
                    //    postprocess_coefs(frame, frame.subframes[ch].best, ch);
                    //}
                }
        }

        unsafe void encode_residual_fixed_sub(FlacFrame frame, int order, int ch)
        {
            if ((frame.subframes[ch].done_fixed & (1U << order)) != 0)
                return; // already calculated;

            frame.current.order = order;
            frame.current.type = SubframeType.Fixed;

#if XXX
            int best_order = order;
            if (frame.subframes[ch].done_fixed == 0)
            {
                best_order = encode_residual_fixed_estimate_best_order(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order);
                if (best_order != order)
                {
                    //frame.subframes[ch].done_fixed |= (1U << order);
                    order = best_order;
                    frame.current.order = order;
                    encode_residual_fixed(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order);
                }
            }
            else
#endif
            int pmax = get_max_p_order(m_settings.MaxPartitionOrder, frame.blocksize, frame.current.order);
            int pmin = Math.Min(m_settings.MinPartitionOrder, pmax);
            ulong* sums = stackalloc ulong[(pmax + 1) * FlakeConstants.MAX_PARTITIONS];
            encode_residual_fixed(frame.current.residual, frame.subframes[ch].samples, frame.blocksize, frame.current.order, sums + pmax * FlakeConstants.MAX_PARTITIONS, pmax);

            frame.current.size = (uint)(frame.current.order * frame.subframes[ch].obits) + 6
                + calc_rice_params_sums(frame.current.rc, pmin, pmax, sums, (uint)frame.blocksize, (uint)frame.current.order, Settings.PCM.BitsPerSample);

            frame.subframes[ch].done_fixed |= (1U << order);

            frame.ChooseBestSubframe(ch);
        }

        unsafe void fixed_compute_best_predictor(int* data, uint data_len, ulong* errors)//, float* residual_bits_per_sample)
        {
            long last_error_0 = data[-1];
            long last_error_1 = data[-1] - data[-2];
            long last_error_2 = last_error_1 - (data[-2] - data[-3]);
            long last_error_3 = last_error_2 - (data[-2] - 2 * data[-3] + data[-4]);
            ulong total_error_0 = 0, total_error_1 = 0, total_error_2 = 0, total_error_3 = 0, total_error_4 = 0;

#if VARIANT1
            long error, save;
            int* finish = data + data_len;
            while (data < finish)
            {
                error = *(data++); total_error_0 += (ulong)((error << 1) ^ (error >> 63)); save = error;
                error -= last_error_0; total_error_1 += (ulong)((error << 1) ^ (error >> 63)); last_error_0 = save; save = error;
                error -= last_error_1; total_error_2 += (ulong)((error << 1) ^ (error >> 63)); last_error_1 = save; save = error;
                error -= last_error_2; total_error_3 += (ulong)((error << 1) ^ (error >> 63)); last_error_2 = save; save = error;
                error -= last_error_3; total_error_4 += (ulong)((error << 1) ^ (error >> 63)); last_error_3 = save;
            }
#else
            int* finish = data + data_len;
            while (data < finish)
            {
                long next_error_0 = *(data++);
                long next_error_1 = next_error_0 - last_error_0;
                long next_error_2 = next_error_1 - last_error_1;
                long next_error_3 = next_error_2 - last_error_2;
                long next_error_4 = next_error_3 - last_error_3;

                last_error_0 = next_error_0;
                last_error_1 = next_error_1;
                last_error_2 = next_error_2;
                last_error_3 = next_error_3;

                total_error_0 += (ulong)((last_error_0 << 1) ^ (last_error_0 >> 63));
                total_error_1 += (ulong)((last_error_1 << 1) ^ (last_error_1 >> 63));
                total_error_2 += (ulong)((last_error_2 << 1) ^ (last_error_2 >> 63));
                total_error_3 += (ulong)((last_error_3 << 1) ^ (last_error_3 >> 63));
                total_error_4 += (ulong)((next_error_4 << 1) ^ (next_error_4 >> 63));
            }
#endif

            errors[0] = total_error_0;
            errors[1] = total_error_1;
            errors[2] = total_error_2;
            errors[3] = total_error_3;
            errors[4] = total_error_4;

            //residual_bits_per_sample[0] = (float)((total_error_0 > 0) ? log(M_LN2 * (FLAC__double)total_error_0 / (FLAC__double)data_len) / M_LN2 : 0.0);
            //residual_bits_per_sample[1] = (float)((total_error_1 > 0) ? log(M_LN2 * (FLAC__double)total_error_1 / (FLAC__double)data_len) / M_LN2 : 0.0);
            //residual_bits_per_sample[2] = (float)((total_error_2 > 0) ? log(M_LN2 * (FLAC__double)total_error_2 / (FLAC__double)data_len) / M_LN2 : 0.0);
            //residual_bits_per_sample[3] = (float)((total_error_3 > 0) ? log(M_LN2 * (FLAC__double)total_error_3 / (FLAC__double)data_len) / M_LN2 : 0.0);
            //residual_bits_per_sample[4] = (float)((total_error_4 > 0) ? log(M_LN2 * (FLAC__double)total_error_4 / (FLAC__double)data_len) / M_LN2 : 0.0);
        }

        unsafe int fixed_compute_best_predictor_order(ulong* error)
        {
            int order;
            if ((error[0] < error[1]) & (error[0] < error[2]) & (error[0] < error[3]) & (error[0] < error[4]))
                order = 0;
            else if ((error[1] < error[2]) & (error[1] < error[3]) & (error[1] < error[4]))
                order = 1;
            else if ((error[2] < error[3]) & (error[2] < error[4]))
                order = 2;
            else if (error[3] < error[4])
                order = 3;
            else
                order = 4;
            return order;
        }

		unsafe void encode_residual(FlacFrame frame, int ch, PredictionType predict, OrderMethod omethod, int pass, int windows_mask)
		{
			int* smp = frame.subframes[ch].samples;
			int i, n = frame.blocksize;
			// save best.window, because we can overwrite it later with fixed frame

			// CONSTANT
			for (i = 1; i < n; i++)
			{
				if (smp[i] != smp[0]) break;
			}
			if (i == n)
			{
				frame.subframes[ch].best.type = SubframeType.Constant;
				frame.subframes[ch].best.residual[0] = smp[0];
				frame.subframes[ch].best.size = (uint)frame.subframes[ch].obits;
				return;
			}

			// VERBATIM
			frame.current.type = SubframeType.Verbatim;
			frame.current.size = (uint)(frame.subframes[ch].obits * frame.blocksize);
			frame.ChooseBestSubframe(ch);

			if (n < 5 || predict == PredictionType.None)
				return;

			// LPC
            if (n > m_settings.MaxLPCOrder &&
			   (predict == PredictionType.Levinson ||
				predict == PredictionType.Search)
				//predict == PredictionType.Search ||
				//(pass == 2 && frame.subframes[ch].best.type == SubframeType.LPC))
				)
			{
				float* lpcs = stackalloc float[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
                int min_order = m_settings.MinLPCOrder;
                int max_order = m_settings.MaxLPCOrder;

				for (int iWindow = 0; iWindow < _windowcount; iWindow++)
				{
                    if ((windows_mask & (1 << iWindow)) == 0)
						continue;

					LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[iWindow];
                    fixed (LpcWindowSection* sections = &windowSections[frame.nSeg, iWindow, 0])
                        lpc_ctx.GetReflection(
                            frame.subframes[ch].sf, max_order, frame.blocksize, 
                            smp, frame.window_buffer + iWindow * FlakeConstants.MAX_BLOCKSIZE * 2, sections);
                    lpc_ctx.ComputeLPC(lpcs);

					//int frameSize = n;
					//float* F = stackalloc float[frameSize];
					//float* B = stackalloc float[frameSize];
					//float* PE = stackalloc float[max_order + 1];
					//float* arp = stackalloc float[max_order];
					//float* rc = stackalloc float[max_order];

					//for (int j = 0; j < frameSize; j++)
					//    F[j] = B[j] = smp[j];

					//for (int K = 1; K <= max_order; K++)
					//{
					//    // BURG:
					//    float denominator = 0.0f;
					//    //float denominator = F[K - 1] * F[K - 1] + B[frameSize - K] * B[frameSize - K];
					//    for (int j = 0; j < frameSize - K; j++)
					//        denominator += F[j + K] * F[j + K] + B[j] * B[j];
					//    denominator /= 2;

					//    // Estimate error
					//    PE[K - 1] = denominator / (frameSize - K);

					//    float reflectionCoeff = 0.0f;
					//    for (int j = 0; j < frameSize - K; j++)
					//        reflectionCoeff += F[j + K] * B[j];
					//    reflectionCoeff /= denominator;
					//    rc[K - 1] = arp[K - 1] = reflectionCoeff;

					//    // Levinson-Durbin
					//    for (int j = 0; j < (K - 1) >> 1; j++)
					//    {
					//        float arptmp = arp[j];
					//        arp[j] -= reflectionCoeff * arp[K - 2 - j];
					//        arp[K - 2 - j] -= reflectionCoeff * arptmp;
					//    }
					//    if (((K - 1) & 1) != 0)
					//        arp[(K - 1) >> 1] -= reflectionCoeff * arp[(K - 1) >> 1];

					//    for (int j = 0; j < frameSize - K; j++)
					//    {
					//        float f = F[j + K];
					//        float b = B[j];
					//        F[j + K] = f - reflectionCoeff * b;
					//        B[j] = b - reflectionCoeff * f;
					//    }

					//    for (int j = 0; j < K; j++)
					//        lpcs[(K - 1) * lpc.MAX_LPC_ORDER + j] = (float)arp[j];
					//}

					switch (omethod)
					{
						case OrderMethod.Akaike:
                            //lpc_ctx.SortOrdersAkaike(frame.blocksize, m_settings.EstimationDepth, max_order, 7.1, 0.0);
							lpc_ctx.SortOrdersAkaike(frame.blocksize, m_settings.EstimationDepth, min_order, max_order, 4.5, 0);
							break;
						default:
							throw new Exception("unknown order method");
					}

                    for (i = 0; i < m_settings.EstimationDepth && i < max_order; i++)
						encode_residual_lpc_sub(frame, lpcs, iWindow, lpc_ctx.best_orders[i], ch);
                }

                postprocess_coefs(frame, frame.subframes[ch].best, ch);
            }
        
            // FIXED
            if (predict == PredictionType.Fixed ||
                (predict == PredictionType.Search && pass != 1) ||
                //predict == PredictionType.Search ||
                //(pass == 2 && frame.subframes[ch].best.type == SubframeType.Fixed) ||
                (n > m_settings.MaxFixedOrder && n <= m_settings.MaxLPCOrder))
            {
                int max_fixed_order = Math.Min(m_settings.MaxFixedOrder, 4);
                int min_fixed_order = Math.Min(m_settings.MinFixedOrder, max_fixed_order);

                if (min_fixed_order == 0 && max_fixed_order == 4)
                {
                    fixed (ulong* fixed_errors = frame.subframes[ch].best_fixed)
                    {
                        if ((frame.subframes[ch].done_fixed & (1U << 5)) == 0)
                        {
                            fixed_compute_best_predictor(smp + 4, (uint)n - 4, fixed_errors);
                            frame.subframes[ch].done_fixed |= (1U << 5);
                        }
                        i = fixed_compute_best_predictor_order(fixed_errors);
                        encode_residual_fixed_sub(frame, i, ch);
                    }
                }
                else
                {
                    for (i = max_fixed_order; i >= min_fixed_order; i--)
                        encode_residual_fixed_sub(frame, i, ch);
                }
            }

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
			bitwriter.writebits(2, sub.best.rc.coding_method);

			// partition order
			int porder = sub.best.rc.porder;
			int psize = frame.blocksize >> porder;
			//assert(porder >= 0);
			bitwriter.writebits(4, porder);
			int res_cnt = psize - sub.best.order;

			int rice_len = 4 + sub.best.rc.coding_method;
			// residual
			int j = sub.best.order;
			fixed (byte* fixbuf = &frame_buffer[0])
			for (int p = 0; p < (1 << porder); p++)
			{
				int k = sub.best.rc.rparams[p];
				bitwriter.writebits(rice_len, k);
				if (p == 1) res_cnt = psize;
				int cnt = Math.Min(res_cnt, frame.blocksize - j);
				bitwriter.write_rice_block_signed(fixbuf, k, sub.best.residual + j, cnt);
				j += cnt;
			}
		}

		unsafe void 
		output_subframe_constant(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			bitwriter.writebits_signed(sub.obits, sub.best.residual[0]);
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
				bitwriter.writebits_signed(sub.obits, sub.best.residual[i]);

			// residual
			output_residual(frame, bitwriter, sub);
		}

		unsafe void
		output_subframe_lpc(FlacFrame frame, BitWriter bitwriter, FlacSubframeInfo sub)
		{
			// warm-up samples
			for (int i = 0; i < sub.best.order; i++)
				bitwriter.writebits_signed(sub.obits, sub.best.residual[i]);

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
            ushort crc = bitwriter.get_crc16();
            bitwriter.writebits(16, crc);
			bitwriter.flush();
		}

        unsafe void encode_residual_pass1(FlacFrame frame, int ch, int windows_mask)
		{
            int max_prediction_order = m_settings.MaxLPCOrder;
            //int max_fixed_order = m_settings.MaxFixedOrder;
            //int min_fixed_order = m_settings.MinFixedOrder;
			int lpc_min_precision_search = m_settings.MinPrecisionSearch;
            int lpc_max_precision_search = m_settings.MaxPrecisionSearch;
            int max_partition_order = m_settings.MaxPartitionOrder;
            int estimation_depth = m_settings.EstimationDepth;
            var development_mode = eparams.development_mode;
            //m_settings.MinFixedOrder = 2;
            //m_settings.MaxFixedOrder = 2;
            m_settings.MinPrecisionSearch = m_settings.MaxPrecisionSearch;
            m_settings.MaxLPCOrder = Math.Min(m_settings.MaxLPCOrder, Math.Max(m_settings.MinLPCOrder, 8));
            m_settings.EstimationDepth = 1;
            eparams.development_mode = Math.Min(eparams.development_mode, -1);
            encode_residual(frame, ch, m_settings.PredictionType, OrderMethod.Akaike, 1, windows_mask);
            //m_settings.MinFixedOrder = min_fixed_order;
            //m_settings.MaxFixedOrder = max_fixed_order;
            m_settings.MaxLPCOrder = max_prediction_order;
            m_settings.MinPrecisionSearch = lpc_min_precision_search;
            m_settings.MaxPrecisionSearch = lpc_max_precision_search;
            m_settings.MaxPartitionOrder = max_partition_order;
            m_settings.EstimationDepth = estimation_depth;
            eparams.development_mode = development_mode;
		}

		unsafe void encode_residual_pass2(FlacFrame frame, int ch)
		{
            encode_residual(frame, ch, m_settings.PredictionType, eparams.order_method, 2, estimate_best_windows(frame, ch));
		}

        unsafe int estimate_best_windows_akaike(FlacFrame frame, int ch, int count, bool onePerType)
        {
            int* windows_present = stackalloc int[_windowcount];
            for (int i = 0; i < _windowcount; i++)
                windows_present[i] = 0;
            if (onePerType)
            {
                for (int i = 0; i < _windowcount; i++)
                    for (int j = 0; j < _windowcount; j++)
                        if (windowType[j] == windowType[i])
                            windows_present[j]++;
            }

            float* err = stackalloc float[lpc.MAX_LPC_ORDER];
            for (int i = 0; i < _windowcount; i++)
            {
                LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[i];
                if (onePerType && windows_present[i] <= count)
                {
                    err[i] = 0;
                    continue;
                }
                int estimate_order = 4;
                fixed (LpcWindowSection* sections = &windowSections[frame.nSeg, i, 0])
                lpc_ctx.GetReflection(
                    frame.subframes[ch].sf, estimate_order, frame.blocksize,
                    frame.subframes[ch].samples, frame.window_buffer + i * FlakeConstants.MAX_BLOCKSIZE * 2, sections);
                lpc_ctx.SortOrdersAkaike(frame.blocksize, 1, 1, estimate_order, 4.5, 0.0);
                //err[i] = (float)(lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], 4.5, 0.0));
                //err[i] = (float)((frame.blocksize * lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1] / windowScale[i]) + lpc_ctx.best_orders[0] * 4.5);
                //err[i] = (float)((frame.blocksize * lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1] / windowScale[i]) + lpc_ctx.best_orders[0] * frame.subframes[ch].obits);
                
                // realistic
                //err[i] = (float)(frame.blocksize * Math.Log(lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1]) / Math.Log(2) / 2.5
                    //- windowScale[i] / 2 + lpc_ctx.best_orders[0] * frame.subframes[ch].obits / 2);

                //err[i] = (float)(frame.blocksize * Math.Log(lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1]) / Math.Log(2) / 2.5
                    //- frame.blocksize * Math.Log(lpc_ctx.autocorr_values[0]) / 2.1
                    //+ Math.Log(frame.blocksize) * lpc_ctx.best_orders[0] * 4.5 / 2.5 / Math.Log(2));

                // Akaike
                //err[i] = (float)(frame.blocksize * (Math.Log(lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1])) + Math.Log(frame.blocksize) * lpc_ctx.best_orders[0] * 4.5);

                //err[i] = (float)(lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], 4.5, 0.0) - frame.blocksize * (frame.subframes[ch].obits + Math.Log(windowScale[i] / frame.blocksize) / 2));

                // tested good
                err[i] = (float)(lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], 4.5, 0.0) - frame.blocksize * Math.Log(lpc_ctx.autocorr_values[0]) / 2);
            }
            int* best_windows = stackalloc int[lpc.MAX_LPC_ORDER];
            for (int i = 0; i < _windowcount; i++)
                best_windows[i] = i;
            for (int i = 0; i < _windowcount; i++)
            {
                for (int j = i + 1; j < _windowcount; j++)
                {
                    if (err[best_windows[i]] > err[best_windows[j]])
                    {
                        int tmp = best_windows[j];
                        best_windows[j] = best_windows[i];
                        best_windows[i] = tmp;
                    }
                }
            }
            int window_mask = 0;
            if (onePerType)
            {
                for (int i = 0; i < _windowcount; i++)
                    windows_present[i] = count;
                for (int i = 0; i < _windowcount; i++)
                {
                    int w = best_windows[i];
                    if (windows_present[w] > 0)
                    {
                        for (int j = 0; j < _windowcount; j++)
                            if (windowType[j] == windowType[w])
                                windows_present[j]--;
                        window_mask |= 1 << w;
                    }
                }
            }
            else
            {
                for (int i = 0; i < _windowcount && i < count; i++)
                    window_mask |= 1 << best_windows[i];
            }
            return window_mask;
        }

		unsafe int estimate_best_windows(FlacFrame frame, int ch)
		{
            if (_windowcount == 1 || m_settings.PredictionType == PredictionType.Fixed)
				return 1;
			switch (m_settings.WindowMethod)
			{
                case WindowMethod.Estimate:
                    return estimate_best_windows_akaike(frame, ch, 1, false);
                case WindowMethod.Estimate2:
                    return estimate_best_windows_akaike(frame, ch, 2, false);
                case WindowMethod.Estimate3:
                    return estimate_best_windows_akaike(frame, ch, 3, false);
                case WindowMethod.EstimateN:
                    return estimate_best_windows_akaike(frame, ch, 1, true);
                case WindowMethod.Evaluate2:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 2, false));
                    return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.Evaluate3:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 3, false));
                    return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.EvaluateN:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 1, true));
#if XXX
                    if (frame.subframes[ch].best.type == SubframeType.LPC && frame.subframes[ch].best.order <= 4)
                    {
                        LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[frame.subframes[ch].best.window];
                        double err = lpc_ctx.prediction_error[frame.subframes[ch].best.order - 1] / lpc_ctx.autocorr_values[0];
                        double est = frame.blocksize * (frame.subframes[ch].obits * (1 - err));
                        double est1 = frame.blocksize * (frame.subframes[ch].obits * (err));
                        if (est < 0 || est1 < 0) return -1;
                    }
#endif
                    return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.Evaluate2N:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 2, true));
                    return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.Evaluate3N:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 3, true));
                    return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.Evaluate:
					encode_residual_pass1(frame, ch, -1);
					return frame.subframes[ch].best.type == SubframeType.LPC ? 1 << frame.subframes[ch].best.window : 0;
                case WindowMethod.Search:
					return -1;
			}
			return -1;
		}

		unsafe void estimate_frame(FlacFrame frame, bool do_midside)
		{
			int subframes = do_midside ? channels * 2 : channels;

			switch (m_settings.StereoMethod)
			{
				case StereoMethod.Estimate:
					for (int ch = 0; ch < subframes; ch++)
					{
                        LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[0];
                        int estimate_order = 4;
                        int iWindow = 0;
                        fixed (LpcWindowSection* sections = &windowSections[frame.nSeg, iWindow, 0])
                            lpc_ctx.GetReflection(
                                frame.subframes[ch].sf, estimate_order, frame.blocksize,
                                frame.subframes[ch].samples, frame.window_buffer + iWindow * FlakeConstants.MAX_BLOCKSIZE * 2, sections);
                        lpc_ctx.SortOrdersAkaike(frame.blocksize, 1, 1, estimate_order, 4.5, 0.0);
                        frame.subframes[ch].best.size 
                            = (uint)Math.Max(0, frame.blocksize * (Math.Log(lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1])) + Math.Log(frame.blocksize) * lpc_ctx.best_orders[0] * 4.5
                            //= (uint)Math.Max(0, lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], 4.5, 0.0)
                            //* 2.0 / Math.Log(windowScale[0] / frame.blocksize)
                            + 7.1 * frame.subframes[ch].obits * m_settings.MaxLPCOrder);
                    }
					break;
                case StereoMethod.EstimateFixed:
                    for (int ch = 0; ch < subframes; ch++)
                    {
                        fixed (ulong* fixed_errors = frame.subframes[ch].best_fixed)
                        {
                            if ((frame.subframes[ch].done_fixed & (1U << 5)) == 0)
                            {
                                fixed_compute_best_predictor(frame.subframes[ch].samples + 4, (uint)frame.blocksize - 4, fixed_errors);
                                frame.subframes[ch].done_fixed |= (1U << 5);
                            }
                            int best_order = fixed_compute_best_predictor_order(fixed_errors);
                            //residual_bits_per_sample[0] = (float)((total_error_0 > 0) ? log(M_LN2 * (FLAC__double)total_error_0 / (FLAC__double)data_len) / M_LN2 : 0.0);
                            frame.subframes[ch].best.size = (uint)fixed_errors[best_order];
                        }
                    }
                    break;
                case StereoMethod.EstimateX:
                    for (int ch = 0; ch < subframes; ch++)
                    {
                        for (int iWindow = 0; iWindow < _windowcount; iWindow++)
                        {
                            LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[iWindow];
                            int estimate_order = 4;
                            fixed (LpcWindowSection* sections = &windowSections[frame.nSeg, iWindow, 0])
                                lpc_ctx.GetReflection(
                                    frame.subframes[ch].sf, estimate_order, frame.blocksize,
                                    frame.subframes[ch].samples, frame.window_buffer + iWindow * FlakeConstants.MAX_BLOCKSIZE * 2, sections);
                            lpc_ctx.SortOrdersAkaike(frame.blocksize, 1, 1, estimate_order, 4.5, 0.0);
                            uint estimate
                                = (uint)Math.Max(0, frame.blocksize * (Math.Log(lpc_ctx.prediction_error[lpc_ctx.best_orders[0] - 1])) + Math.Log(frame.blocksize) * lpc_ctx.best_orders[0] * 4.5
                                //= (uint)Math.Max(0, lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], 4.5, 0.0)
                                //* 2.0 / Math.Log(windowScale[0] / frame.blocksize)
                                + 7.1 * frame.subframes[ch].obits * m_settings.MaxLPCOrder);
                            if (iWindow == 0 || frame.subframes[ch].best.size > estimate)
                                frame.subframes[ch].best.size = estimate;
                        }
                    }
                    break;
                case StereoMethod.Evaluate:
                    for (int ch = 0; ch < subframes; ch++)
                        encode_residual_pass1(frame, ch, 1);
					break;
                case StereoMethod.EvaluateX:
                    for (int ch = 0; ch < subframes; ch++)
                        encode_residual_pass1(frame, ch,
                            estimate_best_windows_akaike(frame, ch, 1, false));
					break;                    
				case StereoMethod.Search:
					for (int ch = 0; ch < subframes; ch++)
					    encode_residual_pass2(frame, ch);
					break;
			}
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

		unsafe void encode_estimated_frame(FlacFrame frame)
		{
			switch (m_settings.StereoMethod)
			{
				case StereoMethod.Estimate:
                case StereoMethod.EstimateX:
                case StereoMethod.EstimateFixed:
                    for (int ch = 0; ch < channels; ch++)
					{
						frame.subframes[ch].best.size = AudioSamples.UINT32_MAX;
						encode_residual_pass2(frame, ch);
					}
					break;
                case StereoMethod.Evaluate:
                case StereoMethod.EvaluateX:
                    for (int ch = 0; ch < channels; ch++)
						encode_residual_pass2(frame, ch);
					break;
				case StereoMethod.Search:
					break;
			}
		}

		unsafe delegate void window_function(float* window, int size);

		unsafe void calculate_window(float* window, window_function func, WindowFunction flag)
		{
			if ((m_settings.WindowFunctions & flag) == 0 || _windowcount == lpc.MAX_LPC_WINDOWS)
				return;
			int sz = _windowsize;
			float* pos1 = window + _windowcount * FlakeConstants.MAX_BLOCKSIZE * 2;
			float* pos = pos1;
            int nSeg = 0;
			do
			{
                windowSections[nSeg, _windowcount, 0].setData(0, sz);
                for (int j = 1; j < lpc.MAX_LPC_SECTIONS; j++)
                    windowSections[nSeg, _windowcount, j].setZero(sz, sz);

                fixed (LpcWindowSection* sections = &windowSections[nSeg, _windowcount, 0])
				func(pos, sz);
                if ((sz & 1) != 0)
					break;
                nSeg++;
                pos += sz;
				sz >>= 1;
			} while (sz >= 32);
			double scale = 0.0;
			for (int i = 0; i < _windowsize; i++)
				scale += pos1[i] * pos1[i];
			windowScale[_windowcount] = scale;
            windowType[_windowcount] = flag;
			_windowcount++;
		}

        class PunchoutTukeyVariant
        {
            public PunchoutTukeyVariant(
                WindowFunction _type,
                int _parts, double _overlap, double _p)
            {
                parts = _parts;
                type = _type;
                overlap = _overlap;
                p = _p;
            }            
            public WindowFunction type;
            public int parts;
            public double overlap;
            public double p;
        };

		unsafe int encode_frame(out int size)
		{
			fixed (int* s = samplesBuffer, r = residualBuffer)
			fixed (float* window = windowBuffer)
			{
				frame.InitSize(m_blockSize, eparams.variable_block_size != 0);

                if (frame.blocksize != _windowsize && frame.blocksize > 4 && m_settings.PredictionType != PredictionType.Fixed)
                {
                    _windowsize = frame.blocksize;
                    _windowcount = 0;
                    calculate_window(window, lpc.window_welch, WindowFunction.Welch);
                    calculate_window(window, lpc.window_flattop, WindowFunction.Flattop);
                    calculate_window(window, lpc.window_hann, WindowFunction.Hann);
                    calculate_window(window, lpc.window_bartlett, WindowFunction.Bartlett);
                    var tukeys = new PunchoutTukeyVariant[]
                    {
                        new PunchoutTukeyVariant(WindowFunction.Tukey4,  4, 0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey4A, 4, 0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey4B, 4, 0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey4X, 4, m_settings.TukeyOverlap, m_settings.TukeyP),
                        new PunchoutTukeyVariant(WindowFunction.Tukey3,  3, 1.0/3, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey3A, 3, 1.0/3, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey3B, 3, 1.0/3, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey3X, 3, m_settings.TukeyOverlap, m_settings.TukeyP),
                        new PunchoutTukeyVariant(WindowFunction.Tukey2,  2, 0.25, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey2A, 2, 0.25, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey2B, 2, 0.25, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey2X, 2, m_settings.TukeyOverlap, m_settings.TukeyP),
                        new PunchoutTukeyVariant(WindowFunction.Tukey,   1, 0.0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey1A, 1, 0.0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey1B, 1, 0.0, 0.03),
                        new PunchoutTukeyVariant(WindowFunction.Tukey1X, 1, m_settings.TukeyOverlap, m_settings.TukeyP),
                    };

                    foreach (var tukey in tukeys)
                    {
                        if (tukey.parts == 0 || (m_settings.WindowFunctions & tukey.type) == 0) continue;
                        if (tukey.parts == 1)
                        {
                            calculate_window(window, (w, wsz) =>
                            {
                                lpc.window_tukey(w, wsz, tukey.p);
                            }, tukey.type);
                            continue;
                        }
                        double overlap = tukey.overlap;
                        double overlap_units = overlap / (1.0 - overlap);
                        for (int m = 0; m < tukey.parts; m++)
                            calculate_window(window, (w, wsz) =>
                            {
                                lpc.window_punchout_tukey(w, wsz, tukey.p, tukey.p,
                                    m / (tukey.parts + overlap_units),
                                    (m + 1 + overlap_units) / (tukey.parts + overlap_units));
                            }, tukey.type);
                    }

                    if (_windowcount == 0)
                        throw new Exception("invalid windowfunction");
                    int nSeg = 0;
                    int sz = _windowsize;
                    float* window_segment = window;
                    do
        			{
                        fixed (LpcWindowSection* sections = &windowSections[nSeg, 0, 0])
                            LpcWindowSection.Detect(_windowcount, window_segment, FlakeConstants.MAX_BLOCKSIZE * 2, sz, Settings.PCM.BitsPerSample, sections);
                        if ((sz & 1) != 0)
                            break;
                        window_segment += sz;
                        nSeg++;
                        sz >>= 1;
                    } while (sz >= 32);
#if NONONO
                    using (TextWriter tx = File.CreateText(@"H:\ubuntu\flac\w.txt"))
                    {
#if !NONONO
                        int totaltotal = 0;
                        for (int i = 0; i < _windowcount; i++)
                        {
                            int total = 0;
                            for (int sec = 0; sec < lpc.MAX_LPC_SECTIONS; sec++)
                                if (windowSections[0, i, sec].m_type != LpcWindowSection.SectionType.Zero || windowSections[0, i, sec].m_start != windowSections[0, i, sec].m_end)
                                {
                                    tx.WriteLine("{0}\t{1}\t{2}\t{3}", windowSections[0, i, sec].m_start, windowSections[0, i, sec].m_end, windowSections[0, i, sec].m_type, windowSections[0, i, sec].m_id);
                                    if (windowSections[0, i, sec].m_type != LpcWindowSection.SectionType.One)
                                        total += windowSections[0, i, sec].m_end - windowSections[0, i, sec].m_start;
                                }
                            totaltotal += total;
                            tx.WriteLine("{0} total window data", total);
                        }
                        tx.WriteLine("{0} grand total window data", totaltotal);
#endif
                        for (int x = 0; x < frame.blocksize; x++)       
                        {
                            tx.Write("{0}", x);
                            for (int i = 0; i < _windowcount; i++)
                                tx.Write("\t{0}", window[i * FlakeConstants.MAX_BLOCKSIZE * 2 + x]);
                            tx.WriteLine();
                        }
                    }
#endif
                }

				if (channels != 2 || frame.blocksize <= 32 || m_settings.StereoMethod == StereoMethod.Independent)
				{
					frame.window_buffer = window;
                    frame.nSeg = 0;
					frame.current.residual = r + channels * FlakeConstants.MAX_BLOCKSIZE;
					frame.ch_mode = channels != 2 ? ChannelMode.NotStereo : ChannelMode.LeftRight;
					for (int ch = 0; ch < channels; ch++)
						frame.subframes[ch].Init(s + ch * FlakeConstants.MAX_BLOCKSIZE, r + ch * FlakeConstants.MAX_BLOCKSIZE,
                            Settings.PCM.BitsPerSample, get_wasted_bits(s + ch * FlakeConstants.MAX_BLOCKSIZE, frame.blocksize));

					for (int ch = 0; ch < channels; ch++)
						encode_residual_pass2(frame, ch);
				}
				else
				{
					//channel_decorrelation(s, s + FlakeConstants.MAX_BLOCKSIZE, s + 2 * FlakeConstants.MAX_BLOCKSIZE, s + 3 * FlakeConstants.MAX_BLOCKSIZE, frame.blocksize);
					frame.window_buffer = window;
                    frame.nSeg = 0;
					frame.current.residual = r + 4 * FlakeConstants.MAX_BLOCKSIZE;
					for (int ch = 0; ch < 4; ch++)
						frame.subframes[ch].Init(s + ch * FlakeConstants.MAX_BLOCKSIZE, r + ch * FlakeConstants.MAX_BLOCKSIZE,
                            Settings.PCM.BitsPerSample + (ch == 3 ? 1 : 0), get_wasted_bits(s + ch * FlakeConstants.MAX_BLOCKSIZE, frame.blocksize));

					//for (int ch = 0; ch < 4; ch++)
					//    for (int iWindow = 0; iWindow < _windowcount; iWindow++)
					//        frame.subframes[ch].lpc_ctx[iWindow].GetReflection(32, frame.subframes[ch].samples, frame.blocksize, frame.window_buffer + iWindow * FlakeConstants.MAX_BLOCKSIZE * 2);

					estimate_frame(frame, true);
					uint fs = measure_frame_size(frame, true);

					if (0 != eparams.variable_block_size)
					{
						FlacFrame frame2 = new FlacFrame(channels * 2);
						FlacFrame frame3 = new FlacFrame(channels * 2);
						int tumbler = 1;
						while ((frame.blocksize & 1) == 0 && frame.blocksize >= 1024)
						{
							frame2.InitSize(frame.blocksize / 2, true);
							frame2.window_buffer = frame.window_buffer + frame.blocksize;
                            frame2.nSeg = frame.nSeg + 1;
							frame2.current.residual = r + tumbler * 5 * FlakeConstants.MAX_BLOCKSIZE;
							for (int ch = 0; ch < 4; ch++)
								frame2.subframes[ch].Init(frame.subframes[ch].samples, frame2.current.residual + (ch + 1) * frame2.blocksize,
									frame.subframes[ch].obits + frame.subframes[ch].wbits, frame.subframes[ch].wbits);
							estimate_frame(frame2, true);
                            //measure_frame_size(frame2, true);
                            //frame2.ChooseSubframes();
                            //encode_estimated_frame(frame2);
                            //uint fs2 = measure_frame_size(frame2, false);
                            uint fs2 = measure_frame_size(frame2, true);
							uint fs3 = fs2;
							if (eparams.variable_block_size == 2 || eparams.variable_block_size == 4)
							{
								frame3.InitSize(frame2.blocksize, true);
								frame3.window_buffer = frame2.window_buffer;
                                frame3.nSeg = frame2.nSeg;
								frame3.current.residual = frame2.current.residual + 5 * frame2.blocksize;
								for (int ch = 0; ch < 4; ch++)
									frame3.subframes[ch].Init(frame2.subframes[ch].samples + frame2.blocksize, frame3.current.residual + (ch + 1) * frame3.blocksize,
										frame.subframes[ch].obits + frame.subframes[ch].wbits, frame.subframes[ch].wbits);
								estimate_frame(frame3, true);
								fs3 = measure_frame_size(frame3, true);
							}
							if (fs2 + fs3 > fs)
								break;
							FlacFrame tmp = frame;
							frame = frame2;
							frame2 = tmp;
							fs = fs2;
							if (eparams.variable_block_size <= 2)
								break;
							tumbler = 1 - tumbler;
						}
					}

					frame.ChooseSubframes();
					encode_estimated_frame(frame);
				}

				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);

				output_frame_header(frame, bitwriter);
				output_subframes(frame, bitwriter);
				output_frame_footer(bitwriter);

				if (bitwriter.Length >= max_frame_size)
					throw new Exception("buffer overflow");

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
                        AudioSamples.MemCpy(s + ch * FlakeConstants.MAX_BLOCKSIZE, r + ch * FlakeConstants.MAX_BLOCKSIZE, m_blockSize);
			}

			int fs, bs;
            //if (0 != eparams.variable_block_size && 0 == (m_blockSize & 7) && m_blockSize >= 128)
			//    fs = encode_frame_vbs();
			//else
			fs = encode_frame(out bs);

			if (seek_table != null && _IO.CanSeek)
			{
				for (int sp = 0; sp < seek_table.Length; sp++)
				{
					if (seek_table[sp].framesize != 0)
						continue;
					if (seek_table[sp].number > _position + bs)
						break;
					if (seek_table[sp].number >= _position)
					{
						seek_table[sp].number = _position;
						seek_table[sp].offset = _IO.Position - first_frame_offset;
						seek_table[sp].framesize = bs;
					}
				}
			}

			_position += bs;
			_IO.Write(frame_buffer, 0, fs);
			_totalSize += fs;

			if (verify != null)
				try
				{
					int decoded = verify.DecodeFrame(frame_buffer, 0, fs);
					if (decoded != fs || verify.Remaining != bs)
						throw new Exception(Properties.Resources.ExceptionValidationFailed);
					fixed (int* s = verifyBuffer, r = verify.Samples)
					{
						for (int ch = 0; ch < channels; ch++)
							if (AudioSamples.MemCmp(s + ch * FlakeConstants.MAX_BLOCKSIZE, r + ch * FlakeConstants.MAX_BLOCKSIZE, bs))
								throw new Exception(Properties.Resources.ExceptionValidationFailed);
					}
				}
				catch (Exception ex)
				{
					//if (channels == 2)
					//{
					//    var sw = new WAVWriter(string.Format("verify_{0}.wav", this.frame_count), new WAVWriterSettings(this.Settings.PCM));
					//    sw.FinalSampleCount = this.frame.blocksize;
					//    var ab = new AudioBuffer(this.Settings.PCM, this.frame.blocksize);
					//    ab.Prepare(this.frame.blocksize);
					//    fixed (int* abs = ab.Samples, s = verifyBuffer)
					//        AudioSamples.Interlace(abs, s, s + FlakeConstants.MAX_BLOCKSIZE, this.frame.blocksize);
					//    sw.Write(ab);
					//    sw.Close();
					//} else
					throw ex;
				}

            if (bs < m_blockSize)
			{
				for (int ch = 0; ch < (channels == 2 ? 4 : channels); ch++)
                    Buffer.BlockCopy(samplesBuffer, (bs + ch * FlakeConstants.MAX_BLOCKSIZE) * sizeof(int), samplesBuffer, ch * FlakeConstants.MAX_BLOCKSIZE * sizeof(int), (m_blockSize - bs) * sizeof(int));
				//fixed (int* s = samplesBuffer)
				//    for (int ch = 0; ch < channels; ch++)
                //        AudioSamples.MemCpy(s + ch * FlakeConstants.MAX_BLOCKSIZE, s + bs + ch * FlakeConstants.MAX_BLOCKSIZE, m_blockSize - bs);
			}

			samplesInBuffer -= bs;

			return bs;
		}

		public void Write(AudioBuffer buff)
		{
			if (!inited)
			{
				if (_IO == null)
					_IO = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read, 0x20000);
                inited = true;
                int header_size = flake_encode_init();
				_IO.Write(header, 0, header_size);
				if (_IO.CanSeek)
					first_frame_offset = _IO.Position;
			}

			buff.Prepare(this);

			int pos = 0;
			while (pos < buff.Length)
			{
                int block = Math.Min(buff.Length - pos, m_blockSize - samplesInBuffer);

				copy_samples(buff.Samples, pos, block);

				pos += block;

                while (samplesInBuffer >= m_blockSize)
					output_frame();
			}

			if (md5 != null)
				md5.TransformBlock(buff.Bytes, 0, buff.ByteLength, null, 0);
		}

		public string Path { get { return _path; } }

        public static string Vendor
        {
            get
            {
                var version = typeof(AudioEncoder).Assembly.GetName().Version;
                return vendor_string ?? "CUETools " + version.Major + "." + version.Minor + "." + version.Build;
            }
            set
            {
                vendor_string = value;
            }
        }

        static string vendor_string = null;

		int select_blocksize(int samplerate, int time_ms)
		{
			int blocksize = FlakeConstants.flac_blocksizes[1];
			int target = (samplerate * time_ms) / 1000;
			if (eparams.variable_block_size > 0)
			{
				blocksize = 1024;
				while (target >= blocksize)
					blocksize <<= 1;
				return blocksize >> 1;
			}

            for (int i = 8; i < FlakeConstants.flac_blocksizes.Length - 1; i++)
				if (target >= FlakeConstants.flac_blocksizes[i] && FlakeConstants.flac_blocksizes[i] > blocksize)
				{
					blocksize = FlakeConstants.flac_blocksizes[i];
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
                bitwriter.writebits(16, m_blockSize);

            bitwriter.writebits(16, m_blockSize);
			bitwriter.writebits(24, 0);
			bitwriter.writebits(24, max_frame_size);
            bitwriter.writebits(20, Settings.PCM.SampleRate);
			bitwriter.writebits(3, channels - 1);
            bitwriter.writebits(5, Settings.PCM.BitsPerSample - 1);

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
		int write_vorbis_comment(byte[] comment, int pos, int len, int last)
		{
			BitWriter bitwriter = new BitWriter(comment, pos, len);
			Encoding enc = new UTF8Encoding();
            byte[] str = enc.GetBytes(Vendor);

			// metadata header
			bitwriter.writebits(1, last);
			bitwriter.writebits(7, (int)MetadataType.VorbisComment);
            int tagsLen = 0;
            if (m_settings.Tags != null)
                foreach (var t in m_settings.Tags)
                    tagsLen += 4 + enc.GetByteCount(t);
            bitwriter.writebits(24, 8 + str.Length + tagsLen);
            for (int i = 0; i < 4; i++)
                bitwriter.writebits(8, (str.Length >> (i * 8)) & 0xff);
            bitwriter.write(str);
            int nTags = m_settings.Tags != null ? m_settings.Tags.Length : 0;
            for (int i = 0; i < 4; i++)
                bitwriter.writebits(8, (nTags >> (i * 8)) & 0xff);
            if (m_settings.Tags != null)
                foreach (var tag in m_settings.Tags)
                {
                    str = enc.GetBytes(tag);
                    for (int i = 0; i < 4; i++)
                        bitwriter.writebits(8, (str.Length >> (i * 8)) & 0xff);
                    bitwriter.write(str);
                }
			bitwriter.flush();
            return bitwriter.Length;
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
				bitwriter.writebits(FlakeConstants.FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN, (ulong)seek_table[i].number);
				bitwriter.writebits(FlakeConstants.FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN, (ulong)seek_table[i].offset);
				bitwriter.writebits(FlakeConstants.FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN, seek_table[i].framesize);
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

            bitwriter.flush();
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

			// vorbis comments
            if (m_settings.Padding == 0) last = 1;
            header_size += write_vorbis_comment(header, header_size, header.Length - header_size, last);

			// padding
			if (m_settings.Padding > 0)
			{
				last = 1;
				header_size += write_padding(header, header_size, last, m_settings.Padding);
			}

			return header_size;
		}

		int flake_encode_init()
		{
			int i, header_len;

			//if(flake_validate_params(s) < 0)

			ch_code = channels - 1;

			// find samplerate in table
			for (i = 1; i < 12; i++)
			{
                if (Settings.PCM.SampleRate == FlakeConstants.flac_samplerates[i])
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
                if (Settings.PCM.BitsPerSample == FlakeConstants.flac_bitdepths[i])
				{
					bps_code = i;
					break;
				}
			}
			if (i == 8)
				throw new Exception("non-standard bps");

            m_blockSize = m_settings.BlockSize != 0 ? m_settings.BlockSize :
                select_blocksize(Settings.PCM.SampleRate, eparams.block_time_ms);

			// set maximum encoded frame size (if larger, re-encodes in verbatim mode)
			if (channels == 2)
                max_frame_size = 16 + ((m_blockSize * (Settings.PCM.BitsPerSample + Settings.PCM.BitsPerSample + 1) + 7) >> 3);
			else
                max_frame_size = 16 + ((m_blockSize * channels * Settings.PCM.BitsPerSample + 7) >> 3);

			if (_IO.CanSeek && eparams.do_seektable && sample_count > 0)
			{
                int seek_points_distance = Settings.PCM.SampleRate * 10;
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
            int tagsLen = 0;
			Encoding enc = new UTF8Encoding();
            if (m_settings.Tags != null)
                foreach (var t in m_settings.Tags)
                    tagsLen += 4 + enc.GetByteCount(t);
            header = new byte[m_settings.Padding + 1024 + (seek_table == null ? 0 : seek_table.Length * 18) + tagsLen];
			header_len = write_headers();

			// initialize CRC & MD5
			if (_IO.CanSeek && m_settings.DoMD5)
				md5 = new MD5CryptoServiceProvider();

			if (m_settings.DoVerify)
			{
                verify = new AudioDecoder(Settings.PCM);
				verifyBuffer = new int[FlakeConstants.MAX_BLOCKSIZE * channels];
			}

			frame_buffer = new byte[max_frame_size];

			return header_len;
		}
	}

	struct FlakeEncodeParams
	{
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

		// block time in milliseconds
		// set by the user prior to calling flake_encode_init
		// used to calculate block_size based on sample rate
		// can also be changed by user before encoding a frame
		public int block_time_ms;

		// whether to use variable block sizes
		// set by user prior to calling flake_encode_init
		// 0 = fixed block size
		// 1 = variable block size
		public int variable_block_size;

		public bool do_seektable;

        public int development_mode;

        public int flake_set_defaults(EncoderSettings settings)
        {
            order_method = OrderMethod.Akaike;
            block_time_ms = 105;
            variable_block_size = 0;
            do_seektable = true;
            development_mode = -1;

            if (settings.GetEncoderModeIndex() == 11)
                variable_block_size = 4;

            return 0;
        }
	}
}
