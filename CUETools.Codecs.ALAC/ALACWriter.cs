/**
 * CUETools.Codecs.ALAC: pure managed ALAC audio encoder
 * Copyright (c) 2009 Grigory Chudov
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

#define INTEROP

using System;
using System.ComponentModel;
using System.Text;
using System.IO;
using System.Collections.Generic;
#if INTEROP
using System.Runtime.InteropServices;
#endif
using CUETools.Codecs;

namespace CUETools.Codecs.ALAC
{
	public class ALACWriterSettings: AudioEncoderSettings
	{
        public ALACWriterSettings()
            : base("0 1 2 3 4 5 6 7 8 9 10", "5")
        {
        }

        public void Validate()
        {
            if (EncoderModeIndex < 0
                || Padding < 0
                || (BlockSize != 0 && (BlockSize < 256 || BlockSize >= Int32.MaxValue)))
                throw new Exception("unsupported encoder settings");
        }

		[DefaultValue(false)]
		[DisplayName("Verify")]
		[Description("Decode each frame and compare with original")]
		public bool DoVerify { get; set; }
	}

	[AudioEncoderClass("cuetools", "m4a", true, 1, typeof(ALACWriterSettings))]
	public class ALACWriter : IAudioDest
	{
		Stream _IO = null;
		bool _pathGiven = false;
		string _path;
		long _position;

		const int max_header_len = 709 + 38; // minimum 38 bytes in padding

		// total stream samples
		// if < 0, stream length is unknown
		int sample_count = -1;

		ALACEncodeParams eparams;

		// maximum frame size in bytes
		// this can be used to allocate memory for output
		int max_frame_size;

		int initial_history = 10, history_mult = 40, k_modifier = 14;

		byte[] frame_buffer = null;

		int frame_count = 0;

		int first_frame_offset = 0;

#if INTEROP
		TimeSpan _userProcessorTime;
#endif

		uint[] _sample_byte_size;
		int[] samplesBuffer;
		int[] verifyBuffer;
		int[] residualBuffer;
		float[] windowBuffer;
        WindowFunction[] windowType;
        LpcWindowSection[,] windowSections;
        int samplesInBuffer = 0;

		int m_blockSize = 0;
		int _totalSize = 0;
		int _windowsize = 0, _windowcount = 0;

		ALACFrame frame;
		ALACReader verify;

		bool inited = false;

		List<int> chunk_pos;

        public ALACWriter(string path, Stream IO, ALACWriterSettings settings)
		{
            m_settings = settings;
            m_settings.Validate();

            if (Settings.PCM.BitsPerSample != 16)
				throw new Exception("Bits per sample must be 16.");
            if (Settings.PCM.ChannelCount != 2)
				throw new Exception("ChannelCount must be 2.");

			_path = path;
			_IO = IO;
			_pathGiven = _IO == null;
			if (_IO != null && !_IO.CanSeek)
				throw new NotSupportedException("stream doesn't support seeking");

            samplesBuffer = new int[Alac.MAX_BLOCKSIZE * (Settings.PCM.ChannelCount == 2 ? 5 : Settings.PCM.ChannelCount)];
            residualBuffer = new int[Alac.MAX_BLOCKSIZE * (Settings.PCM.ChannelCount == 2 ? 6 : Settings.PCM.ChannelCount + 1)];
			windowBuffer = new float[Alac.MAX_BLOCKSIZE * 2 * lpc.MAX_LPC_WINDOWS];
            windowType = new WindowFunction[lpc.MAX_LPC_WINDOWS];
            windowSections = new LpcWindowSection[lpc.MAX_LPC_WINDOWS, lpc.MAX_LPC_SECTIONS];

            eparams.set_defaults(m_settings.EncoderModeIndex);

            frame = new ALACFrame(Settings.PCM.ChannelCount == 2 ? 5 : Settings.PCM.ChannelCount);
			chunk_pos = new List<int>();
		}

        public ALACWriter(string path, ALACWriterSettings settings)
            : this(path, null, settings)
		{
		}

		public int TotalSize
		{
			get
			{
				return _totalSize;
			}
		}

		ALACWriterSettings m_settings;

		public AudioEncoderSettings Settings
		{
			get
			{
				return m_settings;
			}
		}

#if INTEROP
		[DllImport("kernel32.dll")]
		static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
		[DllImport("kernel32.dll")]
		static extern IntPtr GetCurrentThread();
#endif

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

				int mdat_len = (int)_IO.Position - first_frame_offset;
				int header_len = first_frame_offset;

				if (sample_count <= 0 && _position != 0) 
				{
					sample_count = (int)_position;
					header_len = max_header_len
						+ m_settings.Padding
						+ frame_count * 4 // stsz
						+ frame_count * 4 / eparams.chunk_size; // stco
					//if (header_len % 0x400 != 0)
					//    header_len += 0x400 - (header_len % 0x400);
				}

				if (!_creationTime.HasValue)
					_creationTime = DateTime.Now;

				if (header_len > first_frame_offset)
				{
					// if frame_count is high, need to rewrite 
					// the whole file to increase first_frame_offset

					//System.Diagnostics.Trace.WriteLine(String.Format("Rewriting whole file: {0}/{1} + {2}", header_len, first_frame_offset, mdat_len));

					// assert(_pathGiven);					
					string tmpPath = _path + ".tmp"; // TODO: make sure tmpPath is unique?
					FileStream IO2 = new FileStream(tmpPath, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
					byte[] header = write_headers(header_len, mdat_len);
					IO2.Write(header, 0, header_len);
					_IO.Position = first_frame_offset;
					int bufSize = Math.Min(mdat_len, 0x2000);
					byte[] buffer = new byte[bufSize];
					int n;
					do
					{
						n = _IO.Read(buffer, 0, buffer.Length);
						IO2.Write(buffer, 0, n);
					} while (n != 0);
					IO2.Close();
					_IO.Close();
					File.Delete(_path);
					File.Move(tmpPath, _path);
				}
				else
				{
					//System.Diagnostics.Trace.WriteLine(String.Format("{0}/{1}", header_len, first_frame_offset));
					byte[] header = write_headers(first_frame_offset, mdat_len);
					_IO.Position = 0;
					_IO.Write(header, 0, first_frame_offset);
					_IO.Close();
				}
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

		public WindowMethod WindowMethod
		{
			get { return eparams.window_method; }
			set { eparams.window_method = value; }
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
				if (value > 7)
					throw new Exception("invalid MaxHistoryModifier " + value.ToString());
				eparams.max_modifier = value;
				if (eparams.min_modifier > value)
					eparams.min_modifier = value;
			}
		}

		public int HistoryMult
		{
			get
			{
				return history_mult;
			}
			set
			{
				if (value < 1 || value > 255)
					throw new Exception("invalid history_mult");
				history_mult = value;
			}
		}

		public int InitialHistory
		{
			get
			{
				return initial_history;
			}
			set
			{
				if (value < 1 || value > 255)
					throw new Exception("invalid initial_history");
				initial_history = value;
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

		public int AdaptivePasses
		{
			get
			{
				return eparams.adaptive_passes;
			}
			set
			{
				if (value >= lpc.MAX_LPC_PRECISIONS || value < 0)
					throw new Exception("invalid adaptive_passes " + value.ToString());
				eparams.adaptive_passes = value;
			}			
		}

		public TimeSpan UserProcessorTime
		{
			get
			{
#if INTEROP
				return _userProcessorTime;
#else
				return TimeSpan(0);
#endif
			}
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
				if (Settings.PCM.ChannelCount == 2)
					AudioSamples.Deinterlace(fsamples + samplesInBuffer, fsamples + Alac.MAX_BLOCKSIZE + samplesInBuffer, src, block);
				else
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
					{
						int* psamples = fsamples + ch * Alac.MAX_BLOCKSIZE + samplesInBuffer;
                        int channels = Settings.PCM.ChannelCount;
						for (int i = 0; i < block; i++)
							psamples[i] = src[i * channels + ch];
					}
			}
			samplesInBuffer += block;
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

		private static int sign_only(int val)
		{
			return (val >> 31) + ((val - 1) >> 31) + 1;
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

#if aaa
            // contains errors: if (resval * orig_sign <= 0) continue; is not the same as sign(resval) * sign(orig), because 0x80000000 * 0xffffffff < 0!
            // probably should be if (resval == 0 || sign_only(resval) != orig_sign)
            // sign_only(d0 ^ resval); is the same as sign_only(d0) * sign_only(resval);
            // but sign_only(d0) * orig_sign is not the same (when d0 == 0x80000000)
            switch (order)
            {
                case 8:
                    {
                        const int constOrder = 8;
                        int c0 = coefs[0], c1 = coefs[1], c2 = coefs[2], c3 = coefs[3], c4 = coefs[4], c5 = coefs[5], c6 = coefs[6], c7 = coefs[7];
                        int denhalf = 1 << (shift - 1);
                        for (int i = constOrder + 1; i < n; i++)
                        {
                            int sample = *(smp++);
                            int d0 = smp[0] - sample, d1 = smp[1] - sample, d2 = smp[2] - sample, d3 = smp[3] - sample, d4 = smp[4] - sample, d5 = smp[5] - sample, d6 = smp[6] - sample, d7 = smp[7] - sample;
                            int sum = denhalf + d0 * c0 + d1 * c1 + d2 * c2 + d3 * c3 + d4 * c4 + d5 * c5 + d6 * c6 + d7 * c7;
                            int resval = extend_sign32(smp[constOrder] - sample - (int)(sum >> shift), bps);
                            res[i] = resval;

                            if (resval == 0) continue;
                            int orig_sign = sign_only(resval);

                            int sign = sign_only(d0 ^ resval);
                            c0 += sign;
                            resval -= (d0 * sign >> shift) * (0 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d1 ^ resval);
                            c1 += sign;
                            resval -= (d1 * sign >> shift) * (1 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d2 ^ resval);
                            c2 += sign;
                            resval -= (d2 * sign >> shift) * (2 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d3 ^ resval);
                            c3 += sign;
                            resval -= (d3 * sign >> shift) * (3 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d4 ^ resval);
                            c4 += sign;
                            resval -= (d4 * sign >> shift) * (4 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d5 ^ resval);
                            c5 += sign;
                            resval -= (d5 * sign >> shift) * (5 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d6 ^ resval);
                            c6 += sign;
                            resval -= (d6 * sign >> shift) * (6 + 1);
                            if (resval * orig_sign <= 0) continue;

                            sign = sign_only(d7 ^ resval);
                            c7 += sign;
                            resval -= (d7 * sign >> shift) * (7 + 1);
                        }

                        coefs[0] = c0;
                        coefs[1] = c1;
                        coefs[2] = c2;
                        coefs[3] = c3;
                        coefs[4] = c4;
                        coefs[5] = c5;
                        coefs[6] = c6;
                        coefs[7] = c7;
                        res[n] = 1; // Stop byte to help alac_entropy_coder;
                        return;
                    }
            }
#endif

            /* general case */
			for (int i = order + 1; i < n; i++)
			{
				int sample = *(smp++);
				int/*long*/ sum = (1 << (shift - 1));
				for (int j = 0; j < order; j++)
					sum += (smp[j] - sample) * coefs[j];
				int resval = extend_sign32(smp[order] - sample - (int)(sum >> shift), bps);
				res[i] = resval;

				if (resval > 0)
				{
					for (int j = 0; j < order && resval > 0; j++)
					{
						int val = smp[j] - sample;
						int sign = sign_only(val);
						coefs[j] += sign;
						resval -= (val * sign >> shift) * (j + 1);
					}
				}
				else
				{
					for (int j = 0; j < order && resval < 0; j++)
					{
						int val = smp[j] - sample;
						int sign = -sign_only(val);
						coefs[j] += sign;
						resval -= (val * sign >> shift) * (j + 1);
					}
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
				int newsize = alac_entropy_estimate(res, n, bps, i);
				if (size > newsize)
				{
					size = newsize;
					modifier = i;
				}
			}
			return size;
		}

        //unsafe int alac_entropy_coder(int* res, int n, int bps, int modifier)
        //{
        //    int history = initial_history;
        //    int sign_modifier = 0;
        //    int rice_historymult = modifier * history_mult / 4;
        //    int size = 0;
        //    int* fin = res + n;

        //    while (res < fin)
        //    {
        //        int k = BitReader.log2i((history >> 9) + 3);
        //        int x = *(res++);
        //        x = (x << 1) ^ (x >> 31);

        //        size += encode_scalar(x - sign_modifier, Math.Min(k, k_modifier), bps);

        //        history += x * rice_historymult - ((history * rice_historymult) >> 9);

        //        sign_modifier = 0;
        //        if (x > 0xFFFF)
        //            history = 0xFFFF;

        //        if (history < 128 && res < fin)
        //        {
        //            k = 7 - BitReader.log2i(history) + ((history + 16) >> 6);
        //            int* res1 = res;
        //            while (*res == 0) // we have a stop byte, so need not check if res < fin
        //                res++;
        //            int block_size = (int)(res - res1);
        //            size += encode_scalar(block_size, Math.Min(k, k_modifier), 16);
        //            //sign_modifier = (block_size <= 0xFFFF) ? 1 : 0; //never happens
        //            sign_modifier = 1;
        //            history = 0;
        //        }
        //    }
        //    return size;
        //}

		/// <summary>
		/// Crude estimation of entropy length
		/// </summary>
		/// <param name="res"></param>
		/// <param name="n"></param>
		/// <param name="bps"></param>
		/// <param name="modifier"></param>
		/// <returns></returns>
		unsafe int alac_entropy_estimate(int* res, int n, int bps, int modifier)
		{
			int history = initial_history;
			int rice_historymult = modifier * history_mult / 4;
			int size = 0;
			int* fin = res + n;

			while (res < fin)
			{
				int x = *(res++);
				x = (x << 1) ^ (x >> 31);
				int k = BitReader.log2i((history >> 9) + 3);
				k = k > k_modifier ? k_modifier : k;
				size += (x >> k) > 8 ? 9 + bps : (x >> k) + k + 1;
				history += x * rice_historymult - ((history * rice_historymult) >> 9);
			}
			return size;
		}

		unsafe void alac_entropy_coder(BitWriter bitwriter, int* res, int n, int bps, int modifier)
		{
			int history = initial_history;
			int sign_modifier = 0;
			int rice_historymult = modifier * history_mult / 4;
			int* fin = res + n;

			while (res < fin)
			{
				int k = BitReader.log2i((history >> 9) + 3);
				int x = *(res++);
				x = (x << 1) ^ (x >> 31);

				encode_scalar(bitwriter, x - sign_modifier, k, bps);

				history += x * rice_historymult - ((history * rice_historymult) >> 9);

				sign_modifier = 0;
				if (x > 0xFFFF)
					history = 0xFFFF;

				if (history < 128 && res < fin)
				{
					k = 7 - BitReader.log2i(history) + ((history + 16) >> 6);
					int* res1 = res;
					while (*res == 0) // we have a stop byte, so need not check if res < fin
						res++;
					int block_size = (int)(res - res1);
					encode_scalar(bitwriter, block_size, k, 16);
					sign_modifier = (block_size <= 0xFFFF) ? 1 : 0;
					history = 0;
				}
			}
		}

		unsafe void encode_residual_lpc_sub(ALACFrame frame, float* lpcs, int iWindow, int order, int ch)
		{
			// check if we already calculated with this order, window and precision
			if ((frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[eparams.adaptive_passes] & (1U << (order - 1))) == 0)
			{
				frame.subframes[ch].lpc_ctx[iWindow].done_lpcs[eparams.adaptive_passes] |= (1U << (order - 1));

				uint cbits = 15U;

				frame.current.order = order;
				frame.current.window = iWindow;

                int bps = Settings.PCM.BitsPerSample + Settings.PCM.ChannelCount - 1;

				int* coefs = stackalloc int[lpc.MAX_LPC_ORDER];

				//if (frame.subframes[ch].best.order == order && frame.subframes[ch].best.window == iWindow)
				//{
				//    frame.current.shift = frame.subframes[ch].best.shift;
				//    for (int i = 0; i < frame.current.order; i++)
				//        frame.current.coefs[i] = frame.subframes[ch].best.coefs_adapted[i];
				//}
				//else
				{
					lpc.quantize_lpc_coefs(lpcs + (frame.current.order - 1) * lpc.MAX_LPC_ORDER,
						frame.current.order, cbits, coefs, out frame.current.shift, 15, 1);

					if (frame.current.shift < 0 || frame.current.shift > 15)
						throw new Exception("negative shift");

					for (int i = 0; i < frame.current.order; i++)
						frame.current.coefs[i] = coefs[i];
				}

				for (int i = 0; i < frame.current.order; i++)
					coefs[i] = frame.current.coefs[frame.current.order - 1 - i];
				for (int i = frame.current.order; i < lpc.MAX_LPC_ORDER;  i++)
					coefs[i] = 0;

				alac_encode_residual(frame.current.residual, frame.subframes[ch].samples, frame.blocksize,
					frame.current.order, coefs, frame.current.shift, bps);

				for (int i = 0; i < frame.current.order; i++)
					frame.current.coefs_adapted[i] = coefs[frame.current.order - 1 - i];

				for (int adaptive_pass = 0; adaptive_pass < eparams.adaptive_passes; adaptive_pass++)
				{
					for (int i = 0; i < frame.current.order; i++)
						frame.current.coefs[i] = frame.current.coefs_adapted[i];

					alac_encode_residual(frame.current.residual, frame.subframes[ch].samples, frame.blocksize,
						frame.current.order, coefs, frame.current.shift, bps);

					for (int i = 0; i < frame.current.order; i++)
						frame.current.coefs_adapted[i] = coefs[frame.current.order - 1 - i];
				}

				frame.current.size = (uint)(alac_entropy_estimate(frame.current.residual, frame.blocksize, bps, eparams.max_modifier) + 16 + 16 * order);
				
				frame.ChooseBestSubframe(ch);
			}
		}

		unsafe void encode_residual(ALACFrame frame, int ch, int pass, int best_windows)
		{
			int* smp = frame.subframes[ch].samples;
			int i, n = frame.blocksize;
            int bps = Settings.PCM.BitsPerSample + Settings.PCM.ChannelCount - 1;

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

			float* lpcs = stackalloc float[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_ORDER];
			int min_order = eparams.min_prediction_order;
			int max_order = eparams.max_prediction_order;

			for (int iWindow = 0; iWindow < _windowcount; iWindow++)
			{
				if (0 == (best_windows & (1 << iWindow)))
					continue;

				LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[iWindow];

                fixed (LpcWindowSection* sections = &windowSections[iWindow, 0])
                    lpc_ctx.GetReflection(
                        frame.subframes[ch].sf, max_order, frame.blocksize, smp, 
                        frame.window_buffer + iWindow * Alac.MAX_BLOCKSIZE * 2, sections);
				lpc_ctx.ComputeLPC(lpcs);
				lpc_ctx.SortOrdersAkaike(frame.blocksize, eparams.estimation_depth, min_order, max_order, 5.0, 1.0/18);
				for (i = 0; i < eparams.estimation_depth && i < max_order; i++)
					encode_residual_lpc_sub(frame, lpcs, iWindow, lpc_ctx.best_orders[i], ch);
			}
		}

		unsafe void output_frame_header(ALACFrame frame, BitWriter bitwriter)
		{
            bitwriter.writebits(3, Settings.PCM.ChannelCount - 1);
			bitwriter.writebits(16, 0);
			bitwriter.writebits(1, frame.blocksize != m_blockSize ? 1 : 0); // sample count is in the header
			bitwriter.writebits(2, 0); // wasted bytes
			bitwriter.writebits(1, frame.type == FrameType.Verbatim ? 1 : 0); // is verbatim
			if (frame.blocksize != m_blockSize)
				bitwriter.writebits(32, frame.blocksize);
			if (frame.type != FrameType.Verbatim)
			{
				bitwriter.writebits(8, frame.interlacing_shift);
				bitwriter.writebits(8, frame.interlacing_leftweight);
                for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
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

		unsafe void encode_residual_pass1(ALACFrame frame, int ch, int best_windows)
		{
			int max_prediction_order = eparams.max_prediction_order;
			int estimation_depth = eparams.estimation_depth;
			int min_modifier = eparams.min_modifier;
			int adaptive_passes = eparams.adaptive_passes;
			eparams.max_prediction_order = Math.Min(8,eparams.max_prediction_order);
			eparams.estimation_depth = 1;
			eparams.min_modifier = eparams.max_modifier;
			eparams.adaptive_passes = 0;
			encode_residual(frame, ch, 1, best_windows);
			eparams.max_prediction_order = max_prediction_order;
			eparams.estimation_depth = estimation_depth;
			eparams.min_modifier = min_modifier;
			eparams.adaptive_passes = adaptive_passes;
		}

		unsafe void encode_residual_pass2(ALACFrame frame, int ch)
		{
			encode_residual(frame, ch, 2, estimate_best_window(frame, ch));
		}

        unsafe int estimate_best_windows_akaike(ALACFrame frame, int ch, int count, bool onePerType)
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

            int order = Math.Min(4, eparams.max_prediction_order);
            float* err = stackalloc float[lpc.MAX_LPC_ORDER];
            for (int i = 0; i < _windowcount; i++)
            {
                LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[i];
                fixed (LpcWindowSection* sections = &windowSections[i, 0])
                    lpc_ctx.GetReflection(
                        frame.subframes[ch].sf, order, frame.blocksize,
                        frame.subframes[ch].samples,
                        frame.window_buffer + i * Alac.MAX_BLOCKSIZE * 2, sections);
                lpc_ctx.SortOrdersAkaike(frame.blocksize, 1, 1, order, 4.5, 0.0);
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

		unsafe int estimate_best_window(ALACFrame frame, int ch)
		{
			if (_windowcount == 1)
				return 1;
			switch (eparams.window_method)
			{
				case WindowMethod.Estimate:
                    return estimate_best_windows_akaike(frame, ch, 1, false);
                case WindowMethod.EstimateN:
                    return estimate_best_windows_akaike(frame, ch, 1, true);
                case WindowMethod.EvaluateN:
                    encode_residual_pass1(frame, ch, estimate_best_windows_akaike(frame, ch, 1, true));
                    return 1 << frame.subframes[ch].best.window;
                case WindowMethod.Evaluate:
					encode_residual_pass1(frame, ch, -1);
					return 1 << frame.subframes[ch].best.window;
				case WindowMethod.Search:
					return -1;
			}
			return -1;
		}

		unsafe void estimate_frame(ALACFrame frame, bool do_midside)
		{
            int subframes = do_midside ? 5 : Settings.PCM.ChannelCount;

			switch (eparams.stereo_method)
			{
				case StereoMethod.Estimate:
					for (int ch = 0; ch < subframes; ch++)
					{
                        int iWindow = 0;
                        LpcContext lpc_ctx = frame.subframes[ch].lpc_ctx[iWindow];
                        int stereo_order = Math.Min(8, eparams.max_prediction_order);
						double alpha = 1.5; // 4.5 + eparams.max_prediction_order / 10.0;
                        fixed (LpcWindowSection* sections = &windowSections[iWindow, 0])
                            lpc_ctx.GetReflection(
                                frame.subframes[ch].sf, stereo_order, frame.blocksize,
                                frame.subframes[ch].samples,
                                frame.window_buffer + iWindow * Alac.MAX_BLOCKSIZE * 2, sections);
						lpc_ctx.SortOrdersAkaike(frame.blocksize, 1, 1, stereo_order, alpha, 0);
						frame.subframes[ch].best.size = (uint)Math.Max(0, lpc_ctx.Akaike(frame.blocksize, lpc_ctx.best_orders[0], alpha, 0));
					}
					break;
				case StereoMethod.Evaluate:
					for (int ch = 0; ch < subframes; ch++)
						encode_residual_pass1(frame, ch, 1);
					break;
				case StereoMethod.Search:
					for (int ch = 0; ch < subframes; ch++)
						encode_residual_pass2(frame, ch);
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

            for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
				total += frame.subframes[ch].best.size;

			return total;
		}

		unsafe void encode_estimated_frame(ALACFrame frame)
		{
			switch (eparams.stereo_method)
			{
				case StereoMethod.Estimate:
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
					{
						frame.subframes[ch].best.size = AudioSamples.UINT32_MAX;
						encode_residual_pass2(frame, ch);
					}
					break;
				case StereoMethod.Evaluate:
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						encode_residual_pass2(frame, ch);
					break;
				case StereoMethod.Search:
					break;
			}
		}

		unsafe delegate void window_function(float* window, int size);

		unsafe void calculate_window(float * window, window_function func, WindowFunction flag)
		{
			if ((eparams.window_function & flag) == 0 || _windowcount == lpc.MAX_LPC_WINDOWS)
				return;
			int sz = _windowsize;
			float* pos = window + _windowcount * Alac.MAX_BLOCKSIZE * 2;
			do
			{
                windowSections[_windowcount, 0].setData(0, sz);
                for (int j = 1; j < lpc.MAX_LPC_SECTIONS; j++)
                    windowSections[_windowcount, j].setZero(sz, sz);
                func(pos, sz);
                break;
				if ((sz & 1) != 0)
					break;
				pos += sz;
				sz >>= 1;
			} while (sz >= 32);
            windowType[_windowcount] = flag;
            _windowcount++;
		}

		unsafe int encode_frame(ref int size)
		{
			fixed (int* s = samplesBuffer, r = residualBuffer)
			fixed (float * window = windowBuffer)
			{
				frame.InitSize(size);

				if (frame.blocksize != _windowsize && frame.blocksize > 4)
				{
					_windowsize = frame.blocksize;
					_windowcount = 0;
					calculate_window(window, lpc.window_welch, WindowFunction.Welch);
					calculate_window(window, lpc.window_bartlett, WindowFunction.Bartlett);
					calculate_window(window, lpc.window_tukey, WindowFunction.Tukey);
					calculate_window(window, lpc.window_hann, WindowFunction.Hann);
					calculate_window(window, lpc.window_flattop, WindowFunction.Flattop);
                    int tukey_parts = 2;
                    double overlap = -0.3;
                    double overlap_units = overlap / (1.0 - overlap);
                    for (int m = 0; m < tukey_parts; m++)
                        calculate_window(window, (w, wsz) =>
                        {
                            lpc.window_punchout_tukey(w, wsz, 0.1,
                                m / (tukey_parts + overlap_units),
                                (m + 1 + overlap_units) / (tukey_parts + overlap_units));
                        }, WindowFunction.PartialTukey);

                    tukey_parts = 3;
                    overlap = -0.1;
                    //overlap = 0.1;
                    overlap_units = overlap / (1.0 - overlap);
                    for (int m = 0; m < tukey_parts; m++)
                        calculate_window(window, (w, wsz) =>
                        {
                            lpc.window_punchout_tukey(w, wsz, 0.1,
                                m / (tukey_parts + overlap_units),
                                (m + 1 + overlap_units) / (tukey_parts + overlap_units));
                        }, WindowFunction.PunchoutTukey);
                    if (_windowcount == 0)
						throw new Exception("invalid windowfunction");
                    fixed (LpcWindowSection* sections = &windowSections[0, 0])
                        LpcWindowSection.Detect(_windowcount, window, Alac.MAX_BLOCKSIZE * 2, _windowsize, Settings.PCM.BitsPerSample, sections);
                }
				frame.window_buffer = window;

                int bps = Settings.PCM.BitsPerSample + Settings.PCM.ChannelCount - 1;
                if (Settings.PCM.ChannelCount != 2 || frame.blocksize <= 32 || eparams.stereo_method == StereoMethod.Independent)
				{
                    frame.current.residual = r + Settings.PCM.ChannelCount * Alac.MAX_BLOCKSIZE;

                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						frame.subframes[ch].Init(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE);

                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						encode_residual_pass2(frame, ch);
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

                for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
				{
					if (eparams.min_modifier == eparams.max_modifier)
						frame.subframes[ch].best.ricemodifier = eparams.max_modifier;
					else
						/*frame.subframes[ch].best.size = 16 + 16 * order + */
						alac_entropy_coder(frame.subframes[ch].best.residual, frame.blocksize, bps, out frame.subframes[ch].best.ricemodifier);
				}

				uint fs = measure_frame_size(frame, false);
                frame.type = ((int)fs > frame.blocksize * Settings.PCM.ChannelCount * bps) ? FrameType.Verbatim : FrameType.Compressed;
				BitWriter bitwriter = new BitWriter(frame_buffer, 0, max_frame_size);
				output_frame_header(frame, bitwriter);
				if (frame.type == FrameType.Verbatim)
				{
                    int obps = Settings.PCM.BitsPerSample;
					for (int i = 0; i < frame.blocksize; i++)
                        for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
							bitwriter.writebits_signed(obps, frame.subframes[ch].samples[i]);
				}
				else if (frame.type == FrameType.Compressed)
				{
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						alac_entropy_coder(bitwriter, frame.subframes[ch].best.residual, frame.blocksize, 
							bps, frame.subframes[ch].best.ricemodifier);
				}
				output_frame_footer(bitwriter);

				if (_sample_byte_size.Length <= frame_count)
				{
					uint[] tmp = new uint[frame_count * 2];
					Array.Copy(_sample_byte_size, tmp, _sample_byte_size.Length);
					_sample_byte_size = tmp;
				}
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
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						AudioSamples.MemCpy(s + ch * Alac.MAX_BLOCKSIZE, r + ch * Alac.MAX_BLOCKSIZE, blocksize);
			}

			//if (0 != eparams.variable_block_size && 0 == (m_blockSize & 7) && m_blockSize >= 128)
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
				if (decoded != fs || verify.Remaining != bs)
					throw new Exception("validation failed!");
                int[,] deinterlaced = new int[bs, Settings.PCM.ChannelCount];
				verify.deinterlace(deinterlaced, 0, bs);
				fixed (int* s = verifyBuffer, r = deinterlaced)
				{
                    int channels = Settings.PCM.ChannelCount;
					for (int i = 0; i < bs; i++)
                        for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
							if (r[i * channels + ch] != s[ch * Alac.MAX_BLOCKSIZE + i])
								throw new Exception("validation failed!");
				}
			}

			if (bs < blocksize)
			{
				fixed (int* s = samplesBuffer)
                    for (int ch = 0; ch < Settings.PCM.ChannelCount; ch++)
						AudioSamples.MemCpy(s + ch * Alac.MAX_BLOCKSIZE, s + bs + ch * Alac.MAX_BLOCKSIZE, blocksize - bs);
			}

			samplesInBuffer -= bs;

			return bs;
		}

		public void Write(AudioBuffer buff)
		{
			if (!inited)
			{
				if (!_pathGiven && sample_count <= 0)
					throw new NotSupportedException("input and output are both pipes");
				if (_IO == null)
					_IO = new FileStream(_path, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
				if (_IO != null && !_IO.CanSeek)
					throw new NotSupportedException("stream doesn't support seeking");
                encode_init();
				inited = true;
			}

			buff.Prepare(this);

			int pos = 0;
			int len = buff.Length;
			while (len > 0)
			{
				int block = Math.Min(len, m_blockSize - samplesInBuffer);

				copy_samples(buff.Samples, pos, block);

				len -= block;
				pos += block;

				while (samplesInBuffer >= m_blockSize)
					output_frame(m_blockSize);
			}
		}

		public string Path { get { return _path; } }

		private DateTime? _creationTime = null;

		public DateTime CreationTime
		{
			set
			{
				_creationTime = value;
			}
		}

		public static string Vendor
		{
			get
			{
                var version = typeof(ALACWriter).Assembly.GetName().Version;
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
			int target = (samplerate * time_ms) / 1000;
			int blocksize = 1024;
			while (target >= blocksize)
				blocksize <<= 1;
			return blocksize >> 1;
		}

		void write_chunk_mvhd(BitWriter bitwriter)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'v', 'h', 'd');
				bitwriter.writebits(32, 0);
				bitwriter.writebits(_creationTime.Value);
				bitwriter.writebits(_creationTime.Value);
                bitwriter.writebits(32, Settings.PCM.SampleRate);
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

		void write_chunk_minf(BitWriter bitwriter, int header_len)
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
                            bitwriter.writebits(16, Settings.PCM.SampleRate); // time scale
							bitwriter.writebits(16, 0); // reserved
							chunk_start(bitwriter);
							{
								int max_fs = 0;
								long sum_fs = 0;
								for (int i = 0; i < frame_count; i++)
								{
									max_fs = Math.Max(max_fs, (int)_sample_byte_size[i]);
									sum_fs += (int)_sample_byte_size[i];
								}
								bitwriter.write('a', 'l', 'a', 'c');
								bitwriter.writebits(32, 0); // reserved
								bitwriter.writebits(32, m_blockSize); // max frame size
								bitwriter.writebits(8, 0); // reserved
                                bitwriter.writebits(8, Settings.PCM.BitsPerSample);
								bitwriter.writebits(8, history_mult);
								bitwriter.writebits(8, initial_history);
								bitwriter.writebits(8, k_modifier);
                                bitwriter.writebits(8, Settings.PCM.ChannelCount); // channels
								bitwriter.writebits(16, 0); // reserved or 0x00 0xff????
								bitwriter.writebits(32, max_fs);
                                bitwriter.writebits(32, (int)(8 * sum_fs * Settings.PCM.SampleRate / sample_count)); // average bitrate
                                bitwriter.writebits(32, Settings.PCM.SampleRate);
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
						if (sample_count % m_blockSize == 0)
						{
							bitwriter.writebits(32, 1); // entries
							bitwriter.writebits(32, sample_count / m_blockSize);
							bitwriter.writebits(32, m_blockSize);
						}
						else
						{
							bitwriter.writebits(32, 2); // entries
                            bitwriter.writebits(32, sample_count / m_blockSize);
                            bitwriter.writebits(32, m_blockSize);
							bitwriter.writebits(32, 1);
                            bitwriter.writebits(32, sample_count % m_blockSize);
						}
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 's', 'c');
						bitwriter.writebits(32, 0); // version & flags
						if (frame_count % eparams.chunk_size == 0)
						{
							bitwriter.writebits(32, 1); // entries
							bitwriter.writebits(32, 1); // first chunk
							bitwriter.writebits(32, eparams.chunk_size); // samples in chunk
							bitwriter.writebits(32, 1); // sample description index
						}
						else
						{
							bitwriter.writebits(32, 2); // entries
							bitwriter.writebits(32, 1); // first chunk
							bitwriter.writebits(32, eparams.chunk_size); // samples in chunk
							bitwriter.writebits(32, 1); // sample description index
							bitwriter.writebits(32, 1 + frame_count / eparams.chunk_size); // first chunk
							bitwriter.writebits(32, frame_count % eparams.chunk_size); // samples in chunk
							bitwriter.writebits(32, 1); // sample description index
						}
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 's', 'z'); // stsz
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, 0); // sample size (0 == variable)
						bitwriter.writebits(32, frame_count); // entry count
						for (int i = 0; i < frame_count; i++)
							bitwriter.writebits(32, _sample_byte_size[i]);
					}
					chunk_end(bitwriter);
					chunk_start(bitwriter);
					{
						bitwriter.write('s', 't', 'c', 'o'); // stco
						bitwriter.writebits(32, 0); // version & flags
						bitwriter.writebits(32, (frame_count + eparams.chunk_size - 1) / eparams.chunk_size); // entry count
						int pos = header_len;
						for (int i = 0; i < frame_count; i++)
						{
							if (i % eparams.chunk_size == 0) bitwriter.writebits(32, pos);
							pos += (int)_sample_byte_size[i];
						}
					}
					chunk_end(bitwriter);
				}
				chunk_end(bitwriter);
			}
			chunk_end(bitwriter);
		}

		void write_chunk_mdia(BitWriter bitwriter, int header_len)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'd', 'i', 'a');
				chunk_start(bitwriter);
				{
					bitwriter.write('m', 'd', 'h', 'd');
					bitwriter.writebits(32, 0); // version & flags
					bitwriter.writebits(_creationTime.Value);
					bitwriter.writebits(_creationTime.Value);
                    bitwriter.writebits(32, Settings.PCM.SampleRate);
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
					bitwriter.writebits(8, 0); //bitwriter.writebits(8, "SoundHandler".Length);
					bitwriter.writebits(8, 0); //bitwriter.write("SoundHandler");
				}
				chunk_end(bitwriter);
				write_chunk_minf(bitwriter, header_len);
			}
			chunk_end(bitwriter);
		}

		void write_chunk_trak(BitWriter bitwriter, int header_len)
		{
			chunk_start(bitwriter);
			{
				bitwriter.write('t', 'r', 'a', 'k');
				chunk_start(bitwriter);
				{
					bitwriter.write('t', 'k', 'h', 'd');
					bitwriter.writebits(32, 7); // version
					bitwriter.writebits(_creationTime.Value);
					bitwriter.writebits(_creationTime.Value);
					bitwriter.writebits(32, 1); // track ID
					bitwriter.writebits(32, 0); // reserved
					bitwriter.writebits(32, sample_count);
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
				write_chunk_mdia(bitwriter, header_len);
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
								bitwriter.write(Vendor);
							}
							chunk_end(bitwriter);
						}
						chunk_end(bitwriter);
					}
					chunk_end(bitwriter);

					chunk_start(bitwriter); // padding
					{
						bitwriter.write('f', 'r', 'e', 'e');
						bitwriter.writebytes(m_settings.Padding, 0);
					}
					chunk_end(bitwriter);					
				}
				chunk_end(bitwriter);
			}
			chunk_end(bitwriter);
		}

		byte[] write_headers(int header_len, int mdat_len)
		{
			byte[] header = new byte[header_len];
			BitWriter bitwriter = new BitWriter(header, 0, header.Length);

			chunk_start(bitwriter);
			{
				bitwriter.write('f', 't', 'y', 'p');
				bitwriter.write('M', '4', 'A', ' ');
				bitwriter.writebits(32, 0x200); // minor version
				bitwriter.write('M', '4', 'A', ' ');
				bitwriter.write('m', 'p', '4', '2');
				bitwriter.write('i', 's', 'o', 'm');
				bitwriter.writebits(32, 0);
			}
			chunk_end(bitwriter);

			chunk_start(bitwriter);
			{
				bitwriter.write('m', 'o', 'o', 'v');
				write_chunk_mvhd(bitwriter);
				write_chunk_trak(bitwriter, header_len);
				write_chunk_udta(bitwriter);
			}
			chunk_end(bitwriter);

			chunk_start(bitwriter); // padding
			{
				bitwriter.write('f', 'r', 'e', 'e');
				int padding_len = header_len - bitwriter.Length - 8;
				if (padding_len < 0)
					throw new Exception("padding length too small");
				bitwriter.writebytes(padding_len, 0);
			}
			chunk_end(bitwriter);

			bitwriter.writebits(32, mdat_len + 8);
			bitwriter.write('m', 'd', 'a', 't');
			bitwriter.flush();

			return header;
		}

		void encode_init()
		{
			// FIXME: For now, only 44100 samplerate is supported
            if (Settings.PCM.SampleRate != 44100)
				throw new Exception("non-standard samplerate");

			// FIXME: For now, only 16-bit encoding is supported
            if (Settings.PCM.BitsPerSample != 16)
				throw new Exception("non-standard bps");

            m_blockSize =
                m_settings.BlockSize != 0 ? m_settings.BlockSize :
                select_blocksize(Settings.PCM.SampleRate, eparams.block_time_ms);

			// set maximum encoded frame size (if larger, re-encodes in verbatim mode)
            if (Settings.PCM.ChannelCount == 2)
                max_frame_size = 16 + ((m_blockSize * (Settings.PCM.BitsPerSample + Settings.PCM.BitsPerSample + 1) + 7) >> 3);
			else
                max_frame_size = 16 + ((m_blockSize * Settings.PCM.ChannelCount * Settings.PCM.BitsPerSample + 7) >> 3);

			frame_buffer = new byte[max_frame_size];
            _sample_byte_size = new uint[Math.Max(0x100, sample_count / m_blockSize + 1)];

			if (m_settings.DoVerify)
			{
                verify = new ALACReader(Settings.PCM, history_mult, initial_history, k_modifier, m_blockSize);
                verifyBuffer = new int[Alac.MAX_BLOCKSIZE * Settings.PCM.ChannelCount];
			}

			if (sample_count < 0)
				throw new InvalidOperationException("FinalSampleCount unknown");
            int frames = sample_count / m_blockSize;
			int header_len = max_header_len
				+ m_settings.Padding
				+ frames * 4 // stsz
				+ frames * 4 / eparams.chunk_size; // stco
			//if (header_len % 0x400 != 0)
			//    header_len += 0x400 - (header_len % 0x400);
			first_frame_offset = header_len;
			_IO.Write(new byte[first_frame_offset], 0, first_frame_offset);
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

		public WindowMethod window_method;

		public int chunk_size;

		// block time in milliseconds
		// set by the user prior to calling encode_init
		// used to calculate block_size based on sample rate
		// can also be changed by user before encoding a frame
		public int block_time_ms;

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

		public int adaptive_passes;

		public int min_modifier, max_modifier;

		public WindowFunction window_function;

		public bool do_seektable;

		public int set_defaults(int lvl)
		{
			compression = lvl;

			if ((lvl < 0 || lvl > 12) && (lvl != 99))
			{
				return -1;
			}

			// default to level 5 params
            window_function = WindowFunction.PartialTukey | WindowFunction.PunchoutTukey;
			order_method = OrderMethod.Estimate;
			stereo_method = StereoMethod.Estimate;
			window_method = WindowMethod.Estimate;
			block_time_ms = 105;
			min_modifier = 4;
			max_modifier = 4;
			min_prediction_order = 1;
			max_prediction_order = 12;
			estimation_depth = 1;
			adaptive_passes = 0;
			do_seektable = false;
			chunk_size = 5;

			// differences from level 6
			switch (lvl)
			{
				case 0:
					stereo_method = StereoMethod.Independent;
					max_prediction_order = 6;
					break;
				case 1:
					stereo_method = StereoMethod.Independent;
					max_prediction_order = 8;
					break;
				case 2:
					max_prediction_order = 6;
					break;
				case 3:
                    window_function = WindowFunction.PartialTukey;
					max_prediction_order = 8;
					break;
				case 4:
                    window_function = WindowFunction.PunchoutTukey;
					max_prediction_order = 8;
					break;
				case 5:
                    window_function = WindowFunction.PunchoutTukey;
					break;
				case 6:
                    window_method = WindowMethod.EvaluateN;
					break;
				case 7:
					window_method = WindowMethod.EvaluateN;
					adaptive_passes = 1;
					min_modifier = 2;
					break;
				case 8:
                    stereo_method = StereoMethod.Evaluate;
					window_method = WindowMethod.EvaluateN;
					adaptive_passes = 1;
					min_modifier = 2;
					break;
				case 9:
                    stereo_method = StereoMethod.Evaluate;
                    window_method = WindowMethod.EvaluateN;
					adaptive_passes = 1;
					max_prediction_order = 30;
					min_modifier = 2;
					break;
				case 10:
                    stereo_method = StereoMethod.Evaluate;
                    window_method = WindowMethod.EvaluateN;
					estimation_depth = 2;
					adaptive_passes = 2;
					max_prediction_order = 30;
					min_modifier = 2;
					break;
			}

			return 0;
		}
	}
}
