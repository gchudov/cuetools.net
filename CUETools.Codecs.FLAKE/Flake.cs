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
	public class Flake
	{
		public const int MAX_BLOCKSIZE = 65535;
		public const int MAX_RICE_PARAM = 14;
		public const int MAX_PARTITION_ORDER = 8;
		public const int MAX_PARTITIONS = 1 << MAX_PARTITION_ORDER;

		public const int FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN = 64; /* bits */
		public const int FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN = 64; /* bits */
		public const int FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN = 16; /* bits */

		public static readonly int[] flac_samplerates = new int[16] {
				0, 0, 0, 0,
				8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
				0, 0, 0, 0
			};
		public static readonly int[] flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };
		public static readonly int[] flac_bitdepths = new int[8] { 0, 8, 12, 0, 16, 20, 24, 0 };

		public static int log2i(int v)
		{
			return log2i((uint)v);
		}
		public static int log2i(uint v)
		{
			int i;
			int n = 0;
			if (0 != (v & 0xffff0000)) { v >>= 16; n += 16; }
			if (0 != (v & 0xff00)) { v >>= 8; n += 8; }
			for (i = 2; i < 256; i <<= 1)
			{
				if (v >= i) n++;
				else break;
			}
			return n;
		}

		public static PredictionType LookupPredictionType(string name)
		{
			return (PredictionType)(Enum.Parse(typeof(PredictionType), name, true));
		}

		public static StereoMethod LookupStereoMethod(string name)
		{
			return (StereoMethod)(Enum.Parse(typeof(StereoMethod), name, true));
		}

		public static OrderMethod LookupOrderMethod(string name)
		{
			return (OrderMethod)(Enum.Parse(typeof(OrderMethod), name, true));
		}

		public static WindowFunction LookupWindowFunction(string name)
		{
			return (WindowFunction)(Enum.Parse(typeof(WindowFunction), name, true));
		}

		unsafe public static bool memcmp(int* res, int* smp, int n)
		{
			for (int i = n; i > 0; i--)
				if (*(res++) != *(smp++))
					return true;
			return false;
		}
		unsafe public static void memcpy(int* res, int* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}
		unsafe public static void memcpy(byte* res, byte* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}
		unsafe public static void memset(int* res, int smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = smp;
		}
		unsafe public static void interlace(int* res, int* src1, int* src2, int n)
		{
			for (int i = n; i > 0; i--)
			{
				*(res++) = *(src1++);
				*(res++) = *(src2++);
			}
		}
		unsafe public static void deinterlace(int* dst1, int* dst2, int* src, int n)
		{
			for (int i = n; i > 0; i--)
			{
				*(dst1++) = *(src++);
				*(dst2++) = *(src++);
			}
		}
	}

	unsafe struct RiceContext
	{
		public int porder;					/* partition order */
		public fixed int rparams[Flake.MAX_PARTITIONS];  /* Rice parameters */
		public fixed int esc_bps[Flake.MAX_PARTITIONS];	/* bps if using escape code */
	};

	unsafe struct FlacSubframe
	{
		public SubframeType type;
		public int order;
		public int* residual;
		public RiceContext rc;
		public uint size;

		public int cbits;
		public int shift;
		public fixed int coefs[lpc.MAX_LPC_ORDER];
		public int window;
	};

	unsafe struct FlacSubframeInfo
	{
		public FlacSubframe best;
		public uint obits;
		public uint wbits;
		public int* samples;
		public fixed uint done_lpcs[lpc.MAX_LPC_WINDOWS * 2];
		public uint done_fixed;
		public fixed double lpcs_reff[lpc.MAX_LPC_ORDER * lpc.MAX_LPC_WINDOWS];
		public fixed int lpcs_order[lpc.MAX_LPC_WINDOWS];
	};

	unsafe struct FlacFrame
	{
		public int blocksize;
		public int bs_code0, bs_code1;
		public ChannelMode ch_mode;
		public int ch_order0, ch_order1;
		public byte crc8;
		public FlacSubframeInfo* subframes;
		public uint frame_count;
		public FlacSubframe current;
		public double* window_buffer;
	}

	public enum OrderMethod
	{
		Max = 0,
		Estimate = 1,
		LogFast = 2,
		LogSearch = 3,
		EstSearch = 4,
		Estimate8 = 6,
		Search = 5
	}

	/// <summary>
	/// Type of linear prediction
	/// </summary>
	public enum PredictionType
	{
		/// <summary>
		/// verbatim
		/// </summary>
		None = 0,
		/// <summary>
		/// Fixed only
		/// </summary>
		Fixed = 1,
		/// <summary>
		/// Levinson-Durbin recursion
		/// </summary>
		Levinson = 2,
		/// <summary>
		/// Exhaustive search
		/// </summary>
		Search = 3,
		/// <summary>
		/// Internal; Use prediction type from previous estimation
		/// </summary>
		Estimated = 4
	}

	public enum StereoMethod
	{
		Independent = 0,
		Estimate = 1,
		Evaluate = 2,
		Search = 3
	}

	public enum SubframeType
	{
		Constant = 0,
		Verbatim = 1,
		Fixed = 8,
		LPC = 32
	};

	public enum ChannelMode
	{
		NotStereo = 0,
		LeftRight = 1,
		LeftSide = 8,
		RightSide = 9,
		MidSide = 10
	}

	public enum WindowFunction
	{
		Welch = 1,
		Tukey = 2,
		Hann = 4,
		Flattop = 8,
		TukFlat = 10
	}

	public struct SeekPoint
	{
		public ulong number;
		public ulong offset;
		public uint framesize;
	}

	public enum MetadataType
	{

		/// <summary>
		/// <A HREF="../format.html#metadata_block_streaminfo">STREAMINFO</A> block
		/// </summary>
		FLAC__METADATA_TYPE_STREAMINFO = 0,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_padding">PADDING</A> block
		/// </summary>
		FLAC__METADATA_TYPE_PADDING = 1,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_application">APPLICATION</A> block 
		/// </summary>
		FLAC__METADATA_TYPE_APPLICATION = 2,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_seektable">SEEKTABLE</A> block
		/// </summary>
		FLAC__METADATA_TYPE_SEEKTABLE = 3,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_vorbis_comment">VORBISCOMMENT</A> block (a.k.a. FLAC tags)
		/// </summary>
		FLAC__METADATA_TYPE_VORBIS_COMMENT = 4,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_cuesheet">CUESHEET</A> block
		/// </summary>
		FLAC__METADATA_TYPE_CUESHEET = 5,

		/// <summary>
		/// <A HREF="../format.html#metadata_block_picture">PICTURE</A> block
		/// </summary>
		FLAC__METADATA_TYPE_PICTURE = 6,

		/// <summary>
		/// marker to denote beginning of undefined type range; this number will increase as new metadata types are added
		/// </summary>
		FLAC__METADATA_TYPE_UNDEFINED = 7
	};
}
