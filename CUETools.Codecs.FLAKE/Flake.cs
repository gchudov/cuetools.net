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
			switch (name)
			{
				case "fixed": return PredictionType.Fixed;
				case "levinson": return PredictionType.Levinson;
				case "search" : return PredictionType.Search;
			}
			return (PredictionType)Int32.Parse(name);
		}

		public static StereoMethod LookupStereoMethod(string name)
		{
			switch (name)
			{
				case "independent": return StereoMethod.Independent;
				case "estimate": return StereoMethod.Estimate;
				case "estimate2": return StereoMethod.Estimate2;
				case "estimate3": return StereoMethod.Estimate3;
				case "estimate4": return StereoMethod.Estimate4;
				case "estimate5": return StereoMethod.Estimate5;
				case "search": return StereoMethod.Search;
			}
			return (StereoMethod)Int32.Parse(name);
		}

		public static OrderMethod LookupOrderMethod(string name)
		{
			switch (name)
			{
				case "estimate": return OrderMethod.Estimate;
				case "logfast": return OrderMethod.LogFast;
				case "logsearch": return OrderMethod.LogSearch;
				case "estsearch": return OrderMethod.EstSearch;
				case "search": return OrderMethod.Search;
			}
			return (OrderMethod)Int32.Parse(name);
		}

		public static WindowFunction LookupWindowFunction(string name)
		{
			string[] parts = name.Split(',');
			WindowFunction res = (WindowFunction)0;
			foreach (string part in parts)
			{
				switch (part)
				{
					case "welch": res |= WindowFunction.Welch; break;
					case "tukey": res |= WindowFunction.Tukey; break;
					case "hann": res |= WindowFunction.Hann; break;
					case "flattop": res |= WindowFunction.Flattop; break;
					default: res |= (WindowFunction)Int32.Parse(name); break;
				}
			}
			return res;
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
		public uint obits;
		public uint wbits;
		public int cbits;
		public int shift;
		public fixed int coefs[lpc.MAX_LPC_ORDER];
		public int* samples;
		public int* residual;
		public RiceContext rc;
		public uint size;
		public fixed uint done_lpcs[lpc.MAX_LPC_WINDOWS];
		public uint done_fixed;
		public int window;
	};

	unsafe struct FlacFrame
	{
		public int blocksize;
		public int bs_code0, bs_code1;
		public ChannelMode ch_mode;
		public int ch_order0, ch_order1;
		public byte crc8;
		public FlacSubframe* subframes;
		public uint frame_count;
		public FlacSubframe current;
	}

	public enum OrderMethod
	{
		Max = 0,
		Estimate = 1,
		LogFast = 2,
		LogSearch = 3,
		EstSearch = 4,
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
		Estimate2 = 2,
		Estimate3 = 3,
		Estimate4 = 4,
		Estimate5 = 5,
		Search = 9
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
		Flattop = 8
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
