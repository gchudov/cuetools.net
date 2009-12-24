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
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.ALAC
{
	public class Alac
	{
		public const int MAX_BLOCKSIZE = 65535;
		public const int MAX_RICE_PARAM = 14;
		public const int MAX_PARTITION_ORDER = 8;
		public const int MAX_PARTITIONS = 1 << MAX_PARTITION_ORDER;
		public const int MAX_LPC_WINDOWS = 4;

		public const uint UINT32_MAX = 0xffffffff;

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

		public static WindowMethod LookupWindowMethod(string name)
		{
			return (WindowMethod)(Enum.Parse(typeof(WindowMethod), name, true));
		}
	}

	unsafe class RiceContext
	{
		public RiceContext()
		{
			rparams = new int[Alac.MAX_PARTITIONS];
			esc_bps = new int[Alac.MAX_PARTITIONS];
		}
		/// <summary>
		/// partition order
		/// </summary>
		public int porder;

		/// <summary>
		/// Rice parameters
		/// </summary>
		public int[] rparams;

		/// <summary>
		/// bps if using escape code
		/// </summary>
		public int[] esc_bps;
	};

	unsafe class ALACSubframe
	{
		public ALACSubframe()
		{
			rc = new RiceContext();
			coefs = new int[lpc.MAX_LPC_ORDER];
			coefs_adapted = new int[lpc.MAX_LPC_ORDER];
		}
		public int order;
		public int* residual;
		public RiceContext rc;
		public uint size;

		public int ricemodifier;
		public int cbits;
		public int shift;
		public int[] coefs;
		public int[] coefs_adapted;
		public int window;
	};

	unsafe class ALACSubframeInfo
	{
		public ALACSubframeInfo()
		{
			best = new ALACSubframe();
			lpc_ctx = new LpcContext[Alac.MAX_LPC_WINDOWS];
			for (int i = 0; i < Alac.MAX_LPC_WINDOWS; i++)
				lpc_ctx[i] = new LpcContext();
		}

		public void Init(int* s, int* r)
		{
			samples = s;
			best.residual = r;
			best.size = AudioSamples.UINT32_MAX;
			best.order = 0;
			for (int iWindow = 0; iWindow < Alac.MAX_LPC_WINDOWS; iWindow++)
				lpc_ctx[iWindow].Reset();
			done_fixed = 0;
		}

		public ALACSubframe best;
		public int* samples;
		public uint done_fixed;
		public LpcContext[] lpc_ctx;
	};

	unsafe class ALACFrame
	{
		public ALACFrame(int subframes_count)
		{
			subframes = new ALACSubframeInfo[subframes_count];
			for (int ch = 0; ch < subframes_count; ch++)
				subframes[ch] = new ALACSubframeInfo();
			current = new ALACSubframe();
		}

		public void InitSize(int bs)
		{
			blocksize = bs;
			type = FrameType.Verbatim;
			interlacing_shift = interlacing_leftweight = 0;
		}

		public void ChooseBestSubframe(int ch)
		{
			if (current.size >= subframes[ch].best.size)
				return;
			ALACSubframe tmp = subframes[ch].best;
			subframes[ch].best = current;
			current = tmp;
		}

		public void SwapSubframes(int ch1, int ch2)
		{
			ALACSubframeInfo tmp = subframes[ch1];
			subframes[ch1] = subframes[ch2];
			subframes[ch2] = tmp;
		}

		/// <summary>
		/// Swap subframes according to channel mode.
		/// It is assumed that we have 4 subframes,
		/// 0 is right, 1 is left, 2 is middle, 3 is difference
		/// </summary>
		public void ChooseSubframes()
		{
			if (interlacing_leftweight != 0)
			{
				SwapSubframes(1, 3);
				switch (interlacing_shift)
				{
					case 0: // leftside						
						break;
					case 1: // midside
						SwapSubframes(0, 2);
						break;
					case 31: // rightside
						SwapSubframes(0, 4);
						break;
				}
			}
		}

		public FrameType type;
		public int blocksize;
		public int interlacing_shift, interlacing_leftweight;
		public ALACSubframeInfo[] subframes;
		public ALACSubframe current;
		public float* window_buffer;
	}

	public enum OrderMethod
	{
		Estimate = 0
	}

	public enum StereoMethod
	{
		Independent = 0,
		Estimate = 1,
		Evaluate = 2,
		Search = 3,
	}

	public enum WindowMethod
	{
		Estimate = 0,
		Evaluate = 1,
		Search = 2
	}

	public enum FrameType
	{
		Verbatim = 0,
		Compressed = 1
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
		Bartlett = 16,
		TukFlat = 10
	}
}
