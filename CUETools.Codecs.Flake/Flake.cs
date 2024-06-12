/**
 * CUETools.Flake: pure managed FLAC audio encoder
 * Copyright (c) 2009-2024 Grigory Chudov
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

namespace CUETools.Codecs.Flake
{
	public class FlakeConstants
	{
		public const int MAX_BLOCKSIZE = 65535;
		public const int MAX_RICE_PARAM = 14;
		public const int MAX_PARTITION_ORDER = 8;
		public const int MAX_PARTITIONS = 1 << MAX_PARTITION_ORDER;

		public const int FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN = 64; /* bits */
		public const int FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN = 64; /* bits */
		public const int FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN = 16; /* bits */

		public static readonly int[] flac_samplerates = new int[16] {
				0, 88200, 176400, 192000,
				8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
				0, 0, 0, 0
			};
        //1100 : get 8 bit sample rate (in kHz) from end of header
        //1101 : get 16 bit sample rate (in Hz) from end of header
        //1110 : get 16 bit sample rate (in tens of Hz) from end of header
		public static readonly int[] flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };
        //0110 : get 8 bit (blocksize-1) from end of header
        //0111 : get 16 bit (blocksize-1) from end of header
		public static readonly int[] flac_bitdepths = new int[8] { 0, 8, 12, 0, 16, 20, 24, 0 };

		public static PredictionType LookupPredictionType(string name)
		{
			return (PredictionType)(Enum.Parse(typeof(PredictionType), name, true));
		}

		public static StereoMethod LookupStereoMethod(string name)
		{
			return (StereoMethod)(Enum.Parse(typeof(StereoMethod), name, true));
		}

		public static WindowMethod LookupWindowMethod(string name)
		{
			return (WindowMethod)(Enum.Parse(typeof(WindowMethod), name, true));
		}

		public static OrderMethod LookupOrderMethod(string name)
		{
			return (OrderMethod)(Enum.Parse(typeof(OrderMethod), name, true));
		}

		public static WindowFunction LookupWindowFunction(string name)
		{
			return (WindowFunction)(Enum.Parse(typeof(WindowFunction), name, true));
		}
	}
}
