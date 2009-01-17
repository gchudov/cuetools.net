/*
 * TTAWriter.h
 *
 * Description: TTA compressor internals
 *
 * Copyright (c) 2004 Alexander Djourik. All rights reserved.
 * Copyright (c) 2004 Pavel Zhilin. All rights reserved.
 *
 */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * aint with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * Please see the file COPYING in this directory for full copyright
 * information.
 */

#pragma once
#include "TTACommon.h"

#define WAVE_FORMAT_PCM	1
#define WAVE_FORMAT_IEEE_FLOAT 3

namespace TTALib 
{
	class BitWriter;

	class TTAWriter
	{
		HANDLE hOutFile;
		TTAHeader ttahdr;
		BitWriter *bitWriter;
		unsigned long *seek_table, st_size, *st;
		unsigned long offset, is_float, framelen, lastlen;
		unsigned long fframes, byte_size, num_chan;
		unsigned long data_pos, max_bytes;
		encoder *tta, *enc;

	public:
		TTAWriter (HANDLE fd, long offset, unsigned short AudioFormat, 
		unsigned short NumChannels,	unsigned short BitsPerSample,
		unsigned long SampleRate, unsigned long DataLength);
		~TTAWriter ();

		unsigned long input_byte_count;
		unsigned long output_byte_count;

		bool CompressBlock (long *data, long data_len);
	};
}
