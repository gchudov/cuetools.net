/*
 * TTAReader.h
 *
 * Description: TTA decompressor internals
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

#include "ttacommon.h"

#define WAVE_FORMAT_PCM	1
#define WAVE_FORMAT_IEEE_FLOAT 3

namespace TTALib 
{
	class BitReader;

	class TTAReader
	{
		HANDLE hInFile;
		long *data;
		unsigned long offset, is_float, framelen, lastlen;
		unsigned long fframes, byte_size, num_chan;
		unsigned long *seek_table;
		bool st_state;
		encoder *tta, *enc;		

		BitReader *bitReader;

	public:
		TTAReader (HANDLE fd);
		~TTAReader ();

		unsigned long input_byte_count;
		unsigned long output_byte_count;
		TTAHeader ttahdr;

		long GetBlock (long **buf);
	};
}
