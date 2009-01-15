/*
 * TTAReader.cpp
 *
 * Description: TTA decompressor functions
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

#include "stdafx.h"
#include "BitReader.h"
#include "TTAReader.h"
#include "filters3.h"

namespace TTALib 
{
	TTAReader::TTAReader (HANDLE fd) : hInFile(fd)  
	{
		unsigned long data_size;
		int st_size;

		// clear statistics
		output_byte_count = 0;
		bitReader = new BitReader (hInFile);
		bitReader->GetHeader (&ttahdr);

		byte_size = (ttahdr.BitsPerSample + 7) / 8;
		framelen = (long) (FRAME_TIME * ttahdr.SampleRate);
		is_float = (ttahdr.AudioFormat == WAVE_FORMAT_IEEE_FLOAT);
		num_chan = ttahdr.NumChannels << is_float;
		data_size = ttahdr.DataLength * byte_size * ttahdr.NumChannels;

		lastlen = ttahdr.DataLength % framelen;
		fframes = ttahdr.DataLength / framelen + (lastlen ? 1 : 0);
		st_size = (fframes + 1);
		st_state = 0;

		enc = tta = new encoder[num_chan];
		seek_table = new unsigned long[st_size];
		data = new long[framelen * num_chan];
		st_state = bitReader->GetSeekTable (seek_table, st_size);
		encoder_init(tta, num_chan, byte_size);
	}

	TTAReader::~TTAReader ()
	{
		delete [] seek_table;
		delete [] tta;
		delete [] data;
		delete bitReader;
	}

	long TTAReader::GetBlock (long **buf)
	{
		long *p, value;
		unsigned long  unary, binary, depth, k;

		if (!fframes--)
			return 0;

		if (!fframes && lastlen) framelen = lastlen;

		encoder_init(tta, num_chan, byte_size);

		for (p = data; p < data + framelen * num_chan; p++) {
			fltst *fst = &enc->fst;
			adapt *rice = &enc->rice;
			long *last = &enc->last;

			// decode Rice unsigned
			bitReader->GetUnary(&unary);

			switch (unary) 
			{
			case 0: depth = 0; k = rice->k0; break;
			default:
				depth = 1; k = rice->k1;
				unary--;
			}

			if (k) {
				bitReader->GetBinary(&binary, k);
				value = (unary << k) + binary;
			} else value = unary;

			switch (depth) 
			{
			case 1: 
				rice->sum1 += value - (rice->sum1 >> 4);
				if (rice->k1 > 0 && rice->sum1 < shift_16[rice->k1])
					rice->k1--;
				else if (rice->sum1 > shift_16[rice->k1 + 1])
					rice->k1++;
				value += bit_shift[rice->k0];
			default:
				rice->sum0 += value - (rice->sum0 >> 4);
				if (rice->k0 > 0 && rice->sum0 < shift_16[rice->k0])
					rice->k0--;
				else if (rice->sum0 > shift_16[rice->k0 + 1])
					rice->k0++;
			}

			*p = DEC(value);

			// decompress stage 1: adaptive hybrid filter
			hybrid_filter(fst, p, 0);

			// decompress stage 2: fixed order 1 prediction
			switch (byte_size) 
			{
			case 1: *p += PREDICTOR1(*last, 4); break;	// bps 8
			case 2: *p += PREDICTOR1(*last, 5); break;	// bps 16
			case 3: *p += PREDICTOR1(*last, 5); break;	// bps 24
			case 4: *p += *last; break;		// bps 32
			} *last = *p;

			// combine data
			if (is_float && ((p - data) & 1)) {
				unsigned long negative = *p & 0x80000000;
				unsigned long data_hi = *(p - 1);
				unsigned long data_lo = abs(*p) - 1;

				data_hi += (data_hi || data_lo) ? 0x3F80 : 0;
				*(p - 1) = (data_hi << 16) | SWAP16(data_lo) | negative;
			}

			if (enc < tta + num_chan - 1) enc++;
			else {
				if (!is_float && num_chan > 1) {
					long *r = p - 1;
					for (*p += *r/2; r > p - num_chan; r--)
						*r = *(r + 1) - *r;
				}
				enc = tta;
			}
		}

		if (bitReader->Done ()) // CRC error
		{
			if (st_state)
			{
				bitReader->SkipFrame ();
				ZeroMemory(data, num_chan * framelen * sizeof(long));
			} 
			else throw TTAException (FILE_ERROR);
		}			

		*buf = data;

		input_byte_count = bitReader->input_byte_count;
		output_byte_count += (p - data) * byte_size;  

		return (p - data) / num_chan;
	}
};
