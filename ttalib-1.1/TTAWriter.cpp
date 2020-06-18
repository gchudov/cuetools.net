/*
 * TTAWriter.cpp
 *
 * Description: TTA compressor functions
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

#include "BitWriter.h"
#include "TTAError.h"
#include "TTAWriter.h"
#include "filters3.h"

namespace TTALib 
{
	TTAWriter::TTAWriter (HANDLE fd, long offset, unsigned short AudioFormat, 
		unsigned short NumChannels,	unsigned short BitsPerSample,
		unsigned long SampleRate, unsigned long DataLength) 
		: hOutFile (fd)
	{			
		ttahdr.AudioFormat = AudioFormat;
		ttahdr.NumChannels = NumChannels;
		ttahdr.BitsPerSample = BitsPerSample;
		ttahdr.SampleRate = SampleRate;
		ttahdr.DataLength = DataLength;

		switch (ttahdr.AudioFormat) 
		{
		case WAVE_FORMAT_IEEE_FLOAT: is_float = 1; break;
		case WAVE_FORMAT_PCM: is_float = 0; break;
		default: throw TTAException (FORMAT_ERROR);
		}

		if ((is_float && ttahdr.BitsPerSample != MAX_BPS) ||
			(!is_float && ttahdr.BitsPerSample == MAX_BPS)) 
			throw TTAException (FORMAT_ERROR);

		framelen = (long) (FRAME_TIME * ttahdr.SampleRate);
		num_chan = ttahdr.NumChannels << is_float;
		byte_size = (ttahdr.BitsPerSample + 7) / 8;

		input_byte_count = 0;
		output_byte_count = 0;

		lastlen = ttahdr.DataLength % framelen;
		fframes = ttahdr.DataLength / framelen + (lastlen ? 1 : 0);
		st_size = (fframes + 1);

		// grab some space for an encoder buffers
		st = seek_table = new unsigned long[st_size];
		enc = tta = new encoder[num_chan];			

		bitWriter = new BitWriter (hOutFile, offset);
		bitWriter->PutHeader (ttahdr);
		bitWriter->PutSeekTable (seek_table, st_size);

		data_pos = 0;
		encoder_init(tta, num_chan, byte_size);
		fframes--;

		max_bytes = (num_chan * byte_size * ttahdr.DataLength) >> is_float;
	}

	TTAWriter::~TTAWriter () 
	{
		bitWriter->PutSeekTable (seek_table, st_size);
		delete bitWriter;

		delete [] seek_table;
		delete [] tta;			
	};

	bool TTAWriter::CompressBlock (long *data, long data_len)
	{
		long *p, tmp, prev;
		unsigned long value, k, unary, binary;
		long len;			
		bool ret = false;

		if (data_len > (long)(max_bytes - input_byte_count))
			data_len = (long) (max_bytes - input_byte_count); 
		
		input_byte_count += (num_chan * byte_size * data_len) >> is_float;

		while (data_len > 0 && fframes >= 0)
		{
			if (data_len < (long)(framelen - data_pos))
				len = data_len;
			else
				len = framelen - data_pos;
			data_len -= len;

			for (p = data, prev = 0; p < data + len * num_chan; p++) 
			{
				fltst *fst = &enc->fst;
				adapt *rice = &enc->rice;
				long *last = &enc->last;

				// transform data
				if (!is_float) {
					if (enc < tta + num_chan - 1)
						*p = prev = *(p + 1) - *p;
					else *p -= prev / 2;
				} else if (!((p - data) & 1)) {
					unsigned long t = *p;
					unsigned long negative = (t & 0x80000000) ? -1 : 1;
					unsigned long data_hi = (t & 0x7FFF0000) >> 16;
					unsigned long data_lo = (t & 0x0000FFFF);

					*p = (data_hi || data_lo) ? (data_hi - 0x3F80) : 0;
					*(p + 1) = (SWAP16(data_lo) + 1) * negative;
				}

				// compress stage 1: fixed order 1 prediction
				tmp = *p; 
				switch (byte_size) 
				{
				case 1:	*p -= PREDICTOR1(*last, 4); break;	// bps 8
				case 2:	*p -= PREDICTOR1(*last, 5);	break;	// bps 16
				case 3: *p -= PREDICTOR1(*last, 5); break;	// bps 24
				case 4: *p -= *last; break;			// bps 32
				}
				*last = tmp;

				// compress stage 2: adaptive hybrid filter
				hybrid_filter(fst, p, 1);

				value = ENC(*p);

				// encode Rice unsigned
				k = rice->k0;

				rice->sum0 += value - (rice->sum0 >> 4);
				if (rice->k0 > 0 && rice->sum0 < shift_16[rice->k0])
					rice->k0--;
				else if (rice->sum0 > shift_16[rice->k0 + 1])
					rice->k0++;

				if (value >= bit_shift[k]) {
					value -= bit_shift[k];
					k = rice->k1;

					rice->sum1 += value - (rice->sum1 >> 4);
					if (rice->k1 > 0 && rice->sum1 < shift_16[rice->k1])
						rice->k1--;
					else if (rice->sum1 > shift_16[rice->k1 + 1])
						rice->k1++;

					unary = 1 + (value >> k);
				} else unary = 0;

				bitWriter->PutUnary(unary);
				if (k) {
					binary = value & bit_mask[k];
					bitWriter->PutBinary(binary, k);
				}

				if (enc < tta + num_chan - 1) enc++;
				else enc = tta;
			}

			data_pos += len;

			if (data_pos == framelen)
			{
				*st++ = bitWriter->Done ();

				fframes--;
				if (!fframes && lastlen) framelen = lastlen;
				encoder_init(tta, num_chan, byte_size);
				data_pos = 0;
			}			

			data += (num_chan * len);

			ret = true; 
		}

		output_byte_count = bitWriter->output_byte_count;
		return ret;
	}

};
