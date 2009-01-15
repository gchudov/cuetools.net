/*
 * BitWriter.h
 *
 * Description: Bit writer internal interface
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
#include <windows.h>
#include "TTACommon.h"
#include "TTAError.h"
#include "crc32.h"

namespace TTALib 
{
	class BitWriter
	{
	protected:
		HANDLE hOutFile;
		unsigned long start_offset;

		unsigned char bit_buffer[BIT_BUFFER_SIZE + 8];
		unsigned char *bit_buffer_end;
		unsigned long frame_crc32;
		unsigned long bit_count;
		unsigned long bit_cache;
		unsigned char *bitpos;
		unsigned long lastpos;

	public:
		unsigned long output_byte_count;

		BitWriter(HANDLE fd, unsigned long offset) :
			frame_crc32(0xFFFFFFFFUL),
			start_offset(offset), hOutFile (fd),
			bit_count(0), bit_cache(0),
			bitpos (bit_buffer), lastpos (0), output_byte_count (0),
			bit_buffer_end (bit_buffer + BIT_BUFFER_SIZE)
		{
	  		SetFilePointer (hOutFile, start_offset, NULL, FILE_BEGIN);
		}

		virtual ~BitWriter(void) {}

		virtual void PutHeader (TTAHeader ttahdr)
		{
			unsigned long result;

			ttahdr.TTAid = ENDSWAP_INT32(TTA1_SIGN);
			ttahdr.AudioFormat = ENDSWAP_INT16(ttahdr.AudioFormat); 
			ttahdr.NumChannels = ENDSWAP_INT16(ttahdr.NumChannels);
			ttahdr.BitsPerSample = ENDSWAP_INT16(ttahdr.BitsPerSample);
			ttahdr.SampleRate = ENDSWAP_INT32(ttahdr.SampleRate);
			ttahdr.DataLength = ENDSWAP_INT32(ttahdr.DataLength);
			ttahdr.CRC32 = crc32((unsigned char *) &ttahdr,
				sizeof(TTAHeader) - sizeof(long));
			ttahdr.CRC32 = ENDSWAP_INT32(ttahdr.CRC32);

			// write TTA header
			if (!WriteFile(hOutFile, &ttahdr, sizeof(TTAHeader), &result, NULL) ||
				result != sizeof (TTAHeader))
			{
				CloseHandle (hOutFile);
				throw TTAException (WRITE_ERROR);
			}
	
			lastpos = (output_byte_count += sizeof(TTAHeader));
		}

		virtual void PutSeekTable (unsigned long *seek_table, long st_size)
		{
			unsigned long result;
			unsigned long *st;

			if (SetFilePointer (hOutFile, 0, NULL, FILE_CURRENT) != start_offset + sizeof(TTAHeader))
				SetFilePointer (hOutFile, start_offset + sizeof(TTAHeader), NULL,  FILE_BEGIN);
			else
				lastpos = (output_byte_count += st_size * sizeof(long));
			
			for (st = seek_table; st < (seek_table + st_size - 1); st++)
				*st = ENDSWAP_INT32(*st);
			seek_table[st_size - 1] = crc32((unsigned char *) seek_table, 
				(st_size - 1) * sizeof(long));
			seek_table[st_size - 1] = ENDSWAP_INT32(seek_table[st_size - 1]);

			if (!WriteFile(hOutFile, seek_table, st_size * sizeof(long), &result, NULL) ||
				result != st_size * sizeof(long))
			{
				CloseHandle (hOutFile);
				throw TTAException (WRITE_ERROR);
			}
		}

		virtual void PutBinary(unsigned long value, unsigned long bits) 
		{
			while (bit_count >= 8) {
				if (bitpos == bit_buffer_end) {
					unsigned long result;
					if (!WriteFile (hOutFile, bit_buffer, BIT_BUFFER_SIZE, &result, NULL) ||
						result != BIT_BUFFER_SIZE) 
						throw TTAException (WRITE_ERROR);
						
					output_byte_count += result;
					bitpos = bit_buffer;
				}

				*bitpos = (unsigned char) (bit_cache & 0xFF);
				UPDATE_CRC32(*bitpos, frame_crc32);
				bit_cache >>= 8;
				bit_count -= 8;
				bitpos++;
			}

			bit_cache |= (value & bit_mask[bits]) << bit_count;
			bit_count += bits;
		}

		virtual void PutUnary(unsigned long value) 
		{
			do {
				while (bit_count >= 8) {
					if (bitpos == bit_buffer_end) {
						unsigned long result;
						if (!WriteFile (hOutFile, bit_buffer, BIT_BUFFER_SIZE, &result, NULL) ||
							result != BIT_BUFFER_SIZE) 
							throw TTAException (WRITE_ERROR);								  
  
						output_byte_count += result;
						bitpos = bit_buffer;
					}

					*bitpos = (unsigned char) (bit_cache & 0xFF);
					UPDATE_CRC32(*bitpos, frame_crc32);
					bit_cache >>= 8;
					bit_count -= 8;
					bitpos++;
				}

				if (value > 23) {
					bit_cache |= bit_mask[23] << bit_count;
					bit_count += 23;
					value -= 23;
				} else {
					bit_cache |= bit_mask[value] << bit_count;
					bit_count += value + 1;
					value = 0;
				}
			} while (value);
		}

		virtual int Done() 
		{
			unsigned long res, bytes_to_write;

			while (bit_count) {
				*bitpos = (unsigned char) (bit_cache & 0xFF);
				UPDATE_CRC32(*bitpos, frame_crc32);
				bit_cache >>= 8;
				bit_count = (bit_count > 8) ? (bit_count - 8) : 0;
				bitpos++;
			}

			frame_crc32 ^= 0xFFFFFFFFUL;
			frame_crc32 = ENDSWAP_INT32(frame_crc32);
			CopyMemory(bitpos, &frame_crc32, 4);
			bytes_to_write = bitpos + sizeof(long) - bit_buffer;
			
			if (!WriteFile(hOutFile, bit_buffer, bytes_to_write, &res, NULL) || 
				res != bytes_to_write)
				throw TTAException (WRITE_ERROR);

			output_byte_count += res;
			bitpos = bit_buffer;
			frame_crc32 = 0xFFFFFFFFUL;

			res = output_byte_count - lastpos;
			lastpos = output_byte_count;

			return res;
		}
	};
};
