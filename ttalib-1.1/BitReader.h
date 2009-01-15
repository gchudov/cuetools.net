/*
 * BitReader.h
 *
 * Description: Bit reader internal interface
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
	class BitReader
	{
	protected:	
		HANDLE hInFile;

		unsigned char bit_buffer[BIT_BUFFER_SIZE + 8];
		unsigned char *bit_buffer_end;

		unsigned long frame_crc32;
		unsigned long bit_count;
		unsigned long bit_cache;
		unsigned char *bitpos;

		unsigned long *st;
		unsigned long next_frame_pos;

	public:
		BitReader(HANDLE fd) :
			frame_crc32(0xFFFFFFFFUL), hInFile (fd),
			bit_count(0), bit_cache(0), bit_buffer_end(bit_buffer + BIT_BUFFER_SIZE),
			bitpos(bit_buffer_end), input_byte_count (0)
		{			
		}

		virtual ~BitReader(void) {}

		virtual void GetHeader (TTAHeader *ttahdr)
		{
			unsigned long result, checksum;

			if (!ReadFile(hInFile, ttahdr, sizeof(TTAHeader), &result, NULL) ||
				result != sizeof (TTAHeader))
			{
				throw TTAException (READ_ERROR);
			} else input_byte_count += sizeof(*ttahdr);

			// check for supported formats
			if (ENDSWAP_INT32(ttahdr->TTAid) != TTA1_SIGN) 
				throw TTAException (FORMAT_ERROR);

			checksum = crc32((unsigned char *) ttahdr, sizeof(TTAHeader) - sizeof(long));
			if (checksum != ttahdr->CRC32) 
				throw TTAException (FILE_ERROR);

			ttahdr->AudioFormat = ENDSWAP_INT16(ttahdr->AudioFormat); 
			ttahdr->NumChannels = ENDSWAP_INT16(ttahdr->NumChannels);
			ttahdr->BitsPerSample = ENDSWAP_INT16(ttahdr->BitsPerSample);
			ttahdr->SampleRate = ENDSWAP_INT32(ttahdr->SampleRate);
			ttahdr->DataLength = ENDSWAP_INT32(ttahdr->DataLength);
		}

		virtual bool GetSeekTable (unsigned long *seek_table, long st_size)
		{
			unsigned long result, checksum;
			bool st_state = false;
		
			if (!ReadFile(hInFile, seek_table, st_size * sizeof(long), &result, NULL) ||
				result != st_size * sizeof(long))
				throw TTAException (READ_ERROR);
			else input_byte_count += st_size * sizeof(long);

			checksum = crc32((unsigned char *) seek_table, (st_size - 1) * sizeof(long));
			if (checksum == ENDSWAP_INT32(seek_table[st_size - 1]))
				st_state = true;

			for (st = seek_table; st < (seek_table + st_size); st++)
				*st = ENDSWAP_INT32(*st);
						
			next_frame_pos = SetFilePointer (hInFile, 0, NULL, FILE_CURRENT);
			st = seek_table;

			return st_state;
		}

		virtual void GetBinary(unsigned long *value, unsigned long bits) {
			while (bit_count < bits) {
				if (bitpos == bit_buffer_end) {
					unsigned long result;			 
					if (!ReadFile (hInFile, bit_buffer, BIT_BUFFER_SIZE, &result, NULL))
						throw TTAException (READ_ERROR);
					input_byte_count += result;
					bitpos = bit_buffer;
				}

				UPDATE_CRC32(*bitpos, frame_crc32);
				bit_cache |= *bitpos << bit_count;
				bit_count += 8;
				bitpos++;
			}

			*value = bit_cache & bit_mask[bits];
			bit_cache >>= bits;
			bit_count -= bits;
			bit_cache &= bit_mask[bit_count];
		}

		virtual void GetUnary(unsigned long *value) 
		{
			*value = 0;

			while (!(bit_cache ^ bit_mask[bit_count])) {
				if (bitpos == bit_buffer_end) {
					unsigned long result;			 
					if (!ReadFile (hInFile, bit_buffer, BIT_BUFFER_SIZE, &result, NULL)) 
						throw TTAException (READ_ERROR);
					input_byte_count += result;
					bitpos = bit_buffer;
				}

				*value += bit_count;
				bit_cache = *bitpos++;
				UPDATE_CRC32(bit_cache, frame_crc32);
				bit_count = 8;
			}

			while (bit_cache & 1) {
				(*value)++;
				bit_cache >>= 1;
				bit_count--;
			}

			bit_cache >>= 1;
			bit_count--;
		}

		virtual int Done ()
		{
			unsigned long crc32, rbytes, result;
			frame_crc32 ^= 0xFFFFFFFFUL;

	  		next_frame_pos += *st++;

			rbytes = bit_buffer_end - bitpos;
			if (rbytes < sizeof(long)) {
				CopyMemory(bit_buffer, bitpos, 4);
				if (!ReadFile(hInFile, bit_buffer + rbytes,
					BIT_BUFFER_SIZE - rbytes, &result, NULL))
					throw TTAException (READ_ERROR);
				input_byte_count += result;
				bitpos = bit_buffer;
			}

			CopyMemory(&crc32, bitpos, 4);
			crc32 = ENDSWAP_INT32(crc32);
			bitpos += sizeof(long);
			result = (crc32 != frame_crc32);

			bit_cache = bit_count = 0;
			frame_crc32 = 0xFFFFFFFFUL;

			return result;
		}

		virtual void SkipFrame ()
		{
			SetFilePointer(hInFile, next_frame_pos, NULL, FILE_BEGIN);
			bitpos = bit_buffer_end;
		}

		unsigned long input_byte_count;
	};
};
