/*
 * WavFile.cpp
 *
 * Description: Wraps working with WAV files
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
#include <windows.h>
#include "TTACommon.h"
#include "TTAError.h"
#include "WavFile.h"

/************************** WAV functions ******************************/

TTALib::WaveFile::WaveFile () : fd (INVALID_HANDLE_VALUE)
{
}

TTALib::WaveFile::~WaveFile () 
{
	if (fd != INVALID_HANDLE_VALUE) 
		CloseHandle (fd);
}

HANDLE TTALib::WaveFile::Create (const char *filename)
{
	errNo = TTALib::TTA_NO_ERROR;
	if ((fd = CreateFile (filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
			FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
	{
		errNo = TTALib::OPEN_ERROR;
		return INVALID_HANDLE_VALUE;
	}
	return fd;
}

HANDLE TTALib::WaveFile::Open (const char *filename)
{
	errNo = TTALib::TTA_NO_ERROR;
	if ((fd = CreateFile (filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
		errNo = TTALib::OPEN_ERROR;
	return fd;
}

bool TTALib::WaveFile::ReadHeaders ()
{
	unsigned long result;

	// Read WAVE header	
	if (!ReadFile(fd, &wave_hdr, sizeof(wave_hdr), &result, NULL)) 
	{
		CloseHandle (fd);
		errNo = TTALib::READ_ERROR;
		return false;
	}

	wave_hdr.ChunkID = ENDSWAP_INT32(wave_hdr.ChunkID);
	wave_hdr.ChunkSize = ENDSWAP_INT32(wave_hdr.ChunkSize);
	wave_hdr.Format = ENDSWAP_INT32(wave_hdr.Format);
	wave_hdr.Subchunk1ID = ENDSWAP_INT32(wave_hdr.Subchunk1ID);
	wave_hdr.Subchunk1Size = ENDSWAP_INT32(wave_hdr.Subchunk1Size);
	wave_hdr.AudioFormat = ENDSWAP_INT16(wave_hdr.AudioFormat);
	wave_hdr.NumChannels = ENDSWAP_INT16(wave_hdr.NumChannels);
	wave_hdr.SampleRate = ENDSWAP_INT32(wave_hdr.SampleRate);
	wave_hdr.ByteRate = ENDSWAP_INT32(wave_hdr.ByteRate);
	wave_hdr.BlockAlign = ENDSWAP_INT16(wave_hdr.BlockAlign);
	wave_hdr.BitsPerSample = ENDSWAP_INT16(wave_hdr.BitsPerSample);

	// skip extra format bytes
	if (wave_hdr.Subchunk1Size > 16) {
		SetFilePointer (fd, wave_hdr.Subchunk1Size - 16, NULL, FILE_CURRENT);
	}

	// skip unsupported chunks
	while (ReadFile(fd, &subchunk_hdr, sizeof(subchunk_hdr), &result, NULL) &&
		subchunk_hdr.SubchunkID != ENDSWAP_INT32(data_SIGN)) {
		char chunk_id[5];

		subchunk_hdr.SubchunkSize = ENDSWAP_INT32(subchunk_hdr.SubchunkSize);
		subchunk_hdr.SubchunkID = ENDSWAP_INT32(subchunk_hdr.SubchunkID);

		if (subchunk_hdr.SubchunkSize & 0x80000000UL) {
			CloseHandle (fd);
			errNo = TTALib::FILE_ERROR;
			return false;
		}

		CopyMemory(chunk_id, &subchunk_hdr.SubchunkID, 4);
		chunk_id[4] = 0;

		SetFilePointer(fd, subchunk_hdr.SubchunkSize, NULL, FILE_CURRENT);
	}
	subchunk_hdr.SubchunkSize = ENDSWAP_INT32(subchunk_hdr.SubchunkSize);
	return true;
}

bool TTALib::WaveFile::WriteHeaders ()
{
	unsigned long result;

	errNo = TTALib::TTA_NO_ERROR;
	wave_hdr.ChunkID = ENDSWAP_INT32(RIFF_SIGN);
	wave_hdr.ChunkSize = ENDSWAP_INT32(wave_hdr.ChunkSize);
	wave_hdr.Format = ENDSWAP_INT32(WAVE_SIGN);
	wave_hdr.Subchunk1ID = ENDSWAP_INT32(fmt_SIGN);
	wave_hdr.Subchunk1Size = ENDSWAP_INT32(16);
	wave_hdr.AudioFormat = ENDSWAP_INT16(wave_hdr.AudioFormat);
	wave_hdr.NumChannels = ENDSWAP_INT16(wave_hdr.NumChannels);
	wave_hdr.SampleRate = ENDSWAP_INT32(wave_hdr.SampleRate);
	wave_hdr.ByteRate = ENDSWAP_INT32(wave_hdr.ByteRate);
	wave_hdr.BlockAlign = ENDSWAP_INT16(wave_hdr.BlockAlign);
	wave_hdr.BitsPerSample = ENDSWAP_INT16(wave_hdr.BitsPerSample);
	subchunk_hdr.SubchunkID = ENDSWAP_INT32(data_SIGN);
	subchunk_hdr.SubchunkSize = ENDSWAP_INT32(subchunk_hdr.SubchunkSize);

	// write WAVE header
	if (!WriteFile(fd, &wave_hdr, sizeof(wave_hdr), &result, NULL) ||
		result != sizeof(wave_hdr))
	{
		errNo = TTALib::WRITE_ERROR;
		return false;
	}
	// write Subchunk header
	if (!WriteFile(fd, &subchunk_hdr, sizeof(subchunk_hdr), &result, NULL) || 
		result != sizeof(subchunk_hdr))
	{
		errNo = TTALib::WRITE_ERROR;
		return false;
	}
	return true;
}

bool TTALib::WaveFile::Read(long *data, long byte_size, unsigned long *len)
{
    unsigned long res;
    unsigned char *buffer, *src;
	long *dst = data;

	errNo = TTALib::TTA_NO_ERROR;
	if (!(src = buffer = (unsigned char *)calloc(*len, byte_size)))
	{
		errNo = TTALib::MEMORY_ERROR;
		return false;
	}

	if (!ReadFile (fd, buffer, *len * byte_size, &res, NULL))
	{
		errNo = TTALib::READ_ERROR;
		return false;
	}

	switch (byte_size) {
	case 1: for (; src < buffer + res; dst++)
				*dst = (signed long) *src++ - 0x80;
			break;
	case 2: for (; src < buffer + res; dst++) {
				*dst = (unsigned char) *src++;
				*dst |= (signed char) *src++ << 8;
			}
			break;
	case 3: for (; src < buffer + res; dst++) {
				*dst = (unsigned char) *src++;
				*dst |= (unsigned char) *src++ << 8;
				*dst |= (signed char) *src++ << 16;
			}
			break;
	case 4: for (; src < buffer + res; dst += 2) {
				*dst = (unsigned char) *src++;
				*dst |= (unsigned char) *src++ << 8;
				*dst |= (unsigned char) *src++ << 16;
				*dst |= (signed char) *src++ << 24;
			}
			break;
	}

	*len = res / byte_size;
    free(buffer);

    return true;
}

bool TTALib::WaveFile::Write(long *data, long byte_size, long num_chan, unsigned long *len)
{
    unsigned long res;
    unsigned char *buffer, *dst;
	long *src = data;

	errNo = TTALib::TTA_NO_ERROR;
	if (!(dst = buffer = (unsigned char *)calloc(*len * num_chan, byte_size)))
	{
		errNo = TTALib::MEMORY_ERROR;
		return false;
	}

	switch (byte_size) {
	case 1: for (; src < data + (*len * num_chan); src++)
				*dst++ = (unsigned char) (*src + 0x80);
			break;
	case 2: for (; src < data + (*len * num_chan); src++) {
				*dst++ = (unsigned char) *src;
				*dst++ = (unsigned char) (*src >> 8);
			}
			break;
	case 3: for (; src < data + (*len * num_chan); src++) {
				*dst++ = (unsigned char) *src;
				*dst++ = (unsigned char) (*src >> 8);
				*dst++ = (unsigned char) (*src >> 16);
			}
			break;
	case 4: for (; src < data + (*len * num_chan * 2); src += 2) {
				*dst++ = (unsigned char) *src;
				*dst++ = (unsigned char) (*src >> 8);
				*dst++ = (unsigned char) (*src >> 16);
				*dst++ = (unsigned char) (*src >> 24);
			}
			break;
    }
	
    if (!WriteFile (fd, buffer, *len * num_chan * byte_size, &res, NULL) 
		|| res != *len * num_chan * byte_size)
	{
		errNo = TTALib::WRITE_ERROR;
		return false;
	}
	
	*len = res / byte_size;
    free(buffer);

    return true;
}

TTALib::TTAError TTALib::WaveFile::GetErrNo () const
{
	return errNo;
}

void TTALib::WaveFile::Close ()
{
	errNo = TTALib::TTA_NO_ERROR;
	CloseHandle (fd);
	fd = INVALID_HANDLE_VALUE;
}
