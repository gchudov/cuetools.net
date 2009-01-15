/*
 * TTALib.cpp
 *
 * Description: TTA library functions
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

#include "TTALib.h"
#include "ttacommon.h"
#include "filters3.h"
#include "ttawriter.h"
#include "ttareader.h"
#include "WavFile.h"
#include "TTAError.h"
#include "TTATester.h"

const char *TTAErrorsStr[] = {
	"no errors found",
	"not compatible file format",
	"file is corrupted",
	"file(s) not found",
	"problem creating directory",
	"can't open file",
	"can't write to output file",
	"can't read from input file",
	"insufficient memory available",
	"operation canceled"
};

// ******************* library functions prototypes *********************

const char *
TTALib::GetErrStr (TTALib::TTAError err) 
{
	return TTAErrorsStr[err];
}

TTALib::TTAEncoder::TTAEncoder(const char *filename, 
					bool append,
					unsigned short AudioFormat, 
					unsigned short NumChannels, 
					unsigned short BitsPerSample,
					unsigned long SampleRate,
					unsigned long DataLength)
{
	long offset = 0;

	if (!append)
		hFile = CreateFile (filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	else
	{
		hFile = CreateFile (filename, GENERIC_WRITE, 0, NULL, OPEN_ALWAYS,
			FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL);
		offset = SetFilePointer (hFile, 0, NULL, FILE_END);
	}

	if (hFile == INVALID_HANDLE_VALUE)
		throw TTAException (OPEN_ERROR);


	try {
		writer = new TTAWriter(hFile, offset, AudioFormat, 
		NumChannels, BitsPerSample, SampleRate, DataLength);
	}

	catch (std::exception ex)
	{
		CloseHandle (hFile);
		throw ex;
	}
}

TTALib::TTAEncoder::TTAEncoder(HANDLE hInFile, 
					bool append,
					unsigned short AudioFormat, 
					unsigned short NumChannels, 
					unsigned short BitsPerSample,
					unsigned long SampleRate,
					unsigned long DataLength)
					: hFile (INVALID_HANDLE_VALUE)
{
	long offset = 0;

	if (hInFile == INVALID_HANDLE_VALUE)
		throw TTAException (OPEN_ERROR);

	if (append)
		offset = SetFilePointer (hInFile, 0, NULL, FILE_END);

	writer = new TTAWriter(hInFile, offset, AudioFormat, 
		NumChannels, BitsPerSample, SampleRate, DataLength);
}

TTALib::TTAEncoder::~TTAEncoder()
{
	if (writer) delete writer;
	if (hFile != INVALID_HANDLE_VALUE)
		CloseHandle (hFile);
}

bool TTALib::TTAEncoder::CompressBlock (long *buf, long bufLen)
{
	return writer->CompressBlock (buf, bufLen);
}

TTALib::TTAStat TTALib::TTAEncoder::GetStat ()
{
	stat.input_bytes = writer->input_byte_count;
	stat.output_bytes = writer->output_byte_count;
	stat.ratio = writer->output_byte_count / (1. + writer->input_byte_count);
	return stat;	
}

TTALib::TTADecoder::TTADecoder (const char *filename)
{
	struct {
		unsigned char id[3];
		unsigned short version;
		unsigned char flags;
		unsigned char size[4];
	} __ATTRIBUTE_PACKED__ id3v2;
	unsigned long result;

	if ((hFile = CreateFile (filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
		throw TTAException (OPEN_ERROR);

	// skip ID3V2 header
	if (!ReadFile (hFile, &id3v2, sizeof(id3v2), &result, NULL))
	{
		CloseHandle (hFile);
		throw TTAException (READ_ERROR);
	}

	if (id3v2.id[0] == 'I' && 
		id3v2.id[1] == 'D' && 
		id3v2.id[2] == '3') {
			long len;

			if (id3v2.size[0] & 0x80) {
				throw TTAException (FILE_ERROR);
			}

			len = (id3v2.size[0] & 0x7f);
			len = (len << 7) | (id3v2.size[1] & 0x7f);
			len = (len << 7) | (id3v2.size[2] & 0x7f);
			len = (len << 7) | (id3v2.size[3] & 0x7f);
			len += 10;
			if (id3v2.flags & (1 << 4)) len += 10;

			SetFilePointer (hFile, len, NULL, FILE_BEGIN);
	} else SetFilePointer (hFile, 0, NULL, FILE_BEGIN);

	try 
	{
		reader = new TTAReader(hFile);
	}

	catch (std::exception ex)
	{
		CloseHandle (hFile);
		throw ex;
	}
}

TTALib::TTADecoder::TTADecoder (HANDLE hInFile)
	: hFile(INVALID_HANDLE_VALUE)
{
	reader = new TTAReader(hInFile);
}

TTALib::TTADecoder::~TTADecoder()
{
	if (reader)
		delete reader;
	if (hFile != INVALID_HANDLE_VALUE)
		CloseHandle (hFile);
}

long TTALib::TTADecoder::GetBlock (long **buf)
{
	return reader->GetBlock (buf);
}

TTALib::TTAStat TTALib::TTADecoder::GetStat ()
{	
	stat.input_bytes = reader->input_byte_count;
	stat.output_bytes = reader->output_byte_count;
	stat.ratio = reader->output_byte_count / (1. + reader->input_byte_count);
	return stat;	
}

long TTALib::TTADecoder::GetAudioFormat ()
{
	return reader->ttahdr.AudioFormat;
}

long TTALib::TTADecoder::GetNumChannels ()
{
	return reader->ttahdr.NumChannels;
}

long TTALib::TTADecoder::GetBitsPerSample ()
{
	return reader->ttahdr.BitsPerSample;
}

long TTALib::TTADecoder::GetSampleRate ()
{
	return reader->ttahdr.SampleRate;
}

long TTALib::TTADecoder::GetDataLength ()
{
	return reader->ttahdr.DataLength;
}

// ************************* basic functions *****************************

void rice_init(adapt *rice, unsigned long k0, unsigned long k1)
{
	rice->k0 = k0;
	rice->k1 = k1;
	rice->sum0 = shift_16[k0];
	rice->sum1 = shift_16[k1];
}

void encoder_init(encoder *tta, long nch, long byte_size) 
{
	long *fset = flt_set[byte_size - 1];
	long i;

	for (i = 0; i < nch; i++) {
		filter_init(&tta[i].fst, fset[0], fset[1]);
		rice_init(&tta[i].rice, 10, 10);
		tta[i].last = 0;
	}
}

TTALib::TTAError TTALib::CopyId3Header (HANDLE hInFile, HANDLE hOutFile, bool CopyID3v2Tag)
{
	struct {
		unsigned char id[3];
		unsigned short version;
		unsigned char flags;
		unsigned char size[4];
	} __ATTRIBUTE_PACKED__ id3v2;
	unsigned long data_len, offset = 0,	result;

	if (!ReadFile (hInFile, &id3v2, sizeof(id3v2), &result, NULL))
		return TTALib::READ_ERROR;

	if (id3v2.id[0] == 'I' && 
		id3v2.id[1] == 'D' && 
		id3v2.id[2] == '3') {
			char buffer[512];

			if (id3v2.size[0] & 0x80) 
				return TTALib::FILE_ERROR;

			offset = (id3v2.size[0] & 0x7f);
			offset = (offset << 7) | (id3v2.size[1] & 0x7f);
			offset = (offset << 7) | (id3v2.size[2] & 0x7f);
			offset = (offset << 7) | (id3v2.size[3] & 0x7f);
			if (id3v2.flags & (1 << 4)) offset += 10;
			data_len = offset, offset += 10;

			// write ID3V2 header
			if (CopyID3v2Tag)
			{
				if (!WriteFile(hOutFile, &id3v2, sizeof(id3v2), &result, NULL) ||
					result != sizeof (id3v2))
					return TTALib::WRITE_ERROR;

				while (data_len > 0) {
					unsigned long len = (data_len > sizeof(buffer))? sizeof(buffer):data_len;
					if (!ReadFile (hInFile, buffer, len, &result, NULL)) 
						return TTALib::READ_ERROR;
					if (!WriteFile(hOutFile, buffer, len, &result, NULL) || 
						result != len)
						return TTALib::WRITE_ERROR;
					data_len -= len;
				}
			} else SetFilePointer (hInFile, offset, NULL, FILE_BEGIN);
		} else SetFilePointer (hInFile, 0, NULL, FILE_BEGIN);
	return TTALib::TTA_NO_ERROR;
}

TTALib::TTAError TTALib::Wav2TTA (const char *infile, const char *outfile,
					bool CopyID3v2Tag, TTACALLBACK TTACallback, void *uParam)
{
	HANDLE hInFile, hOutFile;
	unsigned long data_size, byte_size, data_len, framelen, is_float;
	unsigned long offset = 0;
	long *data = NULL;	
	TTALib::TTAEncoder *encoder;
	TTALib::TTAError err;
	TTALib::WaveFile wav;

	if ((hInFile = wav.Open (infile)) == INVALID_HANDLE_VALUE)
		return wav.GetErrNo ();

	if ((hOutFile = CreateFile (outfile, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
	{
		wav.Close ();
		return TTALib::OPEN_ERROR;
	}

	// copy ID3V2 header if present		
	if ((err = TTALib::CopyId3Header (hInFile, hOutFile, CopyID3v2Tag)) != TTALib::TTA_NO_ERROR)
	{
		wav.Close ();
		CloseHandle (hOutFile);
		DeleteFile (outfile);
		return err;
	}	

	offset = SetFilePointer (hOutFile, 0, NULL, FILE_CURRENT);

	if (!wav.ReadHeaders ())
	{
		DeleteFile (outfile);
		return wav.GetErrNo ();
	}

	// check for supported formats
	if ((wav.wave_hdr.ChunkID != RIFF_SIGN) ||
		(wav.wave_hdr.Format != WAVE_SIGN) ||
		(wav.wave_hdr.Subchunk1ID != fmt_SIGN) ||
		(wav.wave_hdr.Subchunk1Size > wav.wave_hdr.ChunkSize) ||
		(wav.wave_hdr.NumChannels == 0) ||
		(wav.wave_hdr.BitsPerSample > MAX_BPS)) {
		wav.Close ();
		CloseHandle (hOutFile);
		DeleteFile (outfile);		
		return TTALib::FORMAT_ERROR;
	}
	
	switch (wav.wave_hdr.AudioFormat) {
	case WAVE_FORMAT_IEEE_FLOAT: is_float = 1; break;
	case WAVE_FORMAT_PCM: is_float = 0; break;
	default: 
		wav.Close ();
		CloseHandle (hOutFile);
		DeleteFile (outfile);		
		return TTALib::FORMAT_ERROR;
	}

	if ((is_float && wav.wave_hdr.BitsPerSample != MAX_BPS) ||
		(!is_float && wav.wave_hdr.BitsPerSample == MAX_BPS)) {
		wav.Close ();
		CloseHandle (hOutFile);
		DeleteFile (outfile);
		return TTALib::FORMAT_ERROR;
	}

	data_size = wav.subchunk_hdr.SubchunkSize;
	byte_size = (wav.wave_hdr.BitsPerSample + 7) / 8;
	data_len = data_size / (byte_size * wav.wave_hdr.NumChannels);
	framelen = (long) (FRAME_TIME * wav.wave_hdr.SampleRate) - 7;
    
	err = TTALib::TTA_NO_ERROR;
	try {
		data = new long [(wav.wave_hdr.NumChannels << is_float) * framelen * sizeof(long)];

		encoder = new TTALib::TTAEncoder (hOutFile, true, 
		wav.wave_hdr.AudioFormat, wav.wave_hdr.NumChannels, wav.wave_hdr.BitsPerSample,
		wav.wave_hdr.SampleRate, data_len);
		unsigned long len;
		for (;;)
		{
			len = framelen * wav.wave_hdr.NumChannels;
			if (!wav.Read (data, byte_size, &len))
				break;
			if (!encoder->CompressBlock (data, len / wav.wave_hdr.NumChannels))
				break;
			if (TTACallback)
				if(!TTACallback(encoder->GetStat(), uParam))
				{
					err = TTALib::TTA_CANCELED;
					break;
				}
		} 
		delete encoder;
		CloseHandle (hOutFile);
		wav.Close ();	
		delete [] data;

	}

	catch (TTALib::TTAException ex)
	{
		wav.Close ();		
		delete [] data;
		CloseHandle (hOutFile);
		DeleteFile (outfile);
		return ex.GetErrNo ();
	}

	catch (...)
	{
		CloseHandle (hOutFile);
		wav.Close ();
		DeleteFile (outfile);
		return TTALib::MEMORY_ERROR;
	}

	return err;
}

TTALib::TTAError TTALib::TTA2Wav (const char *infile, const char *outfile,
					bool CopyID3v2Tag, TTACALLBACK TTACallback, void *uParam)
{
	HANDLE hInFile, hOutFile;
	long *buf;
	unsigned long byte_size, data_size, buflen;
	TTALib::TTADecoder *decoder;
	TTALib::TTAError err;
	TTALib::WaveFile wav;

	if ((hInFile = CreateFile (infile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
		return TTALib::OPEN_ERROR;
	
	if (((hOutFile = wav.Create (outfile))) == INVALID_HANDLE_VALUE)
	{
		CloseHandle (hInFile);
		return wav.GetErrNo();
	}

	// copy ID3V2 header if present
	if ((err = CopyId3Header (hInFile, hOutFile, CopyID3v2Tag)) != TTALib::TTA_NO_ERROR)
	{
		CloseHandle (hInFile);
		wav.Close ();
		DeleteFile (outfile);
		return err;
	}

	err = TTALib::TTA_NO_ERROR;
	try 
	{
		decoder = new TTADecoder (hInFile);

		byte_size = (decoder->GetBitsPerSample() + 7) / 8;
		data_size = decoder->GetDataLength() * byte_size * decoder->GetNumChannels();

		wav.wave_hdr.ChunkSize = data_size + 36;
		wav.wave_hdr.AudioFormat =(unsigned short)decoder->GetAudioFormat();
		wav.wave_hdr.NumChannels = (unsigned short)decoder->GetNumChannels();
		wav.wave_hdr.SampleRate = decoder->GetSampleRate();
		wav.wave_hdr.BitsPerSample = (unsigned short)decoder->GetBitsPerSample();
		wav.wave_hdr.ByteRate = decoder->GetSampleRate() * byte_size * decoder->GetNumChannels();
		wav.wave_hdr.BlockAlign = (unsigned short) (decoder->GetNumChannels() * byte_size);
		wav.subchunk_hdr.SubchunkSize = ENDSWAP_INT32(data_size);

		wav.WriteHeaders ();

		while ((buflen = decoder->GetBlock (&buf)) > 0)
		{
			if (!wav.Write (buf, byte_size, decoder->GetNumChannels(), &buflen))
				return wav.GetErrNo();
			if (TTACallback)
				if(!TTACallback(decoder->GetStat(), uParam))
				{
					err = TTALib::TTA_CANCELED;
					break;
				}
		}
		wav.Close ();
		delete decoder;
		CloseHandle (hInFile);
	}

	catch (TTAException ex)
	{
		DeleteFile (outfile);
		CloseHandle (hInFile);
		return ex.GetErrNo ();
	}

	return err;
}

TTALib::TTAError TTALib::TTATest (const char *infile,
					TTACALLBACK TTACallback, void *uParam)
{	
	unsigned long result, len = 0;
	TTALib::TTAError err;
	TTALib::TTAStat stat = {0, 0, 0};
	TTATester *tester;
	TTAHeader ttahdr;
	HANDLE hFile;

	if ((hFile = CreateFile (infile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN, NULL)) == INVALID_HANDLE_VALUE)
		return TTALib::OPEN_ERROR;

	// copy ID3V2 header if present
	if ((err = CopyId3Header (hFile, 0, false)) != TTALib::TTA_NO_ERROR)
	{
		CloseHandle (hFile);
		return err;
	}

	tester = NULL;
	try 
	{
		tester  = new TTATester(hFile);

		tester->GetHeader (&ttahdr);
		while (tester->TestFrame())
		{
			stat.input_bytes = tester->input_byte_count + len;

			if (TTACallback)
				if(!TTACallback(stat, uParam))
					return TTALib::TTA_CANCELED;
		}		
	}
	catch (TTAException ex)
	{
		if (tester)
			delete tester;
		CloseHandle (hFile);
		return ex.GetErrNo();
	}
	catch (...)
	{
		return TTALib::MEMORY_ERROR;
	}
	delete tester;

	CloseHandle (hFile);

	return err;
}
