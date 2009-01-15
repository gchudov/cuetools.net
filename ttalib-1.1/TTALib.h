/*
 * TTALib.h
 *
 * Description: TTA library interface
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

#ifndef TTALIB_H_
#define TTALIB_H_

#include "TTAError.h"
#include "WavFile.h"
#include <windows.h>

#pragma pack(push)
#pragma pack(1)

#ifdef _WINDLL
	#define DllExport   __declspec( dllexport )
#else
	#define DllExport
#endif

#define WAVE_FORMAT_PCM	1
#define WAVE_FORMAT_IEEE_FLOAT 3

namespace TTALib
{
	struct DllExport TTAStat 
	{
		double ratio;
		unsigned long input_bytes; 
		unsigned long output_bytes;
	};

	class TTAReader;
	class TTAWriter;

	class DllExport TTAEncoder
	{
		TTAWriter *writer;
		HANDLE hFile;
		TTAStat stat;

	public:
		TTAEncoder(const char *filename, 
					bool append,
					unsigned short AudioFormat, 
					unsigned short NumChannels, 
					unsigned short BitsPerSample,
					unsigned long SampleRate,
					unsigned long DataLength);
		TTAEncoder(HANDLE hInFile, 
					bool append,
					unsigned short AudioFormat, 
					unsigned short NumChannels, 
					unsigned short BitsPerSample,
					unsigned long SampleRate,
					unsigned long DataLength);

		~TTAEncoder();
		
		bool CompressBlock (long *buf, long bufLen);

		TTAStat GetStat ();
	};

	class DllExport TTADecoder
	{
		TTAReader *reader;
		HANDLE hFile;
		TTAStat stat;

	public:
		TTADecoder (const char *filename);
		TTADecoder (HANDLE hInFile);
		~TTADecoder();

		long GetBlock (long **buf);

		long GetAudioFormat ();
		long GetNumChannels ();
		long GetBitsPerSample ();
		long GetSampleRate ();
		long GetDataLength ();

		TTAStat GetStat ();
	};

	typedef bool (*TTACALLBACK)(const TTAStat &stat, void *uParam);
	DllExport const char *GetErrStr (TTAError err);

	DllExport TTAError CopyId3Header (HANDLE hInFile, HANDLE hOutFile, bool CopyID3v2Tag);
	DllExport TTAError Wav2TTA (const char *infile, const char *outfile, 
		bool CopyID3v2Tag=true, TTACALLBACK TTACallback=NULL, void *uParam=NULL);
	DllExport TTAError TTA2Wav (const char *infile, const char *outfile,
		bool CopyID3v2Tag=true, TTACALLBACK TTACallback=NULL, void *uParam=NULL);
	DllExport TTAError TTATest (const char *infile,
					TTACALLBACK TTACallback=NULL, void *uParam=NULL);
}

#pragma pack(pop)
#endif // TTALIB_H_
