/*
 * WavFile.h
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

#pragma once
#define RIFF_SIGN		0x46464952
#define WAVE_SIGN		0x45564157
#define fmt_SIGN		0x20746D66
#define data_SIGN		0x61746164

#ifdef _WINDLL
	#define DllExport   __declspec( dllexport )
#else
	#define DllExport
#endif


namespace TTALib
{
	class DllExport WaveFile 
	{
		TTAError errNo;
		
	public:
		HANDLE fd;
		struct {
			unsigned long ChunkID;
			unsigned long ChunkSize;
			unsigned long Format;
			unsigned long Subchunk1ID;
			unsigned long Subchunk1Size;
			unsigned short AudioFormat;
			unsigned short NumChannels;
			unsigned long SampleRate;
			unsigned long ByteRate;
			unsigned short BlockAlign;
			unsigned short BitsPerSample;
		} wave_hdr;
		struct {
			unsigned long SubchunkID;
			unsigned long SubchunkSize;
		} subchunk_hdr;	


		WaveFile ();
		~WaveFile ();

		HANDLE Create (const char *filename);
		HANDLE Open (const char *filename);
		
		bool ReadHeaders ();
		bool Read(long *data, long byte_size, unsigned long *len);
		
		bool WriteHeaders ();
		bool Write(long *data, long byte_size, long num_chan, unsigned long *len);

		void Close ();
	
		TTAError GetErrNo () const;
	};
};
