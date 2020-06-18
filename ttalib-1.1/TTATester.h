#pragma once
#include "BitReader.h"

namespace TTALib 
{

	class TTATester :
		public BitReader
	{
		unsigned char *data;
		unsigned long *seek_table, *fst;
		long fframes;
	public:

		TTATester(HANDLE hFile) 
			: BitReader(hFile), data(NULL), seek_table(NULL), fframes(-1)
		{		
		}

		virtual ~TTATester(void)
		{
			if (seek_table)
				delete [] seek_table;
			if (data)
				delete [] data;
		}

		virtual void GetHeader (TTAHeader *ttahdr)
		{
			BitReader::GetHeader (ttahdr);

			long framelen  = (long) (FRAME_TIME * ttahdr->SampleRate);
			long framesize =  framelen * ttahdr->NumChannels * 
					(ttahdr->BitsPerSample + 7) / 8 + 4;

			long lastlen = ttahdr->DataLength % framelen;
			fframes = ttahdr->DataLength / framelen + (lastlen ? 1 : 0);

			data = new unsigned char[framesize];
			seek_table = new unsigned long[fframes + 1];

			if (!GetSeekTable (seek_table, fframes + 1))
				throw TTAException (FILE_ERROR);
			fst = seek_table;
		}


		virtual bool TestFrame ()
		{
			unsigned long frame_crc32, result;

			if (fst >= seek_table + fframes)
				return false;

			if (!ReadFile(hInFile, data, *fst, &result, NULL) ||
				result != *fst)
				throw TTAException (READ_ERROR);
			else input_byte_count += result;

			CopyMemory(&frame_crc32, data + (result - 4), 4);		
			if (crc32(data, *fst - 4) != ENDSWAP_INT32(frame_crc32))
				throw TTAException (FILE_ERROR);
			
			fst ++;			
			return true;
		}
	};

};