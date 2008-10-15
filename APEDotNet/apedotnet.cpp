// This is the main DLL file.

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace System::Collections::Specialized;
using namespace System::Runtime::InteropServices;
using namespace APETagsDotNet;

#ifndef _WAVEFORMATEX_
#define _WAVEFORMATEX_

#define BOOL int
#define TRUE 1
#define FALSE 0
#define HWND long

/*
 *  extended waveform format structure used for all non-PCM formats. this
 *  structure is common to all non-PCM formats.
 */
typedef struct tWAVEFORMATEX
{
    Int16        wFormatTag;         /* format type */
    Int16        nChannels;          /* number of channels (i.e. mono, stereo...) */
    Int32       nSamplesPerSec;     /* sample rate */
    Int32       nAvgBytesPerSec;    /* for buffer estimation */
    Int16        nBlockAlign;        /* block size of data */
    Int16        wBitsPerSample;     /* number of bits per sample of mono data */
    Int16        cbSize;             /* the count in bytes of the size of */
                                    /* extra information (after cbSize) */
} WAVEFORMATEX, *PWAVEFORMATEX, *NPWAVEFORMATEX, *LPWAVEFORMATEX;

#endif /* _WAVEFORMATEX_ */

#include "All.h"
#include "MACLib.h"

namespace APEDotNet {

	public ref class APEReader {
	public:
		APEReader(String^ path) {
			IntPtr pathChars;

			pAPEDecompress = NULL;
			_sampleOffset = 0;
			_samplesWaiting = false;
			_path = path;
			pBuffer = NULL;

			int nRetVal = 0;

			pathChars = Marshal::StringToHGlobalUni(path);
			size_t pathLen = wcslen ((const wchar_t*)pathChars.ToPointer())+1;
			wchar_t * pPath = new wchar_t[pathLen];
			memcpy ((void*) pPath, (const wchar_t*)pathChars.ToPointer(), pathLen*sizeof(wchar_t));
			Marshal::FreeHGlobal(pathChars);

			pAPEDecompress = CreateIAPEDecompress (pPath, &nRetVal);
			if (!pAPEDecompress) {
				throw gcnew Exception("Unable to open file.");
			}
			
			_sampleRate = pAPEDecompress->GetInfo (APE_INFO_SAMPLE_RATE, 0, 0);
			_bitsPerSample = pAPEDecompress->GetInfo (APE_INFO_BITS_PER_SAMPLE, 0, 0);
			_channelCount = pAPEDecompress->GetInfo (APE_INFO_CHANNELS, 0, 0);

			// make a buffer to hold 16384 blocks of audio data
			nBlockAlign = pAPEDecompress->GetInfo (APE_INFO_BLOCK_ALIGN, 0, 0);
			pBuffer = new unsigned char [16384 * nBlockAlign];

			// loop through the whole file
			_sampleCount = pAPEDecompress->GetInfo (APE_DECOMPRESS_TOTAL_BLOCKS, 0, 0); // * ?
		}

		~APEReader ()
		{
			if (pBuffer) delete [] pBuffer;
		}

		property Int32 BitsPerSample {
			Int32 get() {
				return _bitsPerSample;
			}
		}

		property Int32 ChannelCount {
			Int32 get() {
				return _channelCount;
			}
		}

		property Int32 SampleRate {
			Int32 get() {
				return _sampleRate;
			}
		}

		property Int64 Length {
			Int64 get() {
				return _sampleCount;
			}
		}

		property Int64 Position {
			Int64 get() {
				return _sampleOffset;
			}
			void set(Int64 offset) {
				_sampleOffset = offset;
				_samplesWaiting = false;
				if (pAPEDecompress->Seek ((int) offset /*? */))
					throw gcnew Exception("Unable to seek.");
			}
		}

		property Int64 Remaining {
			Int64 get() {
				return _sampleCount - _sampleOffset;
			}
		}

		void Close() {
			if (pAPEDecompress) delete pAPEDecompress;
			pAPEDecompress = NULL;
		}

		property NameValueCollection^ Tags {
			NameValueCollection^ get () {
				if (!_tags) 
				{
					APETagDotNet^ apeTag = gcnew APETagDotNet (_path, true, true);
					_tags = apeTag->GetStringTags (true);
					apeTag->Close ();
				}
				return _tags;
			}
			void set (NameValueCollection ^tags) {
				_tags = tags;
			}
		}

		Int32 Read([Out] array<Int32, 2>^% sampleBuffer) {
			int sampleCount;

			int nBlocksRetrieved;
			if (pAPEDecompress->GetData ((char *) pBuffer, 16384, &nBlocksRetrieved))
			    throw gcnew Exception("An error occurred while decoding.");

			sampleCount = nBlocksRetrieved;
			array<Int32,2>^ _sampleBuffer = gcnew array<Int32,2> (nBlocksRetrieved, 2);

			{
			    interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, 0];
			    unsigned short * pAPEBuffer = (unsigned short *) pBuffer;
			    unsigned short * pAPEBufferEnd = (unsigned short *) pBuffer + 2 * nBlocksRetrieved;

			    while (pAPEBuffer < pAPEBufferEnd) {
				    *(pMyBuffer++) = *(pAPEBuffer++);
				    *(pMyBuffer++) = *(pAPEBuffer++);
			    }
			}
#if 0
			for (int i = 0; i < nBlocksRetrieved; i++)
			{
			    _sampleBuffer[i,0] = pBuffer[i*4] + (pBuffer[i*4+1] << 8);
			    _sampleBuffer[i,1] = pBuffer[i*4+2] + (pBuffer[i*4+3] << 8);
			}
#endif
			sampleBuffer = _sampleBuffer;
			_sampleOffset += nBlocksRetrieved;
			_samplesWaiting = false;

			return sampleCount;
		}

	private:
		IAPEDecompress * pAPEDecompress;

		NameValueCollection^ _tags;
		Int64 _sampleCount, _sampleOffset;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		int nBlockAlign;
		bool _samplesWaiting;
		unsigned char * pBuffer;
		String^ _path;

#if 0
		APE__StreamDecoderWriteStatus WriteCallback(const APE__StreamDecoder *decoder,
			const APE__Frame *frame, const APE__int32 * const buffer[], void *client_data)
		{
			if ((_sampleBuffer == nullptr) || (_sampleBuffer->GetLength(0) != sampleCount)) {
				_sampleBuffer = gcnew array<Int32, 2>(sampleCount, _channelCount);
			}

			for (Int32 iChan = 0; iChan < _channelCount; iChan++) {
				interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, iChan];
				const APE__int32 *pAPEBuffer = buffer[iChan];
				const APE__int32 *pAPEBufferEnd = pAPEBuffer + sampleCount;

				while (pAPEBuffer < pAPEBufferEnd) {
					*pMyBuffer = *pAPEBuffer;
					pMyBuffer += _channelCount;
					pAPEBuffer++;
				}
			}
		}
#endif
	};

}
