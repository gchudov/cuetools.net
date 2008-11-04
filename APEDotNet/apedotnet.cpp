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

		property String^ Path { 
			String^ get() { 
				return _path; 
			} 
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
		    interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, 0];

		    unsigned short * pAPEBuffer = (unsigned short *) pBuffer;
		    unsigned short * pAPEBufferEnd = (unsigned short *) pBuffer + 2 * nBlocksRetrieved;

		    while (pAPEBuffer < pAPEBufferEnd) {
			    *(pMyBuffer++) = *(pAPEBuffer++);
			    *(pMyBuffer++) = *(pAPEBuffer++);
		    }

			sampleBuffer = _sampleBuffer;
			_sampleOffset += nBlocksRetrieved;

			return sampleCount;
		}

	private:
		IAPEDecompress * pAPEDecompress;

		NameValueCollection^ _tags;
		Int64 _sampleCount, _sampleOffset;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		int nBlockAlign;
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

	public ref class APEWriter {
	public:
		APEWriter(String^ path, Int32 bitsPerSample, Int32 channelCount, Int32 sampleRate) {

			if ((channelCount != 1) && (channelCount != 2)) {
				throw gcnew Exception("Only stereo and mono audio formats are allowed.");
			}

			_path = path;
			_tags = gcnew NameValueCollection();

			_compressionLevel = COMPRESSION_LEVEL_NORMAL;

			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);

			int nRetVal;
			pAPECompress = CreateIAPECompress (&nRetVal);
			if (!pAPECompress) {
				throw gcnew Exception("Unable to open file.");
			}
		}

		void Close() {
			if (pAPECompress) 
			{
				pAPECompress->Finish (NULL, 0, 0);
				delete pAPECompress;
				pAPECompress = NULL;
			}

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount)) {
				throw gcnew Exception("Samples written differs from the expected sample count.");
			}

			if (_tags->Count > 0)
			{
				APETagDotNet^ apeTag = gcnew APETagDotNet (_path, true, false);
				apeTag->SetStringTags (_tags, true);
				apeTag->Save();
				apeTag->Close();
				_tags->Clear ();
			}
		}

		property Int32 FinalSampleCount {
			Int32 get() {
				return _finalSampleCount;
			}
			void set(Int32 value) {
				if (value < 0) {
					throw gcnew Exception("Invalid final sample count.");
				}
				if (_initialized) {
					throw gcnew Exception("Final sample count cannot be changed after encoding begins.");
				}
				_finalSampleCount = value;
			}
		}

		property Int32 CompressionLevel {
			Int32 get() {
				return _compressionLevel;
			}
			void set(Int32 value) {
				if ((value < 1) || (value > 5)) {
					throw gcnew Exception("Invalid compression mode.");
				}
				_compressionLevel = value * 1000;
			}
		}

		void Write(array<unsigned char>^ sampleBuffer, UInt32 sampleCount) {
			if (!_initialized) Initialize();
			pin_ptr<unsigned char> pSampleBuffer = &sampleBuffer[0];
			if (pAPECompress->AddData (pSampleBuffer, sampleCount * _blockAlign))
				throw gcnew Exception("An error occurred while encoding.");
			_samplesWritten += sampleCount;
		}

		property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		void SetTags (NameValueCollection^ tags) {
			_tags = tags;
		}

	private:
		IAPECompress * pAPECompress;
		bool _initialized;
		Int32 _finalSampleCount, _samplesWritten;
		Int32 _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		Int32 _compressionLevel;
		NameValueCollection^ _tags;
		String^ _path;

		void Initialize() {
			IntPtr pathChars;
			int res;
			WAVEFORMATEX waveFormat;

			pathChars = Marshal::StringToHGlobalUni(_path);
			
			FillWaveFormatEx (&waveFormat, _sampleRate, _bitsPerSample, _channelCount);
			res = pAPECompress->Start ((const wchar_t*)pathChars.ToPointer(), 
				&waveFormat, 
				(_finalSampleCount == 0) ? MAX_AUDIO_BYTES_UNKNOWN : _finalSampleCount * _blockAlign,
				_compressionLevel, 
				NULL, 
				CREATE_WAV_HEADER_ON_DECOMPRESSION);
			Marshal::FreeHGlobal(pathChars);
			if (res)
			{
				throw gcnew Exception("Unable to create the encoder.");
			}

			_initialized = true;
		}
	};
}
