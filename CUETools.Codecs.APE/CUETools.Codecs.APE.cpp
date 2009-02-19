// This is the main DLL file.

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace System::IO;
using namespace CUETools::Codecs;

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
#include "IO.h"

namespace CUETools { namespace Codecs { namespace APE {

	class CWinFileIO : public CIO
	{
	public:

		// construction / destruction
		CWinFileIO(GCHandle gchIO, GCHandle gchBuffer)
		{
			_gchIO = gchIO;
			_gchBuffer = gchBuffer;
		}
		~CWinFileIO()
		{
		}

		// open / close
		int Open(const wchar_t * pName)
		{
			throw gcnew Exception("CIO::Open Unsupported.");
		}
		int Close()
		{
			throw gcnew Exception("CIO::Close Unsupported.");
		}
	    
		// read / write
		int Read(void * pBuffer, unsigned int nBytesToRead, unsigned int * pBytesRead);
		int Write(const void * pBuffer, unsigned int nBytesToWrite, unsigned int * pBytesWritten);
	    
		// seek
		int Seek(int nDistance, unsigned int nMoveMode);
	    
		// other functions
		int SetEOF()
		{
			throw gcnew Exception("CIO::SetEOF unsupported.");
		}

		// creation / destruction
		int Create(const wchar_t * pName)
		{
			throw gcnew Exception("CIO::Create unsupported.");
		}

		int Delete()
		{
			throw gcnew Exception("CIO::Delete unsupported.");
		}

		// attributes
		int GetPosition();
		int GetSize();

		int GetName(wchar_t * pBuffer)
		{
			throw gcnew Exception("CIO::GetName unsupported.");
		}

	private:
		GCHandle _gchIO;
		GCHandle _gchBuffer;
	};

	public ref class APEReader : public IAudioSource
	{
	public:
		APEReader(String^ path, Stream^ IO) 
		{
			pAPEDecompress = NULL;
			_sampleOffset = 0;
			_bufferOffset = 0;
			_bufferLength = 0;
			_path = path;

			int nRetVal = 0;
	
			_IO = (IO != nullptr) ? IO : gcnew FileStream (path, FileMode::Open, FileAccess::Read, FileShare::Read);
			_readBuffer = gcnew array<unsigned char>(0x4000);
			_gchIO = GCHandle::Alloc(_IO);
			_gchReadBuffer = GCHandle::Alloc(_readBuffer);
			_winFileIO = new CWinFileIO(_gchIO, _gchReadBuffer);
			pAPEDecompress = CreateIAPEDecompressEx (_winFileIO, &nRetVal);
			if (!pAPEDecompress) {
				throw gcnew Exception("Unable to open file.");
			}
			
			_sampleRate = pAPEDecompress->GetInfo (APE_INFO_SAMPLE_RATE, 0, 0);
			_bitsPerSample = pAPEDecompress->GetInfo (APE_INFO_BITS_PER_SAMPLE, 0, 0);
			_channelCount = pAPEDecompress->GetInfo (APE_INFO_CHANNELS, 0, 0);

			// make a buffer to hold 16384 blocks of audio data
			nBlockAlign = pAPEDecompress->GetInfo (APE_INFO_BLOCK_ALIGN, 0, 0);
			_samplesBuffer = gcnew array<unsigned char> (16384 * nBlockAlign);

			// loop through the whole file
			_sampleCount = pAPEDecompress->GetInfo (APE_DECOMPRESS_TOTAL_BLOCKS, 0, 0); // * ?
		}

		~APEReader ()
		{
			if (_winFileIO)
				delete _winFileIO;
			if (_gchIO.IsAllocated) 
				_gchIO.Free();		
			if (_gchReadBuffer.IsAllocated)
				_gchReadBuffer.Free();
		}

		virtual property Int32 BitsPerSample {
			Int32 get() {
				return _bitsPerSample;
			}
		}

		virtual property Int32 ChannelCount {
			Int32 get() {
				return _channelCount;
			}
		}

		virtual property Int32 SampleRate {
			Int32 get() {
				return _sampleRate;
			}
		}

		virtual property UInt64 Length {
			UInt64 get() {
				return _sampleCount;
			}
		}

		virtual property UInt64 Position 
		{
			UInt64 get() {
				return _sampleOffset - SamplesInBuffer;
			}
			void set(UInt64 offset) {
				_sampleOffset = offset;
				_bufferOffset = 0;
				_bufferLength = 0;
				if (pAPEDecompress->Seek ((int) offset /*? */))
					throw gcnew Exception("Unable to seek.");
			}
		}

		virtual property UInt64 Remaining {
			UInt64 get() {
				return _sampleCount - _sampleOffset + SamplesInBuffer;
			}
		}

		virtual void Close() 
		{
			if (pAPEDecompress) 
			{
				delete pAPEDecompress;
				pAPEDecompress = NULL;
			}		
			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		virtual array<Int32, 2>^ Read(array<Int32, 2>^ buff)
		{
			return AudioSamples::Read(this, buff);
		}

		virtual UInt32 Read([Out] array<Int32, 2>^ buff, UInt32 sampleCount)
		{
			UInt32 buffOffset = 0;
			UInt32 samplesNeeded = sampleCount;

			while (samplesNeeded != 0) 
			{
				if (SamplesInBuffer == 0) 
				{
					int nBlocksRetrieved;
					pin_ptr<unsigned char> pSampleBuffer = &_samplesBuffer[0];
					if (pAPEDecompress->GetData ((char *) pSampleBuffer, 16384, &nBlocksRetrieved))
						throw gcnew Exception("An error occurred while decoding.");
					_bufferOffset = 0;
					_bufferLength = nBlocksRetrieved;
					_sampleOffset += nBlocksRetrieved;
				}
				UInt32 copyCount = Math::Min(samplesNeeded, SamplesInBuffer);
				AudioSamples::BytesToFLACSamples_16(_samplesBuffer, _bufferOffset*nBlockAlign, buff, buffOffset, copyCount, _channelCount);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}
			return sampleCount;
		}

	private:
		IAPEDecompress * pAPEDecompress;

		Int64 _sampleCount, _sampleOffset;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		UInt32 _bufferOffset, _bufferLength;
		int nBlockAlign;
		array<unsigned char>^ _samplesBuffer;
		String^ _path;
		Stream^ _IO;
		array<unsigned char>^ _readBuffer;
		CWinFileIO* _winFileIO;
		GCHandle _gchIO, _gchReadBuffer;

		property UInt32 SamplesInBuffer 
		{
			UInt32 get () 
			{
				return (UInt32) (_bufferLength - _bufferOffset);
			}
		}
	};

	public ref class APEWriter : IAudioDest
	{
	public:
		APEWriter(String^ path, Int32 bitsPerSample, Int32 channelCount, Int32 sampleRate) 
		{
			if (channelCount != 1 && channelCount != 2)
				throw gcnew Exception("Only stereo and mono audio formats are allowed.");
			if (bitsPerSample != 16 && bitsPerSample != 24)
				throw gcnew Exception("Monkey's Audio doesn't support selected bits per sample value.");

			_path = path;
			_winFileIO = NULL;

			_compressionLevel = COMPRESSION_LEVEL_NORMAL;

			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);

			int nRetVal;
			pAPECompress = CreateIAPECompress (&nRetVal);
			if (!pAPECompress)
				throw gcnew Exception("Unable to open APE compressor.");
		}

		~APEWriter()
		{
			if (_winFileIO)
				delete _winFileIO;
			if (_gchIO.IsAllocated) 
				_gchIO.Free();		
			if (_gchBuffer.IsAllocated)
				_gchBuffer.Free();
		}

		virtual void Close() 
		{
			if (pAPECompress) 
			{
				pAPECompress->Finish (NULL, 0, 0);
				delete pAPECompress;
				pAPECompress = NULL;
			}

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount)) {
				throw gcnew Exception("Samples written differs from the expected sample count.");
			}

			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
		}

		virtual void Delete()
		{
			try { Close (); } catch (Exception^) {}
			File::Delete(_path);
		}

		virtual property Int64 FinalSampleCount 
		{
			Int64 get() 
			{
				return _finalSampleCount;
			}
			void set(Int64 value) 
			{
				if (value < 0)
					throw gcnew Exception("Invalid final sample count.");
				if (_initialized)
					throw gcnew Exception("Final sample count cannot be changed after encoding begins.");
				_finalSampleCount = value;
			}
		}

		virtual property Int64 BlockSize
		{
			void set(Int64 value)
			{
			}
		}

		virtual property int BitsPerSample
		{
			int get() { return _bitsPerSample;  }
		}

		virtual void Write(array<Int32,2>^ buff, UInt32 sampleCount) 
		{
			if (_sampleBuffer == nullptr || _sampleBuffer.Length < sampleCount * _blockAlign)
				_sampleBuffer = gcnew array<unsigned char>(sampleCount * _blockAlign);
			AudioSamples::FLACSamplesToBytes(buff, 0, _sampleBuffer, 0, sampleCount, _channelCount, _bitsPerSample);
			if (!_initialized) Initialize();
			pin_ptr<unsigned char> pSampleBuffer = &_sampleBuffer[0];
			if (pAPECompress->AddData (pSampleBuffer, sampleCount * _blockAlign))
				throw gcnew Exception("An error occurred while encoding.");
			_samplesWritten += sampleCount;
		}

		virtual property String^ Path 
		{
			String^ get() { 
				return _path; 
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

	private:
		IAPECompress * pAPECompress;
		bool _initialized;
		Int32 _finalSampleCount, _samplesWritten;
		Int32 _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		Int32 _compressionLevel;
		String^ _path;
		Stream^ _IO;
		GCHandle _gchIO, _gchBuffer;
		CWinFileIO* _winFileIO;
		array<unsigned char>^ _writeBuffer;
		array<unsigned char>^ _sampleBuffer;

		void Initialize() {
			_IO = gcnew FileStream (_path, FileMode::Create, FileAccess::ReadWrite, FileShare::Read);
			_writeBuffer = gcnew array<unsigned char>(0x4000);

			_gchIO = GCHandle::Alloc(_IO);
			_gchBuffer = GCHandle::Alloc(_writeBuffer);
			_winFileIO = new CWinFileIO(_gchIO, _gchBuffer);

			WAVEFORMATEX waveFormat;
			FillWaveFormatEx (&waveFormat, _sampleRate, _bitsPerSample, _channelCount);

			int res = pAPECompress->StartEx (_winFileIO,
				&waveFormat, 
				(_finalSampleCount == 0) ? MAX_AUDIO_BYTES_UNKNOWN : _finalSampleCount * _blockAlign,
				_compressionLevel, 
				NULL, 
				CREATE_WAV_HEADER_ON_DECOMPRESSION);
			if (res)
				throw gcnew Exception("Unable to create the encoder.");

			_initialized = true;
		}
	};

	int CWinFileIO::Read(void * pBuffer, unsigned int nBytesToRead, unsigned int * pBytesRead)
	{
		array<unsigned char>^ buff = (array<unsigned char>^) _gchBuffer.Target;
		if (buff->Length < nBytesToRead)
		{
			Array::Resize (buff, nBytesToRead);
			_gchBuffer.Target = buff;
		}
		int len = ((Stream^)_gchIO.Target)->Read (buff, 0, nBytesToRead);
		if (len) Marshal::Copy (buff, 0, (IntPtr)pBuffer, len);
		*pBytesRead = len;
		return 0;
	}

	int CWinFileIO::Write(const void * pBuffer, unsigned int nBytesToWrite, unsigned int * pBytesWritten)
	{
		array<unsigned char>^ buff = (array<unsigned char>^) _gchBuffer.Target;
		if (buff->Length < nBytesToWrite)
		{
			Array::Resize (buff, nBytesToWrite);
			_gchBuffer.Target = buff;
		}
		Marshal::Copy ((IntPtr)(void*)pBuffer, buff, 0, nBytesToWrite);
		((Stream^)_gchIO.Target)->Write (buff, 0, nBytesToWrite);
		*pBytesWritten = nBytesToWrite;
		return 0;
	}

	int CWinFileIO::GetPosition()
	{
		return ((Stream^)_gchIO.Target)->Position;
	}

	int CWinFileIO::GetSize()
	{
		return ((Stream^)_gchIO.Target)->Length;
	}

	int CWinFileIO::Seek(int delta, unsigned int mode)
	{
		switch (mode)
		{
		case FILE_BEGIN:
			((Stream^)_gchIO.Target)->Seek (delta, System::IO::SeekOrigin::Begin);
			break;
		case FILE_END:
			((Stream^)_gchIO.Target)->Seek (delta, System::IO::SeekOrigin::End);
			break;
		case FILE_CURRENT:
			((Stream^)_gchIO.Target)->Seek (delta, System::IO::SeekOrigin::Current);
			break;
		default:
			return -1;
		}			
		return 0;
	}

#if 0
extern "C" 
{
    BOOL GetMMXAvailable();
    int CalculateDotProduct8(const short* pA, const short* pB, int nOrder);
};

	public ref class SSE2Functions
	{
	public:
		SSE2Functions ()
		{
			_haveSSE2 = GetMMXAvailable();
		}
		int SumInts (short*a, short*b, int count)
		{
			if (_haveSSE2 && count == 8)
				return CalculateDotProduct8(a, b, count);
			int sum = 0;
			for (int j = 0; j < count; j++)
				sum += a[j] * b[j];
			return sum;
		}
		int SumInts (int*a, short*b, int count)
		{
			int sum = 0;
			for (int j = 0; j < count; j++)
				sum += a[j] * b[j];
			return sum;
		}
	private:
		bool _haveSSE2;
	};
#endif
}}}
