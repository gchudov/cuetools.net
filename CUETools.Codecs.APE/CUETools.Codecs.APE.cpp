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

	[AudioDecoderClass("MAC_SDK", "ape", 1)]
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
			
			pcm = gcnew AudioPCMConfig(
			    pAPEDecompress->GetInfo (APE_INFO_BITS_PER_SAMPLE, 0, 0),
			    pAPEDecompress->GetInfo (APE_INFO_CHANNELS, 0, 0),
			    pAPEDecompress->GetInfo (APE_INFO_SAMPLE_RATE, 0, 0),
				(AudioPCMConfig::SpeakerConfig)0);

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

		virtual property AudioPCMConfig^ PCM {
			AudioPCMConfig^ get() {
				return pcm;
			}
		}

		virtual property Int64 Length {
			Int64 get() {
				return _sampleCount;
			}
		}

		virtual property Int64 Position 
		{
			Int64 get() {
				return _sampleOffset - SamplesInBuffer;
			}
			void set(Int64 offset) {
				_sampleOffset = offset;
				_bufferOffset = 0;
				_bufferLength = 0;
				if (pAPEDecompress->Seek ((int) offset /*? */))
					throw gcnew Exception("Unable to seek.");
			}
		}

		virtual property Int64 Remaining {
			Int64 get() {
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

		virtual int Read(AudioBuffer^ buff, int maxLength)
		{
			buff->Prepare(this, maxLength);

			Int32 buffOffset = 0;
			Int32 samplesNeeded = buff->Length;
			Int32 _channelCount = pcm->ChannelCount;

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
				Int32 copyCount = Math::Min(samplesNeeded, SamplesInBuffer);
				AudioBuffer::BytesToFLACSamples_16(_samplesBuffer, _bufferOffset*nBlockAlign, buff->Samples, buffOffset, copyCount, _channelCount);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}
			return buff->Length;
		}

                virtual property AudioDecoderSettings^ Settings {
                    AudioDecoderSettings^ get(void) {
                        return nullptr;
                    }
                }

	private:
		IAPEDecompress * pAPEDecompress;

		Int64 _sampleCount, _sampleOffset;
		AudioPCMConfig^ pcm;
		Int32 _bufferOffset, _bufferLength;
		int nBlockAlign;
		array<unsigned char>^ _samplesBuffer;
		String^ _path;
		Stream^ _IO;
		array<unsigned char>^ _readBuffer;
		CWinFileIO* _winFileIO;
		GCHandle _gchIO, _gchReadBuffer;

		property Int32 SamplesInBuffer 
		{
			Int32 get () 
			{
				return _bufferLength - _bufferOffset;
			}
		}
	};

	public ref class APEWriterSettings : public AudioEncoderSettings
	{
	    public:
		APEWriterSettings() 
			: AudioEncoderSettings("fast normal high extra insane", "high")
		{
		}
	};

	[AudioEncoderClass("MAC_SDK", "ape", true, 1, APEWriterSettings::typeid)]
	public ref class APEWriter : IAudioDest
	{
	public:
		APEWriter(String^ path, APEWriterSettings^ settings)
		{
			_settings = settings;

		    if (_settings->PCM->ChannelCount != 1 && _settings->PCM->ChannelCount != 2)
				throw gcnew Exception("Only stereo and mono audio formats are allowed.");
		    if (_settings->PCM->BitsPerSample != 16 && _settings->PCM->BitsPerSample != 24)
				throw gcnew Exception("Monkey's Audio doesn't support selected bits per sample value.");

		    _path = path;
		    _winFileIO = NULL;

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

		void DoClose()
		{
			if (pAPECompress) 
			{
				pAPECompress->Finish (NULL, 0, 0);
				delete pAPECompress;
				pAPECompress = NULL;
			}

			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
		}

		virtual void Close() 
		{
			DoClose();

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount)) {
				throw gcnew Exception("Samples written differs from the expected sample count.");
			}
		}

		virtual void Delete()
		{
			DoClose ();

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

		virtual void Write(AudioBuffer^ buff)
		{
			if (!_initialized) 
			    Initialize();
			buff->Prepare(this);
			pin_ptr<unsigned char> pSampleBuffer = &buff->Bytes[0];
			if (pAPECompress->AddData (pSampleBuffer, buff->ByteLength))
				throw gcnew Exception("An error occurred while encoding.");
			_samplesWritten += buff->Length;
		}

		virtual property String^ Path 
		{
			String^ get() { 
				return _path; 
			} 
		}

		virtual property AudioEncoderSettings^ Settings
		{
			AudioEncoderSettings^ get()
			{
			    return _settings;
			}
		}

	private:
		IAPECompress * pAPECompress;
		bool _initialized;
		Int32 _finalSampleCount, _samplesWritten;
		APEWriterSettings^ _settings;
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
			FillWaveFormatEx (&waveFormat, _settings->PCM->SampleRate, _settings->PCM->BitsPerSample, _settings->PCM->ChannelCount);

			Int32 _compressionLevel = (_settings->EncoderModeIndex + 1) * 1000;

			int res = pAPECompress->StartEx (_winFileIO,
				&waveFormat, 
				(_finalSampleCount == 0) ? MAX_AUDIO_BYTES_UNKNOWN : _finalSampleCount * _settings->PCM->BlockAlign,
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
