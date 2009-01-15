// This is the main DLL file.

#include "stdafx.h"

#include "CUETools.Codecs.TTA.h"

typedef void * HANDLE;

#include "../TTALib-1.1/TTAReader.h"
#include "../TTALib-1.1/TTAError.h"

namespace CUETools { 
namespace Codecs { 
namespace TTA {

	static const char *TTAErrorsStr[] = {
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

	public ref class TTAReader : public IAudioSource
	{
	public:
		TTAReader(String^ path, Stream^ IO)
		{
			_tags = gcnew NameValueCollection();
			_sampleOffset = 0;
			_sampleBuffer = nullptr;
			_path = path;
			_bufferOffset = 0;
			_bufferLength = 0;

			_IO = (IO != nullptr) ? IO : gcnew FileStream (path, FileMode::Open, FileAccess::Read, FileShare::Read);

			// skip ID3v2


			if (_IO->ReadByte() == 'I' && _IO->ReadByte() == 'D' && _IO->ReadByte() == '3')
			{
				_IO->ReadByte();
				_IO->ReadByte();
				int flags = _IO->ReadByte();
				int sz = _IO->ReadByte();
				if (sz & 0x80)
					throw gcnew Exception("Invalid ID3 tag.");
				int offset = sz;
				offset = (offset << 7) | (_IO->ReadByte() & 0x7f);
				offset = (offset << 7) | (_IO->ReadByte() & 0x7f);
				offset = (offset << 7) | (_IO->ReadByte() & 0x7f);
				if (flags & (1 << 4))
					offset += 10;
				_IO->Position = 10 + offset;
			} else
				_IO->Position = 0;

			try 
			{
				_ttaReader = new TTALib::TTAReader((HANDLE)((FileStream^) _IO)->Handle);
			} catch (TTALib::TTAException ex)
			{
				throw gcnew Exception(String::Format("TTA decoder: {0}", gcnew String(TTAErrorsStr[ex.GetErrNo()])));
			}
			if (WAVE_FORMAT_PCM != _ttaReader->ttahdr.AudioFormat)
				throw gcnew Exception("floating point format not supported.");
			_channelCount = _ttaReader->ttahdr.NumChannels;
			_bitsPerSample = _ttaReader->ttahdr.BitsPerSample;
			_sampleRate = _ttaReader->ttahdr.SampleRate;
			_sampleCount = _ttaReader->ttahdr.DataLength;
		}

		~TTAReader ()
		{
		    Close ();
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

		virtual property UInt64 Position {
			UInt64 get() 
			{
				return _sampleOffset - SamplesInBuffer;
			}
			void set(UInt64 offset) 
			{
				_sampleOffset = offset;
				_bufferOffset = 0;
				_bufferLength = 0;
				throw gcnew Exception("Unable to seek.");
			}
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		virtual property NameValueCollection^ Tags {
			NameValueCollection^ get () {
				return _tags;
			}
			void set (NameValueCollection ^tags) {
				_tags = tags;
			}
		}

		virtual bool UpdateTags (bool preserveTime)
		{
		    return false;
		}

		virtual property UInt64 Remaining {
			UInt64 get() {
				return _sampleCount - _sampleOffset + SamplesInBuffer;
			}
		}

		virtual void Close() {
			if (_ttaReader) 
			{
				delete _ttaReader;
				_ttaReader = nullptr;
			}
			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
		}

		virtual array<Int32, 2>^ Read(array<Int32, 2>^ buff)
		{
			return AudioSamples::Read(this, buff);
		}

		void processBlock (long * buffer, int sampleCount)
		{
			if (_bufferLength > 0)
				throw gcnew Exception("Received unrequested samples.");

			if ((_sampleBuffer == nullptr) || (_sampleBuffer->GetLength(0) < sampleCount))
				_sampleBuffer = gcnew array<Int32, 2>(sampleCount, _channelCount);

			interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, 0];
			const long *pTTABuffer = buffer;
			const long *pTTABufferEnd = pTTABuffer + sampleCount * _channelCount;
			while (pTTABuffer < pTTABufferEnd) 
				*(pMyBuffer++) = *(pTTABuffer++);
			_bufferLength = sampleCount;
		}

		virtual UInt32 Read([Out] array<Int32, 2>^ buff, UInt32 sampleCount)
		{
			UInt32 buffOffset = 0;
			UInt32 samplesNeeded = sampleCount;

			while (samplesNeeded != 0) 
			{
				if (SamplesInBuffer == 0) 
				{
					_bufferOffset = 0;
					_bufferLength = 0;
					do
					{
						long * buf;
						int samplesInBuf = _ttaReader->GetBlock(&buf);
						if (samplesInBuf == 0)
							throw gcnew Exception("An error occurred while decoding.");
						processBlock (buf, samplesInBuf);
					} while (_bufferLength == 0);
					_sampleOffset += _bufferLength;
				}
				UInt32 copyCount = Math::Min(samplesNeeded, SamplesInBuffer);
				Array::Copy(_sampleBuffer, _bufferOffset * _channelCount, buff, buffOffset * _channelCount, copyCount * _channelCount);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}
			return sampleCount;
		}

	private:
		Int64 _sampleCount, _sampleOffset;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		array<Int32, 2>^ _sampleBuffer;
		array<unsigned char>^ _readBuffer;
		NameValueCollection^ _tags;
		String^ _path;
		Stream^ _IO;
		UInt32 _bufferOffset, _bufferLength;
		TTALib::TTAReader * _ttaReader;

		property UInt32 SamplesInBuffer {
			UInt32 get () 
			{
				return (UInt32) (_bufferLength - _bufferOffset);
			}
		}
	};
}
}
}
