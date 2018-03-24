// This is the main DLL file.

#include "stdafx.h"

#include "CUETools.Codecs.TTA.h"

typedef void * HANDLE;

#include "../TTALib-1.1/TTAReader.h"
#include "../TTALib-1.1/TTAWriter.h"
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

	ref class AudioDecoder;

	public ref class DecoderSettings : public AudioDecoderSettings
	{
	public:
		DecoderSettings()
			: AudioDecoderSettings()
		{
		}

		virtual property String^ Name
		{
			String^ get() override { return "ttalib"; }
		}

		virtual property String^ Extension
		{
			String^ get() override { return "tta"; }
		}

		virtual property Type^ DecoderType
		{
			Type^ get() override { return AudioDecoder::typeid; }
		}

		virtual property int Priority
		{
			int get() override { return 1; }
		}
	};

	public ref class AudioDecoder : public IAudioSource
	{
	public:
		AudioDecoder(DecoderSettings^ settings, String^ path, Stream^ IO)
		{
			m_settings = settings;
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

			pcm = gcnew AudioPCMConfig((int)_ttaReader->ttahdr.BitsPerSample, (int)_ttaReader->ttahdr.NumChannels, (int)_ttaReader->ttahdr.SampleRate, (AudioPCMConfig::SpeakerConfig)0);
			_sampleCount = _ttaReader->ttahdr.DataLength;
		}

		~AudioDecoder ()
		{
		    Close ();
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

		virtual property Int64 Position {
			Int64 get() 
			{
				return _sampleOffset - SamplesInBuffer;
			}
			void set(Int64 offset) 
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

		virtual property Int64 Remaining {
			Int64 get() {
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

		void processBlock (long * buffer, int sampleCount)
		{
			if (_bufferLength > 0)
				throw gcnew Exception("Received unrequested samples.");

			if ((_sampleBuffer == nullptr) || (_sampleBuffer->GetLength(0) < sampleCount))
			    _sampleBuffer = gcnew array<Int32, 2>(sampleCount, pcm->ChannelCount);

			interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, 0];
			const long *pTTABuffer = buffer;
			const long *pTTABufferEnd = pTTABuffer + sampleCount * pcm->ChannelCount;
			while (pTTABuffer < pTTABufferEnd) 
				*(pMyBuffer++) = *(pTTABuffer++);
			_bufferLength = sampleCount;
		}

		virtual int Read(AudioBuffer^ buff, int maxLength)
		{
			buff->Prepare(this, maxLength);

			Int32 buffOffset = 0;
			Int32 samplesNeeded = buff->Length;

			while (samplesNeeded != 0) 
			{
				if (SamplesInBuffer == 0) 
				{
					_bufferOffset = 0;
					_bufferLength = 0;
					do
					{
						long * buf;
						int samplesInBuf;
						try
						{
							samplesInBuf = _ttaReader->GetBlock(&buf);
						} catch (TTALib::TTAException ex)
						{
							throw gcnew Exception(String::Format("TTA decoder: {0}", gcnew String(TTAErrorsStr[ex.GetErrNo()])));
						}
						if (samplesInBuf == 0)
							throw gcnew Exception("An error occurred while decoding.");
						processBlock (buf, samplesInBuf);
					} while (_bufferLength == 0);
					_sampleOffset += _bufferLength;
				}
				Int32 copyCount = Math::Min(samplesNeeded, SamplesInBuffer);
				Array::Copy(_sampleBuffer, _bufferOffset * pcm->ChannelCount, buff->Samples, buffOffset * pcm->ChannelCount, copyCount * pcm->ChannelCount);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}
			return buff->Length;
		}

                virtual property AudioDecoderSettings^ Settings {
                    AudioDecoderSettings^ get(void) {
                        return m_settings;
                    }
                }

	private:
		Int64 _sampleCount, _sampleOffset;
		AudioPCMConfig^ pcm;
		array<Int32, 2>^ _sampleBuffer;
		array<unsigned char>^ _readBuffer;
		String^ _path;
		Stream^ _IO;
		Int32 _bufferOffset, _bufferLength;
		TTALib::TTAReader * _ttaReader;
		DecoderSettings^ m_settings;

		property Int32 SamplesInBuffer {
			Int32 get () 
			{
				return _bufferLength - _bufferOffset;
			}
		}
	};

	ref class AudioEncoder;

	public ref class EncoderSettings : public AudioEncoderSettings
	{
	    public:
		EncoderSettings() : AudioEncoderSettings()
		{
		}

		virtual property String^ Name
		{
			String^ get() override { return "ttalib"; }
		}

		virtual property String^ Extension
		{
			String^ get() override { return "tta"; }
		}

		virtual property Type^ EncoderType
		{
			Type^ get() override { return AudioEncoder::typeid; }
		}

		virtual property bool Lossless
		{
			bool get() override { return true; }
		}

		virtual property int Priority
		{
			int get() override { return 1; }
		}
	};

	public ref class AudioEncoder : public IAudioDest
	{
	public:
		AudioEncoder(EncoderSettings^ settings, String^ path, Stream^ IO)
		{
			_settings = settings;

		    if (_settings->PCM->BitsPerSample < 16 || _settings->PCM->BitsPerSample > 24)
	    		throw gcnew Exception("Bits per sample must be 16..24.");

		    _initialized = false;
		    _sampleBuffer = nullptr;
		    _path = path;
		    _finalSampleCount = 0;
		    _samplesWritten = 0;
		}

		virtual void Close() {
			//FLAC__stream_encoder_finish(_encoder);
			//for (int i = 0; i < _metadataCount; i++) {
			//	FLAC__metadata_object_delete(_metadataList[i]);
			//}

			if (_ttaWriter)
			{
				try
				{
					delete _ttaWriter;
				} catch (TTALib::TTAException ex)
				{
					_ttaWriter = nullptr;
					throw gcnew Exception(String::Format("TTA encoder: {0}", gcnew String(TTAErrorsStr[ex.GetErrNo()])));
				}
				_ttaWriter = nullptr;
			}

			if (_IO)
				_IO->Close();

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount))
				throw gcnew Exception("Samples written differs from the expected sample count.");
		}

		virtual void Delete()
		{
			try { Close (); } catch (Exception^) {}
			File::Delete(_path);
		}

		virtual property Int64 FinalSampleCount {
			Int64 get() {
				return _finalSampleCount;
			}
			void set(Int64 value) {
				if (value < 0) {
					throw gcnew Exception("Invalid final sample count.");
				}
				if (_initialized) {
					throw gcnew Exception("Final sample count cannot be changed after encoding begins.");
				}
				_finalSampleCount = value;
			}
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		virtual void Write(AudioBuffer^ sampleBuffer) {
			if (!_initialized) Initialize();

			sampleBuffer->Prepare(this);

			if ((_sampleBuffer == nullptr) || (_sampleBuffer->Length < sampleBuffer->Length * _settings->PCM->ChannelCount))
				_sampleBuffer = gcnew array<long> (sampleBuffer->Length * _settings->PCM->ChannelCount);

			interior_ptr<Int32> pSampleBuffer = &sampleBuffer->Samples[0, 0];
			interior_ptr<long> pTTABuffer = &_sampleBuffer[0];
			for (int i = 0; i < sampleBuffer->Length * _settings->PCM->ChannelCount; i++)
				pTTABuffer[i] = pSampleBuffer[i];

			pin_ptr<long> buffer = &_sampleBuffer[0];
			try
			{
				_ttaWriter->CompressBlock(buffer, sampleBuffer->Length);
			} catch (TTALib::TTAException ex)
			{
				throw gcnew Exception(String::Format("TTA encoder: {0}", gcnew String(TTAErrorsStr[ex.GetErrNo()])));
			}

			_samplesWritten += sampleBuffer->Length;
		}

		virtual property AudioEncoderSettings^ Settings
		{
			AudioEncoderSettings^ get()
			{
			    return _settings;
			}
		}

	private:
		TTALib::TTAWriter* _ttaWriter;
		FileStream^ _IO;
		array<long>^ _sampleBuffer;
		bool _initialized;
		String^ _path;
		Int64 _finalSampleCount, _samplesWritten;
		AudioEncoderSettings^ _settings;

		void Initialize() 
		{
			if (!_finalSampleCount)
				throw gcnew Exception("FinalSampleCount not set.");

			_IO = gcnew FileStream (_path, FileMode::Create, FileAccess::Write, FileShare::Read);
			try 
			{
			    _ttaWriter = new TTALib::TTAWriter((HANDLE)_IO->Handle, 0, WAVE_FORMAT_PCM, _settings->PCM->ChannelCount, _settings->PCM->BitsPerSample, _settings->PCM->SampleRate, _finalSampleCount);
			} catch (TTALib::TTAException ex)
			{
				throw gcnew Exception(String::Format("TTA encoder: {0}", gcnew String(TTAErrorsStr[ex.GetErrNo()])));
			}
			_initialized = true;
		}
	};
}
}
}
