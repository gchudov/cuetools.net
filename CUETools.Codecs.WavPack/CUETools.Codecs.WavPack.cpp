// ****************************************************************************
// 
// Copyright (c) 2006-2007 Moitah (moitah@yahoo.com)
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//   * Neither the name of the author nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 
// ****************************************************************************

using namespace System;
using namespace System::ComponentModel;
using namespace System::Runtime::InteropServices;
using namespace System::Security::Cryptography;
using namespace System::IO;
using namespace CUETools::Codecs;

#include <stdio.h>
#include <memory.h>
#include "wavpack.h"
#include <string.h>

namespace CUETools { namespace Codecs { namespace WavPack {
	int write_block(void *id, void *data, int32_t length);

	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate int32_t DecoderReadDelegate(void *id, void *data, int32_t bcount);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate uint32_t DecoderTellDelegate(void *id);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate int DecoderSeekDelegate(void *id, uint32_t pos);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate int DecoderSeekRelativeDelegate(void *id, int32_t delta, int mode);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate int DecoderPushBackDelegate(void *id, int c);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate uint32_t DecoderLengthDelegate(void *id);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate int DecoderCanSeekDelegate(void *id);

	[AudioDecoderClass("libwavpack", "wv", 1)]
	public ref class WavPackReader : public IAudioSource 
	{
	public:
		WavPackReader(String^ path, Stream^ IO, Stream^ IO_WVC) 
		{
		    Initialize (path, IO, IO_WVC);
		}

		WavPackReader(String^ path, Stream^ IO)		    
		{
		    Initialize (path, IO, nullptr);
		}

		void Initialize(String^ path, Stream^ IO, Stream^ IO_WVC)
		{
			char errorMessage[256];

			_readDel = gcnew DecoderReadDelegate (this, &WavPackReader::ReadCallback);
			_tellDel = gcnew DecoderTellDelegate (this, &WavPackReader::TellCallback);
			_seekDel = gcnew DecoderSeekDelegate (this, &WavPackReader::SeekCallback);
			_seekRelDel = gcnew DecoderSeekRelativeDelegate (this, &WavPackReader::SeekRelCallback);
			_pushBackDel = gcnew DecoderPushBackDelegate (this, &WavPackReader::PushBackCallback);
			_lengthDel = gcnew DecoderLengthDelegate (this, &WavPackReader::LengthCallback);
			_canSeekDel = gcnew DecoderCanSeekDelegate (this, &WavPackReader::CanSeekCallback);

			ioReader = new WavpackStreamReader;
			ioReader->read_bytes = (int32_t (*)(void *, void *, int32_t)) Marshal::GetFunctionPointerForDelegate(_readDel).ToPointer();
			ioReader->get_pos = (uint32_t (*)(void *)) Marshal::GetFunctionPointerForDelegate(_tellDel).ToPointer();
			ioReader->set_pos_abs = (int (*)(void *, uint32_t)) Marshal::GetFunctionPointerForDelegate(_seekDel).ToPointer();
			ioReader->set_pos_rel = (int (*)(void *, int32_t, int)) Marshal::GetFunctionPointerForDelegate(_seekRelDel).ToPointer();
			ioReader->push_back_byte = (int (*)(void *, int)) Marshal::GetFunctionPointerForDelegate(_pushBackDel).ToPointer();
			ioReader->get_length = (uint32_t (*)(void *)) Marshal::GetFunctionPointerForDelegate(_lengthDel).ToPointer();
			ioReader->can_seek = (int (*)(void *)) Marshal::GetFunctionPointerForDelegate(_canSeekDel).ToPointer();
			ioReader->write_bytes = NULL;

			_IO_ungetc = _IO_WVC_ungetc = -1;

			_path = path;

			_IO = (IO != nullptr) ? IO : gcnew FileStream (path, FileMode::Open, FileAccess::Read, FileShare::Read);
			_IO_WVC = (IO != nullptr) ? IO_WVC : System::IO::File::Exists (path+"c") ? gcnew FileStream (path+"c", FileMode::Open, FileAccess::Read, FileShare::Read) : nullptr;

			_wpc = WavpackOpenFileInputEx (ioReader, "v", _IO_WVC != nullptr ? "c" : NULL, errorMessage, OPEN_WVC, 0);
			if (_wpc == NULL) {
				throw gcnew Exception("Unable to initialize the decoder.");
			}

			pcm = gcnew AudioPCMConfig(
			    WavpackGetBitsPerSample(_wpc), 
			    WavpackGetNumChannels(_wpc), 
			    (int)WavpackGetSampleRate(_wpc));
			_sampleCount = WavpackGetNumSamples(_wpc);
			_sampleOffset = 0;
		}

		~WavPackReader()
		{
			delete ioReader;
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
			Int64 get() {
				return _sampleOffset;
			}
			void set(Int64 offset) {
				_sampleOffset = offset;
				if (!WavpackSeekSample(_wpc, offset)) {
					throw gcnew Exception("Unable to seek.");
				}
			}
		}

		virtual property Int64 Remaining {
			Int64 get() {
				return _sampleCount - _sampleOffset;
			}
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		virtual void Close() 
		{
			if (_wpc != NULL)
				_wpc = WavpackCloseFile(_wpc);
			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
			if (_IO_WVC != nullptr) 
			{
				_IO_WVC->Close ();
				_IO_WVC = nullptr;
			}
		}

		virtual int Read(AudioBuffer^ sampleBuffer, int maxLength)
		{
			sampleBuffer->Prepare(this, maxLength);

			pin_ptr<Int32> pSampleBuffer = &sampleBuffer->Samples[0, 0];
			int samplesRead = WavpackUnpackSamples(_wpc, pSampleBuffer, sampleBuffer->Length);
			_sampleOffset += samplesRead;
			if (samplesRead != sampleBuffer->Length)
				throw gcnew Exception("Decoder returned a different number of samples than requested.");
			return sampleBuffer->Length;
		}

	private:
		WavpackContext *_wpc;
		Int32 _sampleCount, _sampleOffset;
		AudioPCMConfig^ pcm;
		String^ _path;
		Stream^ _IO;
		Stream^ _IO_WVC;
		DecoderReadDelegate^ _readDel;
		DecoderTellDelegate^ _tellDel;
		DecoderSeekDelegate^ _seekDel;
		DecoderSeekRelativeDelegate^ _seekRelDel;
		DecoderPushBackDelegate^ _pushBackDel;
		DecoderLengthDelegate^ _lengthDel;
		DecoderCanSeekDelegate^ _canSeekDel;
		array<unsigned char>^ _readBuffer;
		int _IO_ungetc, _IO_WVC_ungetc;
		WavpackStreamReader* ioReader;

		int32_t ReadCallback (void *id, void *data, int32_t bcount)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			int IO_ungetc = (*(char*)id=='c') ? _IO_WVC_ungetc : _IO_ungetc;
			int unget_len = 0;

			if (IO_ungetc != -1)
			{
				*(unsigned char*)data = (unsigned char) IO_ungetc;
				if (IO == _IO)
					_IO_ungetc = -1;
				else
					_IO_WVC_ungetc = -1;
				bcount --;
				if (!bcount)
					return 1;
				data = 1 + (unsigned char*)data;
				unget_len = 1;
			}

			if (_readBuffer == nullptr || _readBuffer->Length < bcount)
				_readBuffer = gcnew array<unsigned char>(bcount < 0x4000 ? 0x4000 : bcount);		
			int len = IO->Read (_readBuffer, 0, bcount);
			if (len) Marshal::Copy (_readBuffer, 0, (IntPtr)data, len);
			return len + unget_len;
		}

		uint32_t TellCallback(void *id)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			return IO->Position;
		}

		int SeekCallback (void *id, uint32_t pos)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			IO->Position = pos;
			return 0;
		}

		int SeekRelCallback (void *id, int32_t delta, int mode)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			switch (mode)
			{
			case SEEK_SET:
				IO->Seek (delta, System::IO::SeekOrigin::Begin);
				break;
			case SEEK_END:
				IO->Seek (delta, System::IO::SeekOrigin::End);
				break;
			case SEEK_CUR:
				IO->Seek (delta, System::IO::SeekOrigin::Current);
				break;
			default:
				return -1;
			}			
			return 0;
		}

		int PushBackCallback (void *id, int c)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			if (IO == _IO)
			{
				if (_IO_ungetc != -1)
					throw gcnew Exception("Double PushBackCallback unsupported.");
				_IO_ungetc = c;
			} else
			{
				if (_IO_WVC_ungetc != -1)
					throw gcnew Exception("Double PushBackCallback unsupported.");
				_IO_WVC_ungetc = c;
			}

			return 0;
		}

		uint32_t LengthCallback (void *id)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			return IO->Length;
		}

		int CanSeekCallback(void *id)
		{
			Stream^ IO = (*(char*)id=='c') ? _IO_WVC : _IO;
			return IO->CanSeek;
		}
	};

	public ref class WavPackWriterSettings : AudioEncoderSettings
	{
	    public:
		WavPackWriterSettings() 
			: AudioEncoderSettings("fast normal high high+", "normal")
		{ 
		    _md5Sum = true;
		    _extraMode = 0;
		}

		[DefaultValue(0)]
		[DisplayName("ExtraMode")]
		property Int32 ExtraMode {
			Int32 get() {
				return _extraMode;
			}
			void set(Int32 value) {
				if ((value < 0) || (value > 6)) {
					throw gcnew Exception("Invalid extra mode.");
				}
				_extraMode = value;
			}
		}

		[DefaultValue(true)]
		[DisplayName("MD5")]
		[Description("Calculate MD5 hash for audio stream")]
		property bool MD5Sum {
			bool get() {
				return _md5Sum;
			}
			void set(bool value) {
				_md5Sum = value;
			}
		}

	    private:
		bool _md5Sum;
		Int32 _extraMode;
	};

	[AudioEncoderClass("libwavpack", "wv", true, 1, WavPackWriterSettings::typeid)]
	public ref class WavPackWriter : IAudioDest 
	{
	public:
		WavPackWriter(String^ path, AudioPCMConfig^ pcm)
		{
			_settings = gcnew WavPackWriterSettings();

			_pcm = pcm;

			if (_pcm->ChannelCount != 1 && _pcm->ChannelCount != 2)
				throw gcnew Exception("Only stereo and mono audio formats are allowed.");
			if (_pcm->BitsPerSample < 16 || _pcm->BitsPerSample > 24)
				throw gcnew Exception("Bits per sample must be 16..24.");

			_path = path;

			_blockSize = 0;

			IntPtr pathChars = Marshal::StringToHGlobalUni(path);
			_hFile = _wfopen((const wchar_t*)pathChars.ToPointer(), L"w+b");
			Marshal::FreeHGlobal(pathChars);
			if (!_hFile) {
				throw gcnew Exception("Unable to open file.");
			}
		}

		virtual void Close() 
		{
		    if (_settings->MD5Sum)
		    {
			_md5hasher->TransformFinalBlock (gcnew array<unsigned char>(1), 0, 0);
			pin_ptr<unsigned char> md5_digest = &_md5hasher->Hash[0];
			WavpackStoreMD5Sum (_wpc, md5_digest);
		    }

		    WavpackFlushSamples(_wpc);
		    _wpc = WavpackCloseFile(_wpc);
		    fclose(_hFile);

		    if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount))
			    throw gcnew Exception("Samples written differs from the expected sample count.");
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
				_blockSize = value;
			}
		}

		virtual property AudioPCMConfig^ PCM
		{
			AudioPCMConfig^ get() { return _pcm;  }
		}

		virtual void Write(AudioBuffer^ sampleBuffer) 
		{
			if (!_initialized) 
				Initialize();

			sampleBuffer->Prepare(this);

			if (_settings->MD5Sum)
				UpdateHash(sampleBuffer->Bytes, sampleBuffer->ByteLength);

			if ((_pcm->BitsPerSample & 7) != 0)
			{
				if (_shiftedSampleBuffer == nullptr || _shiftedSampleBuffer.GetLength(0) < sampleBuffer->Length)
				    _shiftedSampleBuffer = gcnew array<int,2>(sampleBuffer->Length, _pcm->ChannelCount);
				int shift = 8 - (_pcm->BitsPerSample & 7);
				int ch = _pcm->ChannelCount;
				for (int i = 0; i < sampleBuffer->Length; i++)
					for (int c = 0; c < ch; c++)
					    _shiftedSampleBuffer[i,c] = sampleBuffer->Samples[i,c] << shift;
				pin_ptr<Int32> pSampleBuffer = &_shiftedSampleBuffer[0, 0];
				if (!WavpackPackSamples(_wpc, (int32_t*)pSampleBuffer, sampleBuffer->Length))
					throw gcnew Exception("An error occurred while encoding.");
			} else
			{
				pin_ptr<Int32> pSampleBuffer = &sampleBuffer->Samples[0, 0];
				if (!WavpackPackSamples(_wpc, (int32_t*)pSampleBuffer, sampleBuffer->Length))
					throw gcnew Exception("An error occurred while encoding.");
			}

			_samplesWritten += sampleBuffer->Length;
		}

		virtual property String^ Path 
		{
			String^ get() { 
				return _path; 
			} 
		}

		virtual property __int64 Padding
		{
			void set(__int64 value) {
			}
		}

		virtual property AudioEncoderSettings^ Settings
		{
			AudioEncoderSettings^ get()
			{
			    return _settings;
			}
			
			void set(AudioEncoderSettings^ value)
			{
			    if (value == nullptr || value->GetType() != WavPackWriterSettings::typeid)
				throw gcnew Exception(String::Format("Unsupported options: {0}", value));
			    _settings = (WavPackWriterSettings^)value;
			}
		}

		void UpdateHash(array<unsigned char>^ buff, Int32 len) 
		{
			if (!_initialized) Initialize();

			if (!_settings->MD5Sum || !_md5hasher)
				throw gcnew Exception("MD5 not enabled.");
			_md5hasher->TransformBlock (buff, 0, len,  buff, 0);
		}

	private:
		FILE *_hFile;
		bool _initialized;
		WavpackContext *_wpc;
		Int32 _finalSampleCount, _samplesWritten;
		Int32 _blockSize;
		String^ _path;
		MD5^ _md5hasher;
		array<int,2>^ _shiftedSampleBuffer;
		AudioPCMConfig^ _pcm;
		WavPackWriterSettings^ _settings;

		void Initialize() {
			WavpackConfig config;

			_wpc = WavpackOpenFileOutput(write_block, _hFile, NULL);
			if (!_wpc) {
				throw gcnew Exception("Unable to create the encoder.");
			}

			memset(&config, 0, sizeof(WavpackConfig));
			config.bits_per_sample = _pcm->BitsPerSample;
			config.bytes_per_sample = (_pcm->BitsPerSample + 7) / 8;
			config.num_channels = _pcm->ChannelCount;
			config.channel_mask = 5 - _pcm->ChannelCount;
			config.sample_rate = _pcm->SampleRate;
			Int32 _compressionMode = _settings->EncoderModeIndex;
			if (_compressionMode == 0) config.flags |= CONFIG_FAST_FLAG;
			if (_compressionMode == 2) config.flags |= CONFIG_HIGH_FLAG;
			if (_compressionMode == 3) config.flags |= CONFIG_HIGH_FLAG | CONFIG_VERY_HIGH_FLAG;
			if (_settings->ExtraMode != 0) 
			{
			    config.flags |= CONFIG_EXTRA_MODE;
			    config.xmode = _settings->ExtraMode;
			}
			if (_settings->MD5Sum)
			{
			    _md5hasher = gcnew MD5CryptoServiceProvider ();
			    config.flags |= CONFIG_MD5_CHECKSUM;
			}
			config.block_samples = (int)_blockSize;
			if (_blockSize > 0 && _blockSize < 2048)
				config.flags |= CONFIG_MERGE_BLOCKS;

			if (!WavpackSetConfiguration(_wpc, &config, (_finalSampleCount == 0) ? -1 : _finalSampleCount)) {
				throw gcnew Exception("Invalid configuration setting.");
			}

			if (!WavpackPackInit(_wpc)) {
				throw gcnew Exception("Unable to initialize the encoder.");
			}

			_initialized = true;
		}
	};

#pragma unmanaged
	int write_block(void *id, void *data, int32_t length) {
		return (fwrite(data, 1, length, (FILE*)id) == length);
	}
}}}
