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
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Specialized;
using namespace System::Security::Cryptography;
using namespace APETagsDotNet;

#include <stdio.h>
#include <memory.h>
#include "wavpack.h"
#include <string.h>

namespace WavPackDotNet {
	int write_block(void *id, void *data, int32_t length);

	public ref class WavPackReader {
	public:
		WavPackReader(String^ path) {
			IntPtr pathChars;
			char errorMessage[256];

			_path = path;

			pathChars = Marshal::StringToHGlobalUni(path);
			size_t pathLen = wcslen ((const wchar_t*)pathChars.ToPointer())+1;
			wchar_t * pPath = new wchar_t[pathLen];
			memcpy ((void*) pPath, (const wchar_t*)pathChars.ToPointer(), pathLen*sizeof(wchar_t));
			Marshal::FreeHGlobal(pathChars);

			_wpc = WavpackOpenFileInput (pPath, errorMessage, OPEN_WVC, 0);
			if (_wpc == NULL) {
				throw gcnew Exception("Unable to initialize the decoder.");
			}

			_bitsPerSample = WavpackGetBitsPerSample(_wpc);
			_channelCount = WavpackGetNumChannels(_wpc);
			_sampleRate = WavpackGetSampleRate(_wpc);
			_sampleCount = WavpackGetNumSamples(_wpc);
			_sampleOffset = 0;
		}

		~WavPackReader()
		{
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

		property Int32 Length {
			Int32 get() {
				return _sampleCount;
			}
		}

		property Int32 Position {
			Int32 get() {
				return _sampleOffset;
			}
			void set(Int32 offset) {
				_sampleOffset = offset;
				if (!WavpackSeekSample(_wpc, offset)) {
					throw gcnew Exception("Unable to seek.");
				}
			}
		}

		property Int32 Remaining {
			Int32 get() {
				return _sampleCount - _sampleOffset;
			}
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

		void Close() {
			_wpc = WavpackCloseFile(_wpc);
		}

		void Read(array<Int32, 2>^ sampleBuffer, Int32 sampleCount) {
			pin_ptr<Int32> pSampleBuffer = &sampleBuffer[0, 0];
			int samplesRead;
			
			samplesRead = WavpackUnpackSamples(_wpc, pSampleBuffer, sampleCount);
			_sampleOffset += samplesRead;
			
			if (samplesRead != sampleCount) {
				throw gcnew Exception("Decoder returned a different number of samples than requested.");
			}
		}

	private:
		WavpackContext *_wpc;
		NameValueCollection^ _tags;
		Int32 _sampleCount, _sampleOffset;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		String^ _path;
	};

	public ref class WavPackWriter {
	public:
		WavPackWriter(String^ path, Int32 bitsPerSample, Int32 channelCount, Int32 sampleRate) {
			IntPtr pathChars;

			if ((channelCount != 1) && (channelCount != 2)) {
				throw gcnew Exception("Only stereo and mono audio formats are allowed.");
			}

			_path = path;
			_tags = gcnew NameValueCollection();

			_compressionMode = 1;
			_extraMode = 0;

			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;

			pathChars = Marshal::StringToHGlobalUni(path);
			_hFile = _wfopen((const wchar_t*)pathChars.ToPointer(), L"w+b");
			Marshal::FreeHGlobal(pathChars);
			if (!_hFile) {
				throw gcnew Exception("Unable to open file.");
			}
		}

		void Close() {
			if (_md5Sum)
			{
				_md5hasher->TransformFinalBlock (gcnew array<unsigned char>(1), 0, 0);
				pin_ptr<unsigned char> md5_digest = &_md5hasher->Hash[0];
				WavpackStoreMD5Sum (_wpc, md5_digest);
			}

			WavpackFlushSamples(_wpc);
			_wpc = WavpackCloseFile(_wpc);
			fclose(_hFile);

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

		property Int32 CompressionMode {
			Int32 get() {
				return _compressionMode;
			}
			void set(Int32 value) {
				if ((value < 0) || (value > 3)) {
					throw gcnew Exception("Invalid compression mode.");
				}
				_compressionMode = value;
			}
		}

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

		property bool MD5Sum {
			bool get() {
				return _md5Sum;
			}
			void set(bool value) {
				_md5Sum = value;
			}
		}

		void UpdateHash(array<unsigned char>^ buff, Int32 len) 
		{
			if (!_initialized) Initialize();

			if (!_md5Sum || !_md5hasher)
				throw gcnew Exception("MD5 not enabled.");
			_md5hasher->TransformBlock (buff, 0, len,  buff, 0);
		}

		void Write(array<Int32, 2>^ sampleBuffer, Int32 sampleCount) {
			if (!_initialized) Initialize();

			pin_ptr<Int32> pSampleBuffer = &sampleBuffer[0, 0];
			if (!WavpackPackSamples(_wpc, (int32_t*)pSampleBuffer, sampleCount)) {
				throw gcnew Exception("An error occurred while encoding.");
			}

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
		FILE *_hFile;
		bool _initialized;
		WavpackContext *_wpc;
		Int32 _finalSampleCount, _samplesWritten;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		Int32 _compressionMode, _extraMode;
		NameValueCollection^ _tags;
		String^ _path;
		bool _md5Sum;
		MD5^ _md5hasher;

		void Initialize() {
			WavpackConfig config;

			_wpc = WavpackOpenFileOutput(write_block, _hFile, NULL);
			if (!_wpc) {
				throw gcnew Exception("Unable to create the encoder.");
			}

			memset(&config, 0, sizeof(WavpackConfig));
			config.bits_per_sample = _bitsPerSample;
			config.bytes_per_sample = (_bitsPerSample + 7) / 8;
			config.num_channels = _channelCount;
			config.channel_mask = 5 - _channelCount;
			config.sample_rate = _sampleRate;
			if (_compressionMode == 0) config.flags |= CONFIG_FAST_FLAG;
			if (_compressionMode == 2) config.flags |= CONFIG_HIGH_FLAG;
			if (_compressionMode == 3) config.flags |= CONFIG_HIGH_FLAG | CONFIG_VERY_HIGH_FLAG;
			if (_extraMode != 0) {
				config.flags |= CONFIG_EXTRA_MODE;
				config.xmode = _extraMode;
			}
			if (_md5Sum)
			{
				_md5hasher = gcnew MD5CryptoServiceProvider ();
				config.flags |= CONFIG_MD5_CHECKSUM;
			}

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
}