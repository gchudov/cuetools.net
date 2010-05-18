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
using namespace System::Text;
using namespace System::IO;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace CUETools::Codecs;

#include "FLAC\all.h"
#include <string>

struct FLAC__Metadata_Iterator 
{
	int dummy;
};

struct FLAC__Metadata_Chain 
{
	int dummy;
};

extern "C" {
    FLAC_API FLAC__bool FLAC__stream_encoder_set_do_md5(FLAC__StreamEncoder *encoder, FLAC__bool value);
}

namespace CUETools { namespace Codecs { namespace FLAC {
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__StreamDecoderWriteStatus DecoderWriteDelegate(const FLAC__StreamDecoder *decoder, const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate void DecoderMetadataDelegate(const FLAC__StreamDecoder *decoder, const FLAC__StreamMetadata *metadata, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate void DecoderErrorDelegate(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__StreamDecoderReadStatus DecoderReadDelegate (const FLAC__StreamDecoder *decoder, FLAC__byte buffer[], size_t *bytes, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__StreamDecoderSeekStatus DecoderSeekDelegate (const FLAC__StreamDecoder *decoder, FLAC__uint64 absolute_byte_offset, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__StreamDecoderTellStatus DecoderTellDelegate (const FLAC__StreamDecoder *decoder, FLAC__uint64 *absolute_byte_offset, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__StreamDecoderLengthStatus DecoderLengthDelegate (const FLAC__StreamDecoder *decoder, FLAC__uint64 *stream_length, void *client_data);
	[UnmanagedFunctionPointer(CallingConvention::Cdecl)]
	public delegate FLAC__bool DecoderEofDelegate (const FLAC__StreamDecoder *decoder, void *client_data);

	[AudioDecoderClass("libFLAC", "flac")]
	public ref class FLACReader : public IAudioSource
	{
	public:
		FLACReader(String^ path, Stream^ IO)
		{
			_writeDel = gcnew DecoderWriteDelegate(this, &FLACReader::WriteCallback);
			_metadataDel = gcnew DecoderMetadataDelegate(this, &FLACReader::MetadataCallback);
			_errorDel = gcnew DecoderErrorDelegate(this, &FLACReader::ErrorCallback);
			_readDel = gcnew DecoderReadDelegate(this, &FLACReader::ReadCallback);
			_seekDel = gcnew DecoderSeekDelegate(this, &FLACReader::SeekCallback);
			_tellDel = gcnew DecoderTellDelegate(this, &FLACReader::TellCallback);
			_lengthDel = gcnew DecoderLengthDelegate(this, &FLACReader::LengthCallback);
			_eofDel = gcnew DecoderEofDelegate(this, &FLACReader::EofCallback);

			_decoderActive = false;

			_sampleOffset = 0;
			_sampleBuffer = nullptr;
			_path = path;
			_bufferOffset = 0;
			_bufferLength = 0;

			_IO = (IO != nullptr) ? IO : gcnew FileStream (path, FileMode::Open, FileAccess::Read, FileShare::Read);

			_decoder = FLAC__stream_decoder_new();

			if (!FLAC__stream_decoder_set_metadata_respond (_decoder, FLAC__METADATA_TYPE_VORBIS_COMMENT))
				throw gcnew Exception("unable to setup the decoder");

			if (!FLAC__stream_decoder_set_disable_asm(_decoder, true))//disableAsm))
				throw gcnew Exception("unable to setup the decoder");

			FLAC__StreamDecoderInitStatus st = FLAC__stream_decoder_init_stream(_decoder,
				(FLAC__StreamDecoderReadCallback)Marshal::GetFunctionPointerForDelegate(_readDel).ToPointer(),
				_IO->CanSeek?(FLAC__StreamDecoderSeekCallback)Marshal::GetFunctionPointerForDelegate(_seekDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderTellCallback)Marshal::GetFunctionPointerForDelegate(_tellDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderLengthCallback)Marshal::GetFunctionPointerForDelegate(_lengthDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderEofCallback)Marshal::GetFunctionPointerForDelegate(_eofDel).ToPointer():NULL,
				(FLAC__StreamDecoderWriteCallback)Marshal::GetFunctionPointerForDelegate(_writeDel).ToPointer(),
				(FLAC__StreamDecoderMetadataCallback)Marshal::GetFunctionPointerForDelegate(_metadataDel).ToPointer(),
				(FLAC__StreamDecoderErrorCallback)Marshal::GetFunctionPointerForDelegate(_errorDel).ToPointer(),
				NULL);

			if (st != FLAC__STREAM_DECODER_INIT_STATUS_OK)
				throw gcnew Exception(String::Format("unable to initialize the decoder: {0}", gcnew String(FLAC__StreamDecoderInitStatusString[st])));

			_decoderActive = true;

			if (!FLAC__stream_decoder_process_until_end_of_metadata(_decoder))
				throw gcnew Exception("unable to retrieve metadata");
		}

		~FLACReader ()
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
				if (!FLAC__stream_decoder_seek_absolute(_decoder, offset))
					throw gcnew Exception("unable to seek");
			}
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		//virtual bool UpdateTags (bool preserveTime)
		//{
		//    Close ();
		//    
		//    FLAC__Metadata_Chain* chain = FLAC__metadata_chain_new ();
		//    if (!chain) return false;

		//    IntPtr pathChars = Marshal::StringToHGlobalAnsi(_path);
		//    int res = FLAC__metadata_chain_read (chain, (const char*)pathChars.ToPointer());
		//    Marshal::FreeHGlobal(pathChars);
		//    if (!res) {
		//		FLAC__metadata_chain_delete (chain);
		//		return false;
		//    }
		//    FLAC__Metadata_Iterator* i = FLAC__metadata_iterator_new ();
		//    FLAC__metadata_iterator_init (i, chain);
		//    do {
		//		FLAC__StreamMetadata* metadata = FLAC__metadata_iterator_get_block (i);
		//		if (metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT)
		//			FLAC__metadata_iterator_delete_block (i, false);
		//    } while (FLAC__metadata_iterator_next (i));

		//    FLAC__StreamMetadata * vorbiscomment = FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT);
		//    for (int tagno = 0; tagno <_tags->Count; tagno++)
		//    {
		//		String ^ tag_name = _tags->GetKey(tagno);
		//		int tag_len = tag_name->Length;
		//		char * tag = new char [tag_len + 1];
		//		IntPtr nameChars = Marshal::StringToHGlobalAnsi(tag_name);
		//		memcpy (tag, (const char*)nameChars.ToPointer(), tag_len);
		//		Marshal::FreeHGlobal(nameChars);
		//		tag[tag_len] = 0;

		//		array<String^>^ tag_values = _tags->GetValues(tagno);
		//		for (int valno = 0; valno < tag_values->Length; valno++)
		//		{
		//			UTF8Encoding^ enc = gcnew UTF8Encoding();
		//			array<Byte>^ value_array = enc->GetBytes (tag_values[valno]);
		//			int value_len = value_array->Length;
		//			char * value = new char [value_len + 1];
		//			Marshal::Copy (value_array, 0, (IntPtr) value, value_len);
		//			value[value_len] = 0;

		//			FLAC__StreamMetadata_VorbisComment_Entry entry;
		//			/* create and entry and append it */
		//			if(!FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, tag, value)) {
		//				throw gcnew Exception("Unable to add tags, must be valid utf8.");
		//			}
		//			if(!FLAC__metadata_object_vorbiscomment_append_comment(vorbiscomment, entry, /*copy=*/false)) {
		//				throw gcnew Exception("Unable to add tags.");
		//			}
		//			delete [] value;
		//		}
		//		delete [] tag;
		//    }

		//    FLAC__metadata_iterator_insert_block_after (i, vorbiscomment);
		//    FLAC__metadata_iterator_delete (i);
		//    FLAC__metadata_chain_sort_padding (chain);
		//    res = FLAC__metadata_chain_write (chain, true, preserveTime);
		//    FLAC__metadata_chain_delete (chain);
		//    return 0 != res;
		//}

		virtual property Int64 Remaining {
			Int64 get() {
				return _sampleCount - _sampleOffset + SamplesInBuffer;
			}
		}

		virtual void Close() {
			if (_decoderActive) 
			{
				FLAC__stream_decoder_finish(_decoder);
				FLAC__stream_decoder_delete(_decoder);
				_decoderActive = false;
			}
			if (_IO != nullptr) 
			{
				_IO->Close ();
				_IO = nullptr;
			}
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
						if (FLAC__stream_decoder_get_state(_decoder) == FLAC__STREAM_DECODER_END_OF_STREAM)
						    return buff->Length - samplesNeeded;
						if (!FLAC__stream_decoder_process_single(_decoder))
						{
						    String^ state = gcnew String(FLAC__StreamDecoderStateString[FLAC__stream_decoder_get_state(_decoder)]);
						    throw gcnew Exception(String::Format("an error occurred while decoding: {0}", state));
						}
					} while (_bufferLength == 0);
					_sampleOffset += _bufferLength;
				}
				Int32 copyCount = Math::Min(samplesNeeded, SamplesInBuffer);
				Array::Copy(_sampleBuffer->Bytes, _bufferOffset * pcm->BlockAlign, buff->Bytes, buffOffset * pcm->BlockAlign, copyCount * pcm->BlockAlign);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}
			return buff->Length;
		}

	private:
		DecoderWriteDelegate^ _writeDel;
		DecoderMetadataDelegate^ _metadataDel;
		DecoderErrorDelegate^ _errorDel;
		DecoderReadDelegate^ _readDel;
		DecoderSeekDelegate^ _seekDel;
		DecoderTellDelegate^ _tellDel;
		DecoderLengthDelegate^ _lengthDel;
		DecoderEofDelegate^ _eofDel;
		FLAC__StreamDecoder *_decoder;
		Int64 _sampleCount, _sampleOffset;
		AudioPCMConfig^ pcm;
		AudioBuffer^ _sampleBuffer;
		array<unsigned char>^ _readBuffer;
		String^ _path;
		bool _decoderActive;
		Stream^ _IO;
		Int32 _bufferOffset, _bufferLength;

		property Int32 SamplesInBuffer {
			Int32 get () 
			{
				return _bufferLength - _bufferOffset;
			}
		}

		FLAC__StreamDecoderWriteStatus WriteCallback(const FLAC__StreamDecoder *decoder,
			const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *client_data)
		{
			Int32 sampleCount = frame->header.blocksize;

			if (_bufferLength > 0)
				throw gcnew Exception("received unrequested samples");

			if ((frame->header.bits_per_sample != pcm->BitsPerSample) ||
				(frame->header.channels != pcm->ChannelCount) ||
				(frame->header.sample_rate != pcm->SampleRate))
				throw gcnew Exception("format changes within a file are not allowed");

			if (_bufferOffset != 0)
			    throw gcnew Exception("internal buffer error");

			if (_sampleBuffer == nullptr || _sampleBuffer->Size < sampleCount)
			    _sampleBuffer = gcnew AudioBuffer(pcm, sampleCount);
			_sampleBuffer->Length =  sampleCount;

			if (pcm->ChannelCount == 2)
			    _sampleBuffer->Interlace(0, (int*)buffer[0], (int*)buffer[1], sampleCount);
			else
			{
			    int _channelCount = pcm->ChannelCount;
			    for (Int32 iChan = 0; iChan < _channelCount; iChan++) 
			    {
				    interior_ptr<Int32> pMyBuffer = &_sampleBuffer->Samples[0, iChan];
				    const FLAC__int32 *pFLACBuffer = buffer[iChan];
				    const FLAC__int32 *pFLACBufferEnd = pFLACBuffer + sampleCount;

				    while (pFLACBuffer < pFLACBufferEnd) {
					    *pMyBuffer = *pFLACBuffer;
					    pMyBuffer += _channelCount;
					    pFLACBuffer++;
				    }
			    }
			}
			_bufferLength = sampleCount;
			return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
		}

		void MetadataCallback(const FLAC__StreamDecoder *decoder,
			const FLAC__StreamMetadata *metadata, void *client_data)
		{
			if (metadata->type == FLAC__METADATA_TYPE_STREAMINFO) 
			{
				pcm = gcnew AudioPCMConfig(
				    metadata->data.stream_info.bits_per_sample,
				    metadata->data.stream_info.channels,
				    metadata->data.stream_info.sample_rate);
				_sampleCount = metadata->data.stream_info.total_samples;
			}
			//if (metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT) 
			//{
			//    for (unsigned tagno = 0; tagno < metadata->data.vorbis_comment.num_comments; tagno ++)
			//    {
			//		char * field_name, * field_value;
			//		if(!FLAC__metadata_object_vorbiscomment_entry_to_name_value_pair(metadata->data.vorbis_comment.comments[tagno], &field_name, &field_value)) 
			//			throw gcnew Exception("Unable to parse vorbis comment.");
			//		String^ name = Marshal::PtrToStringAnsi ((IntPtr) field_name);
			//		free (field_name);	    
			//		array<Byte>^ bvalue = gcnew array<Byte>((int) strlen (field_value));
			//		Marshal::Copy ((IntPtr) field_value, bvalue, 0, (int) strlen (field_value));
			//		free (field_value);
			//		UTF8Encoding^ enc = gcnew UTF8Encoding();
			//		String ^value = enc->GetString (bvalue);
			//		_tags->Add (name, value);
			//	}
			//}
		}

		void ErrorCallback(const FLAC__StreamDecoder *decoder,
			FLAC__StreamDecoderErrorStatus status, void *client_data)
		{
			switch (status) {
				case FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC:
					throw gcnew Exception("synchronization was lost");
				case FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER:
					throw gcnew Exception("encountered a corrupted frame header");
				case FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH:
					throw gcnew Exception("frame CRC mismatch");
				default:
					throw gcnew Exception("an unknown error has occurred");
			}
		}

		FLAC__StreamDecoderReadStatus ReadCallback (const FLAC__StreamDecoder *decoder, FLAC__byte buffer[], size_t *bytes, void *client_data)
		{
			if(*bytes == 0)
				return FLAC__STREAM_DECODER_READ_STATUS_ABORT; /* abort to avoid a deadlock */

			if (_readBuffer == nullptr || _readBuffer->Length < *bytes)
				_readBuffer = gcnew array<unsigned char>(*bytes < 0x4000 ? 0x4000 : *bytes);

			*bytes = _IO->Read (_readBuffer, 0, *bytes);
			//if(ferror(decoder->private_->file))
				//return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
			//else 
			if(*bytes == 0)
				return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;

			Marshal::Copy (_readBuffer, 0, (IntPtr)buffer, *bytes);
			return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
		}

		FLAC__StreamDecoderSeekStatus SeekCallback (const FLAC__StreamDecoder *decoder, FLAC__uint64 absolute_byte_offset, void *client_data)
		{
			//if (!_IO->CanSeek)
			//	return FLAC__STREAM_DECODER_SEEK_STATUS_UNSUPPORTED;
			_IO->Position = absolute_byte_offset;
			//if( < 0)
				//return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
			return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
		}

		FLAC__StreamDecoderTellStatus TellCallback (const FLAC__StreamDecoder *decoder, FLAC__uint64 *absolute_byte_offset, void *client_data)
		{
			*absolute_byte_offset = _IO->Position;
			// FLAC__STREAM_DECODER_TELL_STATUS_ERROR;
			return FLAC__STREAM_DECODER_TELL_STATUS_OK;
		}

		FLAC__StreamDecoderLengthStatus LengthCallback (const FLAC__StreamDecoder *decoder, FLAC__uint64 *stream_length, void *client_data)
		{
			//if (!_IO->CanSeek)
			//	return FLAC__STREAM_DECODER_LENGTH_STATUS_UNSUPPORTED;
			// FLAC__STREAM_DECODER_LENGTH_STATUS_ERROR;
			*stream_length = _IO->Length;
			return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
		}

		FLAC__bool EofCallback (const FLAC__StreamDecoder *decoder, void *client_data)
		{
			return _IO->Position == _IO->Length;
		}
	};

	public ref class FLACWriterSettings
	{
	    public:
		FLACWriterSettings() 
		{ 
		    _md5Sum = true;
		    _verify = false;
		    _disableAsm = false;
		}

		[DefaultValue(false)]
		[DisplayName("Verify")]
		[Description("Decode each frame and compare with original")]
		property bool Verify {
			bool get() {
				return _verify;
			}
			void set(bool value) {
				_verify = value;
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

		[DefaultValue(false)]
		[DisplayName("Disable ASM")]
		[Description("Disable MMX/SSE optimizations")]
		property bool DisableAsm {
			bool get() {
				return _disableAsm;
			}
			void set(bool value) {
				_disableAsm = value;
			}
		}

	    private:
		bool _md5Sum, _verify, _disableAsm;
	};
	
	[AudioEncoderClass("libFLAC", "flac", true, "0 1 2 3 4 5 6 7 8", "5", 2, FLACWriterSettings::typeid)]
	public ref class FLACWriter : IAudioDest 
	{
	public:
		FLACWriter(String^ path, AudioPCMConfig^ pcm)
		{
		    _pcm = pcm;

		    _settings = gcnew FLACWriterSettings();

		    if (_pcm->BitsPerSample < 16 || _pcm->BitsPerSample > 24)
				throw gcnew Exception("bits per sample must be 16..24");

			_initialized = false;
			_path = path;
			_finalSampleCount = 0;
			_samplesWritten = 0;
			_compressionLevel = 5;
			_paddingLength = 8192;
			_blockSize = 0;

			_encoder = FLAC__stream_encoder_new();

			FLAC__stream_encoder_set_bits_per_sample(_encoder, _pcm->BitsPerSample);
			FLAC__stream_encoder_set_channels(_encoder, _pcm->ChannelCount);
			FLAC__stream_encoder_set_sample_rate(_encoder, _pcm->SampleRate);
		}

		virtual void Close() {
			FLAC__stream_encoder_finish(_encoder);

			for (int i = 0; i < _metadataCount; i++) {
				FLAC__metadata_object_delete(_metadataList[i]);
			}
			delete[] _metadataList;
			_metadataList = 0;
			_metadataCount = 0;

			FLAC__stream_encoder_delete(_encoder);

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount))
				throw gcnew Exception("samples written differs from the expected sample count");
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
				if (value < 0)
					throw gcnew Exception("invalid final sample count");
				if (_initialized)
					throw gcnew Exception("final sample count cannot be changed after encoding begins");
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

		virtual property AudioPCMConfig^ PCM {
			AudioPCMConfig^ get() {
				return _pcm;
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

			pin_ptr<Int32> pSampleBuffer = &sampleBuffer->Samples[0, 0];

			if (!FLAC__stream_encoder_process_interleaved(_encoder,
			    (const FLAC__int32*)pSampleBuffer, sampleBuffer->Length))
			{
				String^ state = gcnew String(FLAC__StreamEncoderStateString[FLAC__stream_encoder_get_state(_encoder)]);
				if (FLAC__stream_encoder_get_state(_encoder) == FLAC__STREAM_ENCODER_VERIFY_MISMATCH_IN_AUDIO_DATA)
				{
					FLAC__uint64 absolute_sample;
					unsigned frame_number;
					unsigned channel;
					unsigned sample;
					FLAC__int32 expected, got;
					FLAC__stream_encoder_get_verify_decoder_error_stats(_encoder, &absolute_sample, &frame_number, &channel, &sample, &expected, &got);
					state = state + String::Format("({0:x} instead of {1:x} @{2:x})", got, expected, absolute_sample);
				}
				throw gcnew Exception("an error occurred while encoding: " + state);
			}

			_samplesWritten += sampleBuffer->Length;
		}

		virtual property Int32 CompressionLevel {
			Int32 get() {
				return _compressionLevel;
			}
			void set(Int32 value) {
				if ((value < 0) || (value > 8))
					throw gcnew Exception("invalid compression level");
				_compressionLevel = value;
			}
		}

		virtual property Object^ Settings
		{
			Object^ get()
			{
			    return _settings;
			}
			
			void set(Object^ value)
			{
			    if (value == nullptr || value->GetType() != FLACWriterSettings::typeid)
				throw gcnew Exception(String::Format("Unsupported options: {0}", value));
			    _settings = (FLACWriterSettings^)value;
			}
		}

		virtual property __int64 Padding {
			__int64 get() {
				return _paddingLength;
			}
			void set(__int64 value) {
				if (value < 0)
					throw gcnew Exception("invalid padding length");
				_paddingLength = value;
			}
		}

	private:
		FLACWriterSettings^ _settings;
		FLAC__StreamEncoder *_encoder;
		bool _initialized;
		String^ _path;
		Int64 _finalSampleCount, _samplesWritten, _blockSize;
		AudioPCMConfig^ _pcm;
		Int32 _compressionLevel;
		__int64 _paddingLength;
		FLAC__StreamMetadata **_metadataList;
		int _metadataCount;

		void Initialize() {
			FLAC__StreamMetadata *padding, *seektable, *vorbiscomment;
			IntPtr pathChars;
			FILE *hFile;

			_metadataList = new FLAC__StreamMetadata*[8];
			_metadataCount = 0;

			if (_finalSampleCount != 0) {
				seektable = FLAC__metadata_object_new(FLAC__METADATA_TYPE_SEEKTABLE);
				FLAC__metadata_object_seektable_template_append_spaced_points_by_samples(
				    seektable, _pcm->SampleRate * 10, _finalSampleCount);
				FLAC__metadata_object_seektable_template_sort(seektable, true);
				_metadataList[_metadataCount++] = seektable;
			}

			vorbiscomment = FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT);
			//for (int tagno = 0; tagno < _tags->Count; tagno++)
			//{
			//	String ^ tag_name = _tags->GetKey(tagno);
			//	int tag_len = tag_name->Length;
			//    char * tag = new char [tag_len + 1];
			//    IntPtr nameChars = Marshal::StringToHGlobalAnsi(tag_name);
			//    memcpy (tag, (const char*)nameChars.ToPointer(), tag_len);
			//    Marshal::FreeHGlobal(nameChars);
			//    tag[tag_len] = 0;

			//	array<String^>^ tag_values = _tags->GetValues(tagno);
			//	for (int valno = 0; valno < tag_values->Length; valno++)
			//	{
			//		UTF8Encoding^ enc = gcnew UTF8Encoding();
			//		array<Byte>^ value_array = enc->GetBytes (tag_values[valno]);
			//		int value_len = value_array->Length;
			//		char * value = new char [value_len + 1];
			//		Marshal::Copy (value_array, 0, (IntPtr) value, value_len);
			//		value[value_len] = 0;

			//		FLAC__StreamMetadata_VorbisComment_Entry entry;
			//		/* create and entry and append it */
			//		if(!FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, tag, value)) {
			//			throw gcnew Exception("Unable to add tags, must be valid utf8.");
			//		}
			//		if(!FLAC__metadata_object_vorbiscomment_append_comment(vorbiscomment, entry, /*copy=*/false)) {
			//			throw gcnew Exception("Unable to add tags.");
			//		}
			//		delete [] value;
			//	}
			//    delete [] tag;
			//}
			_metadataList[_metadataCount++] = vorbiscomment;
	 
			if (_paddingLength != 0) {
				padding = FLAC__metadata_object_new(FLAC__METADATA_TYPE_PADDING);
				padding->length = (int)_paddingLength;
				_metadataList[_metadataCount++] = padding;
			}

			FLAC__stream_encoder_set_metadata(_encoder, _metadataList, _metadataCount);

			FLAC__stream_encoder_set_verify(_encoder, _settings->Verify);

			FLAC__stream_encoder_set_do_md5(_encoder, _settings->MD5Sum);

			FLAC__stream_encoder_set_disable_asm(_encoder, _settings->DisableAsm);

			if (_finalSampleCount != 0) {
				FLAC__stream_encoder_set_total_samples_estimate(_encoder, _finalSampleCount);
			}

			FLAC__stream_encoder_set_compression_level(_encoder, _compressionLevel);

			if (_blockSize > 0)
				FLAC__stream_encoder_set_blocksize(_encoder, (unsigned)_blockSize);

			pathChars = Marshal::StringToHGlobalUni(_path);
			errno_t err = _wfopen_s(&hFile, (const wchar_t*)pathChars.ToPointer(), L"w+b");
			Marshal::FreeHGlobal(pathChars);
			if (err)
			{
			    wchar_t buffer[256];
			    _wcserror_s(buffer, err);
			    throw gcnew Exception(String::Format("unable to open output file {0}: {1}", _path, gcnew String(buffer)));
			}

			FLAC__StreamEncoderInitStatus st = FLAC__stream_encoder_init_FILE(_encoder, hFile, NULL, NULL);
			if (st != FLAC__STREAM_ENCODER_INIT_STATUS_OK)
				throw gcnew Exception(String::Format("unable to initialize the encoder: {0}", gcnew String(FLAC__StreamEncoderInitStatusString[st])));

			_initialized = true;
		}
	};
}
}
}
