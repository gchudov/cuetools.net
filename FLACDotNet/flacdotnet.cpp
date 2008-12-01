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
using namespace System::Text;
using namespace System::IO;
using namespace System::Collections::Generic;
using namespace System::Collections::Specialized;
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

namespace FLACDotNet {
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

	public ref class FLACReader : public IAudioSource
	{
	public:
		FLACReader(String^ path, Stream^ IO)
		{
			_tags = gcnew NameValueCollection();

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
				throw gcnew Exception("Unable to setup the decoder.");

			if (FLAC__stream_decoder_init_stream(_decoder, 
				(FLAC__StreamDecoderReadCallback)Marshal::GetFunctionPointerForDelegate(_readDel).ToPointer(),
				_IO->CanSeek?(FLAC__StreamDecoderSeekCallback)Marshal::GetFunctionPointerForDelegate(_seekDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderTellCallback)Marshal::GetFunctionPointerForDelegate(_tellDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderLengthCallback)Marshal::GetFunctionPointerForDelegate(_lengthDel).ToPointer():NULL,
				_IO->CanSeek?(FLAC__StreamDecoderEofCallback)Marshal::GetFunctionPointerForDelegate(_eofDel).ToPointer():NULL,
				(FLAC__StreamDecoderWriteCallback)Marshal::GetFunctionPointerForDelegate(_writeDel).ToPointer(),
				(FLAC__StreamDecoderMetadataCallback)Marshal::GetFunctionPointerForDelegate(_metadataDel).ToPointer(),
				(FLAC__StreamDecoderErrorCallback)Marshal::GetFunctionPointerForDelegate(_errorDel).ToPointer(),
				NULL) != FLAC__STREAM_DECODER_INIT_STATUS_OK)
			{
				throw gcnew Exception("Unable to initialize the decoder.");
			}

			_decoderActive = true;

			if (!FLAC__stream_decoder_process_until_end_of_metadata(_decoder)) {
				throw gcnew Exception("Unable to retrieve metadata.");
			}

		}

		~FLACReader ()
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
				if (!FLAC__stream_decoder_seek_absolute(_decoder, offset)) {
					throw gcnew Exception("Unable to seek.");
				}
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
		    Close ();
		    
		    FLAC__Metadata_Chain* chain = FLAC__metadata_chain_new ();
		    if (!chain) return false;

		    IntPtr pathChars = Marshal::StringToHGlobalAnsi(_path);
		    int res = FLAC__metadata_chain_read (chain, (const char*)pathChars.ToPointer());
		    Marshal::FreeHGlobal(pathChars);
		    if (!res) {
				FLAC__metadata_chain_delete (chain);
				return false;
		    }
		    FLAC__Metadata_Iterator* i = FLAC__metadata_iterator_new ();
		    FLAC__metadata_iterator_init (i, chain);
		    do {
				FLAC__StreamMetadata* metadata = FLAC__metadata_iterator_get_block (i);
				if (metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT)
					FLAC__metadata_iterator_delete_block (i, false);
		    } while (FLAC__metadata_iterator_next (i));

		    FLAC__StreamMetadata * vorbiscomment = FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT);
		    for (int tagno = 0; tagno <_tags->Count; tagno++)
		    {
				String ^ tag_name = _tags->GetKey(tagno);
				int tag_len = tag_name->Length;
				char * tag = new char [tag_len + 1];
				IntPtr nameChars = Marshal::StringToHGlobalAnsi(tag_name);
				memcpy (tag, (const char*)nameChars.ToPointer(), tag_len);
				Marshal::FreeHGlobal(nameChars);
				tag[tag_len] = 0;

				array<String^>^ tag_values = _tags->GetValues(tagno);
				for (int valno = 0; valno < tag_values->Length; valno++)
				{
					UTF8Encoding^ enc = gcnew UTF8Encoding();
					array<Byte>^ value_array = enc->GetBytes (tag_values[valno]);
					int value_len = value_array->Length;
					char * value = new char [value_len + 1];
					Marshal::Copy (value_array, 0, (IntPtr) value, value_len);
					value[value_len] = 0;

					FLAC__StreamMetadata_VorbisComment_Entry entry;
					/* create and entry and append it */
					if(!FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, tag, value)) {
						throw gcnew Exception("Unable to add tags, must be valid utf8.");
					}
					if(!FLAC__metadata_object_vorbiscomment_append_comment(vorbiscomment, entry, /*copy=*/false)) {
						throw gcnew Exception("Unable to add tags.");
					}
					delete [] value;
				}
				delete [] tag;
		    }

		    FLAC__metadata_iterator_insert_block_after (i, vorbiscomment);
		    FLAC__metadata_iterator_delete (i);
		    FLAC__metadata_chain_sort_padding (chain);
		    res = FLAC__metadata_chain_write (chain, true, preserveTime);
		    FLAC__metadata_chain_delete (chain);
		    return 0 != res;
		}

		virtual property UInt64 Remaining {
			UInt64 get() {
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
						if (!FLAC__stream_decoder_process_single(_decoder))
							throw gcnew Exception("An error occurred while decoding.");
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
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		array<Int32, 2>^ _sampleBuffer;
		array<unsigned char>^ _readBuffer;
		NameValueCollection^ _tags;
		String^ _path;
		bool _decoderActive;
		Stream^ _IO;
		UInt32 _bufferOffset, _bufferLength;

		property UInt32 SamplesInBuffer {
			UInt32 get () 
			{
				return (UInt32) (_bufferLength - _bufferOffset);
			}
		}

		FLAC__StreamDecoderWriteStatus WriteCallback(const FLAC__StreamDecoder *decoder,
			const FLAC__Frame *frame, const FLAC__int32 * const buffer[], void *client_data)
		{
			Int32 sampleCount = frame->header.blocksize;

			if (_bufferLength > 0) {
				throw gcnew Exception("Received unrequested samples.");
			}

			if ((frame->header.bits_per_sample != _bitsPerSample) ||
				(frame->header.channels != _channelCount) ||
				(frame->header.sample_rate != _sampleRate))
			{
				throw gcnew Exception("Format changes within a file are not allowed.");
			}

			if ((_sampleBuffer == nullptr) || (_sampleBuffer->GetLength(0) < sampleCount)) {
				_sampleBuffer = gcnew array<Int32, 2>(sampleCount, _channelCount);
			}

			for (Int32 iChan = 0; iChan < _channelCount; iChan++) {
				interior_ptr<Int32> pMyBuffer = &_sampleBuffer[0, iChan];
				const FLAC__int32 *pFLACBuffer = buffer[iChan];
				const FLAC__int32 *pFLACBufferEnd = pFLACBuffer + sampleCount;

				while (pFLACBuffer < pFLACBufferEnd) {
					*pMyBuffer = *pFLACBuffer;
					pMyBuffer += _channelCount;
					pFLACBuffer++;
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
				_bitsPerSample = metadata->data.stream_info.bits_per_sample;
				_channelCount = metadata->data.stream_info.channels;
				_sampleRate = metadata->data.stream_info.sample_rate;
				_sampleCount = metadata->data.stream_info.total_samples;
			}
			if (metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT) 
			{
			    for (unsigned tagno = 0; tagno < metadata->data.vorbis_comment.num_comments; tagno ++)
			    {
					char * field_name, * field_value;
					if(!FLAC__metadata_object_vorbiscomment_entry_to_name_value_pair(metadata->data.vorbis_comment.comments[tagno], &field_name, &field_value)) 
						throw gcnew Exception("Unable to parse vorbis comment.");
					String^ name = Marshal::PtrToStringAnsi ((IntPtr) field_name);
					free (field_name);	    
					array<Byte>^ bvalue = gcnew array<Byte>((int) strlen (field_value));
					Marshal::Copy ((IntPtr) field_value, bvalue, 0, (int) strlen (field_value));
					free (field_value);
					UTF8Encoding^ enc = gcnew UTF8Encoding();
					String ^value = enc->GetString (bvalue);
					_tags->Add (name, value);
				}
			}
		}

		void ErrorCallback(const FLAC__StreamDecoder *decoder,
			FLAC__StreamDecoderErrorStatus status, void *client_data)
		{
			switch (status) {
				case FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC:
					throw gcnew Exception("Synchronization was lost.");
				case FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER:
					throw gcnew Exception("Encountered a corrupted frame header.");
				case FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH:
					throw gcnew Exception("Frame CRC mismatch.");
				default:
					throw gcnew Exception("An unknown error has occurred.");
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

	public ref class FLACWriter : IAudioDest 
	{
	public:
		FLACWriter(String^ path, Int32 bitsPerSample, Int32 channelCount, Int32 sampleRate) 
		{
			if (bitsPerSample < 16 || bitsPerSample > 24)
				throw gcnew Exception("Bits per sample must be 16..24.");

			_initialized = false;
			_path = path;
			_finalSampleCount = 0;
			_samplesWritten = 0;
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_compressionLevel = 5;
			_paddingLength = 8192;
			_verify = false;
			_blockSize = 0;
			_tags = gcnew NameValueCollection();

			_encoder = FLAC__stream_encoder_new();

			FLAC__stream_encoder_set_bits_per_sample(_encoder, bitsPerSample);
			FLAC__stream_encoder_set_channels(_encoder, channelCount);
			FLAC__stream_encoder_set_sample_rate(_encoder, sampleRate);
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

			if ((_finalSampleCount != 0) && (_samplesWritten != _finalSampleCount)) {
				throw gcnew Exception("Samples written differs from the expected sample count.");
			}

			_tags->Clear ();
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

		virtual property Int64 BlockSize
		{
			void set(Int64 value) 
			{
				_blockSize = value;
			}
		}

		virtual property int BitsPerSample
		{
			int get() { return _bitsPerSample;  }
		}

		virtual bool SetTags (NameValueCollection^ tags) 
		{
			_tags = tags;
			return true;
		}

		virtual property String^ Path { 
			String^ get() { 
				return _path; 
			} 
		}

		virtual void Write(array<Int32, 2>^ sampleBuffer, UInt32 sampleCount) {
			if (!_initialized) Initialize();

			pin_ptr<Int32> pSampleBuffer = &sampleBuffer[0, 0];

			if (!FLAC__stream_encoder_process_interleaved(_encoder,
				(const FLAC__int32*)pSampleBuffer, sampleCount))
			{
				throw gcnew Exception("An error occurred while encoding.");
			}

			_samplesWritten += sampleCount;
		}

		property Int32 CompressionLevel {
			Int32 get() {
				return _compressionLevel;
			}
			void set(Int32 value) {
				if ((value < 0) || (value > 8)) {
					throw gcnew Exception("Invalid compression level.");
				}
				_compressionLevel = value;
			}
		}

		property Boolean Verify {
			Boolean get() {
				return _verify;
			}
			void set(Boolean value) {
				_verify = value;
			}
		}

		property Int32 PaddingLength {
			Int32 get() {
				return _paddingLength;
			}
			void set(Int32 value) {
				if (value < 0) {
					throw gcnew Exception("Invalid padding length.");
				}
				_paddingLength = value;
			}
		}

	private:
		FLAC__StreamEncoder *_encoder;
		bool _initialized;
		String^ _path;
		Int64 _finalSampleCount, _samplesWritten, _blockSize;
		Int32 _bitsPerSample, _channelCount, _sampleRate;
		Int32 _compressionLevel;
		Int32 _paddingLength;
		Boolean _verify;
		FLAC__StreamMetadata **_metadataList;
		int _metadataCount;
		NameValueCollection^ _tags;

		void Initialize() {
			FLAC__StreamMetadata *padding, *seektable, *vorbiscomment;
			IntPtr pathChars;
			FILE *hFile;

			_metadataList = new FLAC__StreamMetadata*[8];
			_metadataCount = 0;

			if (_finalSampleCount != 0) {
				seektable = FLAC__metadata_object_new(FLAC__METADATA_TYPE_SEEKTABLE);
				FLAC__metadata_object_seektable_template_append_spaced_points_by_samples(
					seektable, _sampleRate * 10, _finalSampleCount);
				FLAC__metadata_object_seektable_template_sort(seektable, true);
				_metadataList[_metadataCount++] = seektable;
			}

			vorbiscomment = FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT);

			for (int tagno = 0; tagno < _tags->Count; tagno++)
			{
				String ^ tag_name = _tags->GetKey(tagno);
				int tag_len = tag_name->Length;
			    char * tag = new char [tag_len + 1];
			    IntPtr nameChars = Marshal::StringToHGlobalAnsi(tag_name);
			    memcpy (tag, (const char*)nameChars.ToPointer(), tag_len);
			    Marshal::FreeHGlobal(nameChars);
			    tag[tag_len] = 0;

				array<String^>^ tag_values = _tags->GetValues(tagno);
				for (int valno = 0; valno < tag_values->Length; valno++)
				{
					UTF8Encoding^ enc = gcnew UTF8Encoding();
					array<Byte>^ value_array = enc->GetBytes (tag_values[valno]);
					int value_len = value_array->Length;
					char * value = new char [value_len + 1];
					Marshal::Copy (value_array, 0, (IntPtr) value, value_len);
					value[value_len] = 0;

					FLAC__StreamMetadata_VorbisComment_Entry entry;
					/* create and entry and append it */
					if(!FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, tag, value)) {
						throw gcnew Exception("Unable to add tags, must be valid utf8.");
					}
					if(!FLAC__metadata_object_vorbiscomment_append_comment(vorbiscomment, entry, /*copy=*/false)) {
						throw gcnew Exception("Unable to add tags.");
					}
					delete [] value;
				}
			    delete [] tag;
			}
			_metadataList[_metadataCount++] = vorbiscomment;
	 
			if (_paddingLength != 0) {
				padding = FLAC__metadata_object_new(FLAC__METADATA_TYPE_PADDING);
				padding->length = _paddingLength;
				_metadataList[_metadataCount++] = padding;
			}

			FLAC__stream_encoder_set_metadata(_encoder, _metadataList, _metadataCount);

			FLAC__stream_encoder_set_verify(_encoder, _verify);

			if (_finalSampleCount != 0) {
				FLAC__stream_encoder_set_total_samples_estimate(_encoder, _finalSampleCount);
			}

			FLAC__stream_encoder_set_compression_level(_encoder, _compressionLevel);

			if (_blockSize > 0)
				FLAC__stream_encoder_set_blocksize(_encoder, (unsigned)_blockSize);

			pathChars = Marshal::StringToHGlobalUni(_path);
			hFile = _wfopen((const wchar_t*)pathChars.ToPointer(), L"w+b");
			Marshal::FreeHGlobal(pathChars);

			if (FLAC__stream_encoder_init_FILE(_encoder, hFile, NULL, NULL) !=
				FLAC__STREAM_ENCODER_INIT_STATUS_OK)
			{
				throw gcnew Exception("Unable to initialize the encoder.");
			}

			_initialized = true;
		}
	};
}
