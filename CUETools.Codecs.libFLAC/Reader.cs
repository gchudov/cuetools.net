using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using CUETools.Codecs;

namespace CUETools.Codecs.libFLAC
{
    [AudioDecoderClass("libFLAC", "flac", 1)]
    public unsafe class Reader : IAudioSource
    {
        public Reader(string path, Stream IO)
        {
            m_writeCallback = WriteCallback;
            m_metadataCallback = MetadataCallback;
            m_errorCallback = ErrorCallback;
            m_readCallback = ReadCallback;
            m_seekCallback = SeekCallback;
            m_tellCallback = TellCallback;
            m_lengthCallback = LengthCallback;
            m_eofCallback = EofCallback;

            m_decoderActive = false;

            m_sampleOffset = 0;
            m_sampleBuffer = null;
            m_path = path;
            m_bufferOffset = 0;
            m_bufferLength = 0;

            m_stream = (IO != null) ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

            m_decoder = FLACDLL.FLAC__stream_decoder_new();

            if (0 == FLACDLL.FLAC__stream_decoder_set_metadata_respond(m_decoder, FLAC__MetadataType.FLAC__METADATA_TYPE_VORBIS_COMMENT))
                throw new Exception("unable to setup the decoder");

            FLAC__StreamDecoderInitStatus st = FLACDLL.FLAC__stream_decoder_init_stream(
                m_decoder, m_readCallback,
                m_stream.CanSeek ? m_seekCallback : null,
                m_stream.CanSeek ? m_tellCallback : null,
                m_stream.CanSeek ? m_lengthCallback : null,
                m_stream.CanSeek ? m_eofCallback : null,
                m_writeCallback, m_metadataCallback, m_errorCallback, null);

            if (st != FLAC__StreamDecoderInitStatus.FLAC__STREAM_DECODER_INIT_STATUS_OK)
                throw new Exception(string.Format("unable to initialize the decoder: {0}", st));

            m_decoderActive = true;

            if (0 == FLACDLL.FLAC__stream_decoder_process_until_end_of_metadata(m_decoder))
                throw new Exception("unable to retrieve metadata");
        }

#if SUPPORTMETADATA
		bool UpdateTags (bool preserveTime)
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

				array<string>^ tag_values = _tags->GetValues(tagno);
				for (int valno = 0; valno < tag_values->Length; valno++)
				{
					UTF8Encoding^ enc = new UTF8Encoding();
					array<Byte>^ value_array = enc->GetBytes (tag_values[valno]);
					int value_len = value_array->Length;
					char * value = new char [value_len + 1];
					Marshal::Copy (value_array, 0, (IntPtr) value, value_len);
					value[value_len] = 0;

					FLAC__StreamMetadata_VorbisComment_Entry entry;
					/* create and entry and append it */
					if(!FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, tag, value)) {
						throw new Exception("Unable to add tags, must be valid utf8.");
					}
					if(!FLAC__metadata_object_vorbiscomment_append_comment(vorbiscomment, entry, /*copy=*/false)) {
						throw new Exception("Unable to add tags.");
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
#endif

        FLAC__StreamDecoderWriteStatus WriteCallback(IntPtr decoder,
            FLAC__Frame* frame, int** buffer, void* client_data)
        {
            int sampleCount = frame->header.blocksize;

            if (m_bufferLength > 0)
                throw new Exception("received unrequested samples");

            if ((frame->header.bits_per_sample != m_pcm.BitsPerSample) ||
                (frame->header.channels != m_pcm.ChannelCount) ||
                (frame->header.sample_rate != m_pcm.SampleRate))
                throw new Exception("format changes within a file are not allowed");

            if (m_bufferOffset != 0)
                throw new Exception("internal buffer error");

            if (m_sampleBuffer == null || m_sampleBuffer.Size < sampleCount)
                m_sampleBuffer = new AudioBuffer(m_pcm, sampleCount);
            m_sampleBuffer.Length = sampleCount;

            if (m_pcm.ChannelCount == 2)
                m_sampleBuffer.Interlace(0, (int*)buffer[0], (int*)buffer[1], sampleCount);
            else
            {
                int _channelCount = m_pcm.ChannelCount;
                for (Int32 iChan = 0; iChan < _channelCount; iChan++)
                {
                    fixed (int* pMyBuffer = &m_sampleBuffer.Samples[0, iChan])
                    {
                        int* pMyBufferPtr = pMyBuffer;
                        int* pFLACBuffer = buffer[iChan];
                        int* pFLACBufferEnd = pFLACBuffer + sampleCount;
                        while (pFLACBuffer < pFLACBufferEnd)
                        {
                            *pMyBufferPtr = *pFLACBuffer;
                            pMyBufferPtr += _channelCount;
                            pFLACBuffer++;
                        }
                    }
                }
            }
            m_bufferLength = sampleCount;
            m_sampleOffset += m_bufferLength;
            return FLAC__StreamDecoderWriteStatus.FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
        }

        void MetadataCallback(IntPtr decoder,
			FLAC__StreamMetadata *metadata, void *client_data)
		{
			if (metadata->type == FLAC__MetadataType.FLAC__METADATA_TYPE_STREAMINFO) 
			{
				m_pcm = new AudioPCMConfig(
				    metadata->stream_info.bits_per_sample,
				    metadata->stream_info.channels,
				    metadata->stream_info.sample_rate,
					(AudioPCMConfig.SpeakerConfig)0);
				m_sampleCount = metadata->stream_info.total_samples;
			}
#if SUPPORTMETADATA
            if (metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT) 
            {
                for (int tagno = 0; tagno < metadata->vorbis_comment.num_comments; tagno ++)
                {
            		char * field_name, * field_value;
            		if(!FLACDLL.FLAC__metadata_object_vorbiscomment_entry_to_name_value_pair(metadata->vorbis_comment.comments[tagno], &field_name, &field_value)) 
            			throw new Exception("Unable to parse vorbis comment.");
            		string name = Marshal::PtrToStringAnsi ((IntPtr) field_name);
            		free (field_name);	    
            		array<Byte>^ bvalue = new array<Byte>((int) strlen (field_value));
            		Marshal.Copy ((IntPtr) field_value, bvalue, 0, (int) strlen (field_value));
            		free (field_value);
            		UTF8Encoding enc = new UTF8Encoding();
            		string value = enc.GetString(bvalue);
            		_tags.Add(name, value);
            	}
            }
#endif
        }

        void ErrorCallback(IntPtr decoder,
			FLAC__StreamDecoderErrorStatus status, void *client_data)
		{
            switch (status)
            {
                case FLAC__StreamDecoderErrorStatus.FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC:
                    throw new Exception("synchronization was lost");
                case FLAC__StreamDecoderErrorStatus.FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER:
                    throw new Exception("encountered a corrupted frame header");
                case FLAC__StreamDecoderErrorStatus.FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH:
                    throw new Exception("frame CRC mismatch");
                default:
                    throw new Exception("an unknown error has occurred");
            }
		}

        FLAC__StreamDecoderReadStatus ReadCallback(IntPtr decoder, byte* buffer, ref long bytes, void* client_data)
        {
            if (bytes <= 0 || bytes > int.MaxValue)
                return FLAC__StreamDecoderReadStatus.FLAC__STREAM_DECODER_READ_STATUS_ABORT; /* abort to avoid a deadlock */

            if (m_readBuffer == null || m_readBuffer.Length < bytes)
                m_readBuffer = new byte[Math.Max(bytes, 0x4000)];

            bytes = m_stream.Read(m_readBuffer, 0, (int)bytes);
            //if(ferror(decoder->private_->file))
            //return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
            //else 
            if (bytes == 0)
                return FLAC__StreamDecoderReadStatus.FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;

            Marshal.Copy(m_readBuffer, 0, (IntPtr)buffer, (int)bytes);
            return FLAC__StreamDecoderReadStatus.FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
        }

        FLAC__StreamDecoderSeekStatus SeekCallback(IntPtr decoder, long absolute_byte_offset, void* client_data)
        {
            //if (!_IO.CanSeek)
            //	return FLAC__STREAM_DECODER_SEEK_STATUS_UNSUPPORTED;
            m_stream.Position = absolute_byte_offset;
            //catch(Exception) {
            //  return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
            //}
            return FLAC__StreamDecoderSeekStatus.FLAC__STREAM_DECODER_SEEK_STATUS_OK;
        }

        FLAC__StreamDecoderTellStatus TellCallback(IntPtr decoder, out long absolute_byte_offset, void* client_data)
        {
            //if (!_IO.CanSeek)
            //	return FLAC__STREAM_DECODER_TELL_STATUS_UNSUPPORTED;
            absolute_byte_offset = m_stream.Position;
            // if (_IO.Position < 0)
            //  return FLAC__STREAM_DECODER_TELL_STATUS_ERROR;
            return FLAC__StreamDecoderTellStatus.FLAC__STREAM_DECODER_TELL_STATUS_OK;
        }

        FLAC__StreamDecoderLengthStatus LengthCallback(IntPtr decoder, out long stream_length, void* client_data)
        {
            //if (!_IO.CanSeek)
            //	return FLAC__STREAM_DECODER_LENGTH_STATUS_UNSUPPORTED;
            // if (_IO.Length < 0)
            //  return FLAC__STREAM_DECODER_LENGTH_STATUS_ERROR;
            stream_length = m_stream.Length;
            return FLAC__StreamDecoderLengthStatus.FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
        }

        int EofCallback (IntPtr decoder, void *client_data)
		{
            return m_stream.Position == m_stream.Length ? 1 : 0;
		}

        public AudioDecoderSettings Settings => null;

        public AudioPCMConfig PCM => m_pcm;

        public string Path => m_path;

        public long Length => m_sampleCount;

        private int SamplesInBuffer => m_bufferLength - m_bufferOffset;

        public long Position
        {
            get => m_sampleOffset - SamplesInBuffer;

            set
            {
                m_sampleOffset = value;
                m_bufferOffset = 0;
                m_bufferLength = 0;
                if (0 == FLACDLL.FLAC__stream_decoder_seek_absolute(m_decoder, value))
                    throw new Exception("unable to seek");
            }
        }

        public long Remaining { get => m_sampleCount - Position; }

        public void Close()
        {
            if (m_decoderActive)
            {
                FLACDLL.FLAC__stream_decoder_finish(m_decoder);
                FLACDLL.FLAC__stream_decoder_delete(m_decoder);
                m_decoderActive = false;
            }
            if (m_stream != null)
            {
                m_stream.Close();
                m_stream = null;
            }
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);
			int buffOffset = 0;
			int samplesNeeded = buff.Length;

			while (samplesNeeded != 0) 
			{
				if (SamplesInBuffer == 0) 
				{
					m_bufferOffset = 0;
					m_bufferLength = 0;
					do
					{
						if (FLACDLL.FLAC__stream_decoder_get_state(m_decoder) ==  FLAC__StreamDecoderState.FLAC__STREAM_DECODER_END_OF_STREAM)
						{
						    buff.Length -= samplesNeeded;
						    return buff.Length;
						}
						if (0 == FLACDLL.FLAC__stream_decoder_process_single(m_decoder))
						    throw new Exception(string.Format("an error occurred while decoding: {0}", FLACDLL.FLAC__stream_decoder_get_state(m_decoder)));
					} while (m_bufferLength == 0);
				}
				int copyCount = Math.Min(samplesNeeded, SamplesInBuffer);
				Array.Copy(m_sampleBuffer.Bytes, m_bufferOffset * m_pcm.BlockAlign, buff.Bytes, buffOffset * m_pcm.BlockAlign, copyCount * m_pcm.BlockAlign);
				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				m_bufferOffset += copyCount;
			}
			return buff.Length;
        }

        AudioBuffer m_sampleBuffer;
        byte[] m_readBuffer;
        long m_sampleCount, m_sampleOffset;
        int m_bufferOffset, m_bufferLength;
        IntPtr m_decoder;
        string m_path;
        Stream m_stream;
        bool m_decoderActive;
        AudioPCMConfig m_pcm;
        FLACDLL.FLAC__StreamDecoderReadCallback m_readCallback;
        FLACDLL.FLAC__StreamDecoderSeekCallback m_seekCallback;
        FLACDLL.FLAC__StreamDecoderTellCallback m_tellCallback;
        FLACDLL.FLAC__StreamDecoderLengthCallback m_lengthCallback;
        FLACDLL.FLAC__StreamDecoderEofCallback m_eofCallback;
        FLACDLL.FLAC__StreamDecoderWriteCallback m_writeCallback;
        FLACDLL.FLAC__StreamDecoderMetadataCallback m_metadataCallback;
        FLACDLL.FLAC__StreamDecoderErrorCallback m_errorCallback;
    }
}
