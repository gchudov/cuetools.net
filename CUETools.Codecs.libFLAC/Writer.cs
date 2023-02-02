using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using CUETools.Codecs;
using Newtonsoft.Json;

namespace CUETools.Codecs.libFLAC
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "flac";

        [Browsable(false)]
        public string Name => "libFLAC";

        [Browsable(false)]
        public Type EncoderType => typeof(Encoder);

        [Browsable(false)]
        public bool Lossless => true;

        [Browsable(false)]
        public int Priority => 2;

        [Browsable(false)]
        public string SupportedModes => "0 1 2 3 4 5 6 7 8";

        [Browsable(false)]
        public string DefaultMode => "5";

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
        public string EncoderMode { get; set; }

        [Browsable(false)]
        public AudioPCMConfig PCM { get; set; }

        [Browsable(false)]
        public int BlockSize { get; set; }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding { get; set; }

        public IAudioEncoderSettings Clone()
        {
            return MemberwiseClone() as IAudioEncoderSettings;
        }
        #endregion

        public EncoderSettings()
        {
            this.Init();
        }

        [DefaultValue(false)]
        [DisplayName("Verify")]
        [Description("Decode each frame and compare with original")]
        [JsonProperty]
        public bool Verify { get; set; }

        [DefaultValue(true)]
        [DisplayName("MD5")]
        [Description("Calculate MD5 hash for audio stream")]
        [JsonProperty]
        public bool MD5Sum { get; set; }

        [DisplayName("Version")]
        [Description("Library version")]
        public string Version => FLACDLL.GetVersion;
    };

    public unsafe class Encoder : IAudioDest
    {
        public Encoder(EncoderSettings settings, string path, Stream output = null)
        {
            m_path = path;
            m_stream = output;
            m_settings = settings;
            m_streamGiven = output != null;
            m_initialized = false;
            m_finalSampleCount = -1;
            m_samplesWritten = 0;
            m_write_callback = StreamEncoderWriteCallback;
            m_seek_callback = StreamEncoderSeekCallback;
            m_tell_callback = StreamEncoderTellCallback;

            if (m_settings.PCM.BitsPerSample < 16 || m_settings.PCM.BitsPerSample > 24)
                throw new Exception("bits per sample must be 16..24");

            m_encoder = FLACDLL.FLAC__stream_encoder_new();

            FLACDLL.FLAC__stream_encoder_set_bits_per_sample(m_encoder, (uint)m_settings.PCM.BitsPerSample);
            FLACDLL.FLAC__stream_encoder_set_channels(m_encoder, (uint)m_settings.PCM.ChannelCount);
            FLACDLL.FLAC__stream_encoder_set_sample_rate(m_encoder, (uint)m_settings.PCM.SampleRate);
        }

        public IAudioEncoderSettings Settings => m_settings;

        public string Path { get => m_path; }

        public long FinalSampleCount
        {
            get => m_finalSampleCount;
            set
            {
                if (m_initialized)
                    throw new Exception("final sample count cannot be changed after encoding begins");
                m_finalSampleCount = value;
            }
        }

        public void Close()
        {
            if (m_initialized)
            {
                FLACDLL.FLAC__stream_encoder_finish(m_encoder);
                FLACDLL.FLAC__stream_encoder_delete(m_encoder);
                m_encoder = IntPtr.Zero;
                m_initialized = false;
            }
            if (m_stream != null)
            {
                m_stream.Close();
                m_stream = null;
            }
            if ((m_finalSampleCount > 0) && (m_samplesWritten != m_finalSampleCount))
                throw new Exception("samples written differs from the expected sample count");
        }

        public void Delete()
        {
            try
            {
                if (m_initialized)
                {
                    FLACDLL.FLAC__stream_encoder_delete(m_encoder);
                    m_encoder = IntPtr.Zero;
                    m_initialized = false;
                }
                if (m_stream != null)
                {
                    m_stream.Close();
                    m_stream = null;
                }
            }
            catch (Exception)
            {
            }
            if (m_path != "")
                File.Delete(m_path);
        }

        public void Write(AudioBuffer sampleBuffer)
        {
            if (!m_initialized) Initialize();

            sampleBuffer.Prepare(this);

            fixed (int* pSampleBuffer = &sampleBuffer.Samples[0, 0])
            {
                if (0 == FLACDLL.FLAC__stream_encoder_process_interleaved(m_encoder,
                    pSampleBuffer, sampleBuffer.Length))
                {
                    var state = FLACDLL.FLAC__stream_encoder_get_state(m_encoder);
                    string status = state.ToString();
                    if (state == FLAC__StreamEncoderState.FLAC__STREAM_ENCODER_VERIFY_MISMATCH_IN_AUDIO_DATA)
                    {
                        ulong absolute_sample;
                        uint frame_number;
                        uint channel;
                        uint sample;
                        int expected, got;
                        FLACDLL.FLAC__stream_encoder_get_verify_decoder_error_stats(m_encoder, out absolute_sample, out frame_number, out channel, out sample, out expected, out got);
                        status = status + String.Format("({0:x} instead of {1:x} @{2:x})", got, expected, absolute_sample);
                    }
                    throw new Exception("an error occurred while encoding: " + status);
                }
            }

            m_samplesWritten += sampleBuffer.Length;
        }

        internal FLAC__StreamEncoderWriteStatus StreamEncoderWriteCallback(IntPtr encoder, byte[] buffer, UIntPtr bytes, int samples, int current_frame, void* client_data)
        {
            try
            {
                m_stream.Write(buffer, 0, (int)bytes);
            }
            catch (Exception)
            {
                return FLAC__StreamEncoderWriteStatus.FLAC__STREAM_ENCODER_WRITE_STATUS_FATAL_ERROR;
            }
            return FLAC__StreamEncoderWriteStatus.FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
        }

        internal FLAC__StreamEncoderSeekStatus StreamEncoderSeekCallback(IntPtr encoder, long absolute_byte_offset, void* client_data)
        {
            if (!m_stream.CanSeek) return  FLAC__StreamEncoderSeekStatus.FLAC__STREAM_ENCODER_SEEK_STATUS_UNSUPPORTED;
            try
            {
                m_stream.Position = absolute_byte_offset;
            }
            catch (Exception)
            {
                return FLAC__StreamEncoderSeekStatus.FLAC__STREAM_ENCODER_SEEK_STATUS_ERROR;
            }
            return FLAC__StreamEncoderSeekStatus.FLAC__STREAM_ENCODER_SEEK_STATUS_OK;
        }

        internal FLAC__StreamEncoderTellStatus StreamEncoderTellCallback(IntPtr encoder, out long absolute_byte_offset, void* client_data)
        {
            if (!m_stream.CanSeek)
            {
                absolute_byte_offset = -1;
                return FLAC__StreamEncoderTellStatus.FLAC__STREAM_ENCODER_TELL_STATUS_UNSUPPORTED;
            }
            try
            {
                absolute_byte_offset = m_stream.Position;
            }
            catch (Exception)
            {
                absolute_byte_offset = -1;
                return FLAC__StreamEncoderTellStatus.FLAC__STREAM_ENCODER_TELL_STATUS_ERROR;
            }
            return FLAC__StreamEncoderTellStatus.FLAC__STREAM_ENCODER_TELL_STATUS_OK;
        }

        void Initialize()
        {
            if (m_stream == null)
                m_stream = new FileStream(m_path, FileMode.Create, FileAccess.Write, FileShare.Read, 0x10000);

            var metadata = stackalloc FLAC__StreamMetadata*[4];
            int metadataCount = 0;
            FLAC__StreamMetadata* padding, seektable, vorbiscomment;

            if (m_finalSampleCount > 0)
            {
                seektable = FLACDLL.FLAC__metadata_object_new(FLAC__MetadataType.FLAC__METADATA_TYPE_SEEKTABLE);
                FLACDLL.FLAC__metadata_object_seektable_template_append_spaced_points_by_samples(
                    seektable, m_settings.PCM.SampleRate * 10, m_finalSampleCount);
                FLACDLL.FLAC__metadata_object_seektable_template_sort(seektable, 1);
                metadata[metadataCount++] = seektable;
            }

            vorbiscomment = FLACDLL.FLAC__metadata_object_new(FLAC__MetadataType.FLAC__METADATA_TYPE_VORBIS_COMMENT);
            metadata[metadataCount++] = vorbiscomment;

            if (m_settings.Padding != 0)
            {
                padding = FLACDLL.FLAC__metadata_object_new(FLAC__MetadataType.FLAC__METADATA_TYPE_PADDING);
                padding->length = (uint)m_settings.Padding;
                metadata[metadataCount++] = padding;
            }

            FLACDLL.FLAC__stream_encoder_set_metadata(m_encoder, metadata, metadataCount);
            FLACDLL.FLAC__stream_encoder_set_verify(m_encoder, m_settings.Verify ? 1 : 0);
            FLACDLL.FLAC__stream_encoder_set_do_md5(m_encoder, m_settings.MD5Sum ? 1 : 0);
            FLACDLL.FLAC__stream_encoder_set_compression_level(m_encoder, m_settings.GetEncoderModeIndex());
            if (m_finalSampleCount > 0)
                FLACDLL.FLAC__stream_encoder_set_total_samples_estimate(m_encoder, m_finalSampleCount);
            if (m_settings.BlockSize > 0)
                FLACDLL.FLAC__stream_encoder_set_blocksize(m_encoder, m_settings.BlockSize);

            FLAC__StreamEncoderInitStatus st = FLACDLL.FLAC__stream_encoder_init_stream(
                m_encoder, m_write_callback, m_stream.CanSeek ? m_seek_callback : null,
                m_stream.CanSeek ? m_tell_callback : null, null, null);
            if (st != FLAC__StreamEncoderInitStatus.FLAC__STREAM_ENCODER_INIT_STATUS_OK)
                throw new Exception(string.Format("unable to initialize the encoder: {0}", st));

            m_initialized = true;
        }

        EncoderSettings m_settings;
        Stream m_stream;
        bool m_streamGiven;
        IntPtr m_encoder;
        bool m_initialized;
        string m_path;
        Int64 m_finalSampleCount, m_samplesWritten;
        FLACDLL.FLAC__StreamEncoderWriteCallback m_write_callback;
        FLACDLL.FLAC__StreamEncoderSeekCallback m_seek_callback;
        FLACDLL.FLAC__StreamEncoderTellCallback m_tell_callback;
    }
}
