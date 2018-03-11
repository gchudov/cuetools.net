using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.libFLAC
{
    internal struct FLAC__FrameHeader
    {
        internal int blocksize;
        internal int sample_rate;
        internal int channels;
        internal FLAC__ChannelAssignment channel_assignment;
        internal int bits_per_sample;
        internal FLAC__FrameNumberType number_type;
        internal ulong sample_number; // can be uint frame_number depending on number_type
        internal byte crc;
    };

    //internal struct FLAC__Subframe
    //{
    //    FLAC__SubframeType type;
    //    [FieldOffset(4)]
    //    FLAC__Subframe_Constant data_constant;
    //    [FieldOffset(4)]
    //    FLAC__Subframe_Fixed data_fixed;
    //    [FieldOffset(4)]
    //    FLAC__Subframe_LPC data_lpc;
    //    [FieldOffset(4)]
    //    FLAC__Subframe_Verbatim data_verbatim;
	   // uint wasted_bits;
    //};

    internal unsafe struct FLAC__Frame
    {
        internal FLAC__FrameHeader header;
        //fixed FLAC__Subframe subframes[FLACDLL.FLAC__MAX_CHANNELS];
        //FLAC__FrameFooter footer;
    };

    [StructLayout(LayoutKind.Explicit), Serializable]
    internal struct FLAC__StreamMetadata
    {
        [FieldOffset(0)]
        internal FLAC__MetadataType type;
        [FieldOffset(4)]
        internal int is_last;
        [FieldOffset(8)]
        internal uint length;
        [FieldOffset(16)]
        internal FLAC__StreamMetadata_StreamInfo stream_info;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_Padding padding;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_Application application;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_SeekTable seek_table;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_VorbisComment vorbis_comment;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_CueSheet cue_sheet;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_Picture picture;
        // [FieldOffset(16)]
        // internal FLAC__StreamMetadata_Unknown unknown;
    };

    [StructLayout(LayoutKind.Sequential), Serializable]
    internal unsafe struct FLAC__StreamMetadata_StreamInfo
    {
        internal int min_blocksize, max_blocksize;
        internal int min_framesize, max_framesize;
        internal int sample_rate;
        internal int channels;
        internal int bits_per_sample;
        internal long total_samples;
        internal fixed byte md5sum[16];
    };

    internal enum FLAC__ChannelAssignment
    {
        FLAC__CHANNEL_ASSIGNMENT_INDEPENDENT = 0,
        FLAC__CHANNEL_ASSIGNMENT_LEFT_SIDE = 1,
        FLAC__CHANNEL_ASSIGNMENT_RIGHT_SIDE = 2,
        FLAC__CHANNEL_ASSIGNMENT_MID_SIDE = 3
    };

    internal enum FLAC__FrameNumberType
    {
        FLAC__FRAME_NUMBER_TYPE_FRAME_NUMBER,
        FLAC__FRAME_NUMBER_TYPE_SAMPLE_NUMBER
    };

    internal enum FLAC__StreamDecoderInitStatus
    {
        FLAC__STREAM_DECODER_INIT_STATUS_OK = 0,
        FLAC__STREAM_DECODER_INIT_STATUS_UNSUPPORTED_CONTAINER,
        FLAC__STREAM_DECODER_INIT_STATUS_INVALID_CALLBACKS,
        FLAC__STREAM_DECODER_INIT_STATUS_MEMORY_ALLOCATION_ERROR,
        FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE,
        FLAC__STREAM_DECODER_INIT_STATUS_ALREADY_INITIALIZED
    };

    internal enum FLAC__StreamDecoderWriteStatus
    {
        FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE,
        FLAC__STREAM_DECODER_WRITE_STATUS_ABORT
    };

    internal enum FLAC__StreamDecoderReadStatus
    {
        FLAC__STREAM_DECODER_READ_STATUS_CONTINUE,
        FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM,
        FLAC__STREAM_DECODER_READ_STATUS_ABORT
    };

    internal enum FLAC__StreamDecoderSeekStatus
    {
        FLAC__STREAM_DECODER_SEEK_STATUS_OK,
        FLAC__STREAM_DECODER_SEEK_STATUS_ERROR,
        FLAC__STREAM_DECODER_SEEK_STATUS_UNSUPPORTED
    };

    internal enum FLAC__StreamDecoderTellStatus
    {
        FLAC__STREAM_DECODER_TELL_STATUS_OK,
        FLAC__STREAM_DECODER_TELL_STATUS_ERROR,
        FLAC__STREAM_DECODER_TELL_STATUS_UNSUPPORTED
    };

    internal enum FLAC__StreamDecoderLengthStatus
    {
        FLAC__STREAM_DECODER_LENGTH_STATUS_OK,
        FLAC__STREAM_DECODER_LENGTH_STATUS_ERROR,
        FLAC__STREAM_DECODER_LENGTH_STATUS_UNSUPPORTED
    };

    internal enum FLAC__StreamDecoderErrorStatus
    {
        FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC,
        FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER,
        FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH,
        FLAC__STREAM_DECODER_ERROR_STATUS_UNPARSEABLE_STREAM
    };

    internal enum FLAC__StreamDecoderState
    {
        FLAC__STREAM_DECODER_SEARCH_FOR_METADATA = 0,
        FLAC__STREAM_DECODER_READ_METADATA,
        FLAC__STREAM_DECODER_SEARCH_FOR_FRAME_SYNC,
        FLAC__STREAM_DECODER_READ_FRAME,
        FLAC__STREAM_DECODER_END_OF_STREAM,
        FLAC__STREAM_DECODER_OGG_ERROR,
        FLAC__STREAM_DECODER_SEEK_ERROR,
        FLAC__STREAM_DECODER_ABORTED,
        FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR,
        FLAC__STREAM_DECODER_UNINITIALIZED
    };

    internal enum FLAC__StreamEncoderState : int
    {
        FLAC__STREAM_ENCODER_OK = 0,
        FLAC__STREAM_ENCODER_UNINITIALIZED,
        FLAC__STREAM_ENCODER_OGG_ERROR,
        FLAC__STREAM_ENCODER_VERIFY_DECODER_ERROR,
        FLAC__STREAM_ENCODER_VERIFY_MISMATCH_IN_AUDIO_DATA,
        FLAC__STREAM_ENCODER_CLIENT_ERROR,
        FLAC__STREAM_ENCODER_IO_ERROR,
        FLAC__STREAM_ENCODER_FRAMING_ERROR,
        FLAC__STREAM_ENCODER_MEMORY_ALLOCATION_ERROR
    };

    internal enum FLAC__StreamEncoderInitStatus
    {
        FLAC__STREAM_ENCODER_INIT_STATUS_OK = 0,
        FLAC__STREAM_ENCODER_INIT_STATUS_ENCODER_ERROR,
        FLAC__STREAM_ENCODER_INIT_STATUS_UNSUPPORTED_CONTAINER,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_CALLBACKS,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_NUMBER_OF_CHANNELS,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_BITS_PER_SAMPLE,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_SAMPLE_RATE,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_BLOCK_SIZE,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_MAX_LPC_ORDER,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_QLP_COEFF_PRECISION,
        FLAC__STREAM_ENCODER_INIT_STATUS_BLOCK_SIZE_TOO_SMALL_FOR_LPC_ORDER,
        FLAC__STREAM_ENCODER_INIT_STATUS_NOT_STREAMABLE,
        FLAC__STREAM_ENCODER_INIT_STATUS_INVALID_METADATA,
        FLAC__STREAM_ENCODER_INIT_STATUS_ALREADY_INITIALIZED
    };

    internal enum FLAC__MetadataType : int
    {
        FLAC__METADATA_TYPE_STREAMINFO = 0,
        FLAC__METADATA_TYPE_PADDING = 1,
        FLAC__METADATA_TYPE_APPLICATION = 2,
        FLAC__METADATA_TYPE_SEEKTABLE = 3,
        FLAC__METADATA_TYPE_VORBIS_COMMENT = 4,
        FLAC__METADATA_TYPE_CUESHEET = 5,
        FLAC__METADATA_TYPE_PICTURE = 6,
        FLAC__METADATA_TYPE_UNDEFINED = 7,
        FLAC__MAX_METADATA_TYPE = 126,
    };

    internal enum FLAC__StreamEncoderWriteStatus
    {
        FLAC__STREAM_ENCODER_WRITE_STATUS_OK = 0,
        FLAC__STREAM_ENCODER_WRITE_STATUS_FATAL_ERROR
    };

    internal enum FLAC__StreamEncoderSeekStatus
    {
        FLAC__STREAM_ENCODER_SEEK_STATUS_OK,
        FLAC__STREAM_ENCODER_SEEK_STATUS_ERROR,
        FLAC__STREAM_ENCODER_SEEK_STATUS_UNSUPPORTED
    };

    internal enum FLAC__StreamEncoderTellStatus
    {
        FLAC__STREAM_ENCODER_TELL_STATUS_OK,
        FLAC__STREAM_ENCODER_TELL_STATUS_ERROR,
        FLAC__STREAM_ENCODER_TELL_STATUS_UNSUPPORTED
    };
}
