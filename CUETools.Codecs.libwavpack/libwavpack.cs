using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.libwavpack
{
    internal enum OpenFlags : int
    {
        OPEN_WVC = 0x1,             // open/read "correction" file
        OPEN_TAGS = 0x2,            // read ID3v1 / APEv2 tags (seekable file)
        OPEN_WRAPPER = 0x4,         // make audio wrapper available (i.e. RIFF)
        OPEN_2CH_MAX = 0x8,         // open multichannel as stereo (no downmix)
        OPEN_NORMALIZE = 0x10,      // normalize floating point data to +/- 1.0
        OPEN_STREAMING = 0x20,      // "streaming" mode blindly unpacks blocks
                                    // w/o regard to header file position info
        OPEN_EDIT_TAGS = 0x40,      // allow editing of tags
        OPEN_FILE_UTF8 = 0x80,      // assume filenames are UTF-8 encoded, not ANSI (Windows only)
                                    // new for version 5
        OPEN_DSD_NATIVE = 0x100,    // open DSD files as bitstreams
                                    // (returned as 8-bit "samples" stored in 32-bit words)
        OPEN_DSD_AS_PCM = 0x200,    // open DSD files as 24-bit PCM (decimated 8x)
        OPEN_ALT_TYPES = 0x400,     // application is aware of alternate file types & qmode
                                    // (just affects retrieving wrappers & MD5 checksums)
        OPEN_NO_CHECKSUM = 0x800,   // don't verify block checksums before decoding
    };

    internal enum ConfigFlags : uint
    {
        CONFIG_BYTES_STORED = 3,            // 1-4 bytes/sample
        CONFIG_MONO_FLAG = 4,               // not stereo
        CONFIG_HYBRID_FLAG = 8,             // hybrid mode
        CONFIG_JOINT_STEREO = 0x10,         // joint stereo
        CONFIG_CROSS_DECORR = 0x20,         // no-delay cross decorrelation
        CONFIG_HYBRID_SHAPE = 0x40,         // noise shape (hybrid mode only)
        CONFIG_FLOAT_DATA = 0x80,           // ieee 32-bit floating point data

        CONFIG_FAST_FLAG = 0x200,           // fast mode
        CONFIG_HIGH_FLAG = 0x800,           // high quality mode
        CONFIG_VERY_HIGH_FLAG = 0x1000,     // very high
        CONFIG_BITRATE_KBPS = 0x2000,       // bitrate is kbps, not bits / sample
        CONFIG_AUTO_SHAPING = 0x4000,       // automatic noise shaping
        CONFIG_SHAPE_OVERRIDE = 0x8000,     // shaping mode specified
        CONFIG_JOINT_OVERRIDE = 0x10000,    // joint-stereo mode specified
        CONFIG_DYNAMIC_SHAPING = 0x20000,   // dynamic noise shaping
        CONFIG_CREATE_EXE = 0x40000,        // create executable
        CONFIG_CREATE_WVC = 0x80000,        // create correction file
        CONFIG_OPTIMIZE_WVC = 0x100000,     // maximize bybrid compression
        CONFIG_COMPATIBLE_WRITE = 0x400000, // write files for decoders < 4.3
        CONFIG_CALC_NOISE = 0x800000,       // calc noise in hybrid mode
        CONFIG_LOSSY_MODE = 0x1000000,      // obsolete (for information)
        CONFIG_EXTRA_MODE = 0x2000000,      // extra processing mode
        CONFIG_SKIP_WVX = 0x4000000,        // no wvx stream w/ floats & big ints
        CONFIG_MD5_CHECKSUM = 0x8000000,    // compute & store MD5 signature
        CONFIG_MERGE_BLOCKS = 0x10000000,   // merge blocks of equal redundancy (for lossyWAV)
        CONFIG_PAIR_UNDEF_CHANS = 0x20000000, // encode undefined channels in stereo pairs
        CONFIG_OPTIMIZE_MONO = 0x80000000,  // optimize for mono streams posing as stereo
    };

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderReadDelegate(void* id, void* data, int bcount);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate UInt32 DecoderTellDelegate(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderSeekDelegate(void* id, uint pos);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderSeekRelativeDelegate(void* id, int delta, int mode);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderPushBackDelegate(void* id, int c);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate UInt32 DecoderLengthDelegate(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderCanSeekDelegate(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int WriteBytesDelegate(void* id, void* data, int bcount);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate long DecoderTellDelegate64(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderSeekDelegate64(void* id, long pos);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int DecoderSeekRelativeDelegate64(void* id, long delta, int mode);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate long DecoderLengthDelegate64(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate long DecoderTruncateDelegate(void* id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate long DecoderCloseDelegate(void* id);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate int EncoderBlockOutput(void* id, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2), In] byte[] data, int bcount);

    [StructLayout(LayoutKind.Sequential), Serializable]
    internal unsafe struct WavpackStreamReader64
    {
        internal IntPtr read_bytes;
        internal IntPtr write_bytes;
        internal IntPtr get_pos;
        internal IntPtr set_pos_abs;
        internal IntPtr set_pos_rel;
        internal IntPtr push_back_byte;
        internal IntPtr get_length;
        internal IntPtr can_seek;
        internal IntPtr truncate_here;
        internal IntPtr close;
    };

    [StructLayout(LayoutKind.Sequential), Serializable]
    internal unsafe struct WavpackConfig
    {
        internal float bitrate, shaping_weight;
        internal int bits_per_sample, bytes_per_sample;
        internal int qmode;
        internal ConfigFlags flags;
        internal int xmode, num_channels, float_norm_exp;
        internal int block_samples, extra_flags, sample_rate, channel_mask;
        internal fixed byte md5_checksum[16];
        internal byte md5_read;
        internal int num_tag_strings;
        internal char** tag_strings;
    };

    internal struct WavpackContext
    {
    }
}
