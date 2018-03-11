using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.libFLAC
{
    internal unsafe static class FLACDLL
    {
        internal const string libFLACDll = "libFLAC_dynamic";
        internal const CallingConvention libFLACCallingConvention = CallingConvention.Cdecl;
        internal const int FLAC__MAX_CHANNELS = 8;
        private static string version;

        internal static string GetVersion => version;

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32.dll")]
        public static extern IntPtr GetProcAddress(IntPtr hModule, string procedureName);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamDecoderReadStatus FLAC__StreamDecoderReadCallback(IntPtr decoder, byte* buffer, ref long bytes, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamDecoderSeekStatus FLAC__StreamDecoderSeekCallback(IntPtr decoder, long absolute_byte_offset, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamDecoderTellStatus FLAC__StreamDecoderTellCallback(IntPtr decoder, out long absolute_byte_offset, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamDecoderLengthStatus FLAC__StreamDecoderLengthCallback(IntPtr decoder, out long stream_length, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int FLAC__StreamDecoderEofCallback(IntPtr decoder, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamDecoderWriteStatus FLAC__StreamDecoderWriteCallback(IntPtr decoder, FLAC__Frame* frame, int** buffer, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void FLAC__StreamDecoderMetadataCallback(IntPtr decoder, FLAC__StreamMetadata* metadata, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void FLAC__StreamDecoderErrorCallback(IntPtr decoder, FLAC__StreamDecoderErrorStatus status, void* client_data);


        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern IntPtr FLAC__stream_decoder_new();

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_set_metadata_respond(IntPtr decoder, FLAC__MetadataType type);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_process_until_end_of_metadata(IntPtr decoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern  FLAC__StreamDecoderInitStatus FLAC__stream_decoder_init_stream(
            IntPtr decoder,
            FLAC__StreamDecoderReadCallback read_callback,
            FLAC__StreamDecoderSeekCallback seek_callback,
            FLAC__StreamDecoderTellCallback tell_callback,
            FLAC__StreamDecoderLengthCallback length_callback,
            FLAC__StreamDecoderEofCallback eof_callback,
            FLAC__StreamDecoderWriteCallback write_callback,
            FLAC__StreamDecoderMetadataCallback metadata_callback,
            FLAC__StreamDecoderErrorCallback error_callback,
            void* client_data
        );

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_finish(IntPtr decoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_delete(IntPtr decoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_seek_absolute(IntPtr decoder, long sample);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern FLAC__StreamDecoderState FLAC__stream_decoder_get_state(IntPtr decoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_decoder_process_single(IntPtr decoder);



        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern IntPtr FLAC__stream_encoder_new();

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_bits_per_sample(IntPtr encoder, uint value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_sample_rate(IntPtr encoder, uint value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_channels(IntPtr encoder, uint value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_finish(IntPtr encoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_delete(IntPtr encoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_process_interleaved(IntPtr encoder, int* buffer, int samples);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern FLAC__StreamEncoderState FLAC__stream_encoder_get_state(IntPtr encoder);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern void FLAC__stream_encoder_get_verify_decoder_error_stats(IntPtr encoder, out ulong absolute_sample, out uint frame_number, out uint channel, out uint sample, out int expected, out int got);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern FLAC__StreamMetadata* FLAC__metadata_object_new(FLAC__MetadataType type);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__metadata_object_seektable_template_append_spaced_points_by_samples(FLAC__StreamMetadata* metadata, int samples, long total_samples);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__metadata_object_seektable_template_sort(FLAC__StreamMetadata* metadata, int compact);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_metadata(IntPtr encoder, FLAC__StreamMetadata** metadata, int num_blocks);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_verify(IntPtr encoder, int value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_do_md5(IntPtr encoder, int value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_total_samples_estimate(IntPtr encoder, long value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_compression_level(IntPtr encoder, int value);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern int FLAC__stream_encoder_set_blocksize(IntPtr encoder, int value);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamEncoderWriteStatus FLAC__StreamEncoderWriteCallback(IntPtr encoder, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] byte[] buffer, long bytes, int samples, int current_frame, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamEncoderSeekStatus FLAC__StreamEncoderSeekCallback(IntPtr encoder, long absolute_byte_offset, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate FLAC__StreamEncoderTellStatus FLAC__StreamEncoderTellCallback(IntPtr encoder, out long absolute_byte_offset, void* client_data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void FLAC__StreamEncoderMetadataCallback(IntPtr encoder, FLAC__StreamMetadata* metadata, void* client_data);

        [DllImport(libFLACDll, CallingConvention = libFLACCallingConvention)]
        internal static extern FLAC__StreamEncoderInitStatus FLAC__stream_encoder_init_stream(IntPtr encoder,
            FLAC__StreamEncoderWriteCallback write_callback,
            FLAC__StreamEncoderSeekCallback seek_callback,
            FLAC__StreamEncoderTellCallback tell_callback,
            FLAC__StreamEncoderMetadataCallback metadata_callback,
            void* client_data);

        static FLACDLL()
        {
            var myPath = new Uri(typeof(FLACDLL).Assembly.CodeBase).LocalPath;
            var myFolder = System.IO.Path.GetDirectoryName(myPath);
            var is64 = IntPtr.Size == 8;
            var subfolder = is64 ? "plugins (x64)" : "plugins (win32)";
#if NET40
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(myFolder, "..", subfolder, libFLACDll + ".dll"));
#else
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(System.IO.Path.Combine(System.IO.Path.Combine(myFolder, ".."), subfolder), libFLACDll + ".dll"));
#endif
            if (Dll == IntPtr.Zero)
                Dll = LoadLibrary(libFLACDll + ".dll");
            if (Dll == IntPtr.Zero)
                throw new DllNotFoundException();
            IntPtr addr = GetProcAddress(Dll, "FLAC__VERSION_STRING");
            IntPtr ptr = Marshal.ReadIntPtr(addr);
            version = Marshal.PtrToStringAnsi(ptr);
        }
    };
}
