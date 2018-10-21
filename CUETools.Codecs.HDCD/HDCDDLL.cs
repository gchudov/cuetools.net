using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.HDCD
{
    internal static class HDCDDLL
    {
        internal const string DllName = "hdcd";

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);
        [DllImport("kernel32.dll")]
        public static extern IntPtr GetProcAddress(IntPtr hModule, string procedureName);

        internal delegate bool hdcd_decoder_write_callback(IntPtr decoder, IntPtr buffer, int samples, IntPtr client_data);

        [DllImport(DllName)]
        internal static extern IntPtr hdcd_decoder_new();
        [DllImport(DllName)]
        internal static extern void hdcd_decoder_delete(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern hdcd_decoder_state hdcd_decoder_get_state(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_set_num_channels(IntPtr decoder, Int16 num_channels);
        //HDCD_API uint16_t hdcd_decoder_get_num_channels(const hdcd_decoder *const _decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_set_sample_rate(IntPtr decoder, Int32 sample_rate);
        //HDCD_API uint32_t hdcd_decoder_get_sample_rate(const hdcd_decoder *const _decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_set_input_bps(IntPtr decoder, Int16 input_bps);
        //HDCD_API uint16_t hdcd_decoder_get_input_bps(const hdcd_decoder *const _decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_set_output_bps(IntPtr decoder, Int16 output_bps);
        //HDCD_API uint16_t hdcd_decoder_get_output_bps(const hdcd_decoder *const _decoder);
        [DllImport(DllName)]
        internal static extern hdcd_decoder_init_status hdcd_decoder_init(IntPtr decoder, IntPtr unused, hdcd_decoder_write_callback write_callback, IntPtr client_data);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_finish(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_process_buffer_interleaved(IntPtr decoder, [In, Out] int[,] input_buffer, Int32 samples);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_flush_buffer(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_reset(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern bool hdcd_decoder_detected_hdcd(IntPtr decoder);
        [DllImport(DllName)]
        internal static extern IntPtr hdcd_decoder_get_statistics(IntPtr decoder);

        static HDCDDLL()
        {
            var myPath = new Uri(typeof(HDCDDLL).Assembly.CodeBase).LocalPath;
            var myFolder = System.IO.Path.GetDirectoryName(myPath);
            var is64 = IntPtr.Size == 8;
            var subfolder = is64 ? "x64" : "win32";
#if NET47
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(myFolder, subfolder, DllName + ".dll"));
#else
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(System.IO.Path.Combine(myFolder, subfolder), DllName + ".dll"));
#endif
            if (Dll == IntPtr.Zero)
                Dll = LoadLibrary(DllName + ".dll");
            if (Dll == IntPtr.Zero)
                throw new DllNotFoundException();
        }
    }

    /** \brief Statistics for decoding. */
    [StructLayout(LayoutKind.Sequential)]
    public struct hdcd_decoder_statistics
    {
        public UInt32 num_packets;
        /**<Total number of samples processed. */
        public bool enabled_peak_extend;
        /**< True if peak extend was enabled during decoding. */
        public bool disabled_peak_extend;
        /**< True if peak extend was disabled during decoding. */
        public double min_gain_adjustment;
        /**< Minimum dynamic gain used during decoding. */
        public double max_gain_adjustment;
        /**< Maximum dynamic gain used during decoding. */
        public bool enabled_transient_filter;
        /**< True if the transient filter was enabled during decoding. */
        public bool disabled_transient_filter;
        /**< True if the transient filter was disabled during decoding. */
    };

    /** \brief Return values from hdcd_decoder_init. */
    internal enum hdcd_decoder_init_status
    {
        HDCD_DECODER_INIT_STATUS_OK = 0,
        /**< Initialization was successful. */
        HDCD_DECODER_INIT_STATUS_INVALID_STATE,
        /**< The _decoder was already initialised. */
        HDCD_DECODER_INIT_STATUS_MEMORY_ALOCATION_ERROR,
        /**< Initialization failed due to a memory allocation error. */
        HDCD_DECODER_INIT_STATUS_INVALID_NUM_CHANNELS,
        /**< Initialization failed because the configured number of channels was invalid. */
        HDCD_DECODER_INIT_STATUS_INVALID_SAMPLE_RATE,
        /**< Initialization failed because the configured sample rate was invalid. */
        HDCD_DECODER_INIT_STATUS_INVALID_INPUT_BPS,
        /**< Initialization failed because the configured input bits per sample was invalid. */
        HDCD_DECODER_INIT_STATUS_INVALID_OUTPUT_BPS
        /**< Initialization failed because the configured output bits per sample was invalid. */
    }

    /** \brief State values for a decoder.
     *
     * The decoder's state can be obtained by calling hdcd_decoder_get_state().
     */
    internal enum hdcd_decoder_state
    {
        HDCD_DECODER_STATE_UNINITIALISED = 1,
        /**< The decoder is uninitialised. */
        HDCD_DECODER_STATE_READY,
        /**< The decoder is initialised and ready to process data. */
        HDCD_DECODER_STATE_DIRTY,
        /**< The decoder has processed data, but has not yet been flushed. */
        HDCD_DECODER_STATE_FLUSHED,
        /**< The decoder has been flushed. */
        HDCD_DECODER_STATE_WRITE_ERROR,
        /**< An error was returned by the write callback. */
        HDCD_DECODER_STATE_MEMORY_ALOCATION_ERROR
        /**< Processing failed due to a memory allocation error. */
    };
}
