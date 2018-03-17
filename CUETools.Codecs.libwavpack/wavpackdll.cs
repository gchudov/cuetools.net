using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.libwavpack
{
    internal unsafe static class wavpackdll
    {
        internal const string DllName = "wavpackdll";
        internal const CallingConvention DllCallingConvention = CallingConvention.Cdecl;

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern WavpackContext* WavpackOpenFileInputEx64 (WavpackStreamReader64 *reader, void *wv_id, void *wvc_id,
            [param: MarshalAs(UnmanagedType.LPTStr), Out()] out string error, 
            OpenFlags flags, int norm_offset);
        
        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern long WavpackGetNumSamples64 (WavpackContext *wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackSeekSample64 (WavpackContext *wpc, long sample);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern WavpackContext* WavpackCloseFile (WavpackContext *wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern uint WavpackUnpackSamples (WavpackContext *wpc, int *buffer, uint samples);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackGetBitsPerSample(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackGetNumChannels(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern uint WavpackGetSampleRate(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackGetChannelMask(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr WavpackGetLibraryVersionString();

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackStoreMD5Sum(WavpackContext* wpc, byte* data);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackFlushSamples(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackPackSamples(WavpackContext* wpc, int* sample_buffer, uint sample_count);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern string WavpackGetErrorMessage(WavpackContext* wpc);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern WavpackContext* WavpackOpenFileOutput(EncoderBlockOutput blockout, void* wv_id, void* wvc_id);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackSetConfiguration64(WavpackContext* wpc, WavpackConfig* config, long total_samples, byte* chan_ids);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int WavpackPackInit(WavpackContext* wpc);


        static wavpackdll()
        {
            var myPath = new Uri(typeof(wavpackdll).Assembly.CodeBase).LocalPath;
            var myFolder = System.IO.Path.GetDirectoryName(myPath);
            var is64 = IntPtr.Size == 8;
            var subfolder = is64 ? "x64" : "win32";
#if NET40
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(myFolder, subfolder, DllName + ".dll"));
#else
            IntPtr Dll = LoadLibrary(System.IO.Path.Combine(System.IO.Path.Combine(myFolder, subfolder), DllName + ".dll"));
#endif
            if (Dll == IntPtr.Zero)
                Dll = LoadLibrary(DllName + ".dll");
            if (Dll == IntPtr.Zero)
                throw new DllNotFoundException();
        }
    };
}
