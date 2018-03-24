using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace CUETools.Codecs.libmp3lame
{
    class libmp3lamedll
    {
        private const string DllName = "libmp3lame";
        private const CallingConvention LameCallingConvention = CallingConvention.Cdecl;

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern IntPtr lame_init();
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_close(IntPtr handle);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_num_channels(IntPtr handle, int channels);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_in_samplerate(IntPtr handle, int sampleRate);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_quality(IntPtr handle, int quality);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR(IntPtr handle, int vbrMode);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR_mean_bitrate_kbps(IntPtr handle, int meanBitrate);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_init_params(IntPtr handle);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_num_samples(IntPtr handle, uint numSamples);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_encode_buffer_interleaved(IntPtr handle, IntPtr pcm, int num_samples, IntPtr mp3buf, int mp3buf_size);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_encode_flush(IntPtr handle, IntPtr mp3buf, int size);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern uint lame_get_lametag_frame(IntPtr handle, IntPtr buffer, uint size);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_VBR_quality(IntPtr handle, float vbrQuality);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_brate(IntPtr handle, int bitrate);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_bWriteVbrTag(IntPtr handle, int writeVbrTag);
        [DllImport(DllName, CallingConvention = LameCallingConvention)]
        internal static extern int lame_set_write_id3tag_automatic(IntPtr handle, int automaticWriteId3Tag);

        static libmp3lamedll()
        {
            var myPath = new Uri(typeof(libmp3lamedll).Assembly.CodeBase).LocalPath;
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
    }
}
