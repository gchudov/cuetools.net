using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.MACLib
{
    internal unsafe static class MACLibDll
    {
        internal const string DllName = "MACLibDll";
        internal const CallingConvention DllCallingConvention = CallingConvention.StdCall;

        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int GetVersionNumber();

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr GetVersionString();

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr c_APECIO_Create(
            void* pUserData,
            CIO_ReadDelegate CIO_Read,
            CIO_WriteDelegate CIO_Write,
            CIO_SeekDelegate CIO_Seek,
            CIO_GetPositionDelegate CIO_GetPosition,
            CIO_GetSizeDelegate CIO_GetSize);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern void c_APECIO_Destroy(IntPtr hCIO);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr c_APECompress_Create(out int error);
        
        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern void c_APECompress_Destroy(IntPtr hAPECompress);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int c_APECompress_Finish(IntPtr hAPECompress, char* pTerminatingData, int nTerminatingBytes, int nWAVTerminatingBytes);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int c_APECompress_AddData(IntPtr hAPECompress, byte* pData, int nBytes);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int c_APECompress_StartEx(
            IntPtr hAPECompress, 
            IntPtr hCIO,
            WAVEFORMATEX* pwfeInput,
            int nMaxAudioBytes,
            int nCompressionLevel, 
            void* pHeaderData, 
            int nHeaderBytes);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr c_APEDecompress_CreateEx(IntPtr hCIO, out int pErrorCode);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern IntPtr c_APEDecompress_GetInfo(IntPtr hAPEDecompress, APE_DECOMPRESS_FIELDS Field, int nParam1 = 0, int nParam2 = 0);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int c_APEDecompress_Seek(IntPtr hAPEDecompress, int nBlockOffset);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern int c_APEDecompress_GetData(IntPtr hAPEDecompress, char* pBuffer, IntPtr nBlocks, out long pBlocksRetrieved);

        [DllImport(DllName, CallingConvention = DllCallingConvention)]
        internal static extern void c_APEDecompress_Destroy(IntPtr hAPEDecompress);

        static MACLibDll()
        {
            var myPath = new Uri(typeof(MACLibDll).Assembly.CodeBase).LocalPath;
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
