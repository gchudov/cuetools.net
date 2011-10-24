using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public class BE_VERSION
    {
        public const uint BE_MAX_HOMEPAGE = 256;
        public byte byDLLMajorVersion;
        public byte byDLLMinorVersion;
        public byte byMajorVersion;
        public byte byMinorVersion;
        // DLL Release date
        public byte byDay;
        public byte byMonth;
        public ushort wYear;

        //Homepage URL
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 257/*BE_MAX_HOMEPAGE+1*/)]
        public string zHomepage;

        public byte byAlphaLevel;
        public byte byBetaLevel;
        public byte byMMXEnabled;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 125)]
        public byte[] btReserved;

        public BE_VERSION()
        {
            btReserved = new byte[125];
        }
    }
}
