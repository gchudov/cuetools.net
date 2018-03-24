using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    [StructLayout(LayoutKind.Sequential), Serializable]
    public class BE_CONFIG
    {
        // encoding formats
        public const uint BE_CONFIG_MP3 = 0;
        public const uint BE_CONFIG_LAME = 256;

        public uint dwConfig;
        public Format format;

        public BE_CONFIG(AudioPCMConfig format, uint MpeBitRate, uint quality)
        {
            this.dwConfig = BE_CONFIG_LAME;
            this.format = new Format(format, MpeBitRate, quality);
        }

        public BE_CONFIG(AudioPCMConfig format)
            : this(format, 0, 5)
        {
        }
    }
}
