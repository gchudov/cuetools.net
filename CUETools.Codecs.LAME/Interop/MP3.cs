using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    [StructLayout(LayoutKind.Sequential), Serializable]
    public struct MP3 //BE_CONFIG_MP3
    {
        public uint dwSampleRate;		// 48000, 44100 and 32000 allowed
        public byte byMode;			// BE_MP3_MODE_STEREO, BE_MP3_MODE_DUALCHANNEL, BE_MP3_MODE_MONO
        public ushort wBitrate;		// 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256 and 320 allowed
        public int bPrivate;
        public int bCRC;
        public int bCopyright;
        public int bOriginal;
    }
}
