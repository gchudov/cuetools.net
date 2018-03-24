using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    [StructLayout(LayoutKind.Explicit), Serializable]
    public class Format
    {
        [FieldOffset(0)]
        public MP3 mp3;
        [FieldOffset(0)]
        public LHV1 lhv1;
        [FieldOffset(0)]
        public ACC acc;

        public Format(AudioPCMConfig format, uint MpeBitRate, uint quality)
        {
            lhv1 = new LHV1(format, MpeBitRate, quality);
        }
    }
}
