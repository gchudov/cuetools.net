using System.ComponentModel;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    public class LAMEEncoderCBRSettings : AudioEncoderSettings
    {
        public static readonly uint[] bps_table = new uint[] { 96, 128, 192, 256, 320 };

        [DefaultValue(0)]
        public int CustomBitrate { get; set; }

        [DefaultValue(MpegMode.STEREO)]
        public MpegMode StereoMode { get; set; }

        public LAMEEncoderCBRSettings()
            : base("96 128 192 256 320", "256")
        {
        }
    }
}
