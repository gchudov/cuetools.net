using System.ComponentModel;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    public class LAMEEncoderCBRSettings
    {
        [DefaultValue(0)]
        public int CustomBitrate { get; set; }

        [DefaultValue(MpegMode.STEREO)]
        public MpegMode StereoMode { get; set; }

        public LAMEEncoderCBRSettings()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.ResetValue(this);
            }
        }
    }
}
