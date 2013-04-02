using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterCBRSettings : AudioEncoderSettings
    {
        public static readonly int[] bps_table = new int[] { 96, 128, 192, 256, 320 };

        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public LameWriterCBRSettings()
            : base("96 128 192 256 320", "256")
        {
        }
    }
}
