using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterVBRSettings: AudioEncoderSettings
    {
		[DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public LameWriterVBRSettings()
            : base("V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2")
        {
        }
    }
}
