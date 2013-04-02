using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterVBRSettings : LameWriterSettings
    {
		[DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public LameWriterVBRSettings()
            : base("V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2")
        {
        }
    
        public override void Apply(IntPtr lame)
        {
            LameWriter.lame_set_VBR(lame, (int)LameVbrMode.Default);
            LameWriter.lame_set_VBR_quality(lame, 9 - this.EncoderModeIndex);
            LameWriter.lame_set_quality(lame, (int)this.Quality);
        }
    }
}
