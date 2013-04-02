using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterCBRSettings : LameWriterSettings
    {
        public static readonly int[] bps_table = new int[] { 96, 128, 192, 256, 320 };

        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public LameWriterCBRSettings()
            : base("96 128 192 256 320", "256")
        {
        }

        public override void Apply(IntPtr lame)
        {
            LameWriter.lame_set_VBR(lame, (int)LameVbrMode.Off);
            LameWriter.lame_set_brate(lame, LameWriterCBRSettings.bps_table[this.EncoderModeIndex]);
            LameWriter.lame_set_quality(lame, (int)this.Quality);
        }
    }
}
