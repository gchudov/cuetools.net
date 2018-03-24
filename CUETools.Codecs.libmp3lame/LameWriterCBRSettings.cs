using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.libmp3lame
{
    [JsonObject(MemberSerialization.OptIn)]
    public class CBREncoderSettings : LameEncoderSettings
    {
        public override string Extension => "mp3";

        public override string Name => "libmp3lame-CBR";

        public override int Priority => 1;

        public static readonly int[] bps_table = new int[] { 96, 128, 192, 256, 320 };

        [JsonProperty]
        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public CBREncoderSettings()
            : base("96 128 192 256 320", "256")
        {
        }

        public override void Apply(IntPtr lame)
        {
            libmp3lamedll.lame_set_VBR(lame, (int)LameVbrMode.Off);
            libmp3lamedll.lame_set_brate(lame, CBREncoderSettings.bps_table[this.EncoderModeIndex]);
            libmp3lamedll.lame_set_quality(lame, (int)this.Quality);
        }
    }
}
