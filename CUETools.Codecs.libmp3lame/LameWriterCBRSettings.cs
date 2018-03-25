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
        public override string Name => "libmp3lame-CBR";

        public override int Priority => 1;

        public override string SupportedModes => "96 128 192 256 320";

        public override string DefaultMode => "256";

        public static readonly int[] bps_table = new int[] { 96, 128, 192, 256, 320 };

        [JsonProperty]
        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public CBREncoderSettings()
            : base()
        {
        }

        public override void Apply(IntPtr lame)
        {
            libmp3lamedll.lame_set_VBR(lame, (int)LameVbrMode.Off);
            libmp3lamedll.lame_set_brate(lame, CBREncoderSettings.bps_table[this.GetEncoderModeIndex()]);
            libmp3lamedll.lame_set_quality(lame, (int)this.Quality);
        }
    }
}
