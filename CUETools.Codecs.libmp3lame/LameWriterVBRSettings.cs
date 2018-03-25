using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.libmp3lame
{
    [JsonObject(MemberSerialization.OptIn)]
    public class VBREncoderSettings : LameEncoderSettings
    {
        public override string Name => "libmp3lame-VBR";

        public override int Priority => 2;

        public override string SupportedModes => "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0";

        public override string DefaultMode => "V2";

        [JsonProperty]
        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public VBREncoderSettings()
            : base()
        {
        }
    
        public override void Apply(IntPtr lame)
        {
            libmp3lamedll.lame_set_VBR(lame, (int)LameVbrMode.Default);
            libmp3lamedll.lame_set_VBR_quality(lame, 9 - this.GetEncoderModeIndex());
            libmp3lamedll.lame_set_quality(lame, (int)this.Quality);
        }
    }
}
