using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.BDLPCM
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : AudioDecoderSettings
    {
        public override string Extension => "m2ts";

        public override string Name => "cuetools";

        public override Type DecoderType => typeof(AudioDecoder);

        public override int Priority => 2;

        public bool IgnoreShortItems { get; set; }

        public int? Stream { get; set; }

        public ushort? Pid { get; set; }

        public DecoderSettings() : base()
        {
            IgnoreShortItems = true;
        }
    }
}
