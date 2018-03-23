using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.BDLPCM
{
    [JsonObject(MemberSerialization.OptIn)]
    public class BDLPCMDecoderSettings : AudioDecoderSettings
    {
        public BDLPCMDecoderSettings()
        {
            IgnoreShortItems = true;
        }

        public bool IgnoreShortItems { get; set; }

        public int? Stream { get; set; }

        public ushort? Pid { get; set; }
    }
}
