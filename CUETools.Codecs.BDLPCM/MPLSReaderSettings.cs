using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.MPLS
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation
        [Browsable(false)]
        public string Extension => "mpls";

        [Browsable(false)]
        public string Name => "cuetools";

        [Browsable(false)]
        public Type DecoderType => typeof(BDLPCM.MPLSDecoder);

        [Browsable(false)]
        public int Priority => 2;

        public IAudioDecoderSettings Clone()
        {
            return MemberwiseClone() as IAudioDecoderSettings;
        }
        #endregion

        public DecoderSettings()
        {
            this.Init();
        }

        [DefaultValue(true)]
        public bool IgnoreShortItems { get; set; }

        [Browsable(false)]
        public int? Stream { get; set; }

        [Browsable(false)]
        public ushort? Pid { get; set; }
    }
}
