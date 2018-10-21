using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs.MPEG.ATSI
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation
        [Browsable(false)]
        public string Extension => "ifo";

        [Browsable(false)]
        public string Name => "cuetools";

        [Browsable(false)]
        public Type DecoderType => typeof(AudioDecoder);

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
        public int? StreamId { get; set; }

        [Browsable(false)]
        public int? Title { get; set; }
    }
}
