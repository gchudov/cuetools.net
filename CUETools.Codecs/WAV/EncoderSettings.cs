using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.WAV
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "wav";

        [Browsable(false)]
        public string Name => "cuetools";

        [Browsable(false)]
        public Type EncoderType => typeof(WAV.AudioEncoder);

        [Browsable(false)]
        public bool Lossless => true;

        [Browsable(false)]
        public int Priority => 10;

        [Browsable(false)]
        public string SupportedModes => "";

        [Browsable(false)]
        public string DefaultMode => "";

        [Browsable(false)]
        [DefaultValue("")]
        public string EncoderMode { get; set; }

        [Browsable(false)]
        public AudioPCMConfig PCM { get; set; }

        [Browsable(false)]
        public int BlockSize { get; set; }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding { get; set; }

        public IAudioEncoderSettings Clone()
        {
            return MemberwiseClone() as IAudioEncoderSettings;
        }
        #endregion

        public EncoderSettings()
        {
            this.Init();
        }

        public EncoderSettings(AudioPCMConfig pcm)
        {
            this.Init(pcm);
        }
    }
}
