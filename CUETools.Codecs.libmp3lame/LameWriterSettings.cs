using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.libmp3lame
{
    public abstract class LameEncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "mp3";

        [Browsable(false)]
        public abstract string Name { get; }

        [Browsable(false)]
        public Type EncoderType => typeof(AudioEncoder);

        [Browsable(false)]
        public bool Lossless => false;

        [Browsable(false)]
        public abstract int Priority { get; }

        [Browsable(false)]
        public abstract string SupportedModes { get; }

        [Browsable(false)]
        public abstract string DefaultMode { get; }

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
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

        public LameEncoderSettings()
        {
            this.Init();
        }

        public abstract void Apply(IntPtr lame);
    }
}
