using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.WAV
{
    public class EncoderSettings : AudioEncoderSettings
    {
        public override string Extension => "wav";

        public override string Name => "cuetools";

        public override Type EncoderType => typeof(WAV.AudioEncoder);

        public override int Priority => 10;

        public override bool Lossless => true;

        public EncoderSettings()
            : this(null)
        {
        }

        public EncoderSettings(AudioPCMConfig pcm)
            : base(pcm)
        {
        }
    }
}
