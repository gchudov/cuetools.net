using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
    public class WAVWriterSettings : AudioEncoderSettings
    {
        public WAVWriterSettings()
            : this(null)
        {
        }

        public WAVWriterSettings(AudioPCMConfig pcm)
            : base(pcm)
        {
        }
    }
}
