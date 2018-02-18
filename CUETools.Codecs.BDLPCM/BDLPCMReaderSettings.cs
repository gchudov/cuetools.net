using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.BDLPCM
{
    public class BDLPCMReaderSettings : AudioDecoderSettings
    {
        public BDLPCMReaderSettings()
        {
            IgnoreShortItems = true;
        }

        [Browsable(false)]
        public bool IgnoreShortItems { get; set; }

        [Browsable(false)]
        public int? Stream { get; set; }

        [Browsable(false)]
        public ushort? Pid { get; set; }
    }
}
