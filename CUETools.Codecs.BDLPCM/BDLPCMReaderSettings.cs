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
        }
     
        [Browsable(false)]
        public int? Stream { get; set; }

        [Browsable(false)]
        public ushort? Pid { get; set; }
    }
}
