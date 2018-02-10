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
        public string Stream { get; set; }
    }
}
