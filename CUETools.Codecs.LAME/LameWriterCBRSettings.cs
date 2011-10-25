using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterCBRSettings
    {
        [DefaultValue(LameQuality.High)]
        public LameQuality Quality { get; set; }

        public LameWriterCBRSettings()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.ResetValue(this);
            }
            this.Quality = LameQuality.High;
        }
    }
}
