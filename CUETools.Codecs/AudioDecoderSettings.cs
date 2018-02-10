using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.Text;

namespace CUETools.Codecs
{
    public class AudioDecoderSettings
    {
        public AudioDecoderSettings()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                property.ResetValue(this);
        }

        public AudioDecoderSettings Clone()
        {
            return this.MemberwiseClone() as AudioDecoderSettings;
        }

        public bool HasBrowsableAttributes()
        {
            bool hasBrowsable = false;
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                bool isBrowsable = true;
                foreach (var attribute in property.Attributes)
                {
                    var browsable = attribute as BrowsableAttribute;
                    isBrowsable &= browsable == null || browsable.Browsable;
                }
                hasBrowsable |= isBrowsable;
            }
            return hasBrowsable;
        }
    }
}
