using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs
{
    public interface IAudioDecoderSettings
    {
        string Name { get; }

        string Extension { get; }

        Type DecoderType { get; }

        int Priority { get; }

        IAudioDecoderSettings Clone();
    }

    public static class IAudioDecoderSettingsExtensions
    {
        public static bool HasBrowsableAttributes(this IAudioDecoderSettings settings)
        {
            bool hasBrowsable = false;
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
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

        public static void Init(this IAudioDecoderSettings settings)
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
                property.ResetValue(settings);
        }
    }
}
