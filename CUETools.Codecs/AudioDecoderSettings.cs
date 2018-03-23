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

        bool Lossless { get; }

        int Priority { get; }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class AudioDecoderSettings: IAudioDecoderSettings
    {
        [Browsable(false)]
        public virtual string Name => null;

        [Browsable(false)]
        public virtual string Extension => null;

        [Browsable(false)]
        public virtual Type DecoderType => null;

        [Browsable(false)]
        public virtual bool Lossless => true;

        [Browsable(false)]
        public virtual int Priority => 0;

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
