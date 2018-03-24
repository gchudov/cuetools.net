using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;

namespace CUETools.Codecs
{
    public interface IAudioEncoderSettings
    {
        string Name { get; }

        string Extension { get; }

        Type EncoderType { get; }

        bool Lossless { get; }

        int Priority { get; }
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class AudioEncoderSettings : IAudioEncoderSettings
    {
        [Browsable(false)]
        public virtual string Name => null;

        [Browsable(false)]
        public virtual string Extension => null;

        [Browsable(false)]
        public virtual Type EncoderType => null;

        [Browsable(false)]
        public virtual bool Lossless => false;

        [Browsable(false)]
        public virtual int Priority => 0;

        public AudioEncoderSettings()
            : this("", "")
        {
        }

        public AudioEncoderSettings(AudioPCMConfig pcm)
            : this("", "")
        {
            this.PCM = pcm;
        }

        public AudioEncoderSettings(string supported_modes, string default_mode)
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                property.ResetValue(this);
            this.m_supported_modes = supported_modes;
            this.m_default_mode = default_mode;
            if (default_mode == "")
                GetSupportedModes(out default_mode);
            this.EncoderMode = default_mode;
        }

        protected string m_supported_modes;
        protected string m_default_mode;

        public virtual string GetSupportedModes(out string defaultMode)
        {
            defaultMode = m_default_mode;
            return this.m_supported_modes;
        }

        public AudioEncoderSettings Clone()
        {
            return this.MemberwiseClone() as AudioEncoderSettings;
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

        protected void SetDefaultValuesForMode()
        {
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                if (!property.CanResetValue(this))
                    foreach (var attribute in property.Attributes)
                        if (attribute is DefaultValueForModeAttribute)
                        {
                            var defaultValueForMode = attribute as DefaultValueForModeAttribute;
                            property.SetValue(this, defaultValueForMode.m_values[EncoderModeIndex]);
                        }
        }

        public bool HasDefaultValuesForMode(int index)
        {
            bool res = true;
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                foreach (var attribute in property.Attributes)
                    if (attribute is DefaultValueForModeAttribute)
                    {
                        var defaultValueForMode = attribute as DefaultValueForModeAttribute;
                        res &= property.GetValue(this).Equals(defaultValueForMode.m_values[index]);
                    }
            return res;
        }

        public int GuessEncoderMode()
        {
            string defaultMode;
            string[] modes = this.GetSupportedModes(out defaultMode).Split(' ');
            if (modes == null || modes.Length < 1)
                return -1;
            for (int i = 0; i < modes.Length; i++)
                if (HasDefaultValuesForMode(i))
                    return i;
            return -1;
        }

        [Browsable(false)]
        public AudioPCMConfig PCM
        {
            get;
            set;
        }


        [Browsable(false)]
        [DefaultValue(0)]
        public int BlockSize
        {
            get;
            set;
        }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding
        {
            get;
            set;
        }

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
        public string EncoderMode
        {
            get;
            set;
        }

        [Browsable(false)]
        public int EncoderModeIndex
        {
            get
            {
                string defaultMode;
                string[] modes = this.GetSupportedModes(out defaultMode).Split(' ');
                if (modes == null || modes.Length < 1)
                    return -1;
                for (int i = 0; i < modes.Length; i++)
                    if (modes[i] == this.EncoderMode)
                        return i;
                return -1;
            }

            set
            {
                string defaultMode;
                string[] modes = this.GetSupportedModes(out defaultMode).Split(' ');
                if (modes.Length == 0 && value < 0)
                    return;
                if (value < 0 || value >= modes.Length)
                    throw new IndexOutOfRangeException();
                this.EncoderMode = modes[value];
            }
        }
    }
}
