using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Xml.Serialization;
using System.Text;

namespace CUETools.Codecs
{
    public class AudioEncoderSettings
    {
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
            this.EncoderMode =  default_mode;
        }

        private string m_supported_modes;

        public virtual string GetSupportedModes()
        {
            return this.m_supported_modes;
        }

        public virtual bool IsValid()
        {
            return BlockSize == 0 && Padding >= 0;
        }

        public void Validate()
        {
            if (!IsValid())
                throw new Exception("unsupported encoder settings");
        }

        public AudioEncoderSettings Clone()
        {
            return this.MemberwiseClone() as AudioEncoderSettings;
        }

        [Browsable(false)]
        [XmlIgnore]
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
        [XmlIgnore]
        [DefaultValue(4096)]
        public int Padding
        {
            get;
            set;
        }

        [Browsable(false)]
        [DefaultValue("")]
        public string EncoderMode
        {
            get;
            set;
        }

        [XmlIgnore]
        [Browsable(false)]
        public string[] SupportedModes
        {
            get
            {
                return this.GetSupportedModes().Split(' ');
            }
        }
        
        [XmlIgnore]
        [Browsable(false)]
        public int EncoderModeIndex
        {
            get
            {
                string[] modes = this.SupportedModes;
                if (modes == null || modes.Length < 1)
                    return -1;
                for (int i = 0; i < modes.Length; i++)
                    if (modes[i] == this.EncoderMode)
                        return i;
                return -1;
            }

            set
            {
                string[] modes = this.SupportedModes;
                if (modes.Length == 0 && value < 0)
                    return;
                if (value < 0 || value >= modes.Length)
                    throw new InvalidOperationException();
                this.EncoderMode = modes[value];
            }
        }
    }
}
