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
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                property.ResetValue(this);
            this.supported_modes = "";
            this.EncoderMode = "";
        }

        public AudioEncoderSettings(string _supported_modes, string _default_mode)
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
                property.ResetValue(this);
            this.supported_modes = _supported_modes;
            this.EncoderMode =  _default_mode;
        }

        private string supported_modes;

        public virtual string GetSupportedModes()
        {
            return this.supported_modes;
        }

        [Browsable(false)]
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
