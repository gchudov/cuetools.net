using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame CBR", "mp3", false, 2, typeof(LAMEEncoderCBRSettings))]
    public class LAMEEncoderCBR : LAMEEncoder
    {
        private LAMEEncoderCBRSettings m_settings;

        public override AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
        }

        public LAMEEncoderCBR(string path, Stream IO, AudioEncoderSettings settings)
            : base(path, IO, settings)
        {
        }

        public LAMEEncoderCBR(string path, AudioEncoderSettings settings)
            : base(path, null, settings)
        {
        }

        protected override BE_CONFIG MakeConfig()
        {
            BE_CONFIG Mp3Config = new BE_CONFIG(Settings.PCM, m_settings.CustomBitrate > 0 ? (uint)m_settings.CustomBitrate : LAMEEncoderCBRSettings.bps_table[m_settings.EncoderModeIndex], 5);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = m_settings.StereoMode;
            //Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NONE; // --cbr
            //Mp3Config.format.lhv1.nPreset = LAME_QUALITY_PRESET.LQP_NORMAL_QUALITY;
            return Mp3Config;
        }
    }
}
