using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame CBR", "mp3", false, 2, typeof(LAMEEncoderCBRSettings))]
    public class LAMEEncoderCBR : LAMEEncoder
    {
        private LAMEEncoderCBRSettings _settings = new LAMEEncoderCBRSettings();

        public override AudioEncoderSettings Settings
        {
            get
            {
                return _settings;
            }
            set
            {
                if (value as LAMEEncoderCBRSettings == null)
                    throw new Exception("Unsupported options " + value);
                _settings = value as LAMEEncoderCBRSettings;
            }
        }

        public LAMEEncoderCBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(path, IO, pcm)
        {
        }

        public LAMEEncoderCBR(string path, AudioPCMConfig pcm)
            : base(path, null, pcm)
        {
        }

        protected override BE_CONFIG MakeConfig()
        {
            BE_CONFIG Mp3Config = new BE_CONFIG(PCM, _settings.CustomBitrate > 0 ? (uint)_settings.CustomBitrate : LAMEEncoderCBRSettings.bps_table[_settings.EncoderModeIndex], 5);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = _settings.StereoMode;
            //Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NONE; // --cbr
            //Mp3Config.format.lhv1.nPreset = LAME_QUALITY_PRESET.LQP_NORMAL_QUALITY;
            return Mp3Config;
        }
    }
}
