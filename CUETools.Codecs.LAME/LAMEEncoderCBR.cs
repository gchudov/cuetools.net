using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame CBR", "mp3", false, "96 128 192 256 320", "256", 2, typeof(LAMEEncoderCBRSettings))]
    public class LAMEEncoderCBR : LAMEEncoder
    {
        private static readonly uint[] bps_table = new uint[] { 96, 128, 192, 256, 320 };

        private uint bps;
        private LAMEEncoderCBRSettings _settings = new LAMEEncoderCBRSettings();

        public override object Settings
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

        public override int CompressionLevel
        {
            get
            {
                for (int i = 0; i < bps_table.Length; i++)
                {
                    if (bps == bps_table[i])
                    {
                        return i;
                    }
                }
                return -1;
            }
            set
            {
                if (value < 0 || value > bps_table.Length)
                    throw new Exception("unsupported compression level");
                bps = bps_table[value];
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
            BE_CONFIG Mp3Config = new BE_CONFIG(PCM, _settings.CustomBitrate > 0 ? (uint)_settings.CustomBitrate : bps, 5);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = _settings.StereoMode;
            //Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NONE; // --cbr
            //Mp3Config.format.lhv1.nPreset = LAME_QUALITY_PRESET.LQP_NORMAL_QUALITY;
            return Mp3Config;
        }
    }
}
