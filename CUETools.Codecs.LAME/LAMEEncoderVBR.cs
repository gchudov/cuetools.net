using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame VBR", "mp3", false, "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2", 2, typeof(LAMEEncoderVBRSettings))]
    public class LAMEEncoderVBR : LAMEEncoder
    {
        private int quality = 0;
        private LAMEEncoderVBRSettings _settings = new LAMEEncoderVBRSettings();

        public override int CompressionLevel
        {
            get
            {
                return 9 - quality;
            }
            set
            {
                if (value < 0 || value > 9)
                    throw new Exception("unsupported compression level");
                quality = 9 - value;
            }
        }

        public override object Settings
        {
            get
            {
                return _settings;
            }
            set
            {
                if (value as LAMEEncoderVBRSettings == null)
                    throw new Exception("Unsupported options " + value);
                _settings = value as LAMEEncoderVBRSettings;
            }
        }

        public LAMEEncoderVBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(path, IO, pcm)
        {
        }

        public LAMEEncoderVBR(string path, AudioPCMConfig pcm)
            : base(path, null, pcm)
        {
        }

        protected override BE_CONFIG MakeConfig()
        {
            BE_CONFIG Mp3Config = new BE_CONFIG(PCM, 0, (uint)_settings.Quality);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = MpegMode.JOINT_STEREO;
            Mp3Config.format.lhv1.bEnableVBR = 1;
            Mp3Config.format.lhv1.nVBRQuality = quality;
            Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NEW; // --vbr-new
            return Mp3Config;
        }
    }
}
