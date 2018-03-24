using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame VBR", "mp3", false, 2, typeof(LAMEEncoderVBRSettings))]
    public class LAMEEncoderVBR : LAMEEncoder
    {
        private LAMEEncoderVBRSettings m_settings;

        public override AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
        }

        public LAMEEncoderVBR(string path, Stream IO, AudioEncoderSettings settings)
            : base(path, IO, settings)
        {
        }

        public LAMEEncoderVBR(string path, AudioEncoderSettings settings)
            : base(path, null, settings)
        {
        }

        protected override BE_CONFIG MakeConfig()
        {
            BE_CONFIG Mp3Config = new BE_CONFIG(Settings.PCM, 0, (uint)m_settings.Quality);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = MpegMode.JOINT_STEREO;
            Mp3Config.format.lhv1.bEnableVBR = 1;
            Mp3Config.format.lhv1.nVBRQuality = 9 - m_settings.EncoderModeIndex;
            Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NEW; // --vbr-new
            return Mp3Config;
        }
    }
}
