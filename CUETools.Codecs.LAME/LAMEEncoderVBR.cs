using System;
using System.IO;
using CUETools.Codecs.LAME.Interop;

namespace CUETools.Codecs.LAME
{
    //[AudioEncoderClass("lame VBR", "mp3", false, 2, typeof(LAMEEncoderVBRSettings))]
    public class LAMEEncoderVBR : LAMEEncoder
    {
        private LAMEEncoderVBRSettings m_settings = new LAMEEncoderVBRSettings();

        public override AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
            set
            {
                m_settings = value.Clone<LAMEEncoderVBRSettings>();
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
            BE_CONFIG Mp3Config = new BE_CONFIG(PCM, 0, (uint)m_settings.Quality);
            Mp3Config.format.lhv1.bWriteVBRHeader = 1;
            Mp3Config.format.lhv1.nMode = MpegMode.JOINT_STEREO;
            Mp3Config.format.lhv1.bEnableVBR = 1;
            Mp3Config.format.lhv1.nVBRQuality = 9 - m_settings.EncoderModeIndex;
            Mp3Config.format.lhv1.nVbrMethod = VBRMETHOD.VBR_METHOD_NEW; // --vbr-new
            return Mp3Config;
        }
    }
}
