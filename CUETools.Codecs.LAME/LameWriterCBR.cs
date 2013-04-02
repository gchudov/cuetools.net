using System;
using System.IO;

namespace CUETools.Codecs.LAME
{
    [AudioEncoderClass("CBR (libmp3lame)", "mp3", false, 1, typeof(LameWriterCBRSettings))]
    public class LameWriterCBR : LameWriter
    {
        public LameWriterCBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(IO, pcm)
        {
        }

        public LameWriterCBR(string path, AudioPCMConfig pcm)
            : base(path, pcm)
        {
        }

        LameWriterCBRSettings _settings = new LameWriterCBRSettings();

        public override AudioEncoderSettings Settings
        {
            get
            {
                return _settings;
            }
            set
            {
                if (value as LameWriterCBRSettings == null)
                    throw new Exception("Unsupported options " + value);
                _settings = value as LameWriterCBRSettings;
            }
        }

        protected override LameWriterConfig Config
        {
            get
            {
                return LameWriterConfig.CreateCbr(LameWriterCBRSettings.bps_table[this._settings.EncoderModeIndex], this._settings.Quality);
            }
        }
    }
}
