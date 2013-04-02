using System;
using System.IO;

namespace CUETools.Codecs.LAME
{
    [AudioEncoderClass("VBR (libmp3lame)", "mp3", false, 2, typeof(LameWriterVBRSettings))]
    public class LameWriterVBR : LameWriter
    {
        public LameWriterVBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(IO, pcm)
        {
        }

        public LameWriterVBR(string path, AudioPCMConfig pcm)
            : base(path, pcm)
        {
        }

        LameWriterVBRSettings _settings = new LameWriterVBRSettings();

        public override AudioEncoderSettings Settings
        {
            get
            {
                return _settings;
            }
            set
            {
                if (value as LameWriterVBRSettings == null)
                    throw new Exception("Unsupported options " + value);
                _settings = value as LameWriterVBRSettings;
            }
        }

        protected override LameWriterConfig Config
        {
            get
            {
                return LameWriterConfig.CreateVbr(9 - this._settings.EncoderModeIndex, this._settings.Quality);
            }
        }
    }
}
