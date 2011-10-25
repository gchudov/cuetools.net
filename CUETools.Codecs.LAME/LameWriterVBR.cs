using System;
using System.IO;

namespace CUETools.Codecs.LAME
{
    [AudioEncoderClass("lame2 VBR", "mp3", false, "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2", 2, typeof(LameWriterVBRSettings))]
    public class LameWriterVBR : LameWriter
    {
        private int quality = 0;

        public LameWriterVBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(IO, pcm)
        {
        }

        public LameWriterVBR(string path, AudioPCMConfig pcm)
            : base(path, pcm)
        {
        }

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

        LameWriterVBRSettings _settings = new LameWriterVBRSettings();

        public override object Settings
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
                return LameWriterConfig.CreateVbr(this.quality, this._settings.Quality);
            }
        }
    }
}
