using System;
using System.IO;

namespace CUETools.Codecs.LAME
{
    [AudioEncoderClass("libmp3lame CBR", "mp3", false, "96 128 192 256 320", "256", 2, typeof(LameWriterCBRSettings))]
    public class LameWriterCBR : LameWriter
    {
        private static readonly int[] bps_table = new int[] { 96, 128, 192, 256, 320 };
        private int bps;

        public LameWriterCBR(string path, Stream IO, AudioPCMConfig pcm)
            : base(IO, pcm)
        {
        }

        public LameWriterCBR(string path, AudioPCMConfig pcm)
            : base(path, pcm)
        {
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

        LameWriterCBRSettings _settings = new LameWriterCBRSettings();

        public override object Settings
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
                return LameWriterConfig.CreateCbr(this.bps, this._settings.Quality);
            }
        }
    }
}
