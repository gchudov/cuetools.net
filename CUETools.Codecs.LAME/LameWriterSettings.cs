using System;

namespace CUETools.Codecs.LAME
{
    public class LameWriterConfig
    {
        public LameQuality Quality { get; set; }
        public LameVbrMode VbrMode { get; set; }
        public int VbrQuality { get; set; }
        public int Bitrate { get; set; }

        public LameWriterConfig()
        {
            Quality = LameQuality.High;
            VbrMode = LameVbrMode.Default;
            VbrQuality = 5;
        }

        public static LameWriterConfig CreateCbr(int bitrate, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterConfig() { VbrMode = LameVbrMode.Off, Bitrate = bitrate, Quality = encodeQuality };
        }

        public static LameWriterConfig CreateAbr(int bitrate, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterConfig() { VbrMode = LameVbrMode.Abr, Bitrate = bitrate, Quality = encodeQuality };
        }

        public static LameWriterConfig CreateVbr(int vbrQuality, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterConfig() { VbrMode = LameVbrMode.Default, VbrQuality = vbrQuality, Quality = encodeQuality };
        }
    }
}
