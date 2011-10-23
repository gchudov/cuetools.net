using System;

namespace CUETools.Codecs.LAME
{
    public class LameException: Exception
    {
        public LameException(string message)
            : base(message)
        {
        }
    }

    public enum LameQuality
    {
        High,
    }

    public enum LameVbrMode
    {
        Off,
        Default,
        Abr,
    }

    public class LameWriterSettings
    {
        public LameQuality Quality { get; set; }
        public LameVbrMode VbrMode { get; set; }
        public int VbrQuality { get; set; }
        public int Bitrate { get; set; }

        public LameWriterSettings()
        {
            Quality = LameQuality.High;
            VbrMode = LameVbrMode.Default;
            VbrQuality = 5;
        }

        public static LameWriterSettings CreateCbr(int bitrate, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterSettings() { VbrMode = LameVbrMode.Off, Bitrate = bitrate, Quality = encodeQuality };
        }

        public static LameWriterSettings CreateAbr(int bitrate, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterSettings() { VbrMode = LameVbrMode.Abr, Bitrate = bitrate, Quality = encodeQuality };
        }

        public static LameWriterSettings CreateVbr(int vbrQuality, LameQuality encodeQuality = LameQuality.High)
        {
            return new LameWriterSettings() { VbrMode = LameVbrMode.Default, VbrQuality = vbrQuality, Quality = encodeQuality };
        }
    }
}
