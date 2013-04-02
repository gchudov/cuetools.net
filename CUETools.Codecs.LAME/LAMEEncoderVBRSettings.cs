using System.ComponentModel;

namespace CUETools.Codecs.LAME
{
    public class LAMEEncoderVBRSettings : AudioEncoderSettings
	{
		[DefaultValue(LAMEEncoderVBRProcessingQuality.Normal)]
		public LAMEEncoderVBRProcessingQuality Quality { get; set; }

		public LAMEEncoderVBRSettings()
            : base("V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2")
		{
		}
	}
}
