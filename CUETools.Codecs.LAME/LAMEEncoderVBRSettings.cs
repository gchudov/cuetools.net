using System.ComponentModel;

namespace CUETools.Codecs.LAME
{
	public class LAMEEncoderVBRSettings
	{
		[DefaultValue(LAMEEncoderVBRProcessingQuality.Normal)]
		public LAMEEncoderVBRProcessingQuality Quality { get; set; }

		public LAMEEncoderVBRSettings()
		{
			// Iterate through each property and call ResetValue()
			foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this)) {
				property.ResetValue(this);
            }
		}
	}
}
