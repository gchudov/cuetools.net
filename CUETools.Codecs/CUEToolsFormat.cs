namespace CUETools.Codecs
{
    public class CUEToolsFormat
    {
        public CUEToolsFormat(
            string _extension,
            CUEToolsTagger _tagger,
            bool _allowLossless,
            bool _allowLossy,
            bool _allowEmbed,
            bool _builtin,
            AudioEncoderSettingsViewModel _encoderLossless,
            AudioEncoderSettingsViewModel _encoderLossy,
            AudioDecoderSettingsViewModel _decoder)
        {
            extension = _extension;
            tagger = _tagger;
            allowLossless = _allowLossless;
            allowLossy = _allowLossy;
            allowEmbed = _allowEmbed;
            builtin = _builtin;
            encoderLossless = _encoderLossless;
            encoderLossy = _encoderLossy;
            decoder = _decoder;
        }
        public string DotExtension
        {
            get
            {
                return "." + extension;
            }
        }

        public CUEToolsFormat Clone(CUEToolsCodecsConfig cfg)
        {
            var res = this.MemberwiseClone() as CUEToolsFormat;
            if (decoder != null) cfg.decoders.TryGetValue(decoder.decoderSettings.Extension, decoder.Lossless, decoder.decoderSettings.Name, out res.decoder);
            if (encoderLossy != null) cfg.encoders.TryGetValue(encoderLossy.settings.Extension, encoderLossy.Lossless, encoderLossy.settings.Name, out res.encoderLossy);
            if (encoderLossless != null) cfg.encoders.TryGetValue(encoderLossless.settings.Extension, encoderLossless.Lossless, encoderLossless.settings.Name, out res.encoderLossless);
            return res;
        }

        public override string ToString()
        {
            return extension;
        }

        public string extension;
        public AudioEncoderSettingsViewModel encoderLossless;
        public AudioEncoderSettingsViewModel encoderLossy;
        public AudioDecoderSettingsViewModel decoder;
        public CUEToolsTagger tagger;
        public bool allowLossless, allowLossy, allowEmbed, builtin;
    }
}
