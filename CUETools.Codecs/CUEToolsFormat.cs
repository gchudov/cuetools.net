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
            if (decoder != null) cfg.decodersViewModel.TryGetValue(decoder.Settings.Extension, decoder.Settings.Name, out res.decoder);
            if (encoderLossy != null) cfg.encodersViewModel.TryGetValue(encoderLossy.Settings.Extension, encoderLossy.Lossless, encoderLossy.Settings.Name, out res.encoderLossy);
            if (encoderLossless != null) cfg.encodersViewModel.TryGetValue(encoderLossless.Settings.Extension, encoderLossless.Lossless, encoderLossless.Settings.Name, out res.encoderLossless);
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
