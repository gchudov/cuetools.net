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
            CUEToolsUDC _encoderLossless,
            CUEToolsUDC _encoderLossy,
            CUEToolsUDC _decoder)
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
            if (decoder != null) cfg.decoders.TryGetValue(decoder.extension, decoder.lossless, decoder.name, out res.decoder);
            if (encoderLossy != null) cfg.encoders.TryGetValue(encoderLossy.extension, encoderLossy.lossless, encoderLossy.name, out res.encoderLossy);
            if (encoderLossless != null) cfg.encoders.TryGetValue(encoderLossless.extension, encoderLossless.lossless, encoderLossless.name, out res.encoderLossless);
            return res;
        }

        public override string ToString()
        {
            return extension;
        }

        public string extension;
        public CUEToolsUDC encoderLossless;
        public CUEToolsUDC encoderLossy;
        public CUEToolsUDC decoder;
        public CUEToolsTagger tagger;
        public bool allowLossless, allowLossy, allowEmbed, builtin;
    }
}
