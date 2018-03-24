using System;

namespace CUETools.Codecs.WAV
{
    public class DecoderSettings : AudioDecoderSettings
    {
        public override string Name => "cuetools";

        public override string Extension => "wav";

        public override Type DecoderType => typeof(AudioDecoder);

        public override int Priority => 2;

        public DecoderSettings() : base() { }

        public bool IgnoreChunkSizes;
    }
}
