namespace CUETools.Codecs.LossyWAV
{
    struct channel_rec
    {
        public double this_codec_block_rms;
        public double this_codec_block_bits;
        public short maximum_bits_to_remove;
        public short bits_to_remove;
        public short bits_lost;
        public short clipped_samples;
    }
}
