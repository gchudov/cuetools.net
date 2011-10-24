namespace CUETools.Codecs.LossyWAV
{
    struct analysis_rec
    {
        public double fft_underlap_length;
        public int end_overlap_length, central_block_start, half_number_of_blocks, actual_analysis_blocks_start, analysis_blocks;
        public double bin_width, bin_time;
        public int lo_bins, hi_bins, max_bins, num_bins;
        public int[] spreading_averages_int;
        public float[] spreading_averages_rem;
        public float[] spreading_averages_rec;
        public float[] skewing_function;
        public byte[] threshold_index;
    }
}
