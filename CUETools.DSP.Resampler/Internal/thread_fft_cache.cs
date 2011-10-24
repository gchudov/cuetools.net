namespace CUETools.DSP.Resampler.Internal
{
    class thread_fft_cache
    {
        internal int[] lsx_fft_br;
        internal double[] lsx_fft_sc;
        internal int fft_len;

        internal thread_fft_cache()
        {
            fft_len = 0;
            lsx_fft_br = null;
            lsx_fft_sc = null;
        }
    }
}
