namespace CUETools.DSP.Resampler.Internal
{
    class rate_shared_t
    {
        internal double[] poly_fir_coefs;
        internal dft_filter_t[] half_band = new dft_filter_t[2];
        internal thread_fft_cache info;

        internal rate_shared_t()
        {
            info = new thread_fft_cache();
            half_band[0] = new dft_filter_t();
            half_band[1] = new dft_filter_t();
        }
    }
}
