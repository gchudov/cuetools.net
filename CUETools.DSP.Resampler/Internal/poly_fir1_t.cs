namespace CUETools.DSP.Resampler.Internal
{
    abstract class poly_fir1_t
    {
        internal int phase_bits;
        internal poly_fir_t pf;
        internal poly_fir1_t(int phase_bits)
        {
            this.phase_bits = phase_bits;
        }
        internal abstract void fn(stage_t input, fifo_t output);

        public static int coef_idx(int interp_order, int fir_len, int phase_num, int coef_interp_num, int fir_coef_num)
        {
            return (fir_len) * ((interp_order) + 1) * (phase_num) + ((interp_order) + 1) * (fir_coef_num) + (interp_order - coef_interp_num);
        }
    }
}
