namespace CUETools.DSP.Resampler.Internal
{
    class poly_fir1_1_t : poly_fir1_t
    {
        const int COEF_INTERP = 1;
        const double MULT32 = (65536.0 * 65536.0);

        internal poly_fir1_1_t(int phase_bits)
            : base(phase_bits)
        {
        }

        internal unsafe override void fn(stage_t p, fifo_t output_fifo)
        {
            int i, num_in = p.occupancy, max_num_out = (int)(1 + num_in * p.out_in_ratio);
            int output_offs = output_fifo.reserve(max_num_out);
            fixed (byte* pinput = &p.fifo.data[p.offset], poutput = &output_fifo.data[output_offs])
            {
                double* input = (double*)pinput;
                double* output = (double*)poutput;

                for (i = 0; (p.at >> 32) < num_in; ++i, p.at += p.step)
                {
                    double* at = input + (p.at >> 32);
                    uint fraction = (uint)(p.at & 0xffffffff);
                    int phase = (int)(fraction >> (32 - phase_bits));
                    double x = (double)(fraction << phase_bits) * (1 / MULT32);
                    double sum = 0;
                    for (int j = 0; j < pf.num_coefs; j++)
                    {
                        double a = p.shared.poly_fir_coefs[poly_fir1_t.coef_idx(COEF_INTERP, pf.num_coefs, phase, 0, j)];
                        double b = p.shared.poly_fir_coefs[poly_fir1_t.coef_idx(COEF_INTERP, pf.num_coefs, phase, 1, j)];
                        sum += (b * x + a) * at[j];
                    }
                    output[i] = sum;
                }
                //assert(max_num_out - i >= 0);
                output_fifo.trim_by(max_num_out - i);
                p.fifo.read((int)(p.at >> 32), null);
                p.at &= 0xffffffff;
            }
        }
    }
}
