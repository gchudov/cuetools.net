namespace CUETools.DSP.Resampler.Internal
{
    class poly_fir1_0_t : poly_fir1_t
    {
        internal poly_fir1_0_t()
            : base(0)
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

                for (i = 0; (p.at >> 32) < num_in * p.divisor; ++i, p.at += p.step & ~0xffffffffL)
                {
                    int divided = (int)(p.at >> 32) / p.divisor;
                    int divided_rem = (int)(p.at >> 32) % p.divisor;
                    double* at = input + divided;
                    double sum = 0;
                    for (int j = 0; j < pf.num_coefs; j++)
                        sum += (p.shared.poly_fir_coefs[poly_fir1_t.coef_idx(0, pf.num_coefs, divided_rem, 0, j)]) * at[j];
                    output[i] = sum;
                }
                //assert(max_num_out - i >= 0);
                output_fifo.trim_by(max_num_out - i);
                int divided1 = (int)(p.at >> 32) / p.divisor;
                p.fifo.read(divided1, null);
                p.at -= ((long)divided1 * p.divisor) << 32;
            }
        }
    }
}
