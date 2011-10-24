namespace CUETools.DSP.Resampler.Internal
{
    class poly_fir_t
    {
        internal int num_coefs;
        internal double pass, stop, att;
        internal poly_fir1_t[] interp;

        internal poly_fir_t(int num_coefs, double pass, double stop, double att, int pb1, int pb2, int pb3)
        {
            this.num_coefs = num_coefs;
            this.pass = pass;
            this.stop = stop;
            this.att = att;
            this.interp = new poly_fir1_t[4];
            this.interp[0] = new poly_fir1_0_t();
            this.interp[1] = new poly_fir1_1_t(pb1);
            this.interp[2] = new poly_fir1_2_t(pb2);
            this.interp[3] = new poly_fir1_3_t(pb3);
            foreach (poly_fir1_t f1 in interp)
                f1.pf = this;
        }
    }
}
