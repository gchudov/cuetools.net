using System;

namespace CUETools.DSP.Resampler.Internal
{
    class rate_t
    {
        double factor;
        long samples_in, samples_out;
        int level, input_stage_num, output_stage_num;
        bool upsample;
        stage_t[] stages;
        stage_t pre_stage, last_stage, post_stage;
        int in_samplerate, out_samplerate;

        const double MULT32 = (65536.0 * 65536.0);
        const double LSX_TO_6dB = 0.5869;
        const double LSX_TO_3dB = ((2 / 3.0) * (.5 + LSX_TO_6dB));
        const double LSX_MAX_TBW0 = 36.0;
        const double LSX_MAX_TBW0A = (LSX_MAX_TBW0 / (1 + LSX_TO_3dB));
        //const double LSX_MAX_TBW3 = Math.Floor(LSX_MAX_TBW0 * LSX_TO_3dB);
        //const double LSX_MAX_TBW3A = Math.Floor(LSX_MAX_TBW0A * LSX_TO_3dB);

        static int range_limit(int x, int lower, int upper)
        {
            return Math.Min(Math.Max(x, lower), upper);
        }

        static readonly double[] half_fir_coefs_25 = {
		  4.9866643051942178e-001, 3.1333582318860204e-001, 1.2567743716165585e-003,
		 -9.2035726038137103e-002, -1.0507348255277846e-003, 4.2764945027796687e-002,
		  7.7661461450703555e-004, -2.0673365323361139e-002, -5.0429677622613805e-004,
		  9.4223774565849357e-003, 2.8491539998284476e-004, -3.8562347294894628e-003,
		 -1.3803431143314762e-004, 1.3634218103234187e-003, 5.6110366313398705e-005,
		 -3.9872042837864422e-004, -1.8501044952475473e-005, 9.0580351350892191e-005,
		  4.6764104835321042e-006, -1.4284332593063177e-005, -8.1340436298087893e-007,
		  1.1833367010222812e-006, 7.3979325233687461e-008,
		};

        static readonly double[] half_fir_coefs_low = {
		  4.2759802493108773e-001, 3.0939308096100709e-001, 6.9285325719540158e-002,
		  -8.0642059355533674e-002, -6.0528749718348158e-002, 2.5228940037788555e-002,
		  4.7756850372993369e-002, 8.7463256642532057e-004, -3.3208422093026498e-002,
		  -1.3425983316344854e-002, 1.9188320662637096e-002, 1.7478840713827052e-002,
		  -7.5527851809344612e-003, -1.6145235261724403e-002, -6.3013968965413430e-004,
		  1.1965551091184719e-002, 5.1714613100614501e-003, -6.9898749683755968e-003,
		  -6.6150222806158742e-003, 2.6394681964090937e-003, 5.9365183404658526e-003,
		  3.5567920638016650e-004, -4.2031898513566123e-003, -1.8738555289555877e-003,
		  2.2991238738122328e-003, 2.2058519188488186e-003, -7.7796582498205363e-004,
		  -1.8212814627239918e-003, -1.4964619042558244e-004, 1.1706370821176716e-003,
		  5.3082071395224866e-004, -5.6771020453353900e-004, -5.4472363026668942e-004,
		  1.5914542178505357e-004, 3.8911127354338085e-004, 4.2076035174603683e-005,
		  -2.1015548483049000e-004, -9.5381290156278399e-005, 8.0903081108059553e-005,
		  7.5812875822003258e-005, -1.5004304266040688e-005, -3.9149443482028750e-005,
		  -6.0893901283459912e-006, 1.4040363940567877e-005, 4.9834316581482789e-006,
		};

        static readonly poly_fir_t[] poly_firs = new poly_fir_t[] {
		  new poly_fir_t(16, 0.750, 1.5, 108,  9,  7, 6),
		  new poly_fir_t(30, 1.000, 1.5, 133, 10,  9, 7),
		  new poly_fir_t(38, 1.000, 1.5, 165, 12, 10, 8),
		  new poly_fir_t(42, 0.724, 1.0, 105, 10,  8, 6),
		  new poly_fir_t(10, 0.300, 1.5, 107,  9,  7, 6),
		  new poly_fir_t(14, 0.500, 1.5, 125, 10,  8, 6),
		  new poly_fir_t(20, 0.500, 1.5, 174, 11,  9, 7),
		};

        static unsafe void cubic_spline(stage_t p, fifo_t output_fifo)
        {
            int i, num_in = p.occupancy;
            int max_num_out = 1 + (int)(num_in * p.out_in_ratio);
            int output_offs = output_fifo.reserve(max_num_out);
            fixed (byte* pinput = &p.fifo.data[p.offset], poutput = &output_fifo.data[output_offs])
            {
                double* input = (double*)pinput;
                double* output = (double*)poutput;
                for (i = 0; (p.at >> 32) < num_in; ++i, p.at += p.step)
                {
                    double* s = input + (p.at >> 32);
                    double x = (p.at & 0xffffffff) * (1 / MULT32);
                    double b = 0.5 * (s[1] + s[-1]) - *s, a = (1 / 6.0) * (s[2] - s[1] + s[-1] - *s - 4 * b);
                    double c = s[1] - *s - a - b;
                    output[i] = ((a * x + b) * x + c) * x + *s;
                }
            }
            //assert(max_num_out - i >= 0);
            output_fifo.trim_by(max_num_out - i);
            p.fifo.read((int)(p.at >> 32), null);
            p.at &= 0xffffffff;
        }

        static unsafe void half_sample_low(stage_t p, fifo_t output_fifo)
        {
            int num_out = (p.occupancy + 1) / 2;
            int output_offs = output_fifo.reserve(num_out);
            fixed (byte* pinput = &p.fifo.data[p.offset], poutput = &output_fifo.data[output_offs])
            {
                double* input = (double*)pinput;
                double* output = (double*)poutput;
                for (int i = 0; i < num_out; ++i, input += 2)
                {
                    double sum = input[0] * half_fir_coefs_low[0];
                    for (int j = 1; j < half_fir_coefs_low.Length; j++)
                        sum += (input[-j] + input[j]) * half_fir_coefs_low[j];
                    output[i] = sum;
                }
            }
            p.fifo.read(2 * num_out, null);
        }

        static unsafe void half_sample_25(stage_t p, fifo_t output_fifo)
        {
            int num_out = (p.occupancy + 1) / 2;
            int output_offs = output_fifo.reserve(num_out);
            fixed (byte* pinput = &p.fifo.data[p.offset], poutput = &output_fifo.data[output_offs])
            {
                double* input = (double*)pinput;
                double* output = (double*)poutput;
                for (int i = 0; i < num_out; ++i, input += 2)
                {
                    double sum = input[0] * half_fir_coefs_25[0];
                    for (int j = 1; j < half_fir_coefs_25.Length; j++)
                        sum += (input[-j] + input[j]) * half_fir_coefs_25[j];
                    output[i] = sum;
                }
            }
            p.fifo.read(2 * num_out, null);
        }

        static unsafe void half_sample(stage_t p, fifo_t output_fifo)
        {
            int num_in = Math.Max(0, p.fifo.occupancy);
            dft_filter_t f = p.shared.half_band[p.which];
            int overlap = f.num_taps - 1;

            while (num_in >= f.dft_length)
            {
                int input_offs = p.fifo.offset;
                p.fifo.read(f.dft_length - overlap, null);
                num_in -= f.dft_length - overlap;

                int output_offs = output_fifo.reserve(f.dft_length);
                output_fifo.trim_by((f.dft_length + overlap) >> 1);
                Buffer.BlockCopy(p.fifo.data, input_offs, output_fifo.data, output_offs, f.dft_length * sizeof(double));

                fixed (byte* poutput = &output_fifo.data[output_offs])
                fixed (double* lsx_fft_sc = p.shared.info.lsx_fft_sc)
                fixed (int* lsx_fft_br = p.shared.info.lsx_fft_br)
                {
                    double* output = (double*)poutput;
                    SOXFft.rdft(f.dft_length, 1, output, lsx_fft_br, lsx_fft_sc);
                    output[0] *= f.coefs[0];
                    output[1] *= f.coefs[1];
                    for (int i = 2; i < f.dft_length; i += 2)
                    {
                        double tmp = output[i];
                        output[i] = f.coefs[i] * tmp - f.coefs[i + 1] * output[i + 1];
                        output[i + 1] = f.coefs[i + 1] * tmp + f.coefs[i] * output[i + 1];
                    }
                    SOXFft.rdft(f.dft_length, -1, output, lsx_fft_br, lsx_fft_sc);

                    for (int j = 1, i = 2; i < f.dft_length - overlap; ++j, i += 2)
                        output[j] = output[i];
                }
            }
        }

        static unsafe void double_sample(stage_t p, fifo_t output_fifo)
        {
            int num_in = Math.Max(0, p.fifo.occupancy);
            dft_filter_t f = p.shared.half_band[1];
            int overlap = f.num_taps - 1;

            while (num_in > f.dft_length >> 1)
            {
                int input_offs = p.fifo.offset;
                p.fifo.read((f.dft_length - overlap) >> 1, null);
                num_in -= (f.dft_length - overlap) >> 1;

                int output_offs = output_fifo.reserve(f.dft_length);
                output_fifo.trim_by(overlap);

                fixed (byte* pinput = &p.fifo.data[input_offs])
                fixed (byte* poutput = &output_fifo.data[output_offs])
                fixed (double* lsx_fft_sc = p.shared.info.lsx_fft_sc)
                fixed (int* lsx_fft_br = p.shared.info.lsx_fft_br)
                {
                    double* input = (double*)pinput;
                    double* output = (double*)poutput;

                    for (int j = 0, i = 0; i < f.dft_length; ++j, i += 2)
                    {
                        output[i] = input[j];
                        output[i + 1] = 0;
                    }

                    SOXFft.rdft(f.dft_length, 1, output, lsx_fft_br, lsx_fft_sc);
                    output[0] *= f.coefs[0];
                    output[1] *= f.coefs[1];
                    for (int i = 2; i < f.dft_length; i += 2)
                    {
                        double tmp = output[i];
                        output[i] = f.coefs[i] * tmp - f.coefs[i + 1] * output[i + 1];
                        output[i + 1] = f.coefs[i + 1] * tmp + f.coefs[i] * output[i + 1];
                    }
                    SOXFft.rdft(f.dft_length, -1, output, lsx_fft_br, lsx_fft_sc);
                }
            }
        }

        static int lsx_lpf_num_taps(double att, double tr_bw, int k)
        {                    /* TODO this could be cleaner, esp. for k != 0 */
            int n;
            if (att <= 80)
                n = (int)(0.25 / Math.PI * (att - 7.95) / (2.285 * tr_bw) + .5);
            else
            {
                double n160 = (.0425 * att - 1.4) / tr_bw;   /* Half order for att = 160 */
                n = (int)(n160 * (16.556 / (att - 39.6) + .8625) + .5);  /* For att [80,160) */
            }
            return k != 0 ? 2 * n : 2 * (n + (n & 1)) + 1; /* =1 %4 (0 phase 1/2 band) */
        }

        static double lsx_kaiser_beta(double att)
        {
            if (att > 100) return .1117 * att - 1.11;
            if (att > 50) return .1102 * (att - 8.7);
            if (att > 20.96) return .58417 * Math.Pow(att - 20.96, .4) + .07886 * (att - 20.96);
            return 0;
        }

        static double lsx_bessel_I_0(double x)
        {
            double term = 1, sum = 1, last_sum, x2 = x / 2;
            int i = 1;
            do
            {
                double y = x2 / i++;
                last_sum = sum;
                sum += term *= y * y;
            } while (sum != last_sum);
            return sum;
        }

        static double[] lsx_make_lpf(int num_taps, double Fc, double beta, double scale, bool dc_norm)
        {
            int i, m = num_taps - 1;
            double[] h = new double[num_taps];
            double sum = 0;
            double mult = scale / lsx_bessel_I_0(beta);
            //assert(Fc >= 0 && Fc <= 1);
            //lsx_debug("make_lpf(n=%i, Fc=%g beta=%g dc-norm=%i scale=%g)", num_taps, Fc, beta, dc_norm, scale);
            for (i = 0; i <= m / 2; ++i)
            {
                double x = Math.PI * (i - .5 * m), y = 2.0 * i / m - 1;
                h[i] = x != 0 ? Math.Sin(Fc * x) / x : Fc;
                sum += h[i] *= lsx_bessel_I_0(beta * Math.Sqrt(1 - y * y)) * mult;
                if (m - i != i)
                    sum += h[m - i] = h[i];
            }
            for (i = 0; dc_norm && i < num_taps; ++i) h[i] *= scale / sum;
            return h;
        }

        static double[] lsx_design_lpf(
            double Fp,      /* End of pass-band; ~= 0.01dB point */
            double Fc,      /* Start of stop-band */
            double Fn,      /* Nyquist freq; e.g. 0.5, 1, PI */
            bool allow_aliasing,
            double att,     /* Stop-band attenuation in dB */
            ref int num_taps, /* (Single phase.)  0: value will be estimated */
            int k)          /* Number of phases; 0 for single-phase */
        {
            double tr_bw, beta;

            if (allow_aliasing)
                Fc += (Fc - Fp) * LSX_TO_3dB;
            Fp /= Fn;
            Fc /= Fn;        /* Normalise to Fn = 1 */
            tr_bw = LSX_TO_6dB * (Fc - Fp); /* Transition band-width: 6dB to stop points */

            if (num_taps == 0)
                num_taps = lsx_lpf_num_taps(att, tr_bw, k);
            beta = lsx_kaiser_beta(att);
            if (k != 0)
                num_taps = num_taps * k - 1;
            else k = 1;
            //lsx_debug("%g %g %g", Fp, tr_bw, Fc);

            return lsx_make_lpf(num_taps, (Fc - tr_bw) / k, beta, (double)k, true);
        }

        static int lsx_set_dft_length(int num_taps) /* Set to 4 x nearest power of 2 */
        {
            int result, n = num_taps;
            for (result = 8; n > 2; result <<= 1, n >>= 1) ;
            result = range_limit(result, 4096, 131072);
            //assert(num_taps * 2 < result);
            return result;
        }

        static void lsx_fir_to_phase(ref double[] h, ref int len, ref int post_len, double phase, thread_fft_cache info)
        {
            double phase1 = (phase > 50 ? 100 - phase : phase) / 50;
            int i, work_len, begin, end, imp_peak = 0, peak = 0;
            double imp_sum = 0, peak_imp_sum = 0;
            double prev_angle2 = 0, cum_2pi = 0, prev_angle1 = 0, cum_1pi = 0;

            for (i = len, work_len = 2 * 2 * 8; i > 1; work_len <<= 1, i >>= 1) ;

            double[] pi_wraps = new double[work_len + 2]; /* +2: (UN)PACK */
            double[] work = new double[(work_len + 2) / 2];

            Buffer.BlockCopy(h, 0, work, 0, len * sizeof(double));

            SOXFft.safe_rdft(work_len, 1, work, info); /* Cepstral: */

            work[work_len] = work[1]; work[work_len + 1] = work[0]; //LSX_UNPACK(work, work_len);

            for (i = 0; i <= work_len; i += 2)
            {
                double angle = Math.Atan2(work[i + 1], work[i]);
                double detect = 2 * Math.PI;
                double delta = angle - prev_angle2;
                double adjust = detect * ((delta < -detect * 0.7 ? 1 : 0) - (delta > detect * 0.7 ? 1 : 0));
                prev_angle2 = angle;
                cum_2pi += adjust;
                angle += cum_2pi;
                detect = Math.PI;
                delta = angle - prev_angle1;
                adjust = detect * ((delta < -detect * .7 ? 1 : 0) - (delta > detect * .7 ? 1 : 0));
                prev_angle1 = angle;
                cum_1pi += Math.Abs(adjust); /* fabs for when 2pi and 1pi have combined */
                pi_wraps[i >> 1] = cum_1pi;
                double tt = Math.Sqrt(work[i] * work[i] + work[i + 1] * work[i + 1]);
                // assert(tt >= 0)
                work[i] = tt > 0 ? Math.Log(tt) : -26;
                work[i + 1] = 0;
            }

            work[1] = work[work_len]; // LSX_PACK(work, work_len);

            SOXFft.safe_rdft(work_len, -1, work, info);
            for (i = 0; i < work_len; ++i) work[i] *= 2.0 / work_len;

            for (i = 1; i < work_len / 2; ++i)
            {
                /* Window to reject acausal components */
                work[i] *= 2;
                work[i + work_len / 2] = 0;
            }
            SOXFft.safe_rdft(work_len, 1, work, info);

            for (i = 2; i < work_len; i += 2) /* Interpolate between linear & min phase */
                work[i + 1] = phase1 * i / work_len * pi_wraps[work_len >> 1] +
                    (1 - phase1) * (work[i + 1] + pi_wraps[i >> 1]) - pi_wraps[i >> 1];

            work[0] = Math.Exp(work[0]);
            work[1] = Math.Exp(work[1]);
            for (i = 2; i < work_len; i += 2)
            {
                double x = Math.Exp(work[i]);
                work[i] = x * Math.Cos(work[i + 1]);
                work[i + 1] = x * Math.Sin(work[i + 1]);
            }

            SOXFft.safe_rdft(work_len, -1, work, info);
            for (i = 0; i < work_len; ++i) work[i] *= 2.0 / work_len;

            /* Find peak pos. */
            for (i = 0; i <= (int)(pi_wraps[work_len >> 1] / Math.PI + .5); ++i)
            {
                imp_sum += work[i];
                if (Math.Abs(imp_sum) > Math.Abs(peak_imp_sum))
                {
                    peak_imp_sum = imp_sum;
                    peak = i;
                }
                //if (work[i] > work[imp_peak]) /* For debug check only */
                //imp_peak = i;
            }
            while (peak > 0 && Math.Abs(work[peak - 1]) > Math.Abs(work[peak]) && work[peak - 1] * work[peak] > 0)
                --peak;

            if (phase1 == 0)
                begin = 0;
            else if (phase1 == 1)
                begin = peak - len / 2;
            else
            {
                begin = (int)((.997 - (2 - phase1) * .22) * len + .5);
                end = (int)((.997 + (0 - phase1) * .22) * len + .5);
                begin = peak - begin - (begin & 1);
                end = peak + 1 + end + (end & 1);
                len = end - begin;
                double[] h1 = new double[len];
                Buffer.BlockCopy(h, 0, h1, 0, Math.Min(h.Length, h1.Length) * sizeof(double));
                h = h1;
            }
            for (i = 0; i < len; ++i) h[i] =
              work[(begin + (phase > 50 ? len - 1 - i : i) + work_len) & (work_len - 1)];
            post_len = phase > 50 ? peak - begin : begin + len - (peak + 1);

            //lsx_debug("nPI=%g peak-sum@%i=%g (val@%i=%g); len=%i post=%i (%g%%)",
            //    pi_wraps[work_len >> 1] / M_PI, peak, peak_imp_sum, imp_peak,
            //    work[imp_peak], *len, *post_len, 100 - 100. * *post_len / (*len - 1));
            //free(pi_wraps), free(work);
        }

        static void half_band_filter_init(rate_shared_t p, int which,
            int num_taps, double[] h, double Fp, double att, int multiplier,
            double phase, bool allow_aliasing)
        {
            dft_filter_t f = p.half_band[which];
            int dft_length, i;

            if (f.num_taps != 0)
                return;
            if (h != null)
            {
                dft_length = lsx_set_dft_length(num_taps);
                f.coefs = new double[dft_length];
                for (i = 0; i < num_taps; ++i)
                    f.coefs[(i + dft_length - num_taps + 1) & (dft_length - 1)]
                        = h[Math.Abs(num_taps / 2 - i)] / dft_length * 2 * multiplier;
                f.post_peak = num_taps / 2;
            }
            else
            {
                h = lsx_design_lpf(Fp, 1.0, 2.0, allow_aliasing, att, ref num_taps, 0);

                if (phase != 50)
                    lsx_fir_to_phase(ref h, ref num_taps, ref f.post_peak, phase, p.info);
                else f.post_peak = num_taps / 2;

                dft_length = lsx_set_dft_length(num_taps);
                f.coefs = new double[dft_length];
                for (i = 0; i < num_taps; ++i)
                    f.coefs[(i + dft_length - num_taps + 1) & (dft_length - 1)]
                        = h[i] / dft_length * 2 * multiplier;
            }
            //assert(num_taps & 1);
            f.num_taps = num_taps;
            f.dft_length = dft_length;
            //lsx_debug("fir_len=%i dft_length=%i Fp=%g att=%g mult=%i",
            //num_taps, dft_length, Fp, att, multiplier);
            SOXFft.safe_rdft(dft_length, 1, f.coefs, p.info);
        }

        static double[] prepare_coefs(double[] coefs, int num_coefs,
                int num_phases, int interp_order, int multiplier)
        {
            int i, j, length = num_coefs * num_phases;
            double[] result = new double[length * (interp_order + 1)];
            double fm1 = coefs[0], f1 = 0, f2 = 0;

            for (i = num_coefs - 1; i >= 0; --i)
                for (j = num_phases - 1; j >= 0; --j)
                {
                    double f0 = fm1, b = 0, c = 0, d = 0; /* = 0 to kill compiler warning */
                    int pos = i * num_phases + j - 1;
                    fm1 = (pos > 0 ? coefs[pos - 1] : 0) * multiplier;
                    switch (interp_order)
                    {
                        case 1: b = f1 - f0; break;
                        case 2: b = f1 - (.5 * (f2 + f0) - f1) - f0; c = .5 * (f2 + f0) - f1; break;
                        case 3: c = .5 * (f1 + fm1) - f0; d = (1 / 6.0) * (f2 - f1 + fm1 - f0 - 4 * c); b = f1 - f0 - d - c; break;
                        default: /*if (interp_order) assert(0); */ break;
                    }
                    result[poly_fir1_t.coef_idx(interp_order, num_coefs, j, 0, num_coefs - 1 - i)] = f0;
                    if (interp_order > 0) result[poly_fir1_t.coef_idx(interp_order, num_coefs, j, 1, num_coefs - 1 - i)] = b;
                    if (interp_order > 1) result[poly_fir1_t.coef_idx(interp_order, num_coefs, j, 2, num_coefs - 1 - i)] = c;
                    if (interp_order > 2) result[poly_fir1_t.coef_idx(interp_order, num_coefs, j, 3, num_coefs - 1 - i)] = d;
                    f2 = f1;
                    f1 = f0;
                }
            return result;
        }

        public int output(ref int n)
        {
            fifo_t fifo = stages[output_stage_num].fifo;
            samples_out += (n = Math.Min(n, fifo.occupancy));
            while (samples_in > in_samplerate && samples_out > out_samplerate)
            {
                samples_in -= in_samplerate;
                samples_out -= out_samplerate;
            }
            int off = fifo.read(n, null);
            return off;
        }

        public unsafe void output(float[,] samples, int channel, ref int n)
        {
            int offs = output(ref n);
            //if (samples != null)
            //    Buffer.BlockCopy(fifo.data, off, samples, 0, n * sizeof(double));
            fixed (byte* psamples = &stages[output_stage_num].fifo.data[offs])
            {
                double* s = (double*)psamples;
                for (int i = 0; i < n; ++i)
                    samples[i, channel] = Math.Abs(s[i]) < 1.0 / 0x100000000 ? 0.0f : (float)s[i];
            }
        }

        public unsafe int input(int n)
        {
            samples_in += n;
            while (samples_in > in_samplerate && samples_out > out_samplerate)
            {
                samples_in -= in_samplerate;
                samples_out -= out_samplerate;
            }
            return stages[input_stage_num].fifo.reserve(n);
        }

        public unsafe void input(float[,] samples, int channel, int n)
        {
            int offs = input(n);
            fixed (byte* psamples = &stages[input_stage_num].fifo.data[offs])
            {
                double* s = (double*)psamples;
                for (int i = 0; i < n; ++i)
                    s[i] = samples[i, channel];
            }
        }

        public unsafe void process()
        {
            for (int i = input_stage_num; i < output_stage_num; ++i)
                stages[i].fn(stages[i], stages[i + 1].fifo);
        }

        public rate_t(int in_samplerate, int out_samplerate, rate_shared_t shared, double factor,
            SOXResamplerQuality quality, int interp_order, double phase, double bandwidth,
            bool allow_aliasing)
        {
            this.in_samplerate = in_samplerate;
            this.out_samplerate = out_samplerate;

            int i, mult, divisor = 1;

            //assert(factor > 0);
            this.factor = factor;
            if (quality < SOXResamplerQuality.Quick || quality > SOXResamplerQuality.Very)
                quality = SOXResamplerQuality.High;

            if (quality != SOXResamplerQuality.Quick)
            {
                const int max_divisor = 2048;      /* Keep coef table size ~< 500kb */
                const double epsilon = 4 / MULT32; /* Scaled to half this at max_divisor */
                this.upsample = this.factor < 1;
                this.level = 0;
                for (int fi = (int)factor >> 1; fi != 0; fi >>= 1)
                    ++this.level;/* log base 2 */
                factor /= 1 << (this.level + (this.upsample ? 0 : 1));
                for (i = 2; i <= max_divisor && divisor == 1; ++i)
                {
                    double try_d = factor * i;
                    int itry = (int)(try_d + .5);
                    if (Math.Abs(itry - try_d) < itry * epsilon * (1 - (.5 / max_divisor) * i))
                    {
                        if (itry == i)  /* Rounded to 1:1? */
                        {
                            factor = 1;
                            divisor = 2;
                            this.upsample = false;
                        }
                        else
                        {
                            factor = itry;
                            divisor = i;
                        }
                    }
                }
            }

            this.stages = new stage_t[this.level + 4]; // offset by 1!!! + 3?
            for (i = 0; i < this.level + 4; ++i)
                this.stages[i] = new stage_t(shared);
            this.pre_stage = this.stages[0];
            this.last_stage = this.stages[this.level + 1];
            this.post_stage = this.stages[this.level + 2];
            this.last_stage.step = (long)(factor * MULT32 + .5);
            this.last_stage.out_in_ratio = MULT32 * divisor / this.last_stage.step;

            //if (divisor != 1)
            //  assert(!last_stage.step.parts.fraction);
            //else if (quality != Quick)
            //  assert(!last_stage.step.parts.integer);
            //lsx_debug("i/o=%g; %.9g:%i @ level %i", this.factor, factor, divisor, this.level);

            mult = 1 + (this.upsample ? 1 : 0); /* Compensate for zero-stuffing in double_sample */

            this.input_stage_num = this.upsample ? 0 : 1;
            this.output_stage_num = this.level + 1;
            if (quality == SOXResamplerQuality.Quick)
            {
                ++this.output_stage_num;
                last_stage.fn = cubic_spline;
                last_stage.pre_post = Math.Max(3, (int)(last_stage.step >> 32));
                last_stage.preload = last_stage.pre = 1;
            }
            else if (last_stage.out_in_ratio != 2 || (this.upsample && quality == SOXResamplerQuality.Low))
            {
                int n = (this.upsample ? 4 : 0) + range_limit((int)quality, (int)SOXResamplerQuality.Medium, (int)SOXResamplerQuality.Very) - (int)SOXResamplerQuality.Medium;
                if (interp_order < 0)
                    interp_order = quality > SOXResamplerQuality.High ? 1 : 0;
                interp_order = divisor == 1 ? 1 + interp_order : 0;
                last_stage.divisor = divisor;
                this.output_stage_num += 2;
                if (this.upsample && quality == SOXResamplerQuality.Low)
                {
                    mult = 1;
                    ++this.input_stage_num;
                    --this.output_stage_num;
                    --n;
                }
                poly_fir_t f = poly_firs[n];
                poly_fir1_t f1 = f.interp[interp_order];
                if (last_stage.shared.poly_fir_coefs == null)
                {
                    int num_taps = 0, phases = divisor == 1 ? (1 << f1.phase_bits) : divisor;
                    double[] coefs = lsx_design_lpf(
                        f.pass, f.stop, 1.0, false, f.att, ref num_taps, phases);
                    //assert(num_taps == f->num_coefs * phases - 1);
                    last_stage.shared.poly_fir_coefs =
                        prepare_coefs(coefs, f.num_coefs, phases, interp_order, mult);
                    //lsx_debug("fir_len=%i phases=%i coef_interp=%i mult=%i size=%s",
                    //    f->num_coefs, phases, interp_order, mult,
                    //    lsx_sigfigs3((num_taps +1.) * (interp_order + 1) * sizeof(double)));
                    //free(coefs);
                }
                last_stage.fn = f1.fn;
                last_stage.pre_post = f.num_coefs - 1;
                last_stage.pre = 0;
                last_stage.preload = last_stage.pre_post >> 1;
                mult = 1;
            }
            if (quality > SOXResamplerQuality.Low)
            {
                //  typedef struct {int len; double const * h; double bw, a;} filter_t;
                //  static filter_t const filters[] = {
                //    {2 * array_length(half_fir_coefs_low) - 1, half_fir_coefs_low, 0,0},
                //    {0, NULL, .931, 110}, {0, NULL, .931, 125}, {0, NULL, .931, 170}};
                //  filter_t const * f = &filters[quality - Low];
                int[] fa = new int[] { 0, 110, 125, 170 };
                double[] fbw = new double[] { 0.0, 0.931, 0.931, 0.931 };
                double a = fa[quality - SOXResamplerQuality.Low];
                double att = allow_aliasing ? (34.0 / 33) * a : a; /* negate att degrade */
                double bw = bandwidth != 0 ? 1 - (1 - bandwidth / 100) / LSX_TO_3dB : fbw[quality - SOXResamplerQuality.Low];
                double min = 1 - (allow_aliasing ? LSX_MAX_TBW0A : LSX_MAX_TBW0) / 100;
                //  assert((size_t)(quality - Low) < array_length(filters));
                half_band_filter_init(shared, this.upsample ? 1 : 0, 0, null, bw, att, mult, phase, allow_aliasing);
                if (this.upsample)
                {
                    pre_stage.fn = double_sample; /* Finish off setting up pre-stage */
                    pre_stage.preload = shared.half_band[1].post_peak >> 1;
                    /* Start setting up post-stage; TODO don't use dft for short filters */
                    if ((1 - this.factor) / (1 - bw) > 2)
                        half_band_filter_init(shared, 0, 0, null, Math.Max(this.factor, min), att, 1, phase, allow_aliasing);
                    else shared.half_band[0] = shared.half_band[1];
                }
                else if (this.level > 0 && this.output_stage_num > this.level)
                {
                    double pass = bw * divisor / factor / 2;
                    if ((1 - pass) / (1 - bw) > 2)
                        half_band_filter_init(shared, 1, 0, null, Math.Max(pass, min), att, 1, phase, allow_aliasing);
                }
                post_stage.fn = half_sample;
                post_stage.preload = shared.half_band[0].post_peak;
            }
            else if (quality == SOXResamplerQuality.Low && !this.upsample)
            {    /* dft is slower here, so */
                post_stage.fn = half_sample_low;            /* use normal convolution */
                post_stage.pre_post = 2 * (half_fir_coefs_low.Length - 1);
                post_stage.preload = post_stage.pre = post_stage.pre_post >> 1;
            }
            if (this.level > 0)
            {
                stage_t s = this.stages[this.level];
                if (shared.half_band[1].num_taps != 0)
                {
                    s.fn = half_sample;
                    s.preload = shared.half_band[1].post_peak;
                    s.which = 1;
                }
                else
                {
                    //*s = post_stage
                    this.stages[this.level] = post_stage;
                    // ?????????
                    this.stages[this.level + 2] = s;
                }
            }
            for (i = this.input_stage_num; i <= this.output_stage_num; ++i)
            {
                stage_t s = this.stages[i];
                if (i > 0 && i < this.level)
                {
                    s.fn = half_sample_25;
                    s.pre_post = 2 * (half_fir_coefs_25.Length - 1);
                    s.preload = s.pre = s.pre_post >> 1;
                }
                s.fifo = new fifo_t(sizeof(double));
                s.fifo.reserve(s.preload);
                //  memset(fifo_reserve(&s->fifo, s->preload), 0, sizeof(double)*s->preload);

                //  if (i < this.output_stage_num)
                //    lsx_debug("stage=%-3ipre_post=%-3ipre=%-3ipreload=%i",
                //        i, s->pre_post, s->pre, s->preload);
            }
        }
    }
}
