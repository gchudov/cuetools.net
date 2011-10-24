using System;

namespace CUETools.DSP.Resampler.Internal
{
    class stage_t
    {
        internal rate_shared_t shared;
        internal fifo_t fifo;
        internal int pre;				/* Number of past samples to store */
        internal int pre_post;			/* pre + number of future samples to store */
        internal int preload;			/* Number of zero samples to pre-load the fifo */
        internal int which;				/* Which of the 2 half-band filters to use */
        internal stage_fn_t fn;
        internal long at, step;			/* For poly_fir & spline: */
        internal int divisor;			/* For step: > 1 for rational; 1 otherwise */
        internal double out_in_ratio;

        internal stage_t(rate_shared_t shared)
        {
            this.shared = shared;
        }

        internal int offset
        {
            get
            {
                return fifo.offset + pre * sizeof(double);
            }
        }

        internal int occupancy
        {
            get
            {
                return Math.Max(0, fifo.occupancy - pre_post);
            }
        }
    }
}
