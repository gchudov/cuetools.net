using System;

namespace CUETools.Codecs
{
    /// <summary>
    /// Context for LPC coefficients calculation and order estimation
    /// </summary>
    unsafe public class LpcContext
    {
        public LpcContext()
        {
            coefs = new int[lpc.MAX_LPC_ORDER];
            reflection_coeffs = new double[lpc.MAX_LPC_ORDER];
            prediction_error = new double[lpc.MAX_LPC_ORDER];
            autocorr_values = new double[lpc.MAX_LPC_ORDER + 1];
            best_orders = new int[lpc.MAX_LPC_ORDER];
            done_lpcs = new uint[lpc.MAX_LPC_PRECISIONS];
        }

        /// <summary>
        /// Reset to initial (blank) state
        /// </summary>
        public void Reset()
        {
            autocorr_order = 0;
            for (int iPrecision = 0; iPrecision < lpc.MAX_LPC_PRECISIONS; iPrecision++)
                done_lpcs[iPrecision] = 0;
        }

        /// <summary>
        /// Calculate autocorrelation data and reflection coefficients.
        /// Can be used to incrementaly compute coefficients for higher orders,
        /// because it caches them.
        /// </summary>
        /// <param name="order">Maximum order</param>
        /// <param name="samples">Samples pointer</param>
        /// <param name="blocksize">Block size</param>
        /// <param name="window">Window function</param>
        public void GetReflection(int order, int* samples, int blocksize, float* window)
        {
            if (autocorr_order > order)
                return;
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                lpc.compute_autocorr(samples, blocksize, autocorr_order, order, autoc, window);
                lpc.compute_schur_reflection(autoc, (uint)order, reff, err);
                autocorr_order = order + 1;
            }
        }

        public void GetReflection1(int order, int* samples, int blocksize, float* window)
        {
            if (autocorr_order > order)
                return;
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                lpc.compute_autocorr(samples, blocksize, 0, order + 1, autoc, window);
                for (int i = 1; i <= order; i++)
                    autoc[i] = autoc[i + 1];
                lpc.compute_schur_reflection(autoc, (uint)order, reff, err);
                autocorr_order = order + 1;
            }
        }

        public void ComputeReflection(int order, float* autocorr)
        {
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                for (int i = 0; i <= order; i++)
                    autoc[i] = autocorr[i];
                lpc.compute_schur_reflection(autoc, (uint)order, reff, err);
                autocorr_order = order + 1;
            }
        }

        public void ComputeReflection(int order, double* autocorr)
        {
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                for (int i = 0; i <= order; i++)
                    autoc[i] = autocorr[i];
                lpc.compute_schur_reflection(autoc, (uint)order, reff, err);
                autocorr_order = order + 1;
            }
        }

        public double Akaike(int blocksize, int order, double alpha, double beta)
        {
            //return (blocksize - order) * (Math.Log(prediction_error[order - 1]) - Math.Log(1.0)) + Math.Log(blocksize) * order * (alpha + beta * order);
            return blocksize * Math.Log(prediction_error[order - 1]) + Math.Log(blocksize) * order * (alpha + beta * order);
        }

        /// <summary>
        /// Sorts orders based on Akaike's criteria
        /// </summary>
        /// <param name="blocksize">Frame size</param>
        public void SortOrdersAkaike(int blocksize, int count, int max_order, double alpha, double beta)
        {
            for (int i = 0; i < max_order; i++)
                best_orders[i] = i + 1;
            for (int i = 0; i < max_order && i < count; i++)
            {
                for (int j = i + 1; j < max_order; j++)
                {
                    if (Akaike(blocksize, best_orders[j], alpha, beta) < Akaike(blocksize, best_orders[i], alpha, beta))
                    {
                        int tmp = best_orders[j];
                        best_orders[j] = best_orders[i];
                        best_orders[i] = tmp;
                    }
                }
            }
        }

        /// <summary>
        /// Produces LPC coefficients from autocorrelation data.
        /// </summary>
        /// <param name="lpcs">LPC coefficients buffer (for all orders)</param>
        public void ComputeLPC(float* lpcs)
        {
            fixed (double* reff = reflection_coeffs)
                lpc.compute_lpc_coefs((uint)autocorr_order - 1, reff, lpcs);
        }

        public double[] autocorr_values;
        double[] reflection_coeffs;
        public double[] prediction_error;
        public int[] best_orders;
        public int[] coefs;
        int autocorr_order;
        public int shift;

        public double[] Reflection
        {
            get
            {
                return reflection_coeffs;
            }
        }

        public uint[] done_lpcs;
    }
}
