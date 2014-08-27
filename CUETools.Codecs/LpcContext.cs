using System;
using System.Collections.Generic;

namespace CUETools.Codecs
{
    unsafe public class LpcSubframeInfo
    {
        public LpcSubframeInfo()
        {
            autocorr_section_values = new double[lpc.MAX_LPC_SECTIONS, lpc.MAX_LPC_ORDER + 1];
            autocorr_section_orders = new int[lpc.MAX_LPC_SECTIONS];
        }

        // public LpcContext[] lpc_ctx;
        public double[,] autocorr_section_values;
        public int[] autocorr_section_orders;
        //public int obits;

        public void Reset()
        {
            for (int sec = 0; sec < autocorr_section_orders.Length; sec++)
                autocorr_section_orders[sec] = 0;
        }
    }

    unsafe public struct LpcWindowSection
    {
        public enum SectionType
        {
            Zero,
            One,
            Data,
            Glue
        };
        public int m_start;
        public int m_end;
        public SectionType m_type;
        public int m_id;
        public LpcWindowSection(int end)
        {
            m_id = -1;
            m_start = 0;
            m_end = end;
            m_type = SectionType.Data;
        }
        public void setData(int start, int end)
        {
            m_id = -1;
            m_start = start;
            m_end = end;
            m_type = SectionType.Data;
        }
        public void setOne(int start, int end)
        {
            m_id = -1;
            m_start = start;
            m_end = end;
            m_type = SectionType.One;
        }
        public void setGlue(int start)
        {
            m_id = -1;
            m_start = start;
            m_end = start;
            m_type = SectionType.Glue;
        }
        public void setZero(int start, int end)
        {
            m_id = -1;
            m_start = start;
            m_end = end;
            m_type = SectionType.Zero;
        }

        unsafe public static void Detect(int _windowcount, float* window_segment, int stride, int sz, LpcWindowSection* sections)
        {
            int section_id = 0;
            var boundaries = new List<int>();
            var types = new LpcWindowSection.SectionType[_windowcount, lpc.MAX_LPC_SECTIONS * 2];
            for (int x = 0; x < sz; x++)
            {
                for (int i = 0; i < _windowcount; i++)
                {
                    float w = window_segment[i * stride + x];
                    types[i, boundaries.Count] =
                        boundaries.Count >= lpc.MAX_LPC_SECTIONS * 2 - 2 ?
                        LpcWindowSection.SectionType.Data : w == 0.0 ?
                        LpcWindowSection.SectionType.Zero : w == 1.0 ?
                        LpcWindowSection.SectionType.One :
                        LpcWindowSection.SectionType.Data;
                }
                bool isBoundary = false;
                for (int i = 0; i < _windowcount; i++)
                {
                    isBoundary |= boundaries.Count == 0 ||
                        types[i, boundaries.Count - 1] != types[i, boundaries.Count];
                }
                if (isBoundary) boundaries.Add(x);
            }
            boundaries.Add(sz);
            var ones = new int[boundaries.Count - 1];
            // Reconstruct segments list.
            for (int i = 0; i < _windowcount; i++)
            {
                int secs = 0;
                for (int j = 0; j < boundaries.Count - 1; j++)
                {
                    if (types[i, j] == LpcWindowSection.SectionType.Zero)
                    {
                        if (secs > 0 && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_end == boundaries[j] && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_type == LpcWindowSection.SectionType.Zero)
                        {
                            sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_end = boundaries[j + 1];
                            continue;
                        }
                        sections[i * lpc.MAX_LPC_SECTIONS + secs++].setZero(boundaries[j], boundaries[j + 1]);
                        continue;
                    }
                    if (types[i, j] == LpcWindowSection.SectionType.Data
                        || secs + 1 >= lpc.MAX_LPC_SECTIONS
                        || (boundaries[j + 1] - boundaries[j] < lpc.MAX_LPC_ORDER))
                    {
                        if (secs > 0 && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_end == boundaries[j] && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_type == LpcWindowSection.SectionType.Data)
                        {
                            sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_end = boundaries[j + 1];
                            continue;
                        }
                        sections[i * lpc.MAX_LPC_SECTIONS + secs++].setData(boundaries[j], boundaries[j + 1]);
                        continue;
                    }

                    if (secs > 0 && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_end == boundaries[j] && sections[i * lpc.MAX_LPC_SECTIONS + secs - 1].m_type == LpcWindowSection.SectionType.One)
                        sections[i * lpc.MAX_LPC_SECTIONS + secs++].setGlue(boundaries[j]);
                    sections[i * lpc.MAX_LPC_SECTIONS + secs++].setOne(boundaries[j], boundaries[j + 1]);
                    ones[j] |= 1 << i;
                }
                while (secs < lpc.MAX_LPC_SECTIONS)
                    sections[i * lpc.MAX_LPC_SECTIONS + secs++].setZero(sz, sz);
            }
            for (int j = 0; j < boundaries.Count - 1; j++)
            {
                if (j > 0 && ones[j - 1] == ones[j])
                {
                    for (int i = 0; i < _windowcount; i++)
                    {
                        for (int sec = 0; sec < lpc.MAX_LPC_SECTIONS; sec++)
                            if (sections[i * lpc.MAX_LPC_SECTIONS + sec].m_type == LpcWindowSection.SectionType.Glue &&
                                sections[i * lpc.MAX_LPC_SECTIONS + sec].m_start == boundaries[j])
                            {
                                sections[i * lpc.MAX_LPC_SECTIONS + sec - 1].m_end = sections[i * lpc.MAX_LPC_SECTIONS + sec + 1].m_end;
                                for (int sec1 = sec; sec1 + 2 < lpc.MAX_LPC_SECTIONS; sec1++)
                                    sections[i * lpc.MAX_LPC_SECTIONS + sec1] = sections[i * lpc.MAX_LPC_SECTIONS + sec1 + 2];
                            }
                    }
                    continue;
                }
                if ((ones[j] & (ones[j] - 1)) != 0 && section_id < lpc.MAX_LPC_SECTIONS)
                {
                    for (int i = 0; i < _windowcount; i++)
                        for (int sec = 0; sec < lpc.MAX_LPC_SECTIONS; sec++)
                            if (sections[i * lpc.MAX_LPC_SECTIONS + sec].m_type == LpcWindowSection.SectionType.One &&
                                sections[i * lpc.MAX_LPC_SECTIONS + sec].m_start == boundaries[j])
                            {
                                sections[i * lpc.MAX_LPC_SECTIONS + sec].m_id = section_id;
                            }
                    section_id++;
                }
            }
        }
    }

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
        public void GetReflection(LpcSubframeInfo subframe, int order, int* samples, float* window, LpcWindowSection* sections, bool large)
        {
            if (autocorr_order > order)
                return;
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                for (int i = autocorr_order; i <= order; i++) autoc[i] = 0;
                int prev = 0;
                for (int section = 0; section < lpc.MAX_LPC_SECTIONS; section++)
                {
                    if (sections[section].m_type == LpcWindowSection.SectionType.Zero)
                    {
                        prev = 0;
                        continue;
                    }
                    if (sections[section].m_type == LpcWindowSection.SectionType.Data)
                    {
                        int next = section + 1 < lpc.MAX_LPC_SECTIONS && sections[section + 1].m_type == LpcWindowSection.SectionType.One ? 1 : 0;
                        lpc.compute_autocorr(samples + sections[section].m_start, window + sections[section].m_start, sections[section].m_end - sections[section].m_start, autocorr_order, order, autoc, prev, next);
                    }
                    else if (sections[section].m_type == LpcWindowSection.SectionType.Glue)
                        lpc.compute_autocorr_glue(samples + sections[section].m_start, autocorr_order, order, autoc);
                    else if (sections[section].m_type == LpcWindowSection.SectionType.One)
                    {
                        if (sections[section].m_id >= 0)
                        {
                            if (subframe.autocorr_section_orders[sections[section].m_id] <= order)
                            {
                                fixed (double* autocsec = &subframe.autocorr_section_values[sections[section].m_id, 0])
                                {
                                    for (int i = subframe.autocorr_section_orders[sections[section].m_id]; i <= order; i++) autocsec[i] = 0;
                                    if (large)
                                        lpc.compute_autocorr_windowless_large(samples + sections[section].m_start, sections[section].m_end - sections[section].m_start, subframe.autocorr_section_orders[sections[section].m_id], order, autocsec);
                                    else
                                        lpc.compute_autocorr_windowless(samples + sections[section].m_start, sections[section].m_end - sections[section].m_start, subframe.autocorr_section_orders[sections[section].m_id], order, autocsec);
                                }
                                subframe.autocorr_section_orders[sections[section].m_id] = order + 1;
                            }
                            for (int i = autocorr_order; i <= order; i++)
                                autoc[i] += subframe.autocorr_section_values[sections[section].m_id, i];
                        }
                        else
                        {
                            if (large)
                                lpc.compute_autocorr_windowless_large(samples + sections[section].m_start, sections[section].m_end - sections[section].m_start, autocorr_order, order, autoc);
                            else
                                lpc.compute_autocorr_windowless(samples + sections[section].m_start, sections[section].m_end - sections[section].m_start, autocorr_order, order, autoc);
                        }
                        prev = 1;
                    }
                }
                lpc.compute_schur_reflection(autoc, (uint)order, reff, err);
                autocorr_order = order + 1;
            }
        }
#if XXX
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
#endif
        public double Akaike(int blocksize, int order, double alpha, double beta)
        {
            //return (blocksize - order) * (Math.Log(prediction_error[order - 1]) - Math.Log(1.0)) + Math.Log(blocksize) * order * (alpha + beta * order);
            //return blocksize * (Math.Log(prediction_error[order - 1]) - Math.Log(autocorr_values[0]) / 2) + Math.Log(blocksize) * order * (alpha + beta * order);
            return blocksize * (Math.Log(prediction_error[order - 1])) + Math.Log(blocksize) * order * (alpha + beta * order);
        }

        /// <summary>
        /// Sorts orders based on Akaike's criteria
        /// </summary>
        /// <param name="blocksize">Frame size</param>
        public void SortOrdersAkaike(int blocksize, int count, int min_order, int max_order, double alpha, double beta)
        {
            for (int i = min_order; i <= max_order; i++)
                best_orders[i - min_order] = i;
            int lim = max_order - min_order + 1;
            for (int i = 0; i < lim && i < count; i++)
            {
                for (int j = i + 1; j < lim; j++)
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
