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
            OneLarge,
            Data,
            OneGlue,
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

        unsafe public void compute_autocorr(/*const*/ int* data, float* window, int min_order, int order, int blocksize, double* autoc)
        {
            if (m_type == SectionType.OneLarge)
                lpc.compute_autocorr_windowless_large(data + m_start, m_end - m_start, min_order, order, autoc);
            else if (m_type == SectionType.One)
                lpc.compute_autocorr_windowless(data + m_start, m_end - m_start, min_order, order, autoc);
            else if (m_type == SectionType.Data)
                lpc.compute_autocorr(data + m_start, window + m_start, m_end - m_start, min_order, order, autoc);
            else if (m_type == SectionType.Glue)
                lpc.compute_autocorr_glue(data, window, m_start, m_end, min_order, order, autoc);
            else if (m_type == SectionType.OneGlue)
                lpc.compute_autocorr_glue(data + m_start, min_order, order, autoc);
        }

        unsafe public static void Detect(int _windowcount, float* window_segment, int stride, int sz, int bps, LpcWindowSection* sections)
        {
            int section_id = 0;
            var boundaries = new List<int>();
            var types = new LpcWindowSection.SectionType[_windowcount, lpc.MAX_LPC_SECTIONS * 2];
            var alias = new int[_windowcount, lpc.MAX_LPC_SECTIONS * 2];
            var alias_set = new int[_windowcount, lpc.MAX_LPC_SECTIONS * 2];
            for (int x = 0; x < sz; x++)
            {
                for (int i = 0; i < _windowcount; i++)
                {
                    int a = alias[i, boundaries.Count];
                    float w = window_segment[i * stride + x];
                    float wa = window_segment[a * stride + x];
                    if (wa != w)
                    {
                        for (int i1 = i; i1 < _windowcount; i1++)
                            if (alias[i1, boundaries.Count] == a
                                && w == window_segment[i1 * stride + x])
                                alias[i1, boundaries.Count] = i;
                    }
                    if (boundaries.Count >= lpc.MAX_LPC_SECTIONS * 2) throw new IndexOutOfRangeException();
                    types[i, boundaries.Count] =
                        boundaries.Count >= lpc.MAX_LPC_SECTIONS * 2 - 2 ?
                        LpcWindowSection.SectionType.Data : w == 0.0 ?
                        LpcWindowSection.SectionType.Zero : w != 1.0 ?
                        LpcWindowSection.SectionType.Data : bps * 2 + BitReader.log2i(sz) >= 61 ?
                        LpcWindowSection.SectionType.OneLarge :
                        LpcWindowSection.SectionType.One ;
                }
                bool isBoundary = false;
                for (int i = 0; i < _windowcount; i++)
                {
                    isBoundary |= boundaries.Count == 0 ||
                        types[i, boundaries.Count - 1] != types[i, boundaries.Count];
                }
                if (isBoundary)
                {
                    for (int i = 0; i < _windowcount; i++)
                        for (int i1 = 0; i1 < _windowcount; i1++)
                            if (i != i1 && alias[i, boundaries.Count] == alias[i1, boundaries.Count])
                                alias_set[i, boundaries.Count] |= 1 << i1;
                    boundaries.Add(x);
                }
            }
            boundaries.Add(sz);
            var secs = new int[_windowcount];
            // Reconstruct segments list.
            for (int j = 0; j < boundaries.Count - 1; j++)
            {
                for (int i = 0; i < _windowcount; i++)
                {
                    LpcWindowSection* window_sections = sections + i * lpc.MAX_LPC_SECTIONS;
                    // leave room for glue
                    if (secs[i] >= lpc.MAX_LPC_SECTIONS - 1)
                    {
                        throw new IndexOutOfRangeException();
                        window_sections[secs[i] - 1].m_type = LpcWindowSection.SectionType.Data;
                        window_sections[secs[i] - 1].m_end = boundaries[j + 1];
                        continue;
                    }
                    window_sections[secs[i]].setData(boundaries[j], boundaries[j + 1]);
                    window_sections[secs[i]++].m_type = types[i, j];
                }
                for (int i = 0; i < _windowcount; i++)
                {
                    LpcWindowSection* window_sections = sections + i * lpc.MAX_LPC_SECTIONS;
                    int sec = secs[i] - 1;
                    if (sec > 0
                        && j > 0 && (alias_set[i, j] == alias_set[i, j - 1] || window_sections[sec].m_type == SectionType.Zero)
                        && window_sections[sec].m_start == boundaries[j]
                        && window_sections[sec].m_end == boundaries[j + 1]
                        && window_sections[sec - 1].m_end == boundaries[j]
                        && window_sections[sec - 1].m_type == window_sections[sec].m_type)
                    {
                        window_sections[sec - 1].m_end = window_sections[sec].m_end;
                        secs[i]--;
                        continue;
                    }
                    if (section_id >= lpc.MAX_LPC_SECTIONS) throw new IndexOutOfRangeException();
                    if (alias_set[i, j] != 0
                        && types[i, j] != SectionType.Zero
                        && section_id < lpc.MAX_LPC_SECTIONS)
                    {
                        for (int i1 = i; i1 < _windowcount; i1++)
                            if (alias[i1, j] == i && secs[i1] > 0)
                                sections[i1 * lpc.MAX_LPC_SECTIONS + secs[i1] - 1].m_id = section_id;
                        section_id++;
                    }
                    // TODO: section_id for glue? nontrivial, must be sure next sections are the same size
                    if (sec > 0
                        && (window_sections[sec].m_type == SectionType.One || window_sections[sec].m_type == SectionType.OneLarge)
                        && window_sections[sec].m_end - window_sections[sec].m_start >= lpc.MAX_LPC_ORDER
                        && (window_sections[sec - 1].m_type == SectionType.One || window_sections[sec - 1].m_type == SectionType.OneLarge)
                        && window_sections[sec - 1].m_end - window_sections[sec - 1].m_start >= lpc.MAX_LPC_ORDER)
                    {
                        window_sections[sec + 1] = window_sections[sec];
                        window_sections[sec].m_end = window_sections[sec].m_start;
                        window_sections[sec].m_type = SectionType.OneGlue;
                        window_sections[sec].m_id = -1;
                        secs[i]++;
                        continue;
                    }
                    if (sec > 0
                        && window_sections[sec].m_type != SectionType.Zero
                        && window_sections[sec - 1].m_type != SectionType.Zero)
                    {
                        window_sections[sec + 1] = window_sections[sec];
                        window_sections[sec].m_end = window_sections[sec].m_start;
                        window_sections[sec].m_type = SectionType.Glue;
                        window_sections[sec].m_id = -1;
                        secs[i]++;
                        continue;
                    }
                }
            }
            for (int i = 0; i < _windowcount; i++)
            {
                for (int s = 0; s < secs[i]; s++)
                {
                    LpcWindowSection* window_sections = sections + i * lpc.MAX_LPC_SECTIONS;
                    if (window_sections[s].m_type == SectionType.Glue
                        || window_sections[s].m_type == SectionType.OneGlue)
                    {
                        window_sections[s].m_end = window_sections[s + 1].m_end;
                    }
                }
                while (secs[i] < lpc.MAX_LPC_SECTIONS)
                {
                    LpcWindowSection* window_sections = sections + i * lpc.MAX_LPC_SECTIONS;
                    window_sections[secs[i]++].setZero(sz, sz);
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
        public void GetReflection(LpcSubframeInfo subframe, int order, int blocksize, int* samples, float* window, LpcWindowSection* sections)
        {
            if (autocorr_order > order)
                return;
            fixed (double* reff = reflection_coeffs, autoc = autocorr_values, err = prediction_error)
            {
                for (int i = autocorr_order; i <= order; i++) autoc[i] = 0;
                for (int section = 0; section < lpc.MAX_LPC_SECTIONS; section++)
                {
                    if (sections[section].m_type == LpcWindowSection.SectionType.Zero)
                    {
                        continue;
                    }
                    if (sections[section].m_id >= 0)
                    {
                        if (subframe.autocorr_section_orders[sections[section].m_id] <= order)
                        {
                            fixed (double* autocsec = &subframe.autocorr_section_values[sections[section].m_id, 0])
                            {
                                int min_order = subframe.autocorr_section_orders[sections[section].m_id];
                                for (int i = min_order; i <= order; i++) autocsec[i] = 0;
                                sections[section].compute_autocorr(samples, window, min_order, order, blocksize, autocsec);
                            }
                            subframe.autocorr_section_orders[sections[section].m_id] = order + 1;
                        }
                        for (int i = autocorr_order; i <= order; i++)
                            autoc[i] += subframe.autocorr_section_values[sections[section].m_id, i];
                    }
                    else
                    {
                        sections[section].compute_autocorr(samples, window, autocorr_order, order, blocksize, autoc);
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
