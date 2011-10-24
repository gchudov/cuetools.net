namespace CUETools.Codecs.ALAC
{
    unsafe class RiceContext
    {
        public RiceContext()
        {
            rparams = new int[Alac.MAX_PARTITIONS];
            esc_bps = new int[Alac.MAX_PARTITIONS];
        }
        /// <summary>
        /// partition order
        /// </summary>
        public int porder;

        /// <summary>
        /// Rice parameters
        /// </summary>
        public int[] rparams;

        /// <summary>
        /// bps if using escape code
        /// </summary>
        public int[] esc_bps;
    }
}
