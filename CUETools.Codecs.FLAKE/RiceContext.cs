namespace CUETools.Codecs.FLAKE
{
    unsafe public class RiceContext
    {
        public RiceContext()
        {
            rparams = new int[Flake.MAX_PARTITIONS];
            esc_bps = new int[Flake.MAX_PARTITIONS];
        }
        /// <summary>
        /// partition order
        /// </summary>
        public int porder;

        /// <summary>
        /// coding method: rice parameters use 4 bits for coding_method 0 and 5 bits for coding_method 1
        /// </summary>
        public int coding_method;

        /// <summary>
        /// Rice parameters
        /// </summary>
        public int[] rparams;

        /// <summary>
        /// bps if using escape code
        /// </summary>
        public int[] esc_bps;
    };
}
