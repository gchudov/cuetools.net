namespace CUETools.Codecs.ALAC
{
    unsafe class ALACSubframe
    {
        public ALACSubframe()
        {
            rc = new RiceContext();
            coefs = new int[lpc.MAX_LPC_ORDER];
            coefs_adapted = new int[lpc.MAX_LPC_ORDER];
        }
        public int order;
        public int* residual;
        public RiceContext rc;
        public uint size;

        public int ricemodifier;
        public int cbits;
        public int shift;
        public int[] coefs;
        public int[] coefs_adapted;
        public int window;
    }
}
