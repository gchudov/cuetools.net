namespace CUETools.Codecs.Flake
{
    unsafe public class FlacSubframe
    {
        public FlacSubframe()
        {
            rc = new RiceContext();
            coefs = new int[lpc.MAX_LPC_ORDER];
        }
        public SubframeType type;
        public int order;
        public int* residual;
        public RiceContext rc;
        public uint size;

        public int cbits;
        public int shift;
        public int[] coefs;
        public int window;
    };
}
