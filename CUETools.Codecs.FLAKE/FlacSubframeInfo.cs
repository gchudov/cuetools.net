using System;

namespace CUETools.Codecs.FLAKE
{
    unsafe public class FlacSubframeInfo
    {
        public FlacSubframeInfo()
        {
            best = new FlacSubframe();
            sf = new LpcSubframeInfo();
            lpc_ctx = new LpcContext[lpc.MAX_LPC_WINDOWS];
            for (int i = 0; i < lpc.MAX_LPC_WINDOWS; i++)
                lpc_ctx[i] = new LpcContext();
        }

        public void Init(int* s, int* r, int bps, int w)
        {
            if (w > bps)
                throw new Exception("internal error");
            samples = s;
            obits = bps - w;
            wbits = w;
            best.residual = r;
            best.type = SubframeType.Verbatim;
            best.size = AudioSamples.UINT32_MAX;
            sf.Reset();
            for (int iWindow = 0; iWindow < lpc.MAX_LPC_WINDOWS; iWindow++)
                lpc_ctx[iWindow].Reset();
            //sf.obits = obits;
            done_fixed = 0;
        }

        public FlacSubframe best;
        public int obits;
        public int wbits;
        public int* samples;
        public uint done_fixed;
        public LpcContext[] lpc_ctx;
        public LpcSubframeInfo sf;
    };
}
