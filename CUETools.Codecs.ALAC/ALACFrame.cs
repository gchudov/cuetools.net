namespace CUETools.Codecs.ALAC
{
    unsafe class ALACFrame
    {
        public FrameType type;
        public int blocksize;
        public int interlacing_shift, interlacing_leftweight;
        public ALACSubframeInfo[] subframes;
        public ALACSubframe current;
        public float* window_buffer;

        public ALACFrame(int subframes_count)
        {
            subframes = new ALACSubframeInfo[subframes_count];
            for (int ch = 0; ch < subframes_count; ch++)
                subframes[ch] = new ALACSubframeInfo();
            current = new ALACSubframe();
        }

        public void InitSize(int bs)
        {
            blocksize = bs;
            type = FrameType.Verbatim;
            interlacing_shift = interlacing_leftweight = 0;
        }

        public void ChooseBestSubframe(int ch)
        {
            if (current.size >= subframes[ch].best.size)
                return;
            ALACSubframe tmp = subframes[ch].best;
            subframes[ch].best = current;
            current = tmp;
        }

        public void SwapSubframes(int ch1, int ch2)
        {
            ALACSubframeInfo tmp = subframes[ch1];
            subframes[ch1] = subframes[ch2];
            subframes[ch2] = tmp;
        }

        /// <summary>
        /// Swap subframes according to channel mode.
        /// It is assumed that we have 4 subframes,
        /// 0 is right, 1 is left, 2 is middle, 3 is difference
        /// </summary>
        public void ChooseSubframes()
        {
            if (interlacing_leftweight != 0)
            {
                SwapSubframes(1, 3);
                switch (interlacing_shift)
                {
                    case 0: // leftside						
                        break;
                    case 1: // midside
                        SwapSubframes(0, 2);
                        break;
                    case 31: // rightside
                        SwapSubframes(0, 4);
                        break;
                }
            }
        }
    }
}
