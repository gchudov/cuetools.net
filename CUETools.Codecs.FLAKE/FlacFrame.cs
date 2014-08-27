namespace CUETools.Codecs.FLAKE
{
    unsafe public class FlacFrame
    {
        public int blocksize;
        public int bs_code0, bs_code1;
        public ChannelMode ch_mode;
        //public int ch_order0, ch_order1;
        public byte crc8;
        public FlacSubframeInfo[] subframes;
        public int frame_number;
        public FlacSubframe current;
        public float* window_buffer;
        public int nSeg = 0;

        public BitWriter writer = null;
        public int writer_offset = 0;

        public FlacFrame(int subframes_count)
        {
            subframes = new FlacSubframeInfo[subframes_count];
            for (int ch = 0; ch < subframes_count; ch++)
                subframes[ch] = new FlacSubframeInfo();
            current = new FlacSubframe();
        }

        public void InitSize(int bs, bool vbs)
        {
            blocksize = bs;
            int i = 15;
            if (!vbs)
            {
                for (i = 0; i < 15; i++)
                {
                    if (bs == Flake.flac_blocksizes[i])
                    {
                        bs_code0 = i;
                        bs_code1 = -1;
                        break;
                    }
                }
            }
            if (i == 15)
            {
                if (blocksize <= 256)
                {
                    bs_code0 = 6;
                    bs_code1 = blocksize - 1;
                }
                else
                {
                    bs_code0 = 7;
                    bs_code1 = blocksize - 1;
                }
            }
        }

        public void ChooseBestSubframe(int ch)
        {
            if (current.size >= subframes[ch].best.size)
                return;
            FlacSubframe tmp = subframes[ch].best;
            subframes[ch].best = current;
            current = tmp;
        }

        public void SwapSubframes(int ch1, int ch2)
        {
            FlacSubframeInfo tmp = subframes[ch1];
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
            switch (ch_mode)
            {
                case ChannelMode.MidSide:
                    SwapSubframes(0, 2);
                    SwapSubframes(1, 3);
                    break;
                case ChannelMode.RightSide:
                    SwapSubframes(0, 3);
                    break;
                case ChannelMode.LeftSide:
                    SwapSubframes(1, 3);
                    break;
            }
        }
    }
}
