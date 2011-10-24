namespace CUETools.DSP.Resampler
{
    public struct SOXResamplerConfig
    {
        public double phase;
        public double bandwidth;
        public bool allow_aliasing;
        public SOXResamplerQuality quality;
        /*double coef_interp; -- interpolation...*/
    }
}
