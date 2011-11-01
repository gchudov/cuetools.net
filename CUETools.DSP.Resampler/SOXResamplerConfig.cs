namespace CUETools.DSP.Resampler
{
    public struct SOXResamplerConfig
    {
        public double Phase;
        public double Bandwidth;
        public bool AllowAliasing;
        public SOXResamplerQuality Quality;
        /*double coef_interp; -- interpolation...*/
    }
}
