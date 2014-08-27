namespace CUETools.Codecs.FLAKE
{
    public enum WindowFunction
    {
        None = 0,
        Welch = 1,
        Tukey = 2,
        Hann = 4,
        Flattop = 8,
        Bartlett = 16,
        TukeyFlattop = 10,
        PartialTukey = 32,
        TukeyPartialTukey = 34,
        TukeyFlattopPartialTukey = 42,
        PunchoutTukey = 64,
        TukeyFlattopPartialTukeyPunchoutTukey = 106,
    }
}
