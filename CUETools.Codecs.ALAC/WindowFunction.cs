namespace CUETools.Codecs.ALAC
{
    public enum WindowFunction
    {
        Welch = 1,
        Tukey = 2,
        Hann = 4,
        Flattop = 8,
        Bartlett = 16,
        TukFlat = 10,
        PartialTukey = 32,
        PunchoutTukey = 64,
    }
}
