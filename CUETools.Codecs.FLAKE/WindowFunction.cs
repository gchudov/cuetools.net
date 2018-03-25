namespace CUETools.Codecs.Flake
{
    public enum WindowFunction
    {
        None = 0,
        Welch = 1,
        Tukey = 2,
        Hann = 4,
        Flattop = 8,
        Bartlett = 16,
        Tukey2 = 32,
        Tukey3 = 64,
        Tukey4 = (1 << 7),
        Tukey2A = (1 << 9),
        Tukey2B = (1 << 10),
        Tukey3A = (1 << 11),
        Tukey3B = (1 << 12),
        Tukey4A = (1 << 13),
        Tukey4B = (1 << 14),
        Tukey1A = (1 << 15),
        Tukey1B = (1 << 16),
        Tukey1X = (1 << 17),
        Tukey2X = (1 << 18),
        Tukey3X = (1 << 19),
        Tukey4X = (1 << 20),
    }
}
