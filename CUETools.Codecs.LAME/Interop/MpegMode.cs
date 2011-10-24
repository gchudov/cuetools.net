namespace CUETools.Codecs.LAME.Interop
{
    /* MPEG modes */
    public enum MpegMode : uint
    {
        STEREO = 0,
        JOINT_STEREO,
        DUAL_CHANNEL,	 /* LAME doesn't supports this! */
        MONO,
        NOT_SET,
        MAX_INDICATOR	 /* Don't use this! It's used for sanity checks. */
    }
}
