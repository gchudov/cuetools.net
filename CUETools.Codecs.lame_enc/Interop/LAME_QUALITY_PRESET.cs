namespace CUETools.Codecs.LAME.Interop
{
    public enum LAME_QUALITY_PRESET : int
    {
        LQP_NOPRESET = -1,
        // QUALITY PRESETS
        LQP_NORMAL_QUALITY = 0,
        LQP_LOW_QUALITY = 1,
        LQP_HIGH_QUALITY = 2,
        LQP_VOICE_QUALITY = 3,
        LQP_R3MIX = 4,
        LQP_VERYHIGH_QUALITY = 5,
        LQP_STANDARD = 6,
        LQP_FAST_STANDARD = 7,
        LQP_EXTREME = 8,
        LQP_FAST_EXTREME = 9,
        LQP_INSANE = 10,
        LQP_ABR = 11,
        LQP_CBR = 12,
        LQP_MEDIUM = 13,
        LQP_FAST_MEDIUM = 14,
        // NEW PRESET VALUES
        LQP_PHONE = 1000,
        LQP_SW = 2000,
        LQP_AM = 3000,
        LQP_FM = 4000,
        LQP_VOICE = 5000,
        LQP_RADIO = 6000,
        LQP_TAPE = 7000,
        LQP_HIFI = 8000,
        LQP_CD = 9000,
        LQP_STUDIO = 10000
    }
}
