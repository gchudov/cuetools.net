using System;
using CUETools.Codecs;

namespace CUETools.DSP.Mixer
{
    public class AudioReadEventArgs : EventArgs
    {
        public IAudioSource source;
        public AudioBuffer buffer;
    }
}
