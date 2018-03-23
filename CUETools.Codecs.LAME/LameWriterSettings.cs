using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameEncoderSettings : AudioEncoderSettings
    {
        public override Type EncoderType => typeof(AudioEncoder);

        public LameEncoderSettings(string modes, string defaultMode)
            : base(modes, defaultMode)
        {
        }

        public virtual void Apply(IntPtr lame)
        {
            throw new MethodAccessException();
        }
    }
}
