using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameWriterSettings : AudioEncoderSettings
    {
        public LameWriterSettings(string modes, string defaultMode)
            : base(modes, defaultMode)
        {
        }

        public virtual void Apply(IntPtr lame)
        {
            throw new MethodAccessException();
        }
    }
}
