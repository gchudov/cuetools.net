using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.libmp3lame
{
    public class LameException : Exception
    {
        public LameException(string message)
            : base(message)
        {
        }
    }
}
