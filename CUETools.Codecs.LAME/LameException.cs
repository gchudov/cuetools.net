using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public class LameException : Exception
    {
        public LameException(string message)
            : base(message)
        {
        }
    }
}
