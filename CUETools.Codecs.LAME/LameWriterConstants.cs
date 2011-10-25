using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.LAME
{
    public enum LameQuality
    {
        High = 2,
        Normal = 5,
        Fast = 7,
    }

    public enum LameVbrMode
    {
        Off = 0,
        Mt = 1,
        Rh = 2,
        Abr = 3,
        Default = 4,
    }
}
