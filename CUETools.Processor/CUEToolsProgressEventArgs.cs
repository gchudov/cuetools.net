using System;

namespace CUETools.Processor
{
    public class CUEToolsProgressEventArgs : EventArgs
    {
        public string status = string.Empty;
        public double percent = 0.0;
        public int offset = 0;
        public string input = string.Empty;
        public string output = string.Empty;
        public CUESheet cueSheet;
    }
}
