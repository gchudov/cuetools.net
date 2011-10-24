using System;

namespace CUETools.Processor
{
    public class CUEToolsSelectionEventArgs : EventArgs
    {
        public object[] choices;
        public int selection = -1;
    }
}
