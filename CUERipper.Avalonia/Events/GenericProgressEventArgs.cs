using System;

namespace CUERipper.Avalonia.Events
{
    public class GenericProgressEventArgs : EventArgs
    {
        public float Progress { get; set; }

        public GenericProgressEventArgs(float progress)
        {
            Progress = progress;
        }
    }
}
