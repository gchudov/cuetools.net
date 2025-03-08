using System;

namespace CUETools.Ripper.Exceptions
{
    public class SCSIException : Exception
    {
        public SCSIException(string exceptionMessage) : base(exceptionMessage) { }
    }
}
