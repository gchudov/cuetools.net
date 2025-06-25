using System;

namespace CUETools.Ripper.Exceptions
{
    public class ReadCDException : Exception
    {
        public ReadCDException(string exceptionMessage) : base(exceptionMessage) { }
        public ReadCDException(string exceptionMessage, Exception innerException) : base(exceptionMessage, innerException) { }
    }
}
