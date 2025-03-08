using System;

namespace Bwg.Scsi
{
    internal interface ISysDev : IDisposable
    {
        int LastError { get; }

        void Close();
        bool Control(uint code, IntPtr inbuf, uint insize, IntPtr outbuf, uint outsize, ref uint ret, IntPtr overlapped);
        bool Open(string name);
        bool Open(char letter);
        string ErrorCodeToString(int error);
    }
}
