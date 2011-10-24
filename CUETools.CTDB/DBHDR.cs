using System;
using System.IO;
using System.Text;

namespace CUETools.CTDB
{
    class DBHDR : IDisposable
    {
        private long lenOffs;
        private MemoryStream stream;

        public DBHDR(MemoryStream stream, string name)
        {
            this.stream = stream;
            lenOffs = stream.Position;
            Write(0);
            Write(name);
        }

        public void Dispose()
        {
            long fin = stream.Position;
            stream.Position = lenOffs;
            Write((int)(fin - lenOffs));
            stream.Position = fin;
        }

        public void Write(int value)
        {
            byte[] temp = new byte[4];
            temp[3] = (byte)(value & 0xff);
            temp[2] = (byte)((value >> 8) & 0xff);
            temp[1] = (byte)((value >> 16) & 0xff);
            temp[0] = (byte)((value >> 24) & 0xff);
            Write(temp);
        }

        public void Write(uint value)
        {
            byte[] temp = new byte[4];
            temp[3] = (byte)(value & 0xff);
            temp[2] = (byte)((value >> 8) & 0xff);
            temp[1] = (byte)((value >> 16) & 0xff);
            temp[0] = (byte)((value >> 24) & 0xff);
            Write(temp);
        }

        public void Write(string value)
        {
            Write(Encoding.UTF8.GetBytes(value));
        }

        public void Write(byte[] value)
        {
            stream.Write(value, 0, value.Length);
        }

        public DBHDR HDR(string name)
        {
            return new DBHDR(stream, name);
        }
    }
}
