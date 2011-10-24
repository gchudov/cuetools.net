using System.Text;

namespace CUETools.CTDB
{
    class ReadDB
    {
        private byte[] contents;
        public int pos;

        public ReadDB(byte[] contents)
        {
            this.contents = contents;
            this.pos = 0;
        }

        public string ReadHDR(out int end)
        {
            int size = this.ReadInt();
            string res = Encoding.ASCII.GetString(contents, pos, 4);
            this.pos += 4;
            end = pos + size - 8;
            return res;
        }

        public int ReadInt()
        {
            int value =
                (this.contents[this.pos + 3] +
                (this.contents[this.pos + 2] << 8) +
                (this.contents[this.pos + 1] << 16) +
                (this.contents[this.pos + 0] << 24));
            this.pos += 4;
            return value;
        }

        public uint ReadUInt()
        {
            uint value =
                ((uint)this.contents[this.pos + 3] +
                ((uint)this.contents[this.pos + 2] << 8) +
                ((uint)this.contents[this.pos + 1] << 16) +
                ((uint)this.contents[this.pos + 0] << 24));
            this.pos += 4;
            return value;
        }
    }
}
