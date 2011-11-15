using System;
using System.Collections.Generic;
using System.Text;
using CUETools.Parity;
using System.IO;
using System.Net;

namespace CUETools.CTDB.Converter
{
    public class ReadDB
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

    class Program
    {
        const int stride = 588 * 10 * 2;

        private static byte[] Unparse(ushort[,] syndrome, int version)
        {
            if (version == 1)
            {
                return ParityToSyndrome.Syndrome2Bytes(syndrome);
            }

            var output = ParityToSyndrome.Syndrome2Parity(syndrome);
            var newcontents = new MemoryStream();
            using (DBHDR FTYP = new DBHDR(newcontents, "ftyp"))
                FTYP.Write("CTDB");
            using (DBHDR CTDB = new DBHDR(newcontents, "CTDB"))
            {
                using (DBHDR HEAD = CTDB.HDR("HEAD"))
                {
                    using (DBHDR VERS = HEAD.HDR("VERS")) VERS.Write(0x101);
                }
                using (DBHDR DISC = CTDB.HDR("DISC"))
                {
                    using (DBHDR CONF = DISC.HDR("CONF")) CONF.Write(1);
                    using (DBHDR NPAR = DISC.HDR("NPAR")) NPAR.Write(8);
                    using (DBHDR CRC_ = DISC.HDR("CRC ")) CRC_.Write(0);
                    using (DBHDR PAR_ = DISC.HDR("PAR ")) PAR_.Write(output);
                }
            }
            return newcontents.ToArray();
        }

        private static ushort[,] Parse(byte[] contents, int version)
        {
            if (version == 2)
            {
                int npar = contents.Length / stride / 2;
                if (npar < 8 || npar > 32 || contents.Length != npar * stride * 2)
                    throw new Exception("invalid parity length");
                return ParityToSyndrome.Bytes2Syndrome(stride, 8, contents);
            }

            if (contents.Length < 8 * stride * 2
                || contents.Length > 8 * stride * 4)
                throw new Exception("invalid length");

            ReadDB rdr = new ReadDB(contents);

            int end;
            string hdr = rdr.ReadHDR(out end);
            uint magic = rdr.ReadUInt();
            if (hdr != "ftyp" || magic != 0x43544442 || end != rdr.pos)
                throw new Exception("invalid CTDB file");
            hdr = rdr.ReadHDR(out end);
            if (hdr != "CTDB" || end != contents.Length)
                throw new Exception("invalid CTDB file");
            hdr = rdr.ReadHDR(out end);
            if (hdr != "HEAD")
                throw new Exception("invalid CTDB file");
            int endHead = end;
            while (rdr.pos < endHead)
            {
                hdr = rdr.ReadHDR(out end);
                rdr.pos = end;
            }
            rdr.pos = endHead;
            while (rdr.pos < contents.Length)
            {
                hdr = rdr.ReadHDR(out end);
                if (hdr != "DISC")
                {
                    rdr.pos = end;
                    continue;
                }
                int endDisc = end;
                int parPos = 0, parLen = 0;
                while (rdr.pos < endDisc)
                {
                    hdr = rdr.ReadHDR(out end);
                    if (hdr == "PAR ")
                    {
                        parPos = rdr.pos;
                        parLen = end - rdr.pos;
                    }
                    rdr.pos = end;
                }
                if (parPos != 0)
                {
                    if (parLen != 8 * stride * 2)
                        throw new Exception("invalid parity length");
                    return ParityToSyndrome.Parity2Syndrome(stride, stride, 8, 8, contents, parPos);
                }
            }
            throw new Exception("invalid CTDB file");
        }

        static byte[] Fetch(string url)
        {
            var req = WebRequest.Create(url);
            var resp = (HttpWebResponse)req.GetResponse();
            if (resp.StatusCode != HttpStatusCode.OK)
                throw new Exception(resp.ToString());
            return new BinaryReader(resp.GetResponseStream()).ReadBytes((int)resp.ContentLength);
        }

        static void Main(string[] args)
        {
            if (args.Length == 2 && (0 == string.Compare(args[0], "upconvert", true) || 0 == string.Compare(args[0], "downconvert", true)))
            {
                int id = int.Parse(args[1]);
                var version = 0 == string.Compare(args[0], "upconvert", true) ? 1 : 2;
                var contents = Fetch("http://p.cuetools.net/" + id.ToString());
                var syndrome = Parse(contents, version);
                var stdout = System.Console.OpenStandardOutput();
                var output = Unparse(syndrome, version);
                stdout.Write(output, 0, output.Length);
            }
            else if (args.Length == 2 && 0 == string.Compare(args[0], "p2s", true))
            {
                var p = ParityToSyndrome.Parity2Syndrome(1, 1, 8, 8, Convert.FromBase64String(args[1]));
                var output = ParityToSyndrome.Syndrome2Bytes(p);
                System.Console.Write(Convert.ToBase64String(output));
            }
            else
                throw new Exception("Usage: upconvert <id> | downconvert <id> | p2s <parity>");
        }
    }
}
