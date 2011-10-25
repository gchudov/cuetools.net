using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using CUETools.Codecs;

namespace CUETools.AccurateRip
{
    [Serializable]
    public class OffsetSafeCRCRecord
    {
        private uint[] val;

        public OffsetSafeCRCRecord()
        {
            this.val = new uint[1];
        }

        public OffsetSafeCRCRecord(AccurateRipVerify ar)
            : this(new uint[64 + 64])
        {
            int offset = 64 * 64;
            for (int i = 0; i < 64; i++)
                this.val[i] = ar.CTDBCRC(0, (i + 1) * 64, offset, 2 * offset);
            for (int i = 0; i < 64; i++)
                this.val[i + 64] = ar.CTDBCRC(0, 63 - i, offset, 2 * offset);
        }

        public OffsetSafeCRCRecord(uint[] val)
        {
            this.val = val;
        }

        [XmlIgnore]
        public uint[] Value
        {
            get
            {
                return val;
            }
        }

        public unsafe string Base64
        {
            get
            {
                byte[] res = new byte[val.Length * 4];
                fixed (byte* pres = &res[0])
                fixed (uint* psrc = &val[0])
                    AudioSamples.MemCpy(pres, (byte*)psrc, res.Length);
                var b64 = new char[res.Length * 2 + 4];
                int b64len = Convert.ToBase64CharArray(res, 0, res.Length, b64, 0);
                StringBuilder sb = new StringBuilder(b64len + b64len / 4 + 1);
                for (int i = 0; i < b64len; i += 64)
                {
                    sb.Append(b64, i, Math.Min(64, b64len - i));
                    sb.AppendLine();
                }
                return sb.ToString();
            }

            set
            {
                if (value == null)
                    throw new ArgumentNullException();
                byte[] bytes = Convert.FromBase64String(value);
                if (bytes.Length % 4 != 0)
                    throw new InvalidDataException();
                val = new uint[bytes.Length / 4];
                fixed (byte* pb = &bytes[0])
                fixed (uint* pv = &val[0])
                    AudioSamples.MemCpy((byte*)pv, pb, bytes.Length);
            }
        }

        public override bool Equals(object obj)
        {
            return obj is OffsetSafeCRCRecord && this == (OffsetSafeCRCRecord)obj;
        }

        public override int GetHashCode()
        {
            return (int)val[0];
        }

        public static bool operator ==(OffsetSafeCRCRecord x, OffsetSafeCRCRecord y)
        {
            if (x as object == null || y as object == null) return x as object == null && y as object == null;
            if (x.Value.Length != y.Value.Length) return false;
            for (int i = 0; i < x.Value.Length; i++)
                if (x.Value[i] != y.Value[i])
                    return false;
            return true;
        }

        public static bool operator !=(OffsetSafeCRCRecord x, OffsetSafeCRCRecord y)
        {
            return !(x == y);
        }

        public bool DifferByOffset(OffsetSafeCRCRecord rec)
        {
            int offset;
            return FindOffset(rec, out offset);
        }

        public bool FindOffset(OffsetSafeCRCRecord rec, out int offset)
        {
            if (this.Value.Length != 128 || rec.Value.Length != 128)
            {
                offset = 0;
                return false;
                //throw new InvalidDataException("Unsupported OffsetSafeCRCRecord");
            }

            for (int i = 0; i < 64; i++)
            {
                if (rec.Value[0] == Value[i])
                {
                    offset = i * 64;
                    return true;
                }
                if (Value[0] == rec.Value[i])
                {
                    offset = -i * 64;
                    return true;
                }
                for (int j = 0; j < 64; j++)
                {
                    if (rec.Value[i] == Value[64 + j])
                    {
                        offset = i * 64 + j + 1;
                        return true;
                    }
                    if (Value[i] == rec.Value[64 + j])
                    {
                        offset = -i * 64 - j - 1;
                        return true;
                    }
                }
            }
            offset = 0;
            return false;
        }
    }
}
