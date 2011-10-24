using System.Net;
using CUETools.AccurateRip;
using CUETools.CDImage;

namespace CUETools.CTDB
{
    public class DBEntry
    {
        public byte[] parity;
        public int pos;
        public int len;
        public int conf;
        public int npar;
        public int stride;
        public int offset;
        public uint crc;
        public bool hasErrors;
        public bool canRecover;
        public CDRepairFix repair;
        public HttpStatusCode httpStatus;
        public string id;
        public CDImageLayout toc;
        public string hasParity;

        public DBEntry(byte[] parity, int pos, int len, int conf, int npar, int stride, uint crc, string id, CDImageLayout toc, string hasParity)
        {
            this.parity = parity;
            this.id = id;
            this.pos = pos;
            this.len = len;
            this.conf = conf;
            this.crc = crc;
            this.npar = npar;
            this.stride = stride;
            this.toc = toc;
            this.hasParity = hasParity;
        }

        public string Status
        {
            get
            {
                if (!hasErrors)
                {
                    return string.Format("verified OK, confidence {0}", conf);
                }
                if (canRecover)
                {
                    return string.Format("differs in {1} samples, confidence {0}", conf, repair.CorrectableErrors);
                }
                if (httpStatus == HttpStatusCode.OK)
                {
                    return "could not be verified";
                }
                return "could not be verified: " + httpStatus.ToString();
            }
        }
    }
}
