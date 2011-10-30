using System.Net;
using CUETools.AccurateRip;
using CUETools.CDImage;

namespace CUETools.CTDB
{
    public class DBEntry
    {
        public ushort[,] syndrome;
        public int conf;
        public int stride;
        public int offset;
        public uint crc;
        public bool hasErrors;
        public bool canRecover;
        public CDRepairFix repair;
        public HttpStatusCode httpStatus;
        public long id;
        public CDImageLayout toc;
        public string hasParity;

        public DBEntry(ushort[,] syndrome, int conf, int stride, uint crc, long id, CDImageLayout toc, string hasParity)
        {
            this.syndrome = syndrome;
            this.id = id;
            this.conf = conf;
            this.crc = crc;
            this.stride = stride;
            this.toc = toc;
            this.hasParity = hasParity;
        }

        public int Npar
        {
            get
            {
                return syndrome.GetLength(1);
            }
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
