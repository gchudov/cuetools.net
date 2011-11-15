using System;
using System.Globalization;
using System.Net;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Parity;

namespace CUETools.CTDB
{
    public class DBEntry
    {
        public ushort[,] syndrome;
        public uint[] trackcrcs;
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

        public DBEntry(CTDBResponseEntry ctdbRespEntry)
        {
            this.syndrome = ctdbRespEntry.syndrome == null
                ? ParityToSyndrome.Parity2Syndrome(1, 1, 8, 8, Convert.FromBase64String(ctdbRespEntry.parity))
                : ParityToSyndrome.Bytes2Syndrome(1, Math.Min(AccurateRipVerify.maxNpar, ctdbRespEntry.npar), Convert.FromBase64String(ctdbRespEntry.syndrome));
            this.conf = ctdbRespEntry.confidence;
            this.stride = ctdbRespEntry.stride * 2;
            this.crc = uint.Parse(ctdbRespEntry.crc32, NumberStyles.HexNumber);
            this.id = ctdbRespEntry.id;
            this.toc = CDImageLayout.FromString(ctdbRespEntry.toc);
            this.hasParity = ctdbRespEntry.hasparity;
            if (ctdbRespEntry.trackcrcs != null)
            {
                var crcs = ctdbRespEntry.trackcrcs.Split(' ');
                if (crcs.Length == this.toc.AudioTracks)
                {
                    this.trackcrcs = new uint[crcs.Length];
                    for (int i = 0; i < this.trackcrcs.Length; i++)
                    {
                        this.trackcrcs[i] = uint.Parse(crcs[i], NumberStyles.HexNumber);
                    }
                }
            }
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
