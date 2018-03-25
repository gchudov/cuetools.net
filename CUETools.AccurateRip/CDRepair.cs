using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Parity;
using CUETools.AccurateRip;

namespace CUETools.AccurateRip
{
	public class CDRepair
	{
		protected int sampleCount;
        protected int pregap;
		protected int finalSampleCount;
		internal Galois galois;
		internal int stride;
		internal int laststride;
		internal int stridecount;

		public CDRepair(int pregap, int finalSampleCount, int stride)
		{
			this.stride = stride;
            this.pregap = pregap;
            this.finalSampleCount = finalSampleCount;
			this.sampleCount = 0;
			this.galois = Galois16.instance;
            this.laststride = this.stride + ((this.finalSampleCount - this.pregap) * 2) % this.stride;
            this.stridecount = ((this.finalSampleCount - this.pregap) * 2) / this.stride - 2; // minus one for leadin and one for leadout
            if (((this.finalSampleCount - this.pregap) * 2 + this.stride - 1) / this.stride + AccurateRipVerify.maxNpar > galois.Max)
				throw new Exception("invalid stride");
		}

		public CDRepair(CDRepair src)
			: this(src.pregap, src.finalSampleCount, src.stride)
		{
		}

		public long FinalSampleCount
		{
			get
			{
				return finalSampleCount;
			}
			set
			{
				if (value < 0) // != _toc.Length?
					throw new Exception("invalid FinalSampleCount");
				finalSampleCount = (int)value;
			}
		}

		public int Stride
		{
			get
			{
				return stride;
			}
		}   
    }

	public class CDRepairEncode : CDRepair
	{
		protected AccurateRipVerify ar;

		public CDRepairEncode(AccurateRipVerify ar, int stride)
		    : base ((int)ar.TOC.Pregap * 588, (int)ar.FinalSampleCount, stride)
		{
			this.ar = ar;
			ar.InitCDRepair(stride, laststride, stridecount, true);
		}

		public AccurateRipVerify AR
		{
			get
			{
				return ar;
			}
		}

		public uint CRC
		{
			get
			{
				return ar.CTDBCRC(0);
			}
		}

        public string GetTrackCRCs(int oi)
        {
            var sb = new StringBuilder();
            for (int i = 1; i <= ar.TOC.AudioTracks; i++)
                sb.AppendFormat(" {0:x8}", this.TrackCRC(i, oi));
            return sb.ToString().Substring(1);
        }

        public uint TrackCRC(int iTrack, int oi)
        {
            return this.ar.CTDBCRC(iTrack, oi, this.stride / 2, this.laststride / 2);
        }

        public unsafe bool FindOffset(ushort[,] syn2, uint expectedCRC, out int actualOffset, out bool hasErrors)
        {
            int npar2 = syn2.GetLength(1);
            int npar = Math.Min(AccurateRipVerify.maxNpar, npar2);

            if (npar2 != npar)
                throw new Exception("npar mismatch");
            if (ar.Position != ar.FinalSampleCount)
                throw new Exception("ar.Position != ar.FinalSampleCount");

            var rs = new RsDecode16(npar, this.galois);
            fixed (ushort* syn2part = syn2)
            {
                int* _sigma = stackalloc int[npar];
                int* _errpos = stackalloc int[npar];
                int* syn = stackalloc int[npar];
                int bestOffset = 0;
                int bestOffsetErrors = npar / 2;

                // fast search
                for (int offset = 1 - stride / 2; offset < stride / 2; offset++)
                {
                    var syn1 = ar.GetSyndrome(npar, 1, -offset);
                    int err = 0;
                    for (int i = 0; i < npar; i++)
                    {
                        int synI = syn1[0, i] ^ syn2part[i];
                        syn[i] = synI;
                        err |= synI;
                    }
                    if (err == 0)
                    {
                        actualOffset = offset;
                        hasErrors = ar.CTDBCRC(-offset) != expectedCRC;
                        return true;
                    }
                    int err_count = rs.calcSigmaMBM(_sigma, syn);
                    if (err_count > 0 && err_count < bestOffsetErrors && rs.chienSearch(_errpos, stridecount, err_count, _sigma))
                    {
                        bestOffset = offset;
                        bestOffsetErrors = err_count;
                    }
                }

                if (bestOffsetErrors < npar / 2)
                {
                    actualOffset = bestOffset;
                    hasErrors = true;
                    return true;
                }
            }
            actualOffset = 0;
            hasErrors = true;
            return false;
        }

        public unsafe CDRepairFix VerifyParity(ushort[,] syn2, uint crc, int actualOffset)
		{
            int npar2 = syn2.GetLength(1);
            int npar = Math.Min(AccurateRipVerify.maxNpar, npar2);
            var erroff = new int[stride * npar / 2];
            var forney = new ushort[stride * npar / 2];
            var syn1 = ar.GetSyndrome(npar, -1, -actualOffset);
            var rs = new RsDecode16(npar, this.galois);
            CDRepairFix fix = new CDRepairFix(this, npar);

            fix.actualOffset = actualOffset;
            fix.correctableErrors = 0;
            fix.hasErrors = false;
            fix.canRecover = true;

			fixed (ushort *psyn2 = syn2, psyn1 = syn1)
			{
                int sfLen = npar / 2 + 2;
                int ofLen = npar / 2 + 1;
                int efLen = npar / 2;
                int* _sigma = stackalloc int[npar / 2 + 2];
                int* _omega = stackalloc int[npar / 2 + 1];
                int* _errpos = stackalloc int[npar / 2];
                int* syn = stackalloc int[npar];
				int offset = fix.actualOffset;

				for (int part2 = 0; part2 < stride; part2++)
				{
					ushort* syn1part = psyn1 + part2 * npar;
                    ushort* syn2part = psyn2 + part2 * npar;
					int err = 0;

					for (int i = 0; i < npar; i++)
					{
                        var synI = syn1part[i] ^ syn2part[i];
                        syn[i] = synI;
                        err |= synI;
					}

                    if (err != 0)
                    {
                        int errcount = rs.calcSigmaMBM(_sigma, syn);
                        fix.hasErrors = true;
                        if (errcount <= 0 || errcount > efLen || !rs.chienSearch(_errpos, stridecount, errcount, _sigma))
                        {
                            fix.canRecover = false;
                            return fix;
                        }

                        galois.mulPoly(_omega, _sigma, syn, ofLen, sfLen, npar);

                        for (int i = 0; i < errcount; i++)
                        {
                            int pos = galois.toPos(stridecount, _errpos[i]) * stride + part2;
                            int erroffi = stride + pos + pregap * 2 - actualOffset * 2;
                            ushort diff = (ushort)this.galois.doForney(errcount, _errpos[i], _sigma, _omega);
                            if (erroffi < pregap * 2 || erroffi >= finalSampleCount * 2)
                            {
                                fix.canRecover = false;
                                return fix;
                            }
                            crc ^= Crc32.Combine(Crc32.ComputeChecksum(Crc32.ComputeChecksum(0, (byte)diff), (byte)(diff >> 8)), 0, (stridecount * stride - pos - 1) * 2);
                            erroff[fix.correctableErrors] = erroffi;
                            forney[fix.correctableErrors] = diff;
                            fix.correctableErrors++;
                        }
                    }
				}

                crc ^= ar.CTDBCRC(-actualOffset);
                if (crc != 0)
                {
                    fix.canRecover = false;
                    return fix;
                }
			}

            fix.erroffsorted = new int[fix.correctableErrors];
            fix.forneysorted = new ushort[fix.correctableErrors];
            for (int i = 0; i < fix.correctableErrors; i++)
            {
                fix.erroffsorted[i] = erroff[i];
                fix.forneysorted[i] = forney[i];
            }
            Array.Sort<int, ushort>(fix.erroffsorted, fix.forneysorted, 0, fix.correctableErrors);
            return fix;
		}

		public string OffsetSafeCRC
		{
			get
			{
				return ar.OffsetSafeCRC.Base64;
			}
		}
	}

	public class CDRepairFix : CDRepair, IAudioDest
	{
		internal bool hasErrors = false, canRecover = true;
		internal int actualOffset = 0;
		internal int correctableErrors = 0;
        internal int[] erroffsorted;
		internal ushort[] forneysorted;
		private BitArray affectedSectorArray;
		private int nexterroff = 0;
        private int npar;
		uint crc = 0;

		internal CDRepairFix(CDRepairEncode decode, int npar)
			: base(decode)
		{
            this.npar = npar;
        }

		public string AffectedSectors
		{
			get
			{
                return this.GetAffectedSectors(0, finalSampleCount);
			}
		}

		public BitArray AffectedSectorArray
		{
			get
			{
				if (affectedSectorArray == null)
				{
					affectedSectorArray = new BitArray(finalSampleCount / 588 + 1);
                    for (int i = 0; i < correctableErrors; i++)
						affectedSectorArray[erroffsorted[i] / 2 / 588] = true;
				}
				return affectedSectorArray;
			}
		}

        public int GetAffectedSectorsCount(int min, int max)
        {
            min = Math.Max(2 * min, 2 * pregap + stride - 2 * ActualOffset);
            max = Math.Min(2 * max, 2 * finalSampleCount - laststride - 2 * ActualOffset);
            int count = 0;
            for (int i = 0; i < correctableErrors; i++)
                if (erroffsorted[i] >= min && erroffsorted[i] < max)
                    count++;
            return count;
        }

        public string GetAffectedSectors(int min, int max, int offs = 0, int coalesce = 2 * 588 * 5)
        {
            min = Math.Max(2 * min, 2 * pregap + stride - 2 * ActualOffset);
            max = Math.Min(2 * max, 2 * finalSampleCount - laststride - 2 * ActualOffset);
            offs = offs * 2;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < correctableErrors; i++)
                if (erroffsorted[i] >= min && erroffsorted[i] < max)
                {
                    int j;
                    for (j = i + 1; j < correctableErrors; j++)
                        if (erroffsorted[j] - erroffsorted[j - 1] >= coalesce)
                            break;
                    uint sec1 = (uint)(erroffsorted[i] - offs) / 2 / 588;
                    uint sec2 = (uint)(erroffsorted[j - 1] - offs) / 2 / 588;
                    if (sb.Length != 0) sb.Append(",");
                    sb.Append(CDImageLayout.TimeToString(sec1));
                    if (sec1 != sec2) sb.Append("-");
                    if (sec1 != sec2) sb.Append(CDImageLayout.TimeToString(sec2));
                    i = j - 1;
                }
            return sb.ToString();
        }

		public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			int firstPos = Math.Max(0, stride + (pregap - sampleCount) * 2 - ActualOffset * 2);
			int lastPos = Math.Min(sampleBuffer.ByteLength >> 1, (finalSampleCount - sampleCount) * 2 - laststride - ActualOffset * 2);

			fixed (byte* bytes = sampleBuffer.Bytes)
			fixed (uint* t = Crc32.table)
			{
				ushort* data = (ushort*)bytes;
				for (int pos = firstPos; pos < lastPos; pos++)
				{
                    if (nexterroff < erroffsorted.Length && sampleCount * 2 + pos == erroffsorted[nexterroff])
                    {
                        data[pos] ^= forneysorted[nexterroff++];
                        // When we modify sampleBuffer.Bytes, which might have been
                        // copied from sampleBuffer.Samples, which wasn't modified,
                        // we need to make sure sampleBuffer.Samples will be reset;
                        // This strange call makes sure of that.
                        sampleBuffer.Prepare(sampleBuffer.Bytes, sampleBuffer.Length);
                    }
					
					ushort dd = data[pos];
					crc = (crc >> 8) ^ t[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ t[(byte)(crc ^ (dd >> 8))];
				}
			}
			sampleCount += sampleBuffer.Length;
		}

		public unsafe void Close()
		{
			if (sampleCount != finalSampleCount)
				throw new Exception("sampleCount != finalSampleCount");
		}

		public bool HasErrors
		{
			get
			{
				return hasErrors;
			}
		}

		public bool CanRecover
		{
			get
			{
				return canRecover;
			}
		}

		public int CorrectableErrors
		{
			get
			{
				return correctableErrors;
			}
		}

		public int ActualOffset
		{
			get
			{
				return actualOffset;
			}
		}

		public uint CRC
		{
			get
			{
				return 0xffffffff ^ Crc32.Combine(0xffffffff, crc, stride * stridecount * 2);
			}
		}

		public void Delete()
		{
			throw new Exception("unsupported");
		}

        public IAudioEncoderSettings Settings => new Codecs.WAV.EncoderSettings(AudioPCMConfig.RedBook);

        public string Path
		{
			get { throw new Exception("unsupported"); }
		}
	}
}
