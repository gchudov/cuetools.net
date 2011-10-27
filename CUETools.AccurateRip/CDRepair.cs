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
		protected int finalSampleCount;
		internal Galois galois;
		internal int stride;
		internal int laststride;
		internal int stridecount;

		public CDRepair(int finalSampleCount, int stride)
		{			
			this.stride = stride;
			this.finalSampleCount = finalSampleCount;
			sampleCount = 0;
			galois = Galois16.instance;
			laststride = stride + (finalSampleCount * 2) % stride;
			stridecount = (finalSampleCount * 2) / stride - 2; // minus one for leadin and one for leadout
			if ((finalSampleCount * 2 + stride - 1) / stride + AccurateRipVerify.maxNpar > galois.Max)
				throw new Exception("invalid stride");
		}

		public CDRepair(CDRepair src)
			: this(src.finalSampleCount, src.stride)
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
		    : base ((int)ar.FinalSampleCount, stride)
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

		public string TrackCRCs
		{
			get
			{
				var sb = new StringBuilder();
				for (int i = 1; i <= ar.TOC.AudioTracks; i++)
					sb.AppendFormat(" {0:x8}", ar.CTDBCRC(i, 0, stride / 2, laststride / 2));
				return sb.ToString().Substring(1);
			}
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
            int part2 = 0;
            // find offset
            fixed (ushort* chT = rs.chienTable, syn2part = &syn2[part2, 0])
            {
                int* _sigma = stackalloc int[npar];
                int* _errpos = stackalloc int[npar];
                int* syn = stackalloc int[npar];
                bool foundOffset = false;
                var arSyndrome = ar.GetSyndrome(npar);

                for (int allowed_errors = 0; allowed_errors < npar / 2 && !foundOffset; allowed_errors++)
                {
                    // We can only use offset if Abs(offset * 2) < stride,
                    // else we might need to add/remove more than one sample
                    // from syndrome calculations, and that would be too difficult
                    // and will probably require longer leadin/leadout.
                    for (int offset = 1 - stride / 2; offset < stride / 2; offset++)
                    {
                        int err = 0;
                        int part = (part2 + stride - offset * 2) % stride;

                        for (int i = 0; i < npar; i++)
                        {
                            int synI = arSyndrome[part, i];

                            // offset < 0
                            if (part < -offset * 2)
                            {
                                synI ^= galois.mulExp(ar.leadin[stride + part], (i * (stridecount - 1)) % galois.Max);
                                synI = ar.leadout[laststride - part - 1] ^ galois.mulExp(synI, i);
                            }
                            // offset > 0 
                            if (part >= stride - offset * 2)
                            {
                                synI = galois.divExp(synI ^ ar.leadout[laststride + stride - part - 1], i);
                                synI ^= galois.mulExp(ar.leadin[part], (i * (stridecount - 1)) % galois.Max);
                            }

                            synI = galois.mulExp(synI ^ syn2part[i], i * npar);
                            syn[i] = synI;
                            err |= synI;
                        }
                        int err_count = err == 0 ? 0 : rs.calcSigmaMBM(_sigma, syn);
                        if (err_count == allowed_errors && (err_count == 0 || rs.chienSearch(_errpos, stridecount + npar, err_count, _sigma, chT)))
                        {
                            actualOffset = offset;
                            hasErrors = err_count != 0 || ar.CTDBCRC(-offset) != expectedCRC;
                            return true;
                        }
                    }
                }
            }
            actualOffset = 0;
            hasErrors = true;
            return false;
        }

		public unsafe CDRepairFix VerifyParity(ushort[,] syn2, int actualOffset)
		{
            int npar2 = syn2.GetLength(1);
            int npar = Math.Min(AccurateRipVerify.maxNpar, npar2);

			CDRepairFix fix = new CDRepairFix(this, npar);
			fix.actualOffset = actualOffset;
			fix.correctableErrors = 0;
			fix.hasErrors = false;
			fix.canRecover = true;

			fix.sigma = new int[stride, npar / 2 + 2];
			fix.omega = new int[stride, npar / 2 + 1];
			fix.errpos = new int[stride, npar / 2];
			//fix.erroff = new int[stride, npar / 2];
			fix.errors = new int[stride];

            var syn1 = ar.GetSyndrome(npar);
            var rs = new RsDecode16(npar, this.galois);

			//fixed (byte* par = &parity2[pos])
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl, chT = rs.chienTable, psyn2 = syn2, psyn1 = syn1)
			fixed (int* sf = fix.sigma, of = fix.omega, ef = fix.errpos)
			{
                int sfLen = fix.sigma.GetLength(1);
                int ofLen = fix.omega.GetLength(1);
                int efLen = fix.errpos.GetLength(1);
                int* syn = stackalloc int[npar];
				int offset = fix.actualOffset;

				for (int part = 0; part < stride; part++)
				{
					int part2 = (part + offset * 2 + stride) % stride;
					ushort* syn1part = psyn1 + part * npar;
                    ushort* syn2part = psyn2 + part2 * npar;
					int err = 0;

					for (int i = 0; i < npar; i++)
					{
						int synI = syn1part[i];

						// offset < 0
						if (part < -offset * 2)
						{
							synI ^= galois.mulExp(ar.leadin[stride + part], (i * (stridecount - 1)) % galois.Max);
							synI = ar.leadout[laststride - part - 1] ^ galois.mulExp(synI, i);
						}
						// offset > 0 
						if (part >= stride - offset * 2)
						{
                            synI = galois.divExp(synI ^ ar.leadout[laststride + stride - part - 1], i);
                            synI ^= galois.mulExp(ar.leadin[part], (i * (stridecount - 1)) % galois.Max);
						}

                        synI = galois.mulExp(synI ^ syn2part[i], i * npar);
                        syn[i] = synI;
                        err |= synI;
					}

					//for (int j = 0; j < npar; j++)
					//    if (wr[j] != 0)
					//    {
					//        ushort* myexp = exp + log[wr[j]];
					//        syn[0] ^= wr[j];
					//        for (int i = 1; i < npar; i++)
					//            syn[i] ^= myexp[(npar - j - 1) * i];
					//    }

					//for (int i = 0; i < npar; i++)
					//    err |= syn[i];

					if (err != 0)
					{
                        int* s = sf + part * sfLen;
                        int* o = of + part * ofLen;
                        int* e = ef + part * efLen;
						//fixed (int* s = &fix.sigma[part, 0], o = &fix.omega[part, 0], e = &fix.errpos[part, 0])
						{
							fix.errors[part] = rs.calcSigmaMBM(s, syn);
							fix.hasErrors = true;
							fix.correctableErrors += fix.errors[part];
							if (fix.errors[part] <= 0 || !rs.chienSearch(e, stridecount + npar, fix.errors[part], s, chT))
								fix.canRecover = false;
							else
                                galois.mulPoly(o, s, syn, ofLen, sfLen, npar);
						}
					}
					else
						fix.errors[part] = 0;
				}
			}

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
		internal int[,] sigma;
		internal int[,] omega;
		internal int[,] errpos;
		internal int[] erroffsorted;
		internal ushort[] forneysorted;
		internal int erroffcount;
		internal int[] errors;
		private BitArray affectedSectorArray;
		private int nexterroff;
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
				StringBuilder sb = new StringBuilder();
				SortErrors();
				for (int i = 0; i < erroffcount; i++)
				{
					int j;
					for (j = i + 1; j < erroffcount; j++)
						if (erroffsorted[j] - erroffsorted[j - 1] > 2 * 588 * 5)
							break;
					uint sec1 = (uint)erroffsorted[i] / 2 / 588;
					uint sec2 = (uint)erroffsorted[j - 1] / 2 / 588;
					if (sb.Length != 0) sb.Append(",");
					sb.Append(CDImageLayout.TimeToString(sec1));
					if (sec1 != sec2) sb.Append("-");
					if (sec1 != sec2) sb.Append(CDImageLayout.TimeToString(sec2));
					i = j - 1;
				}
				return sb.ToString();
			}
		}

		public BitArray AffectedSectorArray
		{
			get
			{
				if (affectedSectorArray == null)
				{
					affectedSectorArray = new BitArray(finalSampleCount / 588 + 1);
					SortErrors();
					for (int i = 0; i < erroffcount; i++)
						affectedSectorArray[erroffsorted[i] / 2 / 588] = true;
				}
				return affectedSectorArray;
			}
		}

		private int GetErrOff(int part, int i)
		{
			return (2 + galois.toPos(stridecount + npar, errpos[part, i]) - (stride + part + ActualOffset * 2) / stride) * stride + part;
		}

		private unsafe void SortErrors()
		{
			if (erroffsorted != null)
				return;
			erroffcount = 0;
			erroffsorted = new int[errpos.GetLength(0) * errpos.GetLength(1)];
			forneysorted = new ushort[errpos.GetLength(0) * errpos.GetLength(1)];
            for (int part = 0; part < stride; part++)
			{
				fixed (int* s = &sigma[part, 0], o = &omega[part, 0])
					for (int i = 0; i < errors[part]; i++)
					{
						erroffsorted[erroffcount] = GetErrOff(part, i);
						if (erroffsorted[erroffcount] >= 0 && erroffsorted[erroffcount] < finalSampleCount * 2)
						{
							forneysorted[erroffcount] = (ushort)this.galois.doForney(errors[part], errpos[part, i], s, o);
							erroffcount++;
						}
					}
			}
			Array.Sort<int, ushort>(erroffsorted, forneysorted, 0, erroffcount);
			// assert erroffcount == CorrectableErrors
			nexterroff = 0;
		}

		public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			int firstPos = Math.Max(0, stride - sampleCount * 2 - ActualOffset * 2);
			int lastPos = Math.Min(sampleBuffer.ByteLength >> 1, (finalSampleCount - sampleCount) * 2 - laststride - ActualOffset * 2);

			SortErrors();

			fixed (byte* bytes = sampleBuffer.Bytes)
			fixed (uint* t = Crc32.table)
			{
				ushort* data = (ushort*)bytes;
				for (int pos = firstPos; pos < lastPos; pos++)
				{
					if (sampleCount * 2 + pos == erroffsorted[nexterroff] && nexterroff < erroffsorted.Length)
						data[pos] ^= forneysorted[nexterroff++];
					
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

		public int CompressionLevel
		{
			get { return 0; }
			set { }
		}

		public object Settings
		{
			get
			{
				return null;
			}
			set
			{
				if (value != null && value.GetType() != typeof(object))
					throw new Exception("Unsupported options " + value);
			}
		}

		public long Padding
		{
			set { }
		}

		public AudioPCMConfig PCM
		{
			get { return AudioPCMConfig.RedBook; }
		}

		public long BlockSize
		{
			set { throw new Exception("unsupported"); }
		}

		public string Path
		{
			get { throw new Exception("unsupported"); }
		}
	}
}
