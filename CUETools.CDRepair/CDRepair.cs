using System;
using System.Collections.Generic;
using System.Text;
using CUETools.Codecs;
using CUETools.Parity;

namespace CUETools.CDRepair
{
	public class CDRepair : IAudioDest
	{
		protected int sampleCount;
		protected int finalSampleCount;
		protected Galois galois;
		protected RsDecode rs;
		protected Crc32 crc32;
		protected uint crc;
		protected int[] encodeGx;
		protected int stride;
		protected int laststride;
		protected int stridecount;
		protected int npar;

		public CDRepair(int finalSampleCount, int stride, int npar)
		{
			this.npar = npar;
			this.stride = stride;
			this.finalSampleCount = finalSampleCount;
			sampleCount = 0;
			galois = Galois16.instance;
			rs = new RsDecode16(npar, galois);
			crc32 = new Crc32();
			//crc = 0xffffffff;
			encodeGx = galois.makeEncodeGxLog(npar);
			laststride = stride + (finalSampleCount * 2) % stride;
			stridecount = (finalSampleCount * 2) / stride - 2; // minus one for leadin and one for leadout
			if ((finalSampleCount * 2 + stride - 1) / stride + npar > galois.Max)
				throw new Exception("invalid stride");
		}

		public CDRepair(CDRepair src)
			: this(src.finalSampleCount, src.stride, src.npar)
		{
		}

		public unsafe void Write(AudioBuffer sampleBuffer)
		{
			throw new Exception("unsupported");
		}

		public unsafe void Close()
		{
			if (sampleCount != finalSampleCount)
				throw new Exception("sampleCount != finalSampleCount");
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

		public string Options
		{
			set
			{
				if (value == null || value == "") return;
				throw new Exception("Unsupported options " + value);
			}
		}

		public AudioPCMConfig PCM
		{
			get { return AudioPCMConfig.RedBook; }
		}

		public long FinalSampleCount
		{
			set
			{
				if (value < 0) // != _toc.Length?
					throw new Exception("invalid FinalSampleCount");
				finalSampleCount = (int)value;
			}
		}
		public long BlockSize
		{
			set { throw new Exception("unsupported"); }
		}

		public string Path
		{
			get { throw new Exception("unsupported"); }
		}

		public int NPAR
		{
			get
			{
				return npar;
			}
		}

		public uint CRC
		{
			get
			{
				return crc32.Combine(0xffffffff, crc, stride * 2 * stridecount) ^ 0xffffffff;
			}
		}
	}

	public class CDRepairEncode : CDRepair
	{
		protected byte[] parity;
		protected ushort[,] syndrome;
		protected ushort[] leadin;
		protected ushort[] leadout;
		protected bool verify;
		protected bool encode;
		protected uint crcA, crcB;

		public CDRepairEncode(int finalSampleCount, int stride, int npar, bool verify, bool encode)
		    : base (finalSampleCount, stride, npar)
		{
			this.verify = verify;
			this.encode = encode;
			parity = new byte[stride * npar * 2];
			if (verify)
			{
				syndrome = new ushort[stride, npar];
				leadin = new ushort[stride * 2];
				leadout = new ushort[stride + laststride];
			} else
				syndrome = new ushort[1, npar];
		}

		private unsafe void ProcessStride(int currentStride, int currentPart, int count, ushort* data)
		{
			fixed (uint* crct = crc32.table)
			fixed (byte* bpar = parity)
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl, synptr = syndrome)
			fixed (int* gx = encodeGx)
				for (int pos = 0; pos < count; pos++)
				{
					ushort* par = (ushort*)bpar;
					int part = currentPart + pos;
					ushort* wr = ((ushort*)par) + part * npar;
					ushort dd = data[pos];

					crc = (crc >> 8) ^ crct[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ crct[(byte)(crc ^ (dd >> 8))];

					if (verify)
					{
						ushort* syn = synptr + part * npar;
						syn[0] ^= dd; // wk += data
						for (int i = 1; i < npar; i++)
							syn[i] = (ushort)(dd ^ galois.mulExp(syn[i], i)); // wk = data + wk * α^i
					}

					int ib = wr[0] ^ dd;
					if (ib != 0)
					{
						ushort* myexp = exp + log[ib];
						for (int i = 0; i < npar - 1; i++)
							wr[i] = (ushort)(wr[i + 1] ^ myexp[gx[i]]);
						wr[npar - 1] = myexp[gx[npar - 1]];
					}
					else
					{
						for (int i = 0; i < npar - 1; i++)
							wr[i] = wr[i + 1];
						wr[npar - 1] = 0;
					}
				}
		}

		private unsafe void ProcessStride8(int currentStride, int currentPart, int count, ushort* data)
		{
			fixed (uint* crct = crc32.table)
			fixed (byte* bpar = parity)
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl, synptr = syndrome)
				for (int pos = 0; pos < count; pos++)
				{
					ushort* par = (ushort*)bpar;
					int part = currentPart + pos;
					ushort* wr = par + part * 8;
					ushort dd = data[pos];

					crc = (crc >> 8) ^ crct[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ crct[(byte)(crc ^ (dd >> 8))];

					if (encode)
					{
						int ib = wr[0] ^ dd;
						if (ib != 0)
						{
							ushort* myexp = exp + log[ib];
							wr[0] = (ushort)(wr[1] ^ myexp[19483]);
							wr[1] = (ushort)(wr[2] ^ myexp[41576]);
							wr[2] = (ushort)(wr[3] ^ myexp[9460]);
							wr[3] = (ushort)(wr[4] ^ myexp[52075]);
							wr[4] = (ushort)(wr[5] ^ myexp[9467]);
							wr[5] = (ushort)(wr[6] ^ myexp[41590]);
							wr[6] = (ushort)(wr[7] ^ myexp[19504]);
							wr[7] = myexp[28];
						}
						else
						{
							wr[0] = wr[1];
							wr[1] = wr[2];
							wr[2] = wr[3];
							wr[3] = wr[4];
							wr[4] = wr[5];
							wr[5] = wr[6];
							wr[6] = wr[7];
							wr[7] = 0;
						}
					}

					// syn[i] += data[pos] * α^(n*i)
					if (verify && dd != 0)
					{
						int n = stridecount - currentStride;
						ushort* syn = synptr + part * 8;
						syn[0] ^= dd;
						int idx = log[dd];
						idx += n; syn[1] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[2] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[3] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[4] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[5] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[6] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[7] ^= exp[(idx & 0xffff) + (idx >> 16)];
					}
				}
		}

		private unsafe void ProcessStride16(int currentStride, int currentPart, int count, ushort* data)
		{
			fixed (uint* crct = crc32.table)
			fixed (byte* bpar = parity)
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl, synptr = syndrome)
				for (int pos = 0; pos < count; pos++)
				{
					ushort* par = (ushort*)bpar;
					int part = currentPart + pos;
					ushort* wr = par + part * 16;
					ushort dd = data[pos];

					crc = (crc >> 8) ^ crct[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ crct[(byte)(crc ^ (dd >> 8))];

					int ib = wr[0] ^ dd;
					if (ib != 0)
					{
						ushort* myexp = exp + log[ib];
						wr[0] = (ushort)(wr[1] ^ myexp[0x000059f1]);
						wr[1] = (ushort)(wr[2] ^ myexp[0x0000608f]);
						wr[2] = (ushort)(wr[3] ^ myexp[0x0000918b]);
						wr[3] = (ushort)(wr[4] ^ myexp[0x00004487]);
						wr[4] = (ushort)(wr[5] ^ myexp[0x0000a151]);
						wr[5] = (ushort)(wr[6] ^ myexp[0x0000c074]);
						wr[6] = (ushort)(wr[7] ^ myexp[0x00004178]);
						wr[7] = (ushort)(wr[8] ^ myexp[0x00004730]);
						wr[8] = (ushort)(wr[9] ^ myexp[0x00004187]);
						wr[9] = (ushort)(wr[10] ^ myexp[0x0000c092]);
						wr[10] = (ushort)(wr[11] ^ myexp[0x0000a17e]);
						wr[11] = (ushort)(wr[12] ^ myexp[0x000044c3]);
						wr[12] = (ushort)(wr[13] ^ myexp[0x000091d6]);
						wr[13] = (ushort)(wr[14] ^ myexp[0x000060e9]);
						wr[14] = (ushort)(wr[15] ^ myexp[0x00005a5a]);
						wr[15] = myexp[0x00000078];
					}
					else
					{
						wr[0] = wr[1];
						wr[1] = wr[2];
						wr[2] = wr[3];
						wr[3] = wr[4];
						wr[4] = wr[5];
						wr[5] = wr[6];
						wr[6] = wr[7];
						wr[7] = wr[8];
						wr[8] = wr[9];
						wr[9] = wr[10];
						wr[10] = wr[11];
						wr[11] = wr[12];
						wr[12] = wr[13];
						wr[13] = wr[14];
						wr[14] = wr[15];
						wr[15] = 0;
					}

					// syn[i] += data[pos] * α^(n*i)
					if (verify && dd != 0)
					{
						int n = stridecount - currentStride;
						ushort* syn = synptr + part * 16;
						syn[0] ^= dd;
						int idx = log[dd];
						idx += n; syn[1] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[2] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[3] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[4] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[5] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[6] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[7] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[8] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[9] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[10] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[11] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[12] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[13] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[14] ^= exp[(idx & 0xffff) + (idx >> 16)];
						idx += n; syn[15] ^= exp[(idx & 0xffff) + (idx >> 16)];
					}
				}
		}

		new public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			fixed (byte* bytes = sampleBuffer.Bytes)
			{
				int offs = 0;
				while (offs < sampleBuffer.Length)
				{
					int currentPart = (sampleCount * 2) % stride;
					int currentStride = (sampleCount * 2) / stride;
					// Process no more than there is in the buffer, and no more than up to a stride boundary.
					int copyCount = Math.Min((sampleBuffer.Length - offs) * 2, stride - currentPart);
					ushort* data = ((ushort*)bytes) + offs * 2;

					if (verify)
					{
						// remember CRC after leadin
						if (sampleCount * 2 == stride * 2)
							crcA = crc;

						// remember CRC before leadout
						if ((finalSampleCount - sampleCount) * 2 == stride + laststride)
							crcB = crc;

						if (currentStride < 2)
							for (int pos = 0; pos < copyCount; pos++)
								leadin[sampleCount * 2 + pos] = data[pos];

						if (currentStride >= stridecount)
							for (int pos = 0; pos < copyCount; pos++)
							{
								int remaining = (finalSampleCount - sampleCount) * 2 - pos - 1;
								if (remaining < stride + laststride)
									leadout[remaining] = data[pos];
							}
					}

					if (currentStride >= 1 && currentStride <= stridecount)
					{
						if (npar == 8)
							ProcessStride8(currentStride, currentPart, copyCount, data);
						else if (npar == 16)
							ProcessStride16(currentStride, currentPart, copyCount, data);
						else
							ProcessStride(currentStride, currentPart, copyCount, data);
					}

					sampleCount += copyCount >> 1;
					offs += copyCount >> 1;
				}
			}
		}

		public unsafe CDRepairFix VerifyParity(byte[] parity2, int actualOffset)
		{
			return VerifyParity(npar, parity2, 0, parity2.Length, actualOffset);
		}

		private unsafe uint OffsettedCRC(int actualOffset)
		{
			fixed (uint* crct = crc32.table)
			{
				// calculate leadin CRC
				uint crc0 = 0;
				for (int off = stride - 2 * actualOffset; off < 2 * stride; off++)
				{
					ushort dd = leadin[off];
					crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ dd)];
					crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ (dd >> 8))];
				}
				// calculate leadout CRC
				uint crc2 = 0;
				for (int off = laststride + stride - 1; off >= laststride + 2 * actualOffset; off--)
				{
					ushort dd = leadout[off];
					crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ dd)];
					crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ (dd >> 8))];
				}

				// calculate middle CRC
				uint crc1 = crc32.Combine(crcA, crcB, (stridecount - 2) * stride * 2);
				// calculate offsettedCRC as sum of 0xffffffff, crc0, crc1, crc2;
				return crc32.Combine(
					0xffffffff,
					crc32.Combine(
						crc32.Combine(
							crc0,
							crc1,
							(stridecount - 2) * stride * 2),
						crc2,
						(stride - 2 * actualOffset) * 2),
					stridecount * stride * 2) ^ 0xffffffff;
			}
		}

		public unsafe bool FindOffset(int npar2, byte[] parity2, int pos, uint expectedCRC, out int actualOffset, out bool hasErrors)
		{
			if (npar2 != npar)
				throw new Exception("npar mismatch");
			if (!verify)
				throw new Exception("verify was not enabled");
			if (sampleCount != finalSampleCount)
				throw new Exception("sampleCount != finalSampleCount");

			// find offset
			fixed (byte* par2ptr = &parity2[pos])
			{
				ushort* par2 = (ushort*)par2ptr;
				int* _sigma = stackalloc int[npar];
				int* _errpos = stackalloc int[npar];
				bool foundOffset = false;

				for (int allowed_errors = 0; allowed_errors < npar / 2 && !foundOffset; allowed_errors++)
				{
					int part2 = 0;
					ushort* wr = par2 + part2 * npar;

					// We can only use offset if Abs(offset * 2) < stride,
					// else we might need to add/remove more than one sample
					// from syndrome calculations, and that would be too difficult
					// and will probably require longer leadin/leadout.
					for (int offset = 1 - stride / 2; offset < stride / 2; offset++)
					{
						int err = 0;
						int part = (part2 + stride - offset * 2) % stride;
						int* syn = stackalloc int[npar];

						for (int i = 0; i < npar; i++)
						{
							int synI = syndrome[part, i];

							// offset < 0
							if (part < -offset * 2)
							{
								synI ^= galois.mulExp(leadin[stride + part], (i * (stridecount - 1)) % galois.Max);
								synI = leadout[laststride - part - 1] ^ galois.mulExp(synI, i);
							}
							// offset > 0 
							if (part >= stride - offset * 2)
							{
								synI = galois.divExp(synI ^ leadout[laststride + stride - part - 1], i);
								synI ^= galois.mulExp(leadin[part], (i * (stridecount - 1)) % galois.Max);
							}

							for (int j = 0; j < npar; j++)
								synI = wr[j] ^ galois.mulExp(synI, i);

							syn[i] = synI;
							err |= synI;
						}
						int err_count = err == 0 ? 0 : rs.calcSigmaMBM(_sigma, syn);
						if (err_count == allowed_errors && (err_count == 0 || rs.chienSearch(_errpos, stridecount + npar, err_count, _sigma)))
						{
							actualOffset = offset;
							hasErrors = err_count != 0 || OffsettedCRC(offset) != expectedCRC;
							return true;
						}
					}
				}
			}
			actualOffset = 0;
			hasErrors = true;
			return false;
		}

		public unsafe CDRepairFix VerifyParity(int npar2, byte[] parity2, int pos, int len, int actualOffset)
		{
			if (len != stride * npar * 2)
				throw new Exception("wrong size");

			CDRepairFix fix = new CDRepairFix(this);
			fix.actualOffset = actualOffset;
			fix.correctableErrors = 0;
			fix.hasErrors = false;
			fix.canRecover = true;

			fix.sigma = new int[stride, npar / 2 + 2];
			fix.omega = new int[stride, npar / 2 + 1];
			fix.errpos = new int[stride, npar / 2];
			fix.erroff = new int[stride, npar / 2];
			fix.errors = new int[stride];

			fixed (byte* par = &parity2[pos])
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl)
			{
				int* syn = stackalloc int[npar];
				int offset = fix.actualOffset;

				for (int part = 0; part < stride; part++)
				{
					int part2 = (part + offset * 2 + stride) % stride;
					ushort* wr = (ushort*)par + part2 * npar;
					int err = 0;

					for (int i = 0; i < npar; i++)
					{
						syn[i] = syndrome[part, i];

						// offset < 0
						if (part < -offset * 2)
						{
							syn[i] ^= galois.mulExp(leadin[stride + part], (i * (stridecount - 1)) % galois.Max);
							syn[i] = leadout[laststride - part - 1] ^ galois.mulExp(syn[i], i);
						}
						// offset > 0 
						if (part >= stride - offset * 2)
						{
							syn[i] = galois.divExp(syn[i] ^ leadout[laststride + stride - part - 1], i);
							syn[i] ^= galois.mulExp(leadin[part], (i * (stridecount - 1)) % galois.Max);
						}

						//syn[i] = galois.mulExp(syn[i], i * npar);

						for (int j = 0; j < npar; j++)
							syn[i] = wr[j] ^ galois.mulExp(syn[i], i); // wk = data + wk * α^i

						err |= syn[i];
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
						fixed (int* s = &fix.sigma[part, 0], o = &fix.omega[part, 0], e = &fix.errpos[part, 0], f = &fix.erroff[part, 0])
						{
							fix.errors[part] = rs.calcSigmaMBM(s, syn);
							fix.hasErrors = true;
							fix.correctableErrors += fix.errors[part];
							if (fix.errors[part] <= 0 || !rs.chienSearch(e, stridecount + npar, fix.errors[part], s))
								fix.canRecover = false;
							else
							{
								galois.mulPoly(o, s, syn, npar / 2 + 1, npar, npar);
								for (int i = 0; i < fix.errors[part]; i++)
									f[i] = galois.toPos(stridecount + npar, e[i]);
							}
						}
					}
					else
						fix.errors[part] = 0;
				}
			}

			return fix;
		}

		public byte[] Parity
		{
			get
			{
				return parity;
			}
		}
	}

	public class CDRepairFix : CDRepair
	{
		internal bool hasErrors = false, canRecover = true;
		internal int actualOffset = 0;
		internal int correctableErrors = 0;
		internal int[,] sigma;
		internal int[,] omega;
		internal int[,] errpos;
		internal int[,] erroff;
		internal int[] errors;

		public CDRepairFix(CDRepairEncode decode)
			: base(decode)
		{
		}

		new public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			int firstPos = Math.Max(0, stride - sampleCount * 2 - ActualOffset * 2);
			int lastPos = Math.Min(sampleBuffer.ByteLength >> 1, (finalSampleCount - sampleCount) * 2 - laststride - ActualOffset * 2);

			fixed (byte* bytes = sampleBuffer.Bytes)
			fixed (uint* t = crc32.table)
			{
				ushort* data = (ushort*)bytes;
				for (int pos = firstPos; pos < lastPos; pos++)
				{
					int part = (sampleCount * 2 + pos) % stride;
					int nerrors = errors[part];
					fixed (int* s = &sigma[part, 0], o = &omega[part, 0], f = &erroff[part, 0])
						for (int i = 0; i < nerrors; i++)
							if (f[i] == (sampleCount * 2 + ActualOffset * 2 + pos) / stride - 1)
								data[pos] ^= (ushort)rs.doForney(nerrors, errpos[part, i], s, o);

					ushort dd = data[pos];

					crc = (crc >> 8) ^ t[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ t[(byte)(crc ^ (dd >> 8))];
				}
			}
			sampleCount += sampleBuffer.Length;
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
	}
}
