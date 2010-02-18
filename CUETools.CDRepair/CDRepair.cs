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
			crc = 0xffffffff;
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

		public uint CRC
		{
			get
			{
				return crc ^ 0xffffffff;
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
		protected bool hasErrors = false, canRecover = true;
		protected int actualOffset = 0;

		internal int[,] sigma;
		internal int[,] omega;
		internal int[,] errpos;
		internal int[,] erroff;
		internal int[] errors;

		public CDRepairEncode(int finalSampleCount, int stride, int npar, bool verify)
		    : base (finalSampleCount, stride, npar)
		{
			this.verify = verify;
			parity = new byte[stride * npar * 2];
			if (verify)
			{
				syndrome = new ushort[stride, npar];
				leadin = new ushort[stride * 2];
				leadout = new ushort[stride + laststride];
			} else
				syndrome = new ushort[1, npar];
		}

		new public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			int firstPos = Math.Max(0, stride - sampleCount * 2);
			int lastPos = Math.Min(sampleBuffer.ByteLength >> 1, (finalSampleCount - sampleCount) * 2 - laststride);

			fixed (byte* bytes = sampleBuffer.Bytes, par = parity)
			fixed (int* gx = encodeGx)
			fixed (uint* t = crc32.table)
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl, synptr = syndrome)
			{
				ushort* data = (ushort*)bytes;

				if (verify)
					for (int pos = 0; pos < (sampleBuffer.ByteLength >> 1); pos++)
					{
						ushort dd = data[pos];
						if (sampleCount * 2 + pos < 2 * stride)
							leadin[sampleCount * 2 + pos] = dd;
						int remaining = (finalSampleCount - sampleCount) * 2 - pos - 1;
						if (remaining < stride + laststride)
							leadout[remaining] = dd;
					}

				if (npar == 8)
				{
					for (int pos = firstPos; pos < lastPos; pos++)
					{
						int part = (sampleCount * 2 + pos) % stride;
						ushort* wr = ((ushort*)par) + part * 8;
						ushort dd = data[pos];

						crc = (crc >> 8) ^ t[(byte)(crc ^ dd)];
						crc = (crc >> 8) ^ t[(byte)(crc ^ (dd >> 8))];

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

						// syn[i] += data[pos] * α^(pos*i)
						if (verify && dd != 0)
						{
							ushort* syn = synptr + part * 8;
							ushort* myexp = exp + log[dd];
							int offs = stridecount - (sampleCount * 2 + pos) / stride;
							syn[0] ^= dd;
							syn[1] ^= myexp[offs];
							syn[2] ^= myexp[(offs * 2) % 65535];
							syn[3] ^= myexp[(offs * 3) % 65535];
							syn[4] ^= myexp[(offs * 4) % 65535];
							syn[5] ^= myexp[(offs * 5) % 65535];
							syn[6] ^= myexp[(offs * 6) % 65535];
							syn[7] ^= myexp[(offs * 7) % 65535];
							//ushort logdd = log[dd];
							//syn[1] ^= exp[(logdd + offs) % 65535];
							//syn[2] ^= exp[(logdd + offs * 2) % 65535];
							//syn[3] ^= exp[(logdd + offs * 3) % 65535];
							//syn[4] ^= exp[(logdd + offs * 4) % 65535];
							//syn[5] ^= exp[(logdd + offs * 5) % 65535];
							//syn[6] ^= exp[(logdd + offs * 6) % 65535];
							//syn[7] ^= exp[(logdd + offs * 7) % 65535];
						}
					}
				}
				else
				{
					for (int pos = firstPos; pos < lastPos; pos++)
					{
						int part = (sampleCount * 2 + pos) % stride;
						ushort* wr = ((ushort*)par) + part * npar;
						ushort dd = data[pos];

						crc = (crc >> 8) ^ t[(byte)(crc ^ dd)];
						crc = (crc >> 8) ^ t[(byte)(crc ^ (dd >> 8))];

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
			}
			sampleCount += sampleBuffer.Length;
		}

		public unsafe bool VerifyParity(byte[] parity2)
		{
			return VerifyParity(parity2, 0, parity2.Length);
		}

		public unsafe bool VerifyParity(byte[] parity2, int pos, int len)
		{
			if (!verify)
				throw new Exception("verify was not enabled");
			if (sampleCount != finalSampleCount)
				throw new Exception("sampleCount != finalSampleCount");
			if (len != stride * npar * 2)
				throw new Exception("wrong size");

			sigma = new int[stride, npar / 2 + 2];
			omega = new int[stride, npar / 2 + 1];
			errpos = new int[stride, npar / 2];
			erroff = new int[stride, npar / 2];
			errors = new int[stride];

			actualOffset = 0;

			// find offset
			fixed (byte* par2ptr = &parity2[pos])
			{
				ushort* par2 = (ushort*)par2ptr;
				int* syn = stackalloc int[npar];
				int* _sigma = stackalloc int[npar];
				int* _omega = stackalloc int[npar];
				int* _errpos = stackalloc int[npar];
				int bestErrors = npar;

				// We can only use offset if Abs(offset * 2) < stride,
				// else we might need to add/remove more than one sample
				// from syndrome calculations, and that would be too difficult
				// and will probably require longer leadin/leadout.
				for (int offset = 1 - stride / 2; offset < stride / 2; offset++)
				{
					int err = 0;

					for (int i = 0; i < npar; i++)
					{
						int part = (stride - 1) % stride;
						int part2 = (part + offset * 2 + stride) % stride;
						ushort* wr = par2 + part2 * npar;

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

						for (int j = 0; j < npar; j++)
							syn[i] = wr[j] ^ galois.mulExp(syn[i], i);

						err |= syn[i];
					}
					if (err == 0)
					{
						actualOffset = offset;
						bestErrors = 0;
						break;
					}
					int err_count = rs.calcSigmaMBM(_sigma, _omega, syn);
					if (err_count > 0 && rs.chienSearch(_errpos, stridecount + npar, err_count, _sigma))
					{
						if (err_count < bestErrors)
						{
							actualOffset = offset;
							bestErrors = err_count;
						}
					}
				}
			}

			hasErrors = false;
			fixed (byte* par = &parity2[pos])
			fixed (ushort* exp = galois.ExpTbl, log = galois.LogTbl)
			{
				int* syn = stackalloc int[npar];
				int offset = actualOffset;

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
						hasErrors = true;
						fixed (int* s = &sigma[part, 0], o = &omega[part, 0], e = &errpos[part, 0], f = &erroff[part, 0])
						{
							errors[part] = rs.calcSigmaMBM(s, o, syn);
							if (errors[part] <= 0 || !rs.chienSearch(e, stridecount + npar, errors[part], s))
								canRecover = false;
							else
							{
								for (int i = 0; i < errors[part]; i++)
									f[i] = galois.toPos(stridecount + npar, e[i]);
							}
						}
					}
					else
						errors[part] = 0;
				}
			}

			return !hasErrors;
		}

		public byte[] Parity
		{
			get
			{
				return parity;
			}
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

		public int ActualOffset
		{
			get
			{
				return actualOffset;
			}
		}
	}

	public class CDRepairFix : CDRepair
	{
		CDRepairEncode decode;

		public CDRepairFix(CDRepairEncode decode)
			: base(decode)
		{
			this.decode = decode;
		}

		new public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			if ((sampleBuffer.ByteLength & 1) != 0)
				throw new Exception("never happens");

			int firstPos = Math.Max(0, stride - sampleCount * 2 - decode.ActualOffset * 2);
			int lastPos = Math.Min(sampleBuffer.ByteLength >> 1, (finalSampleCount - sampleCount) * 2 - laststride - decode.ActualOffset * 2);

			fixed (byte* bytes = sampleBuffer.Bytes)
			fixed (uint* t = crc32.table)
			{
				ushort* data = (ushort*)bytes;
				for (int pos = firstPos; pos < lastPos; pos++)
				{
					int part = (sampleCount * 2 + pos) % stride;
					int errors = decode.errors[part];
					fixed (int* s = &decode.sigma[part, 0], o = &decode.omega[part, 0], f = &decode.erroff[part, 0])
						for (int i = 0; i < errors; i++)
							if (f[i] == (sampleCount * 2 + decode.ActualOffset * 2 + pos) / stride - 1)
								data[pos] ^= (ushort)rs.doForney(errors, decode.errpos[part, i], s, o);

					ushort dd = data[pos];

					crc = (crc >> 8) ^ t[(byte)(crc ^ dd)];
					crc = (crc >> 8) ^ t[(byte)(crc ^ (dd >> 8))];
				}
			}
			sampleCount += sampleBuffer.Length;
		}
	}
}
