using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Xml.Serialization;
using CUETools.Parity;
using CUETools.CDImage;
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
				this.val[i] = ar.GetMiddleCRC32(offset + i * 64 + 64, 2 * offset - 64 - i * 64);
			for (int i = 0; i < 64; i++)
				this.val[i + 64] = ar.GetMiddleCRC32(offset + 64 - 1 - i, 2 * offset - 64 + 1 + i);
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
				byte[] bytes  = Convert.FromBase64String(value);
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

		public static bool operator == (OffsetSafeCRCRecord x, OffsetSafeCRCRecord y)
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

		public bool FindOffset (OffsetSafeCRCRecord rec, out int offset)
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

	public class AccurateRipVerify : IAudioDest
	{
		public AccurateRipVerify(CDImageLayout toc, IWebProxy proxy)
		{
			this.proxy = proxy;
			_accDisks = new List<AccDisk>();
			_hasLogCRC = false;
			_CRCLOG = new uint[toc.AudioTracks + 1];
			ExceptionStatus = WebExceptionStatus.Pending;
			Init(toc);
		}

		public uint Confidence(int iTrack)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			int trno = iTrack + _toc.FirstAudio - 1;
			for (int di = 0; di < (int)AccDisks.Count; di++)
				if (trno < AccDisks[di].tracks.Count 
					&& (CRC(iTrack) == AccDisks[di].tracks[trno].CRC
					  || CRCV2(iTrack) == AccDisks[di].tracks[trno].CRC))
					conf += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
			return conf;
		}

		public uint WorstTotal()
		{
			uint worstTotal = 0xffff;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint sumTotal = Total(iTrack);
				if (worstTotal > sumTotal && sumTotal != 0)
					worstTotal = sumTotal;
			}
			return worstTotal == 0xffff ? 0 : worstTotal;
		}

		// TODO: Replace min(sum) with sum(min)!!!
		public uint WorstConfidence()
		{
			uint worstConfidence = 0xffff;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint sumConfidence = SumConfidence(iTrack);
				if (worstConfidence > sumConfidence && (Total(iTrack) != 0 || CRC(iTrack) != 0))
					worstConfidence = sumConfidence;
			}
			return worstConfidence;
		}

		public uint SumConfidence(int iTrack)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			int trno = iTrack + _toc.FirstAudio - 1;
			for (int di = 0; di < AccDisks.Count; di++)
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					if (trno < AccDisks[di].tracks.Count
					&& (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC
					  || oi == 0 && CRCV2(iTrack) == AccDisks[di].tracks[trno].CRC))
						conf += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
			return conf;
		}

		public uint Confidence(int iTrack, int oi)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			int trno = iTrack + _toc.FirstAudio - 1;
			for (int di = 0; di < (int)AccDisks.Count; di++)
				if (trno < AccDisks[di].tracks.Count
					&& (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC
					  || oi == 0 && CRCV2(iTrack) == AccDisks[di].tracks[trno].CRC))
					conf += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
			return conf;
		}

		public uint Total(int iTrack)
		{
			if (ARStatus != null)
				return 0U;
			uint total = 0;
			for (int di = 0; di < (int)AccDisks.Count; di++)
				if (iTrack + _toc.FirstAudio - 1 < AccDisks[di].tracks.Count)
					total += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
			return total;
		}

		public uint DBCRC(int iTrack)
		{
			return ARStatus == null && iTrack +_toc.FirstAudio - 1 < AccDisks[0].tracks.Count 
				? AccDisks[0].tracks[iTrack + _toc.FirstAudio - 1].CRC : 0U;
		}

		public uint CRC(int iTrack)
		{
			return CRC(iTrack, 0);
		}

		public uint CRCV2(int iTrack)
		{
			int offs0 = iTrack == 0 ? 5 * 588 - 1 : 0;
			int offs1 = iTrack == _toc.AudioTracks - 1 ? 20 * 588 - 5 * 588 : 0;
			uint crcA1 = _CRCAR[iTrack + 1, offs1] - (offs0 > 0 ? _CRCAR[iTrack + 1, offs0] : 0);
			uint crcA2 = _CRCV2[iTrack + 1, offs1] - (offs0 > 0 ? _CRCV2[iTrack + 1, offs0] : 0);
			return crcA1 + crcA2;
		}

		public uint CRC(int iTrack, int oi)
		{
			int offs0 = iTrack == 0 ? 5 * 588 + oi - 1 : oi;
			int offs1 = iTrack == _toc.AudioTracks - 1 ? 20 * 588 - 5 * 588 + oi : (oi >= 0 ? 0 : 20 * 588 + oi);
			uint crcA = _CRCAR[iTrack + 1, offs1] - (offs0 > 0 ? _CRCAR[iTrack + 1, offs0] : 0);
			uint sumA = _CRCSM[iTrack + 1, offs1] - (offs0 > 0 ? _CRCSM[iTrack + 1, offs0] : 0);
			uint crc = crcA - sumA * (uint)oi;
			if (oi < 0 && iTrack > 0)
			{
				uint crcB = _CRCAR[iTrack, 0] - _CRCAR[iTrack, 20 * 588 + oi];
				uint sumB = _CRCSM[iTrack, 0] - _CRCSM[iTrack, 20 * 588 + oi];
				uint posB = _toc[iTrack + _toc.FirstAudio - 1].Length * 588 + (uint)oi;
				crc += crcB - sumB * posB;
			}
			if (oi > 0 && iTrack < _toc.AudioTracks - 1)
			{
				uint crcB = _CRCAR[iTrack + 2, oi];
				uint sumB = _CRCSM[iTrack + 2, oi];
				uint posB = _toc[iTrack + _toc.FirstAudio].Length * 588 + (uint)-oi;
				crc += crcB + sumB * posB;
			}
			return crc;
		}

		public uint CRC450(int iTrack, int oi)
		{
			uint crca = _CRCAR[iTrack + 1, 20 * 588 + 5 * 588 + oi];
			uint crcb = _CRCAR[iTrack + 1, 20 * 588 + 6 * 588 + oi];
			uint suma = _CRCSM[iTrack + 1, 20 * 588 + 5 * 588 + oi];
			uint sumb = _CRCSM[iTrack + 1, 20 * 588 + 6 * 588 + oi];
			uint offs = 450 * 588 + (uint)oi;
			return crcb - crca - offs * (sumb - suma);
		}

		public int PeakLevel()
		{
			int peak = 0;
			for (int track = 0; track <= _toc.AudioTracks; track++)
				if (peak < _Peak[track])
					peak = _Peak[track];
			return peak;
		}

		public int PeakLevel(int iTrack)
		{
			return _Peak[iTrack];
		}

		internal uint GetMiddleCRC32(int prefixLen, int suffixLen)
		{
			return CTDBCRC(prefixLen * 2, suffixLen * 2);
		}

		public OffsetSafeCRCRecord OffsetSafeCRC
		{
			get
			{
				return new OffsetSafeCRCRecord(this);
			}
		}

		public uint CRC32(int iTrack)
		{
			return CRC32(iTrack, 0);
		}

		public uint CRC32(int iTrack, int oi)
		{
			if (_CacheCRC32[iTrack, _arOffsetRange + oi] == 0)
			{
				uint crc = 0;
				if (iTrack == 0)
				{
					for (iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
					{
						int trackLength = (int)(iTrack > 0 ? _toc[iTrack + _toc.FirstAudio - 1].Length : _toc[_toc.FirstAudio].Pregap) * 588 * 4;
						if (oi < 0 && iTrack == 0)
							crc = Crc32.Combine(crc, 0, -oi * 4);
						if (trackLength == 0)
							continue;
						if (oi > 0 && (iTrack == 0 || (iTrack == 1 && _toc[_toc.FirstAudio].Pregap == 0)))
						{
							// Calculate track CRC skipping first oi samples by 'subtracting' their CRC
							crc = Crc32.Combine(_CRC32[iTrack, oi], _CRC32[iTrack, 0], trackLength - oi * 4);
						}
						else if (oi < 0 && iTrack == _toc.AudioTracks)
						{
							crc = Crc32.Combine(crc, _CRC32[iTrack, 20 * 588 + oi], trackLength + oi * 4);
						}
						else
						{
							crc = Crc32.Combine(crc, _CRC32[iTrack, 0], trackLength);
						}
						if (oi > 0 && iTrack == _toc.AudioTracks)
							crc = Crc32.Combine(crc, 0, oi * 4);
					}
					iTrack = 0;
					// Use 0xffffffff as an initial state
					crc ^= _CRCMASK[0];
				}
				else
				{
					int trackLength = (int)(iTrack > 0 ? _toc[iTrack + _toc.FirstAudio - 1].Length : _toc[_toc.FirstAudio].Pregap) * 588 * 4;
					if (oi > 0)
					{
						// Calculate track CRC skipping first oi samples by 'subtracting' their CRC
						crc = Crc32.Combine(_CRC32[iTrack, oi], _CRC32[iTrack, 0], trackLength - oi * 4);
						// Add oi samples from next track CRC
						crc = Crc32.Combine(crc, 0, oi * 4);
						if (iTrack < _toc.AudioTracks)
							crc ^= _CRC32[iTrack + 1, oi];
					}
					else if (oi < 0)
					{
						// Calculate CRC of previous track's last oi samples by 'subtracting' it's last CRCs
						crc = Crc32.Combine(_CRC32[iTrack - 1, 20 * 588 + oi], _CRC32[iTrack - 1, 0], -oi * 4);
						// Add this track's CRC without last oi samples
						crc = Crc32.Combine(crc, _CRC32[iTrack, 20 * 588 + oi], trackLength + oi * 4);
					}
					else // oi == 0
					{
						crc = _CRC32[iTrack, 0];
					}
					// Use 0xffffffff as an initial state
					crc ^= _CRCMASK[iTrack];
				}
				_CacheCRC32[iTrack, _arOffsetRange + oi] = crc;
			}
			return _CacheCRC32[iTrack, _arOffsetRange + oi];
		}

		public uint CRCWONULL(int iTrack)
		{
			return CRCWONULL(iTrack, 0);
		}

		public uint CRCWONULL(int iTrack, int oi)
		{
			if (_CacheCRCWN[iTrack, _arOffsetRange + oi] == 0)
			{
				uint crc = 0xffffffff;
				if (iTrack == 0)
				{
					for (iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
					{
						int trackLength = (int)(iTrack > 0 ? _toc[iTrack + _toc.FirstAudio - 1].Length : _toc[_toc.FirstAudio].Pregap) * 588 * 4
							- _CRCNL[iTrack, 0] * 2;
						crc = Crc32.Combine(crc, _CRCWN[iTrack, 0], trackLength);
					}
					iTrack = 0;
				}
				else
				{
					int trackLength = (int)(iTrack > 0 ? _toc[iTrack + _toc.FirstAudio - 1].Length : _toc[_toc.FirstAudio].Pregap) * 588 * 4;
					if (oi > 0)
					{
						int nonzeroPrevLength = trackLength - oi * 4 -
							(_CRCNL[iTrack, 0] - _CRCNL[iTrack, oi]) * 2;
						// Calculate track CRC skipping first oi samples by 'subtracting' their CRC
						crc = Crc32.Combine(
							_CRCWN[iTrack, oi],
							_CRCWN[iTrack, 0],
							nonzeroPrevLength);
						// Use 0xffffffff as an initial state
						crc = Crc32.Combine(0xffffffff, crc, nonzeroPrevLength);
						// Add oi samples from next track CRC
						if (iTrack < _toc.AudioTracks)
							crc = Crc32.Combine(crc,
								_CRCWN[iTrack + 1, oi],
								oi * 4 - _CRCNL[iTrack + 1, oi] * 2);
					}
					else if (oi < 0)
					{
						int nonzeroPrevLength = -oi * 4 -
							(_CRCNL[iTrack - 1, 0] - _CRCNL[iTrack - 1, 20 * 588 + oi]) * 2;
						// Calculate CRC of previous track's last oi samples by 'subtracting' it's last CRCs
						crc = Crc32.Combine(
							_CRCWN[iTrack - 1, 20 * 588 + oi],
							_CRCWN[iTrack - 1, 0],
							nonzeroPrevLength);
						// Use 0xffffffff as an initial state
						crc = Crc32.Combine(0xffffffff, crc, nonzeroPrevLength);
						// Add this track's CRC without last oi samples
						crc = Crc32.Combine(crc,
							_CRCWN[iTrack, 20 * 588 + oi],
							trackLength + oi * 4 - _CRCNL[iTrack, 20 * 588 + oi] * 2);
					}
					else // oi == 0
					{
						// Use 0xffffffff as an initial state
						crc = Crc32.Combine(0xffffffff, _CRCWN[iTrack, 0], trackLength - _CRCNL[iTrack, 0] * 2);
					}
				}
				_CacheCRCWN[iTrack, _arOffsetRange + oi] = crc ^ 0xffffffff;
			}
			return _CacheCRCWN[iTrack, _arOffsetRange + oi];
		}

		public uint CRCLOG(int iTrack)
		{
			return _CRCLOG[iTrack];
		}

		public void CRCLOG(int iTrack, uint value)
		{
			_hasLogCRC = true;
			_CRCLOG[iTrack] = value;
		}

		internal ushort[,] syndrome;
		internal byte[] parity;
		internal ushort[] leadin;
		internal ushort[] leadout;
		private uint[] leadinCrc;
		private uint[] leadoutCrc;
		private uint preLeadoutCrc;
		private int stride = 1, laststride = 1, stridecount = 1, npar = 1;
		private bool calcSyn = false;
		private bool calcParity = false;

		internal void InitCDRepair(int stride, int laststride, int stridecount, int npar, bool calcSyn, bool calcParity)
		{
			if (npar != 8)
				throw new NotSupportedException("npar != 8");
			this.stride = stride;
			this.laststride = laststride;
			this.stridecount = stridecount;
			this.npar = npar;
			this.calcSyn = calcSyn;
			this.calcParity = calcParity;
			Init(_toc);
		}

		public unsafe uint CTDBCRC(int prefixLen, int suffixLen)
		{
			if (prefixLen > leadin.Length || suffixLen > leadout.Length)
				throw new ArgumentOutOfRangeException();
			// stride - 2 * actualOffset
			// laststride + 2 * actualOffset
			int lenAXB = (int)_toc.AudioLength * 588 * 4;

			if (leadinCrc == null)
			{
				leadinCrc = new uint[leadin.Length + 1];
				leadoutCrc = new uint[leadout.Length + 1];

				fixed (uint* crct = Crc32.table)
				{
					// calculate leadin CRC
					uint crc0 = 0;
					leadinCrc[0] = crc0;
					for (int off = 0; off < leadin.Length; off++)
					{
						ushort dd = leadin[off];
						crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ dd)];
						crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ (dd >> 8))];
						leadinCrc[off + 1] = crc0;
					}
					// calculate leadout CRC
					uint crc2 = 0;
					leadoutCrc[0] = crc2;
					for (int off = leadout.Length - 1; off >= 0; off--)
					{
						ushort dd = leadout[off];
						crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ dd)];
						crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ (dd >> 8))];
						leadoutCrc[leadout.Length - off] = crc2;
					}
				}
				preLeadoutCrc = CRC32(0, 0) ^ _CRCMASK[0];
				preLeadoutCrc = Crc32.Substract(preLeadoutCrc, leadoutCrc[leadout.Length], leadout.Length * 2);
			}

			uint crcXE = Crc32.Combine(leadinCrc[prefixLen], preLeadoutCrc, lenAXB - prefixLen * 2 - leadout.Length * 2);
			uint crcX = Crc32.Combine(crcXE, leadoutCrc[leadout.Length - suffixLen], (leadout.Length - suffixLen) * 2);
			return Crc32.Combine(0xffffffff, crcX, lenAXB - prefixLen * 2 - suffixLen * 2) ^ 0xffffffff;
		}

		public uint CTDBCRC(int actualOffset)
		{
			return CTDBCRC(stride - 2 * actualOffset, laststride + 2 * actualOffset);
		}

		private unsafe static void CalcSyn8(ushort* exp, ushort* log, ushort* syn, uint lo, uint n, int npar)
		{
			syn[0] ^= (ushort)lo;
			uint idx = log[lo] + n; syn[1] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[2] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[3] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[4] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[5] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[6] ^= exp[(idx & 0xffff) + (idx >> 16)];
			idx += n; syn[7] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//for (int i = 8; i < npar; i += 8)
			//{
			//    idx += n; syn[i] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 1] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 2] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 3] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 4] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 5] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 6] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//    idx += n; syn[i + 7] ^= exp[(idx & 0xffff) + (idx >> 16)];
			//}
		}

#if alternateSynCalc
		private unsafe static void CalcSyn8Alt(ushort* exp, ushort* log, ushort* syn, uint lo, uint n)
		{
			uint idx = log[lo] + 7 * n;
			ulong x = exp[(idx & 0xffff) + (idx >> 16)];
			x <<= 16; idx -= n; x ^= exp[(idx & 0xffff) + (idx >> 16)];
			x <<= 16; idx -= n; x ^= exp[(idx & 0xffff) + (idx >> 16)];
			x <<= 16; idx -= n; x ^= exp[(idx & 0xffff) + (idx >> 16)];
			((ulong*)syn)[1] ^= x;
			idx -= n; x = exp[(idx & 0xffff) + (idx >> 16)];
			x <<= 16; idx -= n; x ^= exp[(idx & 0xffff) + (idx >> 16)];
			x <<= 16; idx -= n; x ^= exp[(idx & 0xffff) + (idx >> 16)];
			((ulong*)syn)[0] ^= (x << 16) + lo;
		}
#endif

		private unsafe static void CalcPar8(ushort* exp, ushort* log, ushort* wr, uint lo)
		{
			uint ib = wr[0] ^ lo;
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

		/// <summary>
		/// This function calculates three different CRCs and also 
		/// collects some additional information for the purposes of 
		/// offset detection.
		/// 
		/// crcar is AccurateRip CRC
		/// crc32 is CRC32
		/// crcwn is CRC32 without null samples (EAC)
		/// crcsm is sum of samples
		/// crcnl is a count of null samples
		/// </summary>
		/// <param name="pSampleBuff"></param>
		/// <param name="count"></param>
		/// <param name="offs"></param>
		public unsafe void CalculateCRCs(uint* t, ushort* exp, ushort* log, ushort* syn, ushort* wr, uint* pSampleBuff, int count, int offs)
		{
			int currentStride = ((int)_sampleCount * 2) / stride;
			bool doSyn = currentStride >= 1 && currentStride <= stridecount && calcSyn;
			bool doPar = currentStride >= 1 && currentStride <= stridecount && calcParity;
			uint n = (uint)(stridecount - currentStride);

			uint crcar = _CRCAR[_currentTrack, 0];
			uint crcsm = _CRCSM[_currentTrack, 0];
			uint crc32 = _CRC32[_currentTrack, 0];
			uint crcwn = _CRCWN[_currentTrack, 0];
			int crcnl = _CRCNL[_currentTrack, 0];
			uint crcv2 = _CRCV2[_currentTrack, 0];
			int peak = _Peak[_currentTrack];
			
			for (int i = 0; i < count; i++)
			{
				if (offs >= 0)
				{
					_CRCAR[_currentTrack, offs + i] = crcar;
					_CRCSM[_currentTrack, offs + i] = crcsm;
					_CRC32[_currentTrack, offs + i] = crc32;
					_CRCWN[_currentTrack, offs + i] = crcwn;
					_CRCNL[_currentTrack, offs + i] = crcnl;
					_CRCV2[_currentTrack, offs + i] = crcv2;
				}

				uint sample = *(pSampleBuff++);
				crcsm += sample;
				ulong calccrc = sample * (ulong)(_samplesDoneTrack + i + 1);
				crcar += (uint)calccrc;
				crcv2 += (uint)(calccrc >> 32);

				uint lo = sample & 0xffff;
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ lo)];
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ (lo >> 8))];
				if (lo != 0)
				{
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ lo)];
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ (lo >> 8))];
				}
				else crcnl++;

				int pk = ((int)(lo << 16)) >> 16;
				peak = Math.Max(peak, (pk << 1) ^ (pk >> 31));

				if (doSyn && lo != 0) CalcSyn8(exp, log, syn + i * 16, lo, n, npar);
				if (doPar) CalcPar8(exp, log, wr + i * 16, lo);

				uint hi = sample >> 16;
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ hi)];
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ (hi >> 8))];
				if (hi != 0)
				{
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ hi)];
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ (hi >> 8))];
				}
				else crcnl++;

				pk = ((int)(hi << 16)) >> 16;
				peak = Math.Max(peak, (pk << 1) ^ (pk >> 31));

				if (doSyn && hi != 0) CalcSyn8(exp, log, syn + i * 16 + 8, hi, n, npar);
				if (doPar) CalcPar8(exp, log, wr + i * 16 + 8, hi);
			}

			_CRCAR[_currentTrack, 0] = crcar;
			_CRCSM[_currentTrack, 0] = crcsm;
			_CRC32[_currentTrack, 0] = crc32;
			_CRCWN[_currentTrack, 0] = crcwn;
			_CRCNL[_currentTrack, 0] = crcnl;
			_CRCV2[_currentTrack, 0] = crcv2;
			_Peak[_currentTrack] = peak;
		}

		private int _samplesRemTrack = 0;
		private int _samplesDoneTrack = 0;

		public long Position
		{
			get
			{
				return _sampleCount;
			}
			set
			{
				_sampleCount = value;
				int tempLocation = 0; // NOT (int)_toc[_toc.FirstAudio][0].Start * 588;
				for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
				{
					int tempLen = (int)(iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
					if (tempLocation + tempLen > _sampleCount)
					{
						_currentTrack = iTrack;
						_samplesRemTrack = tempLocation + tempLen - (int)_sampleCount;
						_samplesDoneTrack = (int)_sampleCount - tempLocation;
						return;
					}
					tempLocation += tempLen;
				}
				throw new ArgumentOutOfRangeException();
			}
		}

		public unsafe void Write(AudioBuffer sampleBuffer)
		{
			sampleBuffer.Prepare(this);

			int pos = 0;
			fixed (uint* t = Crc32.table)
			fixed (ushort* exp = Galois16.instance.ExpTbl, log = Galois16.instance.LogTbl, synptr1 = syndrome)
			fixed (byte* pSampleBuff = &sampleBuffer.Bytes[0], bpar = parity)
				while (pos < sampleBuffer.Length)
				{
					// Process no more than there is in the buffer, no more than there is in this track, and no more than up to a sector boundary.
					int copyCount = Math.Min(Math.Min(sampleBuffer.Length - pos, (int)_samplesRemTrack), 588 - (int)_sampleCount % 588);
					int currentSector = _samplesDoneTrack / 588;
					int remaingSectors = (_samplesRemTrack - 1) / 588;
					uint* samples = ((uint*)pSampleBuff) + pos;
					int currentPart = ((int)_sampleCount * 2) % stride;
					ushort* synptr = synptr1 + npar * currentPart;
					ushort* wr = ((ushort*)bpar) + npar * currentPart;
					int currentStride = ((int)_sampleCount * 2) / stride;

					for (int i = 0; i < Math.Min(leadin.Length - (int)_sampleCount * 2, copyCount * 2); i++)
						leadin[_sampleCount * 2 + i] = ((ushort*)samples)[i];

					for (int i = Math.Max(0, (int)(_finalSampleCount - _sampleCount) * 2 - leadout.Length); i < copyCount * 2; i++)
					//if (currentStride >= stridecount && leadout != null)
					//for (int i = 0; i < copyCount * 2; i++)
					{
						int remaining = (int)(_finalSampleCount - _sampleCount) * 2 - i - 1;
						leadout[remaining] = ((ushort*)samples)[i];
					}
					
					if (currentSector < 10)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, _samplesDoneTrack);
					else if (remaingSectors < 10)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, 20 * 588 - _samplesRemTrack);
					else if (currentSector >= 445 && currentSector <= 455)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, 20 * 588 + _samplesDoneTrack - 445 * 588);
					else
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, -1);

					pos += copyCount;
					_samplesRemTrack -= copyCount;
					_samplesDoneTrack += copyCount;
					_sampleCount += copyCount;

					while (_samplesRemTrack <= 0)
					{
						if (++_currentTrack > _toc.AudioTracks)
							return;
						_samplesRemTrack = (int)_toc[_currentTrack + _toc.FirstAudio - 1].Length * 588;
						_samplesDoneTrack = 0;
					}
				}
		}

		public void Combine(AccurateRipVerify part, int start, int end)
		{
			for (int i = 0; i < leadin.Length; i++)
			{
				int currentOffset = i / 2;
				if (currentOffset >= start && currentOffset < end)
					this.leadin[i] = part.leadin[i];
			}
			for (int i = 0; i < leadout.Length; i++)
			{
				int currentOffset = (int)_finalSampleCount - i / 2 - 1;
				if (currentOffset >= start && currentOffset < end)
					this.leadout[i] = part.leadout[i];
			}
			for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
			{
				// ??? int tempLocation = (int) (iTrack == 0 ? _toc[_toc.FirstAudio][0].Start : _toc[_toc.FirstAudio + iTrack - 1].Start) * 588;
				int tempLocation = (int) (iTrack == 0 ? 0 : _toc[_toc.FirstAudio + iTrack - 1].Start - _toc[_toc.FirstAudio][0].Start) * 588;
				int tempLen = (int) (iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
				int trStart = Math.Max(tempLocation, start);
				int trEnd = Math.Min(tempLocation + tempLen, end);
				if (trStart >= trEnd)
					continue;

				uint crcar = _CRCAR[iTrack, 0];
				uint crcv2 = _CRCV2[iTrack, 0];
				uint crcsm = _CRCSM[iTrack, 0];
				uint crc32 = _CRC32[iTrack, 0];
				uint crcwn = _CRCWN[iTrack, 0];
				int crcnl = _CRCNL[iTrack, 0];
				_CRCAR[iTrack, 0] = crcar + part._CRCAR[iTrack, 0];
				_CRCSM[iTrack, 0] = crcsm + part._CRCSM[iTrack, 0];
				_CRCNL[iTrack, 0] = crcnl + part._CRCNL[iTrack, 0];
				_CRC32[iTrack, 0] = Crc32.Combine(crc32, part._CRC32[iTrack, 0], 4 * (trEnd - trStart));
				_CRCWN[iTrack, 0] = Crc32.Combine(crcwn, part._CRCWN[iTrack, 0], 4 * (trEnd - trStart) - 2 * part._CRCNL[iTrack, 0]);
				_CRCV2[iTrack, 0] = crcv2 + part._CRCV2[iTrack, 0];
				for (int i = 1; i < 31 * 588; i++)
				{
					int currentOffset;
					if (i < 10 * 588)
					{
						currentOffset = tempLocation + i;
					}
					else if (i < 20 * 588)
					{
						currentOffset = tempLocation + tempLen + i - 20 * 588;
					}
					else
					{
						currentOffset = tempLocation + i - 20 * 588 + 445 * 588;
					}
					if (currentOffset <= trStart)
						continue;

					_CRCAR[iTrack, i] = crcar + part._CRCAR[iTrack, i];
					_CRCV2[iTrack, i] = crcv2 + part._CRCV2[iTrack, i];
					_CRCSM[iTrack, i] = crcsm + part._CRCSM[iTrack, i];
					_CRCNL[iTrack, i] = crcnl + part._CRCNL[iTrack, i];
					_CRC32[iTrack, i] = Crc32.Combine(crc32, part._CRC32[iTrack, i], 4 * (currentOffset - trStart));
					_CRCWN[iTrack, i] = Crc32.Combine(crcwn, part._CRCWN[iTrack, i], 4 * (currentOffset - trStart) - 2 * part._CRCNL[iTrack, i]);
				}
				_Peak[iTrack] = Math.Max(_Peak[iTrack], part._Peak[iTrack]);
			}
		}

		public void Init(CDImageLayout toc)
		{
			_toc = toc;
			_finalSampleCount = _toc.AudioLength * 588;
			_CRCMASK = new uint[_toc.AudioTracks + 1];
			_CRCMASK[0] = 0xffffffff ^ Crc32.Combine(0xffffffff, 0, (int)_finalSampleCount * 4);
			for (int iTrack = 1; iTrack <= _toc.AudioTracks; iTrack++)
				_CRCMASK[iTrack] = 0xffffffff ^ Crc32.Combine(0xffffffff, 0, (int)_toc[iTrack + _toc.FirstAudio - 1].Length * 588 * 4);
			_CRCAR = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CRCSM = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CRC32 = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CacheCRC32 = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CRCWN = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CacheCRCWN = new uint[_toc.AudioTracks + 1, 31 * 588];
			_CRCNL = new int[_toc.AudioTracks + 1, 31 * 588];
			_CRCV2 = new uint[_toc.AudioTracks + 1, 31 * 588];
			_Peak = new int[_toc.AudioTracks + 1];
			syndrome = new ushort[calcSyn ? stride : 1, npar];
			parity = new byte[stride * npar * 2];
			int leadin_len = Math.Max(4096 * 4, (calcSyn || calcParity) ? stride * 2 : 0);
			int leadout_len = Math.Max(4096 * 4, (calcSyn || calcParity) ? stride + laststride : 0);
			leadin = new ushort[leadin_len];
			leadout = new ushort[leadout_len];
			_currentTrack = 0;
			Position = 0; // NOT _toc[_toc.FirstAudio][0].Start * 588;
		}

		private uint readIntLE(byte[] data, int pos)
		{
			return (uint)(data[pos] + (data[pos + 1] << 8) + (data[pos + 2] << 16) + (data[pos + 3] << 24));
		}

        static DateTime last_accessed;
        static readonly TimeSpan min_interval = new TimeSpan (5000000); // 0.5 second
        static readonly object server_mutex = new object ();

		public void ContactAccurateRip(string accurateRipId)
		{
			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;

			string[] n = accurateRipId.Split('-');
			if (n.Length != 3)
			{
				ExceptionStatus = WebExceptionStatus.RequestCanceled;
				throw new Exception("Invalid accurateRipId.");
			}
			discId1 = UInt32.Parse(n[0], NumberStyles.HexNumber);
			discId2 = UInt32.Parse(n[1], NumberStyles.HexNumber);
			cddbDiscId = UInt32.Parse(n[2], NumberStyles.HexNumber);

			string url = String.Format("http://www.accuraterip.com/accuraterip/{0:x}/{1:x}/{2:x}/dBAR-{3:d3}-{4:x8}-{5:x8}-{6:x8}.bin",
				discId1 & 0xF, discId1 >> 4 & 0xF, discId1 >> 8 & 0xF, _toc.AudioTracks, discId1, discId2, cddbDiscId);

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.Method = "GET";
			req.Proxy = proxy;

			lock (server_mutex)
			{
				// Don't access the AR server twice within min_interval
				if (last_accessed != null)
				{
					TimeSpan time = DateTime.Now - last_accessed;
					if (min_interval > time)
						Thread.Sleep((min_interval - time).Milliseconds);
				}

				try
				{
					using (HttpWebResponse response = (HttpWebResponse)req.GetResponse())
					{
						ExceptionStatus = WebExceptionStatus.ProtocolError;
						ResponseStatus = response.StatusCode;
						if (ResponseStatus == HttpStatusCode.OK)
						{
							ExceptionStatus = WebExceptionStatus.Success;

							// Retrieve response stream and wrap in StreamReader
							Stream respStream = response.GetResponseStream();

							// Allocate byte buffer to hold stream contents
							byte[] urlData = new byte[13];
							int urlDataLen, bytesRead;

							_accDisks.Clear();
							while (true)
							{
								for (urlDataLen = 0; urlDataLen < 13; urlDataLen += bytesRead)
								{
									bytesRead = respStream.Read(urlData, urlDataLen, 13 - urlDataLen);
									if (0 == bytesRead)
										break;
								}
								if (urlDataLen == 0)
									break;
								if (urlDataLen < 13)
								{
									ExceptionStatus = WebExceptionStatus.ReceiveFailure;
									return;
								}
								AccDisk dsk = new AccDisk();
								dsk.count = urlData[0];
								dsk.discId1 = readIntLE(urlData, 1);
								dsk.discId2 = readIntLE(urlData, 5);
								dsk.cddbDiscId = readIntLE(urlData, 9);

								for (int i = 0; i < dsk.count; i++)
								{
									for (urlDataLen = 0; urlDataLen < 9; urlDataLen += bytesRead)
									{
										bytesRead = respStream.Read(urlData, urlDataLen, 9 - urlDataLen);
										if (0 == bytesRead)
										{
											ExceptionStatus = WebExceptionStatus.ReceiveFailure;
											return;
										}
									}
									AccTrack trk = new AccTrack();
									trk.count = urlData[0];
									trk.CRC = readIntLE(urlData, 1);
									trk.Frame450CRC = readIntLE(urlData, 5);
									dsk.tracks.Add(trk);
								}
								_accDisks.Add(dsk);
							}
							respStream.Close();
						}
					}
				}
				catch (WebException ex)
				{
					ExceptionStatus = ex.Status;
					ExceptionMessage = ex.Message;
					if (ExceptionStatus == WebExceptionStatus.ProtocolError)
						ResponseStatus = (ex.Response as HttpWebResponse).StatusCode;
				}
				finally
				{
					last_accessed = DateTime.Now;
				}
			}
		}

		public void Close()
		{
			if (_sampleCount != _finalSampleCount)
				throw new Exception("_sampleCount != _finalSampleCount");
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

		public long FinalSampleCount
		{
			get
			{
				return _finalSampleCount;
			}
			set
			{
				if (value != _finalSampleCount) 
					throw new Exception("invalid FinalSampleCount");
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

		public void GenerateLog(TextWriter sw, int oi, bool v2)
		{
			uint maxTotal = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
				maxTotal = Math.Max(maxTotal, Total(iTrack));

			uint maxConf = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint crcOI = v2 ? CRCV2(iTrack) : CRC(iTrack, oi);
				for (int di = 0; di < (int)AccDisks.Count; di++)
				{
					int trno = iTrack + _toc.FirstAudio - 1;
					if (trno < AccDisks[di].tracks.Count
						&& crcOI == AccDisks[di].tracks[trno].CRC
						&& 0 != AccDisks[di].tracks[trno].CRC
						)
						maxConf = Math.Max(maxConf, AccDisks[di].tracks[trno].count);
				}
			}
			if (maxConf == 0 && v2)
				return;
			if (v2)
				sw.WriteLine("AccurateRip v2:");
			string ifmt = maxTotal < 10 ? ":0" : maxTotal < 100 ? ":00" : ":000";
			//string ifmt = maxTotal < 10 ? ",1" : maxTotal < 100 ? ",2" : ",3";
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint count = 0;
				uint partials = 0;
				uint conf = 0;
				uint crcOI = v2 ? CRCV2(iTrack) : CRC(iTrack, oi);
				uint crc450OI = CRC450(iTrack, oi);
				for (int di = 0; di < (int)AccDisks.Count; di++)
				{
					int trno = iTrack + _toc.FirstAudio - 1;
					if (trno >= AccDisks[di].tracks.Count)
						continue;
					count += AccDisks[di].tracks[trno].count;
					if (crcOI == AccDisks[di].tracks[trno].CRC
						&& 0 != AccDisks[di].tracks[trno].CRC)
						conf += AccDisks[di].tracks[trno].count;
					if (crc450OI == AccDisks[di].tracks[trno].Frame450CRC
						&& 0 != AccDisks[di].tracks[trno].Frame450CRC)
						partials ++;
				}
				string status;
				if (conf > 0)
					status = "Accurately ripped";
				else if (count == 0 && crcOI == 0)
					status = "Silent track";
				else if (partials > 0)
					status = "No match but offset";
				else
					status = "No match";
				sw.WriteLine(String.Format(" {0:00}     [{1:x8}] ({3" + ifmt + "}/{2" + ifmt + "}) {4}", iTrack + 1, crcOI, count, conf, status));
			}
		}

		public void GenerateFullLog(TextWriter sw, bool verbose, string id)
		{
			sw.WriteLine("[AccurateRip ID: {0}] {1}.", id, ARStatus ?? "found");
			if (ExceptionStatus == WebExceptionStatus.Success)
			{
				if (verbose)
				{
					sw.WriteLine("Track   [ CRC    ] Status");
					GenerateLog(sw, 0, false);
					GenerateLog(sw, 0, true);
					uint offsets_match = 0;
					for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					{
						uint matches = 0;
						for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
						{
							int trno = iTrack + _toc.FirstAudio - 1;
							for (int di = 0; di < (int)AccDisks.Count; di++)
								if (trno < AccDisks[di].tracks.Count
									&& (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC
									&& AccDisks[di].tracks[trno].CRC != 0))
								{
									matches++;
									break;
								}
						}
						if (matches == _toc.AudioTracks && oi != 0)
						{
							if (offsets_match++ > 16)
							{
								sw.WriteLine("More than 16 offsets match!");
								break;
							}
							sw.WriteLine("Offsetted by {0}:", oi);
							GenerateLog(sw, oi, false);
						}
					}
					offsets_match = 0;
					for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					{
						uint matches = 0, partials = 0;
						for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
						{
							uint crcOI = CRC(iTrack, oi);
							uint crc450OI = CRC450(iTrack, oi);
							for (int di = 0; di < (int)AccDisks.Count; di++)
							{
								int trno = iTrack + _toc.FirstAudio - 1;
								if (trno >= AccDisks[di].tracks.Count)
									continue;
								if (crcOI == AccDisks[di].tracks[trno].CRC
									&& AccDisks[di].tracks[trno].CRC != 0)
								{
									matches++;
									break;
								}
								if (crc450OI == AccDisks[di].tracks[trno].Frame450CRC
									&& AccDisks[di].tracks[trno].Frame450CRC != 0)
									partials++;
							}
						}
						if (matches != _toc.AudioTracks && oi != 0 && matches + partials != 0)
						{
							if (offsets_match++ > 16)
							{
								sw.WriteLine("More than 16 offsets match!");
								break;
							}
							sw.WriteLine("Offsetted by {0}:", oi);
							GenerateLog(sw, oi, false);
						}
					}
				}
				else
				{
					sw.WriteLine("Track    Status");
					for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
					{
						uint total = Total(iTrack);
						uint conf = 0;
						bool zeroOffset = false;
						StringBuilder pressings = new StringBuilder();
						for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
							for (int iDisk = 0; iDisk < AccDisks.Count; iDisk++)
							{
								int trno = iTrack + _toc.FirstAudio - 1;
								if (trno < AccDisks[iDisk].tracks.Count
									&& (CRC(iTrack, oi) == AccDisks[iDisk].tracks[trno].CRC
									  || oi == 0 && CRCV2(iTrack) == AccDisks[iDisk].tracks[trno].CRC
									  )
									&& (AccDisks[iDisk].tracks[trno].CRC != 0 || oi == 0))
								{
									conf += AccDisks[iDisk].tracks[trno].count;
									if (oi == 0)
										zeroOffset = true;
									pressings.AppendFormat("{0}{1}({2})", pressings.Length > 0 ? "," : "", oi, AccDisks[iDisk].tracks[trno].count);
								}
							}
						if (conf > 0 && zeroOffset && pressings.Length == 0)
							sw.WriteLine(String.Format(" {0:00}      ({2:00}/{1:00}) Accurately ripped", iTrack + 1, total, conf));
						else if (conf > 0 && zeroOffset)
							sw.WriteLine(String.Format(" {0:00}      ({2:00}/{1:00}) Accurately ripped, all offset(s) {3}", iTrack + 1, total, conf, pressings));
						else if (conf > 0)
							sw.WriteLine(String.Format(" {0:00}      ({2:00}/{1:00}) Accurately ripped with offset(s) {3}", iTrack + 1, total, conf, pressings));
						else if (total > 0)
							sw.WriteLine(String.Format(" {0:00}      (00/{1:00}) NOT ACCURATE", iTrack + 1, total));
						else
							sw.WriteLine(String.Format(" {0:00}      (00/00) Track not present in database", iTrack + 1));
					}
				}
			}
			if (CRC32(0) != 0 && (_hasLogCRC || verbose))
			{
				sw.WriteLine("");
				sw.WriteLine("Track Peak [ CRC32  ] [W/O NULL] {0:10}", _hasLogCRC ? "[  LOG   ]" : "");
				for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
				{
					string inLog, extra = "";
					if (CRCLOG(iTrack) == 0)
						inLog = "";
					else if (CRCLOG(iTrack) == CRC32(iTrack))
						inLog = "  CRC32   ";
					else if (CRCLOG(iTrack) == CRCWONULL(iTrack))
						inLog = " W/O NULL ";
					else
					{
						inLog = String.Format("[{0:X8}]", CRCLOG(iTrack));
						for (int jTrack = 1; jTrack <= _toc.AudioTracks; jTrack++)
						{
							if (CRCLOG(iTrack) == CRC32(jTrack))
							{
								extra = string.Format(": CRC32 for track {0}", jTrack);
								break;
							}
							if (CRCLOG(iTrack) == CRCWONULL(jTrack))
							{
								extra = string.Format(": W/O NULL for track {0}", jTrack);
								break;
							}
						}
						if (extra == "")
							for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
								if (CRCLOG(iTrack) == CRC32(iTrack, oi))
								{
									inLog = "  CRC32   ";
									extra = string.Format(": offset {0}", oi);
									break;
								}
						if (extra == "")
							for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
								if (CRCLOG(iTrack) == CRCWONULL(iTrack, oi))
								{
									inLog = " W/O NULL ";
									if (extra == "")
										extra = string.Format(": offset {0}", oi);
									else
									{
										extra = string.Format(": with offset");
										break;
									}
								}
					}
					sw.WriteLine(" {0}  {5,5:F1} [{1:X8}] [{2:X8}] {3,10}{4}",
						iTrack == 0 ? "--" : string.Format("{0:00}", iTrack),
						CRC32(iTrack),
						CRCWONULL(iTrack),
						inLog,
						extra,
						((iTrack == 0 ? PeakLevel() : PeakLevel(iTrack)) * 1000 / 65534) * 0.1);
				}
			}
		}

		private static uint sumDigits(uint n)
		{
			uint r = 0;
			while (n > 0)
			{
				r = r + (n % 10);
				n = n / 10;
			}
			return r;
		}

		static string CachePath
		{
			get
			{
				string cache = System.IO.Path.Combine(System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "CUE Tools"), "AccurateRipCache");
				if (!Directory.Exists(cache))
					Directory.CreateDirectory(cache);
				return cache;
			}
		}

		public static bool FindDriveReadOffset(string driveName, out int driveReadOffset)
		{
			string fileName = System.IO.Path.Combine(CachePath, "DriveOffsets.bin");
			if (!File.Exists(fileName) || (DateTime.Now - File.GetLastWriteTime(fileName) > TimeSpan.FromDays(10)) )
			{
				HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://www.accuraterip.com/accuraterip/DriveOffsets.bin");
				req.Method = "GET";
				try
				{
					HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
					if (resp.StatusCode != HttpStatusCode.OK)
					{
						driveReadOffset = 0;
						return false;
					}
					Stream respStream = resp.GetResponseStream();
					FileStream myOffsetsSaved = new FileStream(fileName, FileMode.Create, FileAccess.Write);
					byte[] buff = new byte[0x8000];
					do
					{
						int count = respStream.Read(buff, 0, buff.Length);
						if (count == 0) break;
						myOffsetsSaved.Write(buff, 0, count);
					} while (true);
					respStream.Close();
					myOffsetsSaved.Close();
				}
				catch (WebException ex)
				{
					driveReadOffset = 0;
					return false;
				}
			}
			FileStream myOffsets = new FileStream(fileName, FileMode.Open, FileAccess.Read);
			BinaryReader offsetReader = new BinaryReader(myOffsets);
			do
			{
				short readOffset = offsetReader.ReadInt16();
				byte[] name = offsetReader.ReadBytes(0x21);
				byte[] misc = offsetReader.ReadBytes(0x22);
				int len = name.Length;
				while (len > 0 && name[len - 1] == '\0') len--;
				string strname = Encoding.ASCII.GetString(name, 0, len);
				if (strname == driveName)
				{
					driveReadOffset = readOffset;
					return true;
				}
			} while (myOffsets.Position + 0x45 <= myOffsets.Length);
			offsetReader.Close();
			driveReadOffset = 0;
			return false;
		}

		public static string CalculateCDDBQuery(CDImageLayout toc)
		{
			StringBuilder query = new StringBuilder(CalculateCDDBId(toc));
			query.AppendFormat("+{0}", toc.TrackCount);
			for (int iTrack = 1; iTrack <= toc.TrackCount; iTrack++)
				query.AppendFormat("+{0}", toc[iTrack].Start + 150);
			query.AppendFormat("+{0}", toc.Length / 75 - toc[1].Start / 75);
			return query.ToString();
		}

		public static string CalculateCDDBId(CDImageLayout toc)
		{
			uint cddbDiscId = 0;
			for (int iTrack = 1; iTrack <= toc.TrackCount; iTrack++)
				cddbDiscId += sumDigits(toc[iTrack].Start / 75 + 2); // !!!!!!!!!!!!!!!!! %255 !!
			return string.Format("{0:X8}", (((cddbDiscId % 255) << 24) + ((toc.Length / 75 - toc[1].Start / 75) << 8) + (uint)toc.TrackCount) & 0xFFFFFFFF);
		}

		public static string CalculateAccurateRipId(CDImageLayout toc)
		{
			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint num = 0;

			for (int iTrack = 1; iTrack <= toc.TrackCount; iTrack++)
				if (toc[iTrack].IsAudio)
				{
					discId1 += toc[iTrack].Start;
					discId2 += Math.Max(toc[iTrack].Start, 1) * (++num);
				}
			discId1 += toc.Length;
			discId2 += Math.Max(toc.Length, 1) * (++num);
			discId1 &= 0xFFFFFFFF;
			discId2 &= 0xFFFFFFFF;
			return string.Format("{0:x8}-{1:x8}-{2}", discId1, discId2, CalculateCDDBId(toc).ToLower());
		}

		public List<AccDisk> AccDisks
		{
			get
			{
				return _accDisks;
			}
		}

		private string ExceptionMessage;
		public HttpStatusCode ResponseStatus { get; set; }
		public WebExceptionStatus ExceptionStatus { get; set; }
		public string ARStatus
		{
			get
			{
				return ExceptionStatus == WebExceptionStatus.Success ? null :
					ExceptionStatus != WebExceptionStatus.ProtocolError ? ("database access error: " + (ExceptionMessage ?? ExceptionStatus.ToString())) :
					ResponseStatus != HttpStatusCode.NotFound ? "database access error: " + ResponseStatus.ToString() :
					"disk not present in database";
			}
		}

		CDImageLayout _toc;
		long _sampleCount, _finalSampleCount;
		int _currentTrack;
		private List<AccDisk> _accDisks;
		internal uint[,] _CRCAR;
		private uint[,] _CRCV2;
		internal uint[,] _CRCSM;
		internal uint[,] _CRC32;
		internal uint[,] _CRCWN;
		internal int[,] _CRCNL;
		private uint[,] _CacheCRCWN;
		private uint[,] _CacheCRC32;
		internal int[] _Peak;
		private uint[] _CRCLOG;
		private uint[] _CRCMASK;
		private IWebProxy proxy;

		private bool _hasLogCRC;

		private const int _arOffsetRange = 5 * 588 - 1;
	}

	public struct AccTrack
	{
		public uint count;
		public uint CRC;
		public uint Frame450CRC;
	}

	public class AccDisk
	{
		public uint count;
		public uint discId1;
		public uint discId2;
		public uint cddbDiscId;
		public List<AccTrack> tracks;

		public AccDisk()
		{
			tracks = new List<AccTrack>();
		}
	}
}
