using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using CUETools.Parity;
using CUETools.CDImage;
using CUETools.Codecs;

namespace CUETools.AccurateRip
{
	public class AccurateRipVerify : IAudioDest
	{
		public AccurateRipVerify(CDImageLayout toc, IWebProxy proxy)
		{
			this.proxy = proxy;
			_accDisks = new List<AccDisk>();
			_hasLogCRC = false;
			_CRCLOG = new uint[toc.AudioTracks + 1];
			Init(toc);
		}

		public uint Confidence(int iTrack)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			for (int di = 0; di < (int)AccDisks.Count; di++)
				if (iTrack + _toc.FirstAudio - 1 < AccDisks[di].tracks.Count 
					&& CRC(iTrack) == AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].CRC)
					conf += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
			return conf;
		}

		public uint WorstTotal()
		{
			uint worstTotal = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint sumTotal = Total(iTrack);
				if (iTrack == 0 || worstTotal > sumTotal)
					worstTotal = sumTotal;
			}
			return worstTotal;
		}

		public uint WorstConfidence()
		{
			uint worstConfidence = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint sumConfidence = SumConfidence(iTrack);
				if (iTrack == 0 || worstConfidence > sumConfidence)
					worstConfidence = sumConfidence;
			}
			return worstConfidence;
		}

		public uint SumConfidence(int iTrack)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			for (int iDisk = 0; iDisk < AccDisks.Count; iDisk++)
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					if (iTrack + _toc.FirstAudio - 1 < AccDisks[iDisk].tracks.Count
						&& CRC(iTrack, oi) == AccDisks[iDisk].tracks[iTrack + _toc.FirstAudio - 1].CRC)
						conf += AccDisks[iDisk].tracks[iTrack + _toc.FirstAudio - 1].count;
			return conf;
		}

		public uint Confidence(int iTrack, int oi)
		{
			if (ARStatus != null)
				return 0U;
			uint conf = 0;
			for (int di = 0; di < (int)AccDisks.Count; di++)
				if (iTrack +_toc.FirstAudio - 1 < AccDisks[di].tracks.Count
					&& CRC(iTrack, oi) == AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].CRC)
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

		public uint CRC32(int iTrack)
		{
			return CRC32(iTrack, 0);
		}

		public uint GetCRC32(uint crcA, int lenA, uint crcB, int lenB)
		{
			int lenAXB = (int)_toc.AudioLength * 588 * 4;
			int lenXB = lenAXB - lenA;
			int lenX = lenXB - lenB;
			uint crcAXB = CRC32(0, 0) ^ _CRCMASK[0];
			uint crcXB = Crc32.Combine(crcA, crcAXB, lenXB);
			uint crcX = Crc32.Substract(crcXB, crcB, lenB);
			return Crc32.Combine(0xffffffff, crcX, lenX) ^ 0xffffffff;
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

		internal unsafe uint CTDBCRC(int actualOffset)
		{
			fixed (uint* crct = Crc32.table)
			{
				// calculate leadin CRC
				uint crc0 = 0;
				for (int off = 0; off < stride - 2 * actualOffset; off++)
				{
					ushort dd = leadin[off];
					crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ dd)];
					crc0 = (crc0 >> 8) ^ crct[(byte)(crc0 ^ (dd >> 8))];
				}
				// calculate leadout CRC
				uint crc2 = 0;
				for (int off = laststride + 2 * actualOffset - 1; off >= 0; off--)
				{
					ushort dd = leadout[off];
					crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ dd)];
					crc2 = (crc2 >> 8) ^ crct[(byte)(crc2 ^ (dd >> 8))];
				}

				return GetCRC32(crc0, (stride - 2 * actualOffset) * 2, crc2, (laststride + 2 * actualOffset) * 2);
			}
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
		/// <param name="currentOffset"></param>
		/// <param name="offs"></param>
		public unsafe void CalculateCRCs(uint* t, ushort* exp, ushort* log, ushort* syn, ushort* wr, uint* pSampleBuff, int count, int currentOffset, int offs)
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
			int peak = _Peak[_currentTrack];
			//fixed (ushort* exp = expTbl, log = logTbl, synptr = syndrome)
			for (int i = 0; i < count; i++)
			{
				if (offs >= 0)
				{
					_CRCAR[_currentTrack, offs + i] = crcar;
					_CRCSM[_currentTrack, offs + i] = crcsm;
					_CRC32[_currentTrack, offs + i] = crc32;
					_CRCWN[_currentTrack, offs + i] = crcwn;
					_CRCNL[_currentTrack, offs + i] = crcnl;
				}

				uint sample = *(pSampleBuff++);
				crcsm += sample;
				crcar += sample * (uint)(currentOffset + i + 1);

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
			_Peak[_currentTrack] = peak;
		}

		private long _samplesRemTrack = 0;

		public long Position
		{
			get
			{
				return _sampleCount;
			}
			set
			{
				_sampleCount = value;
				int tempLocation = (int)_toc[_toc.FirstAudio][0].Start * 588;
				for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
				{
					int tempLen = (int)(iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
					if (tempLocation + tempLen > _sampleCount)
					{
						_currentTrack = iTrack;
						_samplesRemTrack = tempLocation + tempLen - _sampleCount;
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
					// Calculate offset within a track
					int currentOffset = (int)_sampleCount - (int)(_currentTrack > 0 ? _toc[_currentTrack + _toc.FirstAudio - 1].Start * 588 : 0);
					int currentSector = currentOffset / 588;
					int remaingSectors = (int)(_samplesRemTrack - 1) / 588;
					uint* samples = ((uint*)pSampleBuff) + pos;
					int currentPart = ((int)_sampleCount * 2) % stride;
					ushort* synptr = synptr1 + npar * currentPart;
					ushort* wr = ((ushort*)bpar) + npar * currentPart;
					int currentStride = ((int)_sampleCount * 2) / stride;

					if (currentStride < 2 && leadin != null)
						for (int i = 0; i < copyCount * 2; i++)
							leadin[_sampleCount * 2 + i] = ((ushort*)samples)[i];

					if (currentStride >= stridecount && leadout != null)
						for (int i = 0; i < copyCount * 2; i++)
						{
							int remaining = (int)(_finalSampleCount - _sampleCount) * 2 - i - 1;
							if (remaining < stride + laststride)
								leadout[remaining] = ((ushort*)samples)[i];
						}

					if (currentSector < 10)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, currentOffset, currentOffset);
					else if (remaingSectors < 10)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, currentOffset, 20 * 588 - (int)_samplesRemTrack);
					else if (currentSector >= 445 && currentSector <= 455)
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, currentOffset, 20 * 588 + currentOffset - 445 * 588);
					else
						CalculateCRCs(t, exp, log, synptr, wr, samples, copyCount, currentOffset, -1);

					pos += copyCount;
					_samplesRemTrack -= copyCount;
					_sampleCount += copyCount;

					while (_samplesRemTrack <= 0)
					{
						if (++_currentTrack > _toc.AudioTracks)
							return;
						_samplesRemTrack = _toc[_currentTrack + _toc.FirstAudio - 1].Length * 588;
					}
				}
		}

		public void Combine(AccurateRipVerify part, int start, int end)
		{
			for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
			{
				int tempLocation = (int) (iTrack == 0 ? _toc[_toc.FirstAudio][0].Start : _toc[_toc.FirstAudio + iTrack - 1].Start) * 588;
				int tempLen = (int) (iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
				int trStart = Math.Max(tempLocation, start);
				int trEnd = Math.Min(tempLocation + tempLen, end);
				if (trStart >= trEnd)
					continue;

				uint crcar = _CRCAR[iTrack, 0];
				uint crcsm = _CRCSM[iTrack, 0];
				uint crc32 = _CRC32[iTrack, 0];
				uint crcwn = _CRCWN[iTrack, 0];
				int crcnl = _CRCNL[iTrack, 0];
				_CRCAR[iTrack, 0] = crcar + part._CRCAR[iTrack, 0];
				_CRCSM[iTrack, 0] = crcsm + part._CRCSM[iTrack, 0];
				_CRCNL[iTrack, 0] = crcnl + part._CRCNL[iTrack, 0];
				_CRC32[iTrack, 0] = Crc32.Combine(crc32, part._CRC32[iTrack, 0], 4 * (trEnd - trStart));
				_CRCWN[iTrack, 0] = Crc32.Combine(crcwn, part._CRCWN[iTrack, 0], 4 * (trEnd - trStart) - 2 * part._CRCNL[iTrack, 0]);
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
			_Peak = new int[_toc.AudioTracks + 1];
			syndrome = new ushort[calcSyn ? stride : 1, npar];
			parity = new byte[stride * npar * 2];
			if (calcSyn || calcParity)
			{
				leadin = new ushort[stride * 2];
				leadout = new ushort[stride + laststride];
			}
			else
			{
				leadin = null;
				leadout = null;
			}
			_currentTrack = 0;
			Position = _toc[_toc.FirstAudio][0].Start * 588;
		}

		private uint readIntLE(byte[] data, int pos)
		{
			return (uint)(data[pos] + (data[pos + 1] << 8) + (data[pos + 2] << 16) + (data[pos + 3] << 24));
		}

		public void ContactAccurateRip(string accurateRipId)
		{
			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;

			string[] n = accurateRipId.Split('-');
			if (n.Length != 3)
			{
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

			try
			{
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				_accResult = resp.StatusCode;

				if (_accResult == HttpStatusCode.OK)
				{
					// Retrieve response stream and wrap in StreamReader
					Stream respStream = resp.GetResponseStream();

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
							_accResult = HttpStatusCode.PartialContent;
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
									_accResult = HttpStatusCode.PartialContent;
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
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					_accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					_accResult = HttpStatusCode.BadRequest;
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

		public void GenerateLog(TextWriter sw, int oi)
		{
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint count = 0;
				uint partials = 0;
				uint conf = 0;
				for (int di = 0; di < (int)AccDisks.Count; di++)
				{
					int trno = iTrack + _toc.FirstAudio - 1;
					if (trno >= AccDisks[di].tracks.Count)
						continue;
					count += AccDisks[di].tracks[trno].count;
					if (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC
						&& 0 != AccDisks[di].tracks[trno].CRC)
						conf += AccDisks[di].tracks[trno].count;
					if (CRC450(iTrack, oi) == AccDisks[di].tracks[trno].Frame450CRC
						&& 0 != AccDisks[di].tracks[trno].Frame450CRC)
						partials += AccDisks[di].tracks[trno].count;
				}
				if (conf > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) Accurately ripped", iTrack + 1, CRC(iTrack, oi), count, conf));
				else if (partials > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) No match but offset", iTrack + 1, CRC(iTrack, oi), count, partials));
				else
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] (00/{2:00}) No match", iTrack + 1, CRC(iTrack, oi), count));
			}
		}

		public void GenerateFullLog(TextWriter sw, bool verbose)
		{
			if (AccResult == HttpStatusCode.NotFound)
			{
				sw.WriteLine("Disk not present in database.");
				//for (iTrack = 0; iTrack < TrackCount; iTrack++)
				//    sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] Disk not present in database", iTrack + 1, _tracks[iTrack].CRC));
			}
			else if (AccResult != HttpStatusCode.OK)
			{
				sw.WriteLine("Database access error: " + AccResult.ToString());
				//for (iTrack = 0; iTrack < TrackCount; iTrack++)
				//    sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] Database access error {2}", iTrack + 1, _tracks[iTrack].CRC, accResult.ToString()));
			}
			else
			{
				if (verbose)
				{
					sw.WriteLine("Track\t[ CRC    ] Status");
					GenerateLog(sw, 0);
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
							GenerateLog(sw, oi);
						}
					}
					offsets_match = 0;
					for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					{
						uint matches = 0, partials = 0;
						for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
							for (int di = 0; di < (int)AccDisks.Count; di++)
							{
								int trno = iTrack + _toc.FirstAudio - 1;
								if (trno < AccDisks[di].tracks.Count
									&& (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC 
									&& AccDisks[di].tracks[trno].CRC != 0))
								{
									matches++;
									break;
								}
								if (trno < AccDisks[di].tracks.Count
									&& (CRC450(iTrack, oi) == AccDisks[di].tracks[trno].Frame450CRC 
									&& AccDisks[di].tracks[trno].Frame450CRC != 0))
									partials++;
							}
						if (matches != _toc.AudioTracks && oi != 0 && matches + partials != 0)
						{
							if (offsets_match++ > 16)
							{
								sw.WriteLine("More than 16 offsets match!");
								break;
							}
							sw.WriteLine("Offsetted by {0}:", oi);
							GenerateLog(sw, oi);
						}
					}
				}
				else
				{
					sw.WriteLine("Track\t Status");
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
									&& CRC(iTrack, oi) == AccDisks[iDisk].tracks[trno].CRC 
									&& (AccDisks[iDisk].tracks[trno].CRC != 0 || oi == 0))
								{
									conf += AccDisks[iDisk].tracks[trno].count;
									if (oi == 0)
										zeroOffset = true;
									pressings.AppendFormat("{0}{1}({2})", pressings.Length > 0 ? "," : "", oi, AccDisks[iDisk].tracks[trno].count);
								}
							}
						if (conf > 0 && zeroOffset && pressings.Length == 0)
							sw.WriteLine(String.Format(" {0:00}\t ({2:00}/{1:00}) Accurately ripped", iTrack + 1, total, conf));
						else if (conf > 0 && zeroOffset)
							sw.WriteLine(String.Format(" {0:00}\t ({2:00}/{1:00}) Accurately ripped, all offset(s) {3}", iTrack + 1, total, conf, pressings));
						else if (conf > 0)
							sw.WriteLine(String.Format(" {0:00}\t ({2:00}/{1:00}) Accurately ripped with offset(s) {3}", iTrack + 1, total, conf, pressings));
						else if (total > 0)
							sw.WriteLine(String.Format(" {0:00}\t (00/{1:00}) NOT ACCURATE", iTrack + 1, total));
						else
							sw.WriteLine(String.Format(" {0:00}\t (00/00) Track not present in database", iTrack + 1));
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

		public HttpStatusCode AccResult
		{
			get
			{
				return _accResult;
			}
		}

		public string ARStatus
		{
			get
			{
				return _accResult == HttpStatusCode.NotFound ? "disk not present in database" :
					_accResult == HttpStatusCode.OK ? null
					: _accResult.ToString();
			}
		}

		CDImageLayout _toc;
		long _sampleCount, _finalSampleCount;
		int _currentTrack;
		private List<AccDisk> _accDisks;
		private HttpStatusCode _accResult;
		internal uint[,] _CRCAR;
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
