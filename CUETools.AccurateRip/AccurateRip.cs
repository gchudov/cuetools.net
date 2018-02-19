using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using CUETools.Parity;
using CUETools.CDImage;
using CUETools.Codecs;

namespace CUETools.AccurateRip
{
	public class AccurateRipVerify : IAudioDest
	{
        public const int maxNpar = 16;

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

		public uint WorstConfidence(int oi)
		{
			uint worstConfidence = 0xffff;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				int trno = iTrack + _toc.FirstAudio - 1;
				uint sum = 0;
				// sum all matches for this track and this offset
				for (int di = 0; di < AccDisks.Count; di++)
					if (trno < AccDisks[di].tracks.Count
					&& (CRC(iTrack, oi) == AccDisks[di].tracks[trno].CRC
					  || oi == 0 && CRCV2(iTrack) == AccDisks[di].tracks[trno].CRC))
						sum += AccDisks[di].tracks[iTrack + _toc.FirstAudio - 1].count;
				// exclude silent tracks
				if (worstConfidence > sum && (Total(iTrack) != 0 || CRC(iTrack, oi) != 0))
					worstConfidence = sum;
			}
			return worstConfidence;
		}

		public uint WorstConfidence()
		{
			if (ARStatus != null)
				return 0U;
			uint worstConfidence = 0;
			for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
				worstConfidence += WorstConfidence(oi);
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
			return ARStatus == null && iTrack + _toc.FirstAudio - 1 < AccDisks[0].tracks.Count
				? AccDisks[0].tracks[iTrack + _toc.FirstAudio - 1].CRC : 0U;
		}

		public uint CRC(int iTrack)
		{
			return CRC(iTrack, 0);
		}

		public uint CRCV2(int iTrack)
		{
			int offs0 = iTrack == 0 ? 5 * 588 - 1 : 0;
			int offs1 = iTrack == _toc.AudioTracks - 1 ? 2 * maxOffset - 5 * 588 : 0;
			uint crcA1 = _CRCAR[iTrack + 1, offs1] - (offs0 > 0 ? _CRCAR[iTrack + 1, offs0] : 0);
			uint crcA2 = _CRCV2[iTrack + 1, offs1] - (offs0 > 0 ? _CRCV2[iTrack + 1, offs0] : 0);
			return crcA1 + crcA2;
		}

		public uint CRC(int iTrack, int oi)
		{
			int offs0 = iTrack == 0 ? 5 * 588 + oi - 1 : oi;
			int offs1 = iTrack == _toc.AudioTracks - 1 ? 2 * maxOffset - 5 * 588 + oi : (oi >= 0 ? 0 : 2 * maxOffset + oi);
			uint crcA = _CRCAR[iTrack + 1, offs1] - (offs0 > 0 ? _CRCAR[iTrack + 1, offs0] : 0);
			uint sumA = _CRCSM[iTrack + 1, offs1] - (offs0 > 0 ? _CRCSM[iTrack + 1, offs0] : 0);
			uint crc = crcA - sumA * (uint)oi;
			if (oi < 0 && iTrack > 0)
			{
				uint crcB = _CRCAR[iTrack, 0] - _CRCAR[iTrack, 2 * maxOffset + oi];
				uint sumB = _CRCSM[iTrack, 0] - _CRCSM[iTrack, 2 * maxOffset + oi];
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
			uint crca = _CRCAR[iTrack + 1, 2 * maxOffset + 1 + 5 * 588 + oi];
			uint crcb = _CRCAR[iTrack + 1, 2 * maxOffset + 1 + 6 * 588 + oi];
			uint suma = _CRCSM[iTrack + 1, 2 * maxOffset + 1 + 5 * 588 + oi];
			uint sumb = _CRCSM[iTrack + 1, 2 * maxOffset + 1 + 6 * 588 + oi];
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
					int dlen = (int)_toc.AudioLength * 588;
					if (oi > 0)
					{
						// whole disc crc
						crc = _CRC32[_toc.AudioTracks, 2 * maxOffset];
						// - prefix
						crc = Crc32.Combine(_CRC32[0, oi], crc, (dlen - oi) * 4);
						// + zero suffix
						crc = Crc32.Combine(crc, 0, oi * 4);
					}
					else // if (oi <= 0)
					{
						crc = _CRC32[_toc.AudioTracks, 2 * maxOffset + oi];
					}

					// Use 0xffffffff as an initial state
					crc ^= _CRCMASK[0];
				}
				else
				{
					int trackLength = (int)(iTrack > 0 ? _toc[iTrack + _toc.FirstAudio - 1].Length : _toc[_toc.FirstAudio].Pregap) * 588 * 4;
					if (oi > 0)
					{
						crc = iTrack < _toc.AudioTracks ? _CRC32[iTrack + 1, oi]
							: Crc32.Combine(_CRC32[iTrack, 2 * maxOffset], 0, oi * 4);
						crc = Crc32.Combine(_CRC32[iTrack, oi], crc, trackLength);
					}
					else //if (oi <= 0)
					{
						crc = Crc32.Combine(_CRC32[iTrack - 1, 2 * maxOffset + oi], _CRC32[iTrack, 2 * maxOffset + oi], trackLength);
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
				uint crc;
				int cnt;
				if (iTrack == 0)
				{
					if (oi > 0)
					{
						// whole disc crc
						cnt = _CRCNL[_toc.AudioTracks, 2 * maxOffset] * 2;
						crc = _CRCWN[_toc.AudioTracks, 2 * maxOffset];
						// - prefix
						cnt -= _CRCNL[0, oi] * 2;
						crc = Crc32.Combine(_CRCWN[0, oi], crc, cnt);
					}
					else // if (oi <= 0)
					{
						cnt = _CRCNL[_toc.AudioTracks, 2 * maxOffset + oi] * 2;
						crc = _CRCWN[_toc.AudioTracks, 2 * maxOffset + oi];
					}
				}
				else
				{
					if (oi > 0)
					{
						cnt = (iTrack < _toc.AudioTracks ? _CRCNL[iTrack + 1, oi] : _CRCNL[iTrack, 2 * maxOffset]) * 2;
						crc = iTrack < _toc.AudioTracks ? _CRCWN[iTrack + 1, oi] : _CRCWN[iTrack, 2 * maxOffset];

						cnt -= _CRCNL[iTrack, oi] * 2;
						crc = Crc32.Combine(_CRCWN[iTrack, oi], crc, cnt);
					}
					else //if (oi <= 0)
					{
						cnt = _CRCNL[iTrack, 2 * maxOffset + oi] * 2;
						crc = _CRCWN[iTrack, 2 * maxOffset + oi];

						cnt -= _CRCNL[iTrack - 1, 2 * maxOffset + oi] * 2;
						crc = Crc32.Combine(_CRCWN[iTrack - 1, 2 * maxOffset + oi], crc, cnt);
					}

				}
				// Use 0xffffffff as an initial state
				crc = Crc32.Combine(0xffffffff, crc, cnt);
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

        public unsafe ushort[,] GetSyndrome(int npar = maxNpar, int strides = -1, int offset = 0)
        {
            // We can only use offset if Abs(offset * 2) < stride,
            // else we might need to add/remove more than one sample
            // from syndrome calculations, and that would be too difficult
            // and will probably require longer leadin/leadout.
            if (!calcParity)
                throw new InvalidOperationException();
            if (strides == -1)
                strides = stride;
            var syn = ParityToSyndrome.Parity2Syndrome(strides, stride, npar, maxNpar, parity, 0, -offset * 2);
            var galois = Galois16.instance;
            for (int part2 = 0; part2 < strides; part2++)
            {
                int part = (part2 + offset * 2 + stride) % stride;
                if (part < offset * 2)
                {
                    for (int i = 0; i < npar; i++)
                    {
                        int synI = syn[part2, i];
                        synI = galois.mulExp(synI, i);
                        synI ^= leadout[laststride - part - 1] ^ galois.mulExp(leadin[stride + part], (i * stridecount) % galois.Max);
                        syn[part2, i] = (ushort)synI;
                    }
                }
                if (part >= stride + offset * 2)
                {
                    for (int i = 0; i < npar; i++)
                    {
                        int synI = syn[part2, i];
                        synI ^= leadout[laststride + stride - part - 1] ^ galois.mulExp(leadin[part], (i * stridecount) % galois.Max);
                        synI = galois.divExp(synI, i);
                        syn[part2, i] = (ushort)synI;
                    }
                }
            }
            //for (int part = 0; part < offset * 2; part++)
            //{
            //    int part2 = (part - offset * 2 + stride) % stride;
            //    if (part2 < strides)
            //    for (int i = 0; i < npar; i++)
            //    {
            //        int synI = syn[part2, i];
            //        synI = galois.mulExp(synI, i);
            //        synI ^= leadout[laststride - part - 1] ^ galois.mulExp(leadin[stride + part], (i * stridecount) % galois.Max);
            //        syn[part2, i] = (ushort)synI;
            //    }
            //}
            //for (int part = stride + offset * 2; part < stride; part++)
            //{
            //    int part2 = (part - offset * 2 + stride) % stride;
            //    if (part2 < strides)
            //    for (int i = 0; i < npar; i++)
            //    {
            //        int synI = syn[part2, i];
            //        synI ^= leadout[laststride + stride - part - 1] ^ galois.mulExp(leadin[part], (i * stridecount) % galois.Max);
            //        synI = galois.divExp(synI, i);
            //        syn[part2, i] = (ushort)synI;
            //    }
            //}
            return syn;
        }

		private byte[] parity;
		internal ushort[, ,] encodeTable;
		private int maxOffset;
		internal ushort[] leadin;
		internal ushort[] leadout;
		private int stride = 1, laststride = 1, stridecount = 1;
		private bool calcParity = false;

		internal void InitCDRepair(int stride, int laststride, int stridecount, bool calcParity)
		{
			if (stride % 2 != 0 || laststride % 2 != 0)
				throw new ArgumentOutOfRangeException("stride");
			this.stride = stride;
			this.laststride = laststride;
			this.stridecount = stridecount;
			this.calcParity = calcParity;
			Init(_toc);
		}

		public unsafe uint CTDBCRC(int iTrack, int oi, int prefixSamples, int suffixSamples)
		{
			prefixSamples += oi;
			suffixSamples -= oi;
			if (prefixSamples < 0 || prefixSamples >= maxOffset || suffixSamples < 0 || suffixSamples > maxOffset)
				throw new ArgumentOutOfRangeException();

			if (iTrack == 0)
			{
				int discLen = ((int)_toc.AudioLength - (int)TOC.Pregap) * 588;
				int chunkLen = discLen - prefixSamples - suffixSamples;
				return 0xffffffff ^ Crc32.Combine(
					0xffffffff ^ _CRC32[1, prefixSamples],
					_CRC32[_toc.AudioTracks, 2 * maxOffset - suffixSamples],
					chunkLen * 4);
			}
			int posA = (int)_toc[iTrack + _toc.FirstAudio - 1].Start * 588 + (iTrack > 1 ? oi : prefixSamples);
			int posB = iTrack < _toc.AudioTracks ?
				(int)_toc[iTrack + 1 + _toc.FirstAudio - 1].Start * 588 + oi :
				(int)_toc.Leadout * 588 - suffixSamples;
			uint crcA, crcB;
			if (oi > 0)
			{
				crcA = iTrack > 1 ?
					_CRC32[iTrack, oi] :
					_CRC32[iTrack, prefixSamples];
				crcB = iTrack < _toc.AudioTracks ?
					_CRC32[iTrack + 1, oi] :
					_CRC32[iTrack, maxOffset * 2 - suffixSamples];
			}
			else //if (oi <= 0)
			{
				crcA = iTrack > 1 ?
					_CRC32[iTrack - 1, maxOffset * 2 + oi] :
					_CRC32[iTrack, prefixSamples];
				crcB = iTrack < _toc.AudioTracks ?
					_CRC32[iTrack, maxOffset * 2 + oi] :
					_CRC32[iTrack, maxOffset * 2 - suffixSamples];
			}
            // Use 0xffffffff as an initial state
            return 0xffffffff ^ Crc32.Combine(0xffffffff ^ crcA, crcB, (posB - posA) * 4);
		}

		public uint CTDBCRC(int offset)
		{
			return CTDBCRC(0, offset, stride / 2, laststride / 2);
		}

        // pt = &encodeTable
        private unsafe delegate void SyndromeCalc(ushort* pt, ushort* wr, ushort lo);

        private unsafe static void SyndromeCalcDummy(ushort* pt, ushort* wr, ushort lo)
        {
        }

        private unsafe static void SyndromeCalc8(ushort* pt, ushort* wr, ushort lo)
        {
            ushort wrlo = (ushort)(wr[0] ^ lo);
            ushort* ptiblo0 = pt + (wrlo & 255) * maxNpar * 2;
            ushort* ptiblo1 = pt + (wrlo >> 8) * maxNpar * 2 + maxNpar;
            ((ulong*)wr)[0] = ((ulong*)(wr + 1))[0] ^ ((ulong*)ptiblo0)[0] ^ ((ulong*)ptiblo1)[0];
            ((ulong*)wr)[1] = (((ulong*)(wr))[1] >> 16) ^ ((ulong*)ptiblo0)[1] ^ ((ulong*)ptiblo1)[1];
        }

        //[System.Runtime.InteropServices.DllImport("CUETools.AVX.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.StdCall)]
        //private unsafe static extern void SyndromeCalc16AVX(ushort* table, ushort* parity, ushort* samples, int n);

        private unsafe static void SyndromeCalc16(ushort* pt, ushort* wr, ushort lo)
        {
            ushort wrlo = (ushort)(wr[0] ^ lo);
            ushort* ptiblo0 = pt + (wrlo & 255) * maxNpar * 2;
            ushort* ptiblo1 = pt + (wrlo >> 8) * maxNpar * 2 + maxNpar;
            ((ulong*)wr)[0] = ((ulong*)(wr + 1))[0] ^ ((ulong*)ptiblo0)[0] ^ ((ulong*)ptiblo1)[0];
            ((ulong*)wr)[1] = ((ulong*)(wr + 1))[1] ^ ((ulong*)ptiblo0)[1] ^ ((ulong*)ptiblo1)[1];
            ((ulong*)wr)[2] = ((ulong*)(wr + 1))[2] ^ ((ulong*)ptiblo0)[2] ^ ((ulong*)ptiblo1)[2];
            ((ulong*)wr)[3] = (((ulong*)(wr))[3] >> 16) ^ ((ulong*)ptiblo0)[3] ^ ((ulong*)ptiblo1)[3];
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
		public unsafe void CalculateCRCs(uint* t, ushort* wr, ushort* pte, uint* pSampleBuff, int count, int offs)
		{
            int currentSample = Math.Max(0, (int)_sampleCount - 588 * (int)this.TOC.Pregap);
            int currentStride = (currentSample * 2) / stride;
			bool doPar = currentStride >= 1 && currentStride <= stridecount && calcParity;
            SyndromeCalc syndromeCalc = doPar ? maxNpar == 8 ? (SyndromeCalc)SyndromeCalc8 : (SyndromeCalc)SyndromeCalc16 : (SyndromeCalc)SyndromeCalcDummy;

			int crcTrack = _currentTrack + (_samplesDoneTrack == 0 && _currentTrack > 0 ? -1 : 0);
			uint crcar = _CRCAR[_currentTrack, 0];
			uint crcsm = _CRCSM[_currentTrack, 0];
			uint crc32 = _CRC32[crcTrack, 2 * maxOffset];
			uint crcwn = _CRCWN[crcTrack, 2 * maxOffset];
			int crcnl = _CRCNL[crcTrack, 2 * maxOffset];
			uint crcv2 = _CRCV2[_currentTrack, 0];
			int peak = _Peak[_currentTrack];

            //if (doPar) SyndromeCalc16AVX(pte, wr, (ushort*)pSampleBuff, count * 2);

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
					crcnl++;
				}
                syndromeCalc(pte, wr + i * maxNpar * 2, (ushort)lo);

				uint hi = sample >> 16;
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ hi)];
				crc32 = (crc32 >> 8) ^ t[(byte)(crc32 ^ (hi >> 8))];
				if (hi != 0)
				{
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ hi)];
					crcwn = (crcwn >> 8) ^ t[(byte)(crcwn ^ (hi >> 8))];
					crcnl++;
				}
                syndromeCalc(pte, wr + i * maxNpar * 2 + maxNpar, (ushort)hi);

				int pk = ((int)(lo << 16)) >> 16;
				peak = Math.Max(peak, (pk << 1) ^ (pk >> 31));
				pk = ((int)(hi << 16)) >> 16;
				peak = Math.Max(peak, (pk << 1) ^ (pk >> 31));
			}

			_CRCAR[_currentTrack, 0] = crcar;
			_CRCSM[_currentTrack, 0] = crcsm;
			_CRC32[_currentTrack, 2 * maxOffset] = crc32;
			_CRCWN[_currentTrack, 2 * maxOffset] = crcwn;
			_CRCNL[_currentTrack, 2 * maxOffset] = crcnl;
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
			fixed (ushort* pte = encodeTable)
			fixed (byte* pSampleBuff = &sampleBuffer.Bytes[0], bpar = parity)
				while (pos < sampleBuffer.Length)
				{
					// Process no more than there is in the buffer, no more than there is in this track, and no more than up to a sector boundary.
					int copyCount = Math.Min(Math.Min(sampleBuffer.Length - pos, (int)_samplesRemTrack), 588 - (int)_sampleCount % 588);
					uint* samples = ((uint*)pSampleBuff) + pos;
                    int currentSample = (int)_sampleCount - 588 * (int)this.TOC.Pregap;
                    int currentPart = currentSample < 0 ? 0 : (currentSample * 2) % stride;
					//ushort* synptr = synptr1 + npar * currentPart;
                    ushort* wr = ((ushort*)bpar) + maxNpar * currentPart;

                    for (int i = Math.Max(0, - currentSample * 2); i < Math.Min(leadin.Length - currentSample * 2, copyCount * 2); i++)
                        leadin[currentSample * 2 + i] = ((ushort*)samples)[i];

					for (int i = Math.Max(0, (int)(_finalSampleCount - _sampleCount) * 2 - leadout.Length); i < copyCount * 2; i++)
					{
						int remaining = (int)(_finalSampleCount - _sampleCount) * 2 - i - 1;
						leadout[remaining] = ((ushort*)samples)[i];
					}

					int offset = _samplesDoneTrack < maxOffset ? _samplesDoneTrack
						: _samplesRemTrack <= maxOffset ? 2 * maxOffset - _samplesRemTrack
						: _samplesDoneTrack >= 445 * 588 && _samplesDoneTrack <= 455 * 588 ? 2 * maxOffset + 1 + _samplesDoneTrack - 445 * 588
						: -1;

					CalculateCRCs(t, wr, pte, samples, copyCount, offset);

					// duplicate prefix to suffix
					if (_samplesDoneTrack < maxOffset && _samplesRemTrack <= maxOffset)
					{
						Array.Copy(_CRC32, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRC32, _currentTrack * 3 * maxOffset + 2 * maxOffset - _samplesRemTrack,
							copyCount);
						Array.Copy(_CRCWN, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRCWN, _currentTrack * 3 * maxOffset + 2 * maxOffset - _samplesRemTrack,
							copyCount);
						Array.Copy(_CRCNL, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRCNL, _currentTrack * 3 * maxOffset + 2 * maxOffset - _samplesRemTrack,
							copyCount);
					}
					// duplicate prefix to pregap
					if (_sampleCount < maxOffset && _currentTrack == 1)
					{
						Array.Copy(_CRC32, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRC32, _sampleCount,
							copyCount);
						Array.Copy(_CRCWN, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRCWN, _sampleCount,
							copyCount);
						Array.Copy(_CRCNL, _currentTrack * 3 * maxOffset + _samplesDoneTrack,
							_CRCNL, _sampleCount,
							copyCount);
					}

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

		public unsafe void Combine(AccurateRipVerify part, int start, int end)
		{
			for (int i = 0; i < leadin.Length; i++)
			{
                int currentOffset = i / 2 + 588 * (int)this.TOC.Pregap;
				if (currentOffset >= start && currentOffset < end)
					this.leadin[i] = part.leadin[i];
			}
			for (int i = 0; i < leadout.Length; i++)
			{
				int currentOffset = (int)_finalSampleCount - i / 2 - 1;
				if (currentOffset >= start && currentOffset < end)
					this.leadout[i] = part.leadout[i];
			}
			int iSplitTrack = -1;
			for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
			{
				int tempLocation = (int)(iTrack == 0 ? 0 : _toc[_toc.FirstAudio + iTrack - 1].Start - _toc[_toc.FirstAudio][0].Start) * 588;
				int tempLen = (int)(iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
				if (start > tempLocation && start <= tempLocation + tempLen)
				{
					iSplitTrack = iTrack;
					break;
				}
			}

			uint crc32 = _CRC32[iSplitTrack, 2 * maxOffset];
			uint crcwn = _CRCWN[iSplitTrack, 2 * maxOffset];
			int crcnl = _CRCNL[iSplitTrack, 2 * maxOffset];

			for (int iTrack = 0; iTrack <= _toc.AudioTracks; iTrack++)
			{
				// ??? int tempLocation = (int) (iTrack == 0 ? _toc[_toc.FirstAudio][0].Start : _toc[_toc.FirstAudio + iTrack - 1].Start) * 588;
				int tempLocation = (int)(iTrack == 0 ? 0 : _toc[_toc.FirstAudio + iTrack - 1].Start - _toc[_toc.FirstAudio][0].Start) * 588;
				int tempLen = (int)(iTrack == 0 ? _toc[_toc.FirstAudio].Pregap : _toc[_toc.FirstAudio + iTrack - 1].Length) * 588;
				int trStart = Math.Max(tempLocation, start);
				int trEnd = Math.Min(tempLocation + tempLen, end);
				if (trStart >= trEnd)
					continue;

				uint crcar = _CRCAR[iTrack, 0];
				uint crcv2 = _CRCV2[iTrack, 0];
				uint crcsm = _CRCSM[iTrack, 0];
				_CRCAR[iTrack, 0] = crcar + part._CRCAR[iTrack, 0];
				_CRCSM[iTrack, 0] = crcsm + part._CRCSM[iTrack, 0];
				_CRCV2[iTrack, 0] = crcv2 + part._CRCV2[iTrack, 0];

				for (int i = 0; i < 3 * maxOffset; i++)
				{
					int currentOffset;
					if (i < maxOffset)
						currentOffset = tempLocation + i;
					else if (i < 2 * maxOffset)
						currentOffset = tempLocation + tempLen + i - 2 * maxOffset;
					else if (i == 2 * maxOffset)
						currentOffset = trEnd;
					else //if (i > 2 * maxOffset)
						currentOffset = tempLocation + i - 1 - 2 * maxOffset + 445 * 588;

					if (currentOffset < trStart || currentOffset > trEnd)
						continue;

					_CRC32[iTrack, i] = Crc32.Combine(crc32, part._CRC32[iTrack, i], 4 * (currentOffset - start));
					_CRCWN[iTrack, i] = Crc32.Combine(crcwn, part._CRCWN[iTrack, i], part._CRCNL[iTrack, i] * 2);
					_CRCNL[iTrack, i] = crcnl + part._CRCNL[iTrack, i];
					if (i == 0 || i == 2 * maxOffset) continue;
					_CRCAR[iTrack, i] = crcar + part._CRCAR[iTrack, i];
					_CRCV2[iTrack, i] = crcv2 + part._CRCV2[iTrack, i];
					_CRCSM[iTrack, i] = crcsm + part._CRCSM[iTrack, i];
				}
				_Peak[iTrack] = Math.Max(_Peak[iTrack], part._Peak[iTrack]);
			}

			if (calcParity)
			{
				var newSyndrome1 = this.GetSyndrome();
				var newSyndrome2 = part.GetSyndrome();
                int firstSample = 588 * (int)TOC.Pregap;
                var i1 = Math.Max(0, (start - firstSample) * 2 - stride);
                var i2 = Math.Min(2 * ((int)_finalSampleCount - firstSample) - laststride - stride, (end - firstSample) * 2 - stride);
				var diff = i2 / stride - i1 / stride;
				var i1s = i1 % stride;
				var i2s = i2 % stride;
				var imin = Math.Min(i1s, i2s);
				var imax = Math.Max(i1s, i2s);
				var diff1 = diff + (imin < i2s ? 1 : 0) - (imin < i1s ? 1 : 0);
				for (int i = 0; i < stride; i++)
                    for (int j = 0; j < maxNpar; j++)
					{
						var d1 = j * (i >= imin && i < imax ? diff1 : diff);
						newSyndrome1[i, j] = (ushort)(newSyndrome2[i, j] ^ Galois16.instance.mulExp(newSyndrome1[i, j], (d1 & 0xffff) + (d1 >> 16)));
					}

                ParityToSyndrome.Syndrome2Parity(newSyndrome1, this.parity);
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

			maxOffset = Math.Max(4096 * 2, calcParity ? stride + laststride : 0);
			if (maxOffset % 588 != 0)
				maxOffset += 588 - maxOffset % 588;
			_CRCAR = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CRCSM = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CRC32 = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CacheCRC32 = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CRCWN = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CacheCRCWN = new uint[_toc.AudioTracks + 1, 3 * maxOffset];
			_CRCNL = new int[_toc.AudioTracks + 1, 3 * maxOffset];
			_CRCV2 = new uint[_toc.AudioTracks + 1, 3 * maxOffset];

			_Peak = new int[_toc.AudioTracks + 1];
			parity = null;
            if (calcParity)
            {
                parity = new byte[stride * maxNpar * 2];
                encodeTable = Galois16.instance.makeEncodeTable(maxNpar);
            }

			int leadin_len = Math.Max(4096 * 4, calcParity ? stride * 2 : 0);
			int leadout_len = Math.Max(4096 * 4, calcParity ? stride + laststride : 0);
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
		static readonly TimeSpan min_interval = new TimeSpan(5000000); // 0.5 second
		static readonly object server_mutex = new object();

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

        public AudioEncoderSettings Settings
		{
			get
			{
				return new AudioEncoderSettings(AudioPCMConfig.RedBook);
			}
		}

		public CDImageLayout TOC
		{
			get { return _toc; }
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

		public string Path
		{
			get { throw new Exception("unsupported"); }
		}

		public void GenerateLog(TextWriter sw, int oi)
		{
			uint maxTotal = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
				maxTotal = Math.Max(maxTotal, Total(iTrack));

			uint maxConf = 0, maxConf2 = 0;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint crcOI = CRC(iTrack, oi);
				uint crcOI2 = CRCV2(iTrack);
				for (int di = 0; di < (int)AccDisks.Count; di++)
				{
					int trno = iTrack + _toc.FirstAudio - 1;
					if (trno < AccDisks[di].tracks.Count
						&& crcOI == AccDisks[di].tracks[trno].CRC
						&& 0 != AccDisks[di].tracks[trno].CRC
						)
						maxConf = Math.Max(maxConf, AccDisks[di].tracks[trno].count);
					if (trno < AccDisks[di].tracks.Count
						&& 0 == oi
						&& crcOI2 == AccDisks[di].tracks[trno].CRC
						&& 0 != AccDisks[di].tracks[trno].CRC
						)
						maxConf2 = Math.Max(maxConf, AccDisks[di].tracks[trno].count);
				}
			}
			string ifmt = maxTotal < 10 ? ":0" : maxTotal < 100 ? ":00" : ":000";
			//string ifmt = maxTotal < 10 ? ",1" : maxTotal < 100 ? ",2" : ",3";
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				uint count = 0;
				uint partials = 0;
				uint conf = 0;
				uint conf2 = 0;
				uint crcOI = CRC(iTrack, oi);
				uint crcOI2 = CRCV2(iTrack);
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
					if (crcOI2 == AccDisks[di].tracks[trno].CRC
						&& 0 == oi
						&& 0 != AccDisks[di].tracks[trno].CRC)
						conf2 += AccDisks[di].tracks[trno].count;
					if (crc450OI == AccDisks[di].tracks[trno].Frame450CRC
						&& 0 != AccDisks[di].tracks[trno].Frame450CRC)
						partials++;
				}
				string status;
				if (conf + conf2 > 0)
					status = "Accurately ripped";
				else if (count == 0 && crcOI == 0)
					status = "Silent track";
				else if (partials > 0 && 0 != oi)
					status = "No match (V2 was not tested)";
				else if (partials > 0)
					status = "No match";
				else
					status = "No match";
				if (oi == 0)
					sw.WriteLine(String.Format(" {0:00}     [{1:x8}|{5:x8}] ({3" + ifmt + "}+{6" + ifmt + "}/{2" + ifmt + "}) {4}", iTrack + 1, crcOI, count, conf, status, crcOI2, conf2));
				else
					sw.WriteLine(String.Format(" {0:00}     [{1:x8}] ({3" + ifmt + "}/{2" + ifmt + "}) {4}", iTrack + 1, crcOI, count, conf, status));
			}
		}

		public void GenerateFullLog(TextWriter sw, bool verbose, string id)
		{
            if (ExceptionStatus != WebExceptionStatus.Pending)
			    sw.WriteLine("[AccurateRip ID: {0}] {1}.", id, ARStatus ?? "found");
			if (ExceptionStatus == WebExceptionStatus.Success)
			{
				if (verbose)
				{
					sw.WriteLine("Track   [  CRC   |   V2   ] Status");
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
							GenerateLog(sw, oi);
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
						{
							int oiMin = _arOffsetRange;
							int oiMax = -_arOffsetRange;
							for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
								if (CRCLOG(iTrack) == CRCWONULL(iTrack, oi))
								{
									oiMin = Math.Min(oiMin, oi);
									oiMax = Math.Max(oiMax, oi);
								}
							if (oiMax >= oiMin)
							{
								inLog = " W/O NULL ";
								extra = oiMax == oiMin
									? string.Format(": offset {0}", oiMin)
									: string.Format(": offset {0}..{1}", oiMin, oiMax);
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
			if (!File.Exists(fileName) || (DateTime.Now - File.GetLastWriteTime(fileName) > TimeSpan.FromDays(10)))
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
				catch (WebException)
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
			query.AppendFormat("+{0}", toc.Length / 75 + 2);
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
