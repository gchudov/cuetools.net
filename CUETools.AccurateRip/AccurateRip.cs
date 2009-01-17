using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.Globalization;
using System.Net;
using System.IO;
using CUETools.Codecs;
using CUETools.CDImage;

namespace CUETools.AccurateRip
{
	public class AccurateRipVerify : IAudioDest
	{
		public AccurateRipVerify(CDImageLayout toc)
		{
			_toc = toc;
			_accDisks = new List<AccDisk>();
			_crc32 = new Crc32();
			Init();
		}

		unsafe private void CalculateAccurateRipCRCsSemifast(int* samples, uint count, int iTrack, uint currentOffset, uint previousOffset, uint trackLength)
		{
			fixed (uint* CRCsA = &_offsetedCRC[Math.Max(0, iTrack - 1), 0],
				CRCsB = &_offsetedCRC[iTrack, 0],
				CRCsC = &_offsetedCRC[Math.Min(_toc.AudioTracks - 1, iTrack + 1), 0]
				//CRC32A = &_offsetedCRC32[Math.Max(0, iTrack - 1), 0],
				//CRC32B = &_offsetedCRC32[iTrack, 0],
				//CRC32C = &_offsetedCRC32[Math.Min(_toc.AudioTracks - 1, iTrack + 1), 0]
				)
			{
				for (uint si = 0; si < count; si++)
				{
					uint sampleValue = (uint)((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
					int i;
					int iB = Math.Max(0, _arOffsetRange - (int)(currentOffset + si));
					int iC = Math.Min(2 * _arOffsetRange + 1, _arOffsetRange + (int)trackLength - (int)(currentOffset + si));
					
					uint baseSumA = sampleValue * (uint)(previousOffset + 1 - iB);
					for (i = 0; i < iB; i++)
					{
						CRCsA[i] += baseSumA;
						baseSumA += sampleValue;
						//CRC32A[i] = _crc32.ComputeChecksum(CRC32A[i], sampleValue);
					}
					uint baseSumB = sampleValue * (uint)Math.Max(1, (int)(currentOffset + si) - _arOffsetRange + 1);
					for (i = iB; i < iC; i++)
					{
						CRCsB[i] += baseSumB;
						baseSumB += sampleValue;
						//CRC32B[i] = _crc32.ComputeChecksum(CRC32B[i], sampleValue);
					}
					uint baseSumC = sampleValue;
					for (i = iC; i <= 2 * _arOffsetRange; i++)
					{
						CRCsC[i] += baseSumC;
						baseSumC += sampleValue;
						//CRC32C[i] = _crc32.ComputeChecksum(CRC32C[i], sampleValue);
					}
				}
				return;
			}
		}

		unsafe private void CalculateAccurateRipCRCs(int* samples, uint count, int iTrack, uint currentOffset, uint previousOffset, uint trackLength)
		{
			for (int si = 0; si < count; si++)
			{
				uint sampleValue = (uint)((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
				
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
				{
					int iTrack2 = iTrack;
					int currentOffset2 = (int)currentOffset + si - oi;

					if (currentOffset2 < 5 * 588 - 1 && iTrack == 0)
					// we are in the skipped area at the start of the disk
					{
						continue;
					}
					else if (currentOffset2 < 0)
					// offset takes us to previous track
					{
						iTrack2--;
						currentOffset2 += (int)previousOffset;
					}
					else if (currentOffset2 >= trackLength - 5 * 588 && iTrack == _toc.AudioTracks - 1)
					// we are in the skipped area at the end of the disc
					{
						continue;
					}
					else if (currentOffset2 >= trackLength)
					// offset takes us to the next track
					{
						iTrack2++;
						currentOffset2 -= (int)trackLength;
					}
					_offsetedCRC[iTrack2, _arOffsetRange - oi] += sampleValue * (uint)(currentOffset2 + 1);
					//_offsetedCRC32[iTrack2, _arOffsetRange - oi] = _crc32.ComputeChecksum(_offsetedCRC32[iTrack2, _arOffsetRange - oi], sampleValue);
				}
			}
		}

		unsafe private void CalculateAccurateRipCRCsFast(int* samples, uint count, int iTrack, uint currentOffset)
		{
			int s1 = (int)Math.Min(count, Math.Max(0, 450 * 588 - _arOffsetRange - (int)currentOffset));
			int s2 = (int)Math.Min(count, Math.Max(0, 451 * 588 + _arOffsetRange - (int)currentOffset));
			if (s1 < s2)
				fixed (uint* FrameCRCs = &_offsetedFrame450CRC[iTrack, 0])
					for (int sj = s1; sj < s2; sj++)
					{
						int magicFrameOffset = (int)currentOffset + sj - 450 * 588 + 1;
						int firstOffset = Math.Max(-_arOffsetRange, magicFrameOffset - 588);
						int lastOffset = Math.Min(magicFrameOffset - 1, _arOffsetRange);
						uint sampleValue = (uint)((samples[2 * sj] & 0xffff) + (samples[2 * sj + 1] << 16));
						for (int oi = firstOffset; oi <= lastOffset; oi++)
							FrameCRCs[_arOffsetRange - oi] += sampleValue * (uint)(magicFrameOffset - oi);
					}
			fixed (uint* CRCs = &_offsetedCRC[iTrack, 0])
			{
				uint baseSum = 0, stepSum = 0;
				currentOffset += (uint)_arOffsetRange + 1;
				for (uint si = 0; si < count; si++)
				{
					uint sampleValue = (uint)((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
					stepSum += sampleValue;
					baseSum += sampleValue * (uint)(currentOffset + si);
				}
				for (int i = 2 * _arOffsetRange; i >= 0; i--)
				{
					CRCs[i] += baseSum;
					baseSum -= stepSum;
					//CRC32[i] = _crc32.ComputeChecksum (CRC32[i], samples, count);
				}
			}
		}

		public uint CRC(int iTrack)
		{
			return _offsetedCRC[iTrack, _arOffsetRange];
		}

		public uint BackupCRC(int iTrack)
		{
			return _backupCRC[iTrack];
		}

		public uint CRC(int iTrack, int oi)
		{
			return _offsetedCRC[iTrack, _arOffsetRange - oi];
		}

		public uint CRC32(int iTrack)
		{
			return _offsetedCRC32[iTrack] ^ 0xffffffff;
		}

		public uint CRC450(int iTrack, int oi)
		{
			return _offsetedFrame450CRC[iTrack, _arOffsetRange - oi];
		}

		public void Write(int[,] sampleBuffer, uint sampleCount)
		{
			for (uint pos = 0; pos < sampleCount; )
			{
				uint copyCount = Math.Min(sampleCount - pos, (uint)_samplesRemTrack);
				unsafe
				{
					fixed (int* pSampleBuff = &sampleBuffer[pos, 0])
					{
						if (_currentTrack != 0)
						{
							uint trackLength = _toc[_currentTrack].Length * 588;
							uint currentOffset = (uint)_sampleCount - _toc[_currentTrack].Start * 588;
							uint previousOffset = _currentTrack > 1 ? _toc[_currentTrack - 1].Length * 588 : _toc.Pregap * 588;
							uint si1 = (uint)Math.Min(copyCount, Math.Max(0, 588 * (_currentTrack == 1 ? 10 : 5) - (int)currentOffset));
							uint si2 = (uint)Math.Min(copyCount, Math.Max(si1, trackLength - (int)currentOffset - 588 * (_currentTrack == _toc.AudioTracks ? 10 : 5)));
							if (_currentTrack == 1)
								CalculateAccurateRipCRCs(pSampleBuff, si1, _currentTrack - 1, currentOffset, previousOffset, trackLength);
							else
								CalculateAccurateRipCRCsSemifast(pSampleBuff, si1, _currentTrack - 1, currentOffset, previousOffset, trackLength);
							if (si2 > si1)
								CalculateAccurateRipCRCsFast(pSampleBuff + si1 * 2, (uint)(si2 - si1), _currentTrack - 1, currentOffset + si1);
							if (_currentTrack == _toc.AudioTracks)
								CalculateAccurateRipCRCs(pSampleBuff + si2 * 2, copyCount - si2, _currentTrack - 1, currentOffset + si2, previousOffset, trackLength);
							else
								CalculateAccurateRipCRCsSemifast(pSampleBuff + si2 * 2, copyCount - si2, _currentTrack - 1, currentOffset + si2, previousOffset, trackLength);

							_offsetedCRC32[_currentTrack] = _crc32.ComputeChecksum(_offsetedCRC32[_currentTrack], pSampleBuff, copyCount);
						}
						_offsetedCRC32[0] = _crc32.ComputeChecksum(_offsetedCRC32[0], pSampleBuff, copyCount);
					}
				}
				pos += copyCount;
				_samplesRemTrack -= copyCount;
				_sampleCount += copyCount;
				CheckPosition();
			}
		}

		public void Init()
		{
			_offsetedCRC = new uint[_toc.AudioTracks, 10 * 588];
			_offsetedFrame450CRC = new uint[_toc.AudioTracks, 10 * 588];
			_offsetedCRC32 = new uint[_toc.AudioTracks + 1];
			for (int i = 0; i <= _toc.AudioTracks; i++)
				_offsetedCRC32[i] = 0xffffffff;
			_currentTrack = 0;
			_sampleCount = 0;
			_samplesRemTrack = _toc.Pregap * 588;
			CheckPosition();
		}

		public void CreateBackup(int writeOffset)
		{
			_backupCRC = new uint[_toc.AudioTracks];
			for (int i = 0; i < _toc.AudioTracks; i++)
				_backupCRC[i] = CRC(i, writeOffset);
		}

		private void CheckPosition()
		{
			while (_samplesRemTrack <= 0)
			{
				if (++_currentTrack > _toc.AudioTracks)
					return;
				_samplesRemTrack = _toc[_currentTrack].Length * 588;
			}
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

		public bool SetTags(NameValueCollection tags)
		{
			throw new Exception("unsupported");
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

		public int BitsPerSample
		{
			get { return 16; }
		}

		public long FinalSampleCount
		{
			set { _finalSampleCount = value; }
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
				string pressings = "";
				string partpressings = "";
				for (int di = 0; di < (int)AccDisks.Count; di++)
				{
					count += AccDisks[di].tracks[iTrack].count;
					if (CRC(iTrack, oi) == AccDisks[di].tracks[iTrack].CRC)
					{
						conf += AccDisks[di].tracks[iTrack].count;
						if (pressings != "")
							pressings = pressings + ",";
						pressings = pressings + (di + 1).ToString();
					}
					if (CRC450(iTrack, oi) == AccDisks[di].tracks[iTrack].Frame450CRC)
					{
						partials += AccDisks[di].tracks[iTrack].count;
						if (partpressings != "")
							partpressings = partpressings + ",";
						partpressings = partpressings + (di + 1).ToString();
					}
				}
				if (conf > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) Accurately ripped as in pressing(s) #{4}", iTrack + 1, CRC(iTrack, oi), count, conf, pressings));
				else if (partials > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) Partial match to pressing(s) #{4} ", iTrack + 1, CRC(iTrack, oi), count, partials, partpressings));
				else
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] (00/{2:00}) No matches", iTrack + 1, CRC(iTrack, oi), count));
			}
		}

		public void GenerateFullLog(TextWriter sw, int offsetApplied)
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
				sw.WriteLine("Track\t[ CRC    ] Status");
				GenerateLog(sw, offsetApplied);
				uint offsets_match = 0;
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
				{
					uint matches = 0;
					for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
						for (int di = 0; di < (int)AccDisks.Count; di++)
							if ((CRC(iTrack, oi) == AccDisks[di].tracks[iTrack].CRC && AccDisks[di].tracks[iTrack].CRC != 0) ||
								 (CRC450(iTrack, oi) == AccDisks[di].tracks[iTrack].Frame450CRC && AccDisks[di].tracks[iTrack].Frame450CRC != 0))
								matches++;
					if (matches != 0 && oi != offsetApplied)
					{
						if (offsets_match++ > 10)
						{
							sw.WriteLine("More than 10 offsets match!");
							break;
						}
						sw.WriteLine("Offsetted by {0}:", oi);
						GenerateLog(sw, oi);
					}
				}
			}
			if (CRC32(0) != 0)
			{
				sw.WriteLine("");
				sw.WriteLine("Track\t[ CRC32  ]");
				sw.WriteLine(String.Format(" --\t[{0:X8}]", CRC32(0)));
				for (int iTrack = 1; iTrack <= _toc.AudioTracks; iTrack++)
					sw.WriteLine(String.Format(" {0:00}\t[{1:X8}]", iTrack, CRC32(iTrack)));
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
			if (!File.Exists(fileName))
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
					FileStream myOffsetsSaved = new FileStream(fileName, FileMode.CreateNew, FileAccess.Write);
					byte [] buff = new byte[0x8000];
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
				string strname = Encoding.ASCII.GetString(name,0,len);
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
			for (int iTrack = 1; iTrack <= toc.TrackCount; iTrack++)
				if (toc[iTrack].IsAudio)
				{
					discId1 += toc[iTrack].Start;
					discId2 += Math.Max(toc[iTrack].Start, 1) * toc[iTrack].Number;
				}
			discId1 += toc.Length;
			discId2 += Math.Max(toc.Length, 1) * ((uint)toc.AudioTracks + 1);
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
		long _sampleCount, _finalSampleCount, _samplesRemTrack;
		int _currentTrack;
		private List<AccDisk> _accDisks;
		private HttpStatusCode _accResult;
		private uint[,] _offsetedCRC;
		private uint[,] _offsetedFrame450CRC;
		private uint[] _offsetedCRC32;
		private uint[] _backupCRC;

		Crc32 _crc32;

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
