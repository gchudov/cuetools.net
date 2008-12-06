// ****************************************************************************
// 
// CUERipper
// Copyright (C) 2008 Gregory S. Chudov (gchudov@gmail.com)
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// ****************************************************************************

using System;
using System.Runtime.InteropServices;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.IO;
using Bwg.Scsi;
using Bwg.Logging;
using CUETools.CDImage;
using CUETools.Codecs;

namespace CUETools.Ripper.SCSI
{
	/// <summary>
	/// 
	/// </summary>
	public class CDDriveReader : IAudioSource
	{
		byte[] cdtext = null;
		private Device m_device;
		int _sampleOffset = 0;
		int _driveOffset = 0;
		int _correctionQuality = 1;
		int _currentStart = -1, _currentEnd = -1, _currentErrorsCount = 0;
		const bool DoC2 = true;
		const int CB_AUDIO = 588 * 4 + 16 + (DoC2 ? 294 : 0);
		//const int REDUNDANCY = 8;
		const int NSECTORS = 64; //255 - REDUNDANCY;
		const int MSECTORS = 10000000 / CB_AUDIO;
		int _currentTrack = -1, _currentIndex = -1, _currentTrackActualStart = -1;
		Logger m_logger;
		CDImageLayout _toc;
		DeviceInfo m_info;
		Crc16Ccitt _crc;
		//RsEncode _rsEncoder;
		//RsDecode _rsDecoder;
		List<ScanResults> _scanResults;
		ScanResults _currentScan;
		BitArray _errors;
		int _errorsCount;
		byte[] _currentData = new byte[MSECTORS * 4 * 588];
		int[] valueScore = new int[256];

		public event EventHandler<ReadProgressArgs> ReadProgress;

		public CDImageLayout TOC
		{
			get
			{
				return _toc;
			}
		}

		public BitArray Errors
		{
			get
			{
				return _errors;
			}
		}

		public int ErrorsCount
		{
			get
			{
				return _errorsCount;
			}
		}

		public CDDriveReader()
		{
			m_logger = new Logger();
			_crc = new Crc16Ccitt(InitialCrcValue.Zeros);
			//_rsEncoder = new RsEncode(REDUNDANCY);
			//_rsDecoder = new RsDecode(REDUNDANCY);
		}

		public bool Open(char Drive)
		{
			Device.CommandStatus st;

			// Open the base device
			m_device = new Device(m_logger);
			if (!m_device.Open(Drive))
				throw new Exception("Open failed: SCSI error");

			// Get device info
			m_info = DeviceInfo.CreateDevice(Drive + ":");
			if (!m_info.ExtractInfo(m_device))
				throw new Exception("ExtractInfo failed: SCSI error");

			//// Open/Initialize the driver
			//Drive m_drive = new Drive(dev);
			//DiskOperationError status = m_drive.Initialize();
			//if (status != null)
			//    throw new Exception("SCSI error");

			// {
			//Drive.FeatureState readfeature = m_drive.GetFeatureState(Feature.FeatureType.CDRead);
			//if (readfeature == Drive.FeatureState.Error || readfeature == Drive.FeatureState.NotPresent)
			//    throw new Exception("SCSI error");
			// }{
			//st = m_device.GetConfiguration(Device.GetConfigType.OneFeature, 0, out flist);
			//if (st != Device.CommandStatus.Success)
			//    return CreateErrorObject(st, m_device);

			//Feature f = flist.Features[0];
			//ParseProfileList(f.Data);
			// }

			//SpeedDescriptorList speed_list;
			//st = m_device.GetSpeed(out speed_list);
			//if (st != Device.CommandStatus.Success)
			//    throw new Exception("GetSpeed failed: SCSI error");

			//int bytesPerSec = 4 * 588 * 75 * (pass > 8 ? 4 : pass > 4 ? 8 : pass > 0 ? 16 : 32);
			//Device.CommandStatus st = m_device.SetStreaming(Device.RotationalControl.CLVandNonPureCav, start, end, bytesPerSec, 1, bytesPerSec, 1);
			//if (st != Device.CommandStatus.Success)
			//    System.Console.WriteLine("SetStreaming: ", (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));
			//st = m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, (ushort)(bytesPerSec / 1024), (ushort)(bytesPerSec / 1024));
			//if (st != Device.CommandStatus.Success)
			//    System.Console.WriteLine("SetCdSpeed: ", (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));


			//st = m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, 32767/*Device.OptimumSpeed*/, Device.OptimumSpeed);
			//if (st != Device.CommandStatus.Success)
			//    throw new Exception("SetCdSpeed failed: SCSI error");

			IList<TocEntry> toc;
			st = m_device.ReadToc((byte)0, false, out toc);
			if (st != Device.CommandStatus.Success)
				throw new Exception("ReadTOC: " + (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));

			st = m_device.ReadCDText(out cdtext);
			// new CDTextEncoderDecoder

			_toc = new CDImageLayout();
			for (int iTrack = 0; iTrack < toc.Count - 1; iTrack++)
				_toc.AddTrack(new CDTrack((uint)iTrack + 1, 
					toc[iTrack].StartSector,
					toc[iTrack + 1].StartSector - toc[iTrack].StartSector - 
					    ((toc[iTrack + 1].Control < 4 || iTrack + 1 == toc.Count - 1) ? 0U : 152U * 75U), 
					toc[iTrack].Control < 4,
					(toc[iTrack].Control & 1) == 1));
			if (_toc[1].IsAudio)
				_toc[1][0].Start = 0;
			if (_toc.AudioLength > 0)
				_errors = new BitArray((int)_toc.AudioLength); // !!!
			return true;
		}

		public void Close()
		{
			m_device.Close();
			m_device = null;
			_toc = null;
		}

		public int BestBlockSize
		{
			get
			{
				return Math.Min(m_device.MaximumTransferLength / CB_AUDIO - 1, NSECTORS) * 588;
			}
		}

		private void ProcessSubchannel(int sector, int Sectors2Read, bool updateMap)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int q_pos = (sector - _currentStart + iSector + 1) * CB_AUDIO - 16;
				int ctl = _currentScan.Data[q_pos + 0] >> 4;
				int adr = _currentScan.Data[q_pos + 0] & 7;
				bool preemph = (ctl & 1) == 1;
				switch (adr)
				{
					case 1: // current position
						{
							int iTrack = fromBCD(_currentScan.Data[q_pos + 1]);
							int iIndex = fromBCD(_currentScan.Data[q_pos + 2]);
							int mm = fromBCD(_currentScan.Data[q_pos + 7]);
							int ss = fromBCD(_currentScan.Data[q_pos + 8]);
							int ff = fromBCD(_currentScan.Data[q_pos + 9]);
							int sec = ff + 75 * (ss + 60 * mm) - 150; // sector + iSector;
							//if (sec != sector + iSector)
							//   System.Console.WriteLine("\rLost sync: {0} vs {1} ({2:X} vs {3:X})", CDImageLayout.TimeToString((uint)(sector + iSector)), CDImageLayout.TimeToString((uint)sec), sector + iSector, sec);
							//ushort crc = _crc.ComputeChecksum(_currentScan.Data, q_pos, 10);
							//crc ^= 0xffff;
							//if (_currentScan.Data[q_pos + 10] != 0 && _currentScan.Data[q_pos + 11] != 0 &&
							//    ((crc & 0xff) != _currentScan.Data[q_pos + 11] ||
							//    (crc >> 8) != _currentScan.Data[q_pos + 10])
							//    )
							//{
							//    System.Console.WriteLine("CRC error at {0}", CDImageLayout.TimeToString((uint)(sector + iSector)));
							//}
							if (iTrack == 110)
							{
								if (sector + iSector + 75 < _toc.AudioLength)
									throw new Exception("lead out area encountred");
								// make sure that data is zero?
								return;
							}
							if (iTrack == 0)
								throw new Exception("lead in area encountred");
							if (!updateMap)
								break;
							if (iTrack != _currentTrack)
							{
								_currentTrack = iTrack;
								_currentTrackActualStart = sec;
								_currentIndex = iIndex;
								if (_currentIndex != 1 && _currentIndex != 0)
									throw new Exception("invalid index");
							}
							else if (iIndex != _currentIndex)
							{
								if (iIndex != _currentIndex + 1)
									throw new Exception("invalid index");
								_currentIndex = iIndex;
								if (_currentIndex == 1)
								{
									uint pregap = (uint)(sec - _currentTrackActualStart);
									_toc[iTrack][0].Start = _toc[iTrack].Start - pregap;
									_currentTrackActualStart = sec;
								} else
									_toc[iTrack].AddIndex(new CDTrackIndex((uint)iIndex, (uint)(_toc[iTrack].Start + sec - _currentTrackActualStart)));
								_currentIndex = iIndex;
							}
							if (preemph)
								_toc[iTrack].PreEmphasis = true;
							break;
						}
					case 2: // catalog
						if (_toc.Catalog == null)
						{
							StringBuilder catalog = new StringBuilder();
							for (int i = 1; i < 8; i++)
								catalog.AppendFormat("{0:x2}", _currentScan.Data[q_pos + i]);
							_toc.Catalog = catalog.ToString(0, 13);
						}
						break;
					case 3: //isrc
						if (_toc[_currentTrack].ISRC == null)
						{
							StringBuilder isrc = new StringBuilder();
							char[] ISRC6 = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#', '#', '#', '#', '#', '#', '#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
							isrc.Append(ISRC6[_currentScan.Data[q_pos + 1] >> 2]);
							isrc.Append(ISRC6[((_currentScan.Data[q_pos + 1] & 0x3) << 4) + (_currentScan.Data[q_pos + 2] >> 4)]);
							isrc.Append(ISRC6[((_currentScan.Data[q_pos + 2] & 0xf) << 2) + (_currentScan.Data[q_pos + 3] >> 6)]);
							isrc.Append(ISRC6[(_currentScan.Data[q_pos + 3] & 0x3f)]);
							isrc.Append(ISRC6[_currentScan.Data[q_pos + 4] >> 2]);
							isrc.Append(ISRC6[((_currentScan.Data[q_pos + 4] & 0x3) << 4) + (_currentScan.Data[q_pos + 5] >> 4)]);
							isrc.AppendFormat("{0:x}", _currentScan.Data[q_pos + 5] & 0xf);
							isrc.AppendFormat("{0:x2}", _currentScan.Data[q_pos + 6]);
							isrc.AppendFormat("{0:x2}", _currentScan.Data[q_pos + 7]);
							isrc.AppendFormat("{0:x}", _currentScan.Data[q_pos + 8] >> 4);
							_toc[_currentTrack].ISRC = isrc.ToString();
						}
						break;
				}
			}
		}

		private unsafe void FetchSectors(int sector, int Sectors2Read)
		{
			fixed (byte* data = _currentScan.Data)
			{
				//Device.CommandStatus st;
				//using (Command cmd = new Command(ScsiCommandCode.Read12, 12, (IntPtr)((void*)data), Sectors2Read * 588 * 4, Command.CmdDirection.In, 5 * 60))
				//{
				//    //cmd.SetCDB8(1, 4); // force from media
				//    cmd.SetCDB32(2, sector);
				//    cmd.SetCDB32(6, Sectors2Read);
				//    //cmd.SetCDB8(10, 0x80); // streaming
				//    st= m_device.SendCommand(cmd);
				//}
				Device.CommandStatus st = m_device.ReadCDAndSubChannel(2, DoC2, 1, false, (uint)sector, (uint)Sectors2Read, (IntPtr)((void*)(data + (sector - _currentStart) * CB_AUDIO)), Sectors2Read * CB_AUDIO);
				if (st != Device.CommandStatus.Success)
				{
					if (st == Device.CommandStatus.DeviceFailed && m_device.GetSenseAsc() == 0x64 && m_device.GetSenseAscq() == 0x00)
					{
						int iErrors = 0;
						for (int iSector = 0; iSector < Sectors2Read; iSector++)
						{
							st = m_device.ReadCDAndSubChannel(2, DoC2, 1, false, (uint)(sector + iSector), 1U, (IntPtr)((void*)(data + (sector + iSector - _currentStart) * CB_AUDIO)), CB_AUDIO);
							if (st != Device.CommandStatus.Success)
							{
								iErrors ++;
								for (int iPos = 0; iPos < CB_AUDIO; iPos++)
									data[(sector + iSector - _currentStart) * CB_AUDIO + iPos] = (DoC2 && iPos >= 4 * 588 && iPos < 4 * 588 + 294) ? (byte)255 : (byte)0;
							}
						}
						if (iErrors < Sectors2Read)
							return;
					}
					string status = (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString());
					st = m_device.ReadCDAndSubChannel(0, false, 1, false, (uint)sector, (uint)Sectors2Read, (IntPtr)((void*)(data + (sector - _currentStart) * 4 * 588)), Sectors2Read * 4 * 588);
					if (st != Device.CommandStatus.Success)
						status += "; ReadWithoutSubchannel: " + (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString());
					else
						status += "; ReadWithoutSubchannel: might work";
					throw new Exception("ReadCD: " + status);
				}

				//int localC2 = 0;
				//if (DoC2)
				//    for (int iSector = 0; iSector < Sectors2Read; iSector++)
				//    {
				//        _currentScan.C2Errors[sector + iSector] = data[(iSector + 1) * CB_AUDIO - 18] != 0;
				//        localC2 += (data[(iSector + 1) * CB_AUDIO - 18] != 0) ? 1 : 0;
				//    }
				//_currentScan.C2Count += localC2;
			}
		}

		private unsafe void CorrectSectors(int sector, int Sectors2Read, bool findWorst, bool markErrors)
		{
			int c2Score = 10;
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					int dataPos = (sector - _currentStart + iSector) * CB_AUDIO + iPar;
					int c2Pos = (sector - _currentStart + iSector) * CB_AUDIO + 4 * 588 + iPar / 8;
					int c2Bit = 0x80 >> (iPar % 8);

					Array.Clear(valueScore, 0, 256);

					byte bestValue = _currentScan.Data[dataPos];
					valueScore[bestValue] += 1 + (((_currentScan.Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
					int totalScore = valueScore[bestValue];
					for (int result = 0; result < _scanResults.Count; result++)
					{
						byte value = _scanResults[result].Data[dataPos];
						valueScore[value] += 1 + (((_scanResults[result].Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
						totalScore += 1 + (((_scanResults[result].Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
						if (valueScore[value] > valueScore[bestValue])
							bestValue = value;
					}
					bool fError = valueScore[bestValue] * 2 <= c2Score + totalScore;
					if (fError)
						_currentErrorsCount++;
					_currentData[(sector - _currentStart + iSector) * 4 * 588 + iPar] = bestValue;
					if (findWorst)
					{
						for (int result = 0; result < _scanResults.Count; result++)
							_scanResults[result].Quality += Math.Min(0, valueScore[_scanResults[result].Data[dataPos]] - c2Score - 1);
						_currentScan.Quality += Math.Min(0, valueScore[_currentScan.Data[dataPos]] - c2Score - 1);
					}
					if (markErrors)
					{
						_errors[sector + iSector] = fError;
						_errorsCount += fError ? 1 : 0;
					}
				}
			}
		}

		//private unsafe int CorrectSectorsTest(int start, int end, int c2Score, byte[] realData, int worstScan)
		//{
		//    int[] valueScore = new int[256];
		//    int[] scoreErrors = new int[256];
		//    int realErrors = 0;
		//    int bestScore = 0;
		//    int _errorsCaught = 0;
		//    int _falsePositives = 0;
		//    for (int iSector = 0; iSector < end - start; iSector++)
		//    {
		//        for (int iPar = 0; iPar < 4 * 588; iPar++)
		//        {
		//            int dataPos = iSector * CB_AUDIO + iPar;
		//            int c2Pos = iSector * CB_AUDIO + 4 * 588 + iPar / 8;
		//            int c2Bit = 0x80 >> (iPar % 8);

		//            Array.Clear(valueScore, 0, 256);

		//            byte bestValue = _currentScan.Data[dataPos];
		//            valueScore[bestValue] += 1 + (((_currentScan.Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
		//            int totalScore = valueScore[bestValue];
		//            for (int result = 0; result < _scanResults.Count; result++)
		//            {
		//                if (result == worstScan)
		//                    continue;
		//                byte value = _scanResults[result].Data[dataPos];
		//                valueScore[value] += 1 + (((_scanResults[result].Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
		//                totalScore += 1 + (((_scanResults[result].Data[c2Pos] & c2Bit) == 0) ? c2Score : 0);
		//                if (valueScore[value] > valueScore[bestValue])
		//                    bestValue = value;
		//            }
		//            if (valueScore[bestValue] < (1 + c2Score + totalScore) / 2)
		//                _currentErrorsCount++;
		//            //_currentData[iSector * 4 * 588 + iPar] = bestValue;
		//            if (realData[iSector * 4 * 588 + iPar] != bestValue)
		//            {
		//                if (valueScore[bestValue] > bestScore)
		//                    scoreErrors[valueScore[bestValue]]++;
		//                realErrors++;
		//                if (valueScore[bestValue] * 2 <= c2Score + totalScore)
		//                    _errorsCaught++;
		//            } else
		//                if (valueScore[bestValue] * 2 <= c2Score + totalScore)
		//                    _falsePositives++;
		//        }
		//    }
		//    //string s = "";
		//    //for (int i = 0; i < 256; i++)
		//    //    if (scoreErrors[i] > 0)
		//    //        s += string.Format("[{0}]={1};", i, scoreErrors[i]);
		//    //System.Console.WriteLine("RE{0:000} EC{1} FP{2}", realErrors, _errorsCaught, _falsePositives);
		//    return realErrors;
		//}

		public void PrefetchSector(int iSector)
		{
			int nPasses = 16 + _correctionQuality * 2;
			int nExtraPasses = 8 + _correctionQuality;

			if (_currentStart == MSECTORS * (iSector / MSECTORS))
				return;

			_currentStart = MSECTORS * (iSector / MSECTORS);
			_currentEnd = Math.Min(_currentStart + MSECTORS, (int)_toc.AudioLength);
			_scanResults = new List<ScanResults>();

			//FileStream correctFile = new FileStream("correct.wav", FileMode.Open);
			//byte[] realData = new byte[MSECTORS * 4 * 588];
			//correctFile.Seek(0x2C + start * 588 * 4, SeekOrigin.Begin);
			//if (correctFile.Read(realData, 0, MSECTORS * 4 * 588) != MSECTORS * 4 * 588)
			//    throw new Exception("read");
			//correctFile.Close();

			for (int pass = 0; pass <= nPasses + nExtraPasses; pass++)
			{
				DateTime PassTime = DateTime.Now;
				_currentScan = new ScanResults(MSECTORS, CB_AUDIO);
				_currentErrorsCount = 0;
				int nSectors = Math.Min(NSECTORS, m_device.MaximumTransferLength / CB_AUDIO - 1);
				for (int sector = _currentStart; sector < _currentEnd; sector += nSectors)
				{
					int Sectors2Read = Math.Min(nSectors, _currentEnd - sector);
					FetchSectors(sector, Sectors2Read);
					CorrectSectors(sector, Sectors2Read, pass > nPasses, pass == nPasses + nExtraPasses);
					ProcessSubchannel(sector, Sectors2Read, pass == 0);
					if (ReadProgress != null)
						ReadProgress(this, new ReadProgressArgs(sector + Sectors2Read, pass, _currentStart, _currentEnd, _currentErrorsCount, PassTime));
				}
				//System.Console.WriteLine();
				//if (CorrectSectorsTest(start, _currentEnd, 10, realData) == 0)
				//    break;
				if (pass == nPasses + nExtraPasses)
					break;
				if (pass > nPasses)
				{
					int worstPass = -1;
					for (int result = 0; result < _scanResults.Count; result++)
						if (_scanResults[result].Quality < (worstPass < 0 ? _currentScan.Quality : _scanResults[worstPass].Quality))
							worstPass = result;
					//if (worstPass < 0)
					//    System.Console.WriteLine("bad scan");
					//else
					//    System.Console.WriteLine("{0}->{1}, {2}->{3}", _scanResults[worstPass].Quality, _currentScan.Quality, CorrectSectorsTest(_currentStart, _currentEnd, 10, realData, -1), CorrectSectorsTest(_currentStart, _currentEnd, 10, realData, worstPass));
					if (worstPass < 0)
						_currentScan = null;
					else
						_scanResults[worstPass] = _currentScan;
					for (int result = 0; result < _scanResults.Count; result++)
						_scanResults[result].Quality = 0;
					continue;
				}
				if (_currentErrorsCount == 0 && pass >= _correctionQuality)
				{
					bool syncOk = true;
					//if (pass == 0)
					//{
					//    ScanResults saved = _currentScan;
					//    _currentScan = new ScanResults(_currentEnd - _currentStart, CB_AUDIO);
					//    for (int sector = _currentStart; sector < _currentEnd && syncOk; sector += 7)
					//    {
					//        FetchSectors(sector, 2);
					//        for (int i = 0; i < 2 * CB_AUDIO; i++)
					//            if (_currentScan.Data[(sector - _currentStart) * CB_AUDIO + i] != saved.Data[(sector - _currentStart) * CB_AUDIO + i])
					//            {
					//                System.Console.WriteLine("Lost Sync");
					//                syncOk = false;
					//                break;
					//            }
					//    }
					//    _currentScan = saved;
					//}
					if (syncOk)
						break;
				}			
				_scanResults.Add(_currentScan);
			}
			_currentScan = null;
			_scanResults = null;
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			if (_toc == null)
				throw new Exception("Read: invalid TOC");
			if (_sampleOffset - _driveOffset >= (uint)Length)
			    return 0;
			if (_sampleOffset > (uint)Length)
			{
			    int samplesRead = _sampleOffset - (int)Length;
			    for (int i = 0; i < samplesRead; i++)
			        for (int c = 0; c < ChannelCount; c++)
			            buff[i, c] = 0;
			    _sampleOffset += samplesRead;
			    return 0;
			}
			if ((uint)(_sampleOffset - _driveOffset + sampleCount) > Length)
			    sampleCount = (uint)((int)Length + _driveOffset - _sampleOffset);
			int silenceCount = 0;
			if ((uint)(_sampleOffset + sampleCount) > Length)
			{
			    silenceCount = _sampleOffset + (int)sampleCount - (int)Length;
			    sampleCount -= (uint) silenceCount;
			}
			uint pos = 0;
			if (_sampleOffset < 0)
			{
			    uint nullSamplesRead = Math.Min((uint)-_sampleOffset, sampleCount);
			    for (int i = 0; i < nullSamplesRead; i++)
			        for (int c = 0; c < ChannelCount; c++)
			            buff[i, c] = 0;
			    pos += nullSamplesRead;
			    sampleCount -= nullSamplesRead;
			    _sampleOffset += (int)nullSamplesRead;
			    if (sampleCount == 0)
			        return pos;
			}
			int firstSector = (int)_sampleOffset / 588;
			int lastSector = Math.Min((int)(_sampleOffset + sampleCount + 587)/588, (int)_toc.AudioLength);
			for (int sector = firstSector; sector < lastSector; sector ++)
			{
				PrefetchSector(sector);
		        uint samplesRead = (uint) (Math.Min((int)sampleCount, 588 - (_sampleOffset % 588)));
				AudioSamples.BytesToFLACSamples_16(_currentData, (sector - _currentStart) * 4 * 588 + ((int)_sampleOffset % 588) * 4, buff, (int)pos, samplesRead, 2);
		        pos += samplesRead;
		        sampleCount -= samplesRead;
		        _sampleOffset += (int) samplesRead;
			}
			if (silenceCount > 0)
			{
			    uint nullSamplesRead = (uint)silenceCount;
			    for (int i = 0; i < nullSamplesRead; i++)
			        for (int c = 0; c < ChannelCount; c++)
			            buff[pos + i, c] = 0;
			    pos += nullSamplesRead;
			    _sampleOffset += (int)nullSamplesRead;
			}
			return pos;
		}

		public ulong Length
		{
			get
			{
				if (_toc == null)
					throw new Exception("invalid TOC");
				return (ulong)588 * _toc.AudioLength;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return 16;
			}
		}

		public int ChannelCount
		{
			get
			{
				return 2;
			}
		}

		public int SampleRate
		{
			get
			{
				return 44100;
			}
		}

		public NameValueCollection Tags
		{
			get
			{
				return null;
			}
			set
			{
			}
		}

		public bool UpdateTags(bool preserveTime)
		{
			return false;
		}

		public string Path
		{
			get
			{
				return m_info.LongDesc;
			}
		}

		public string ARName
		{
			get
			{
				return m_info.InquiryData.VendorIdentification.TrimEnd(' ','\0').PadRight(8,' ') + " - " + m_info.InquiryData.ProductIdentification.TrimEnd(' ','\0');
			}
		}

		public ulong Position
		{
			get
			{
				return (ulong)(_sampleOffset - _driveOffset);
			}
			set
			{
				_sampleOffset = (int) value + _driveOffset;
			}
		}

		public ulong Remaining
		{
			get
			{
				return Length - Position;
			}
		}

		public int DriveOffset
		{
			get
			{
				return _driveOffset;
			}
			set
			{
				_driveOffset = value;
				_sampleOffset = value;
			}
		}

		public int CorrectionQuality
		{
			get
			{
				return _correctionQuality;
			}
			set
			{
				_correctionQuality = value;
			}
		}		

		public static string RipperVersion()
		{
			return "CUERipper v1.9.3 Copyright (C) 2008 Gregory S. Chudov";
			// ripper.GetName().Name + " " + ripper.GetName().Version;
		}

		private int fromBCD(byte hex)
		{
			return (hex >> 4) * 10 + (hex & 15);
		}

		public static char[] DrivesAvailable()
		{
			List<char> result = new List<char>();
			foreach (DriveInfo info in DriveInfo.GetDrives())
				if (info.DriveType == DriveType.CDRom)
					result.Add(info.Name[0]);
			return result.ToArray();
		}
	}

	internal class ScanResults
	{
		public byte[] Data;
		public long Quality;

		public ScanResults(int msector, int cbaudio)
		{
			Data = new byte[msector * cbaudio];
			Quality = 0;
		}
	}

	public sealed class ReadProgressArgs: EventArgs
	{
		public int Position;
		public int Pass;
		public int PassStart, PassEnd;
		public int ErrorsCount;
		public DateTime PassTime;

		public ReadProgressArgs(int position, int pass, int passStart, int passEnd, int errorsCount, DateTime passTime)
		{
			Position = position;
			Pass = pass;
			PassStart = passStart;
			PassEnd = passEnd;
			ErrorsCount = errorsCount;
			PassTime = passTime;
		}
	}
}
