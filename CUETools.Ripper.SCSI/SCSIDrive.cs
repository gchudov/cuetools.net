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
using System.Text;
using System.IO;
using Bwg.Scsi;
using Bwg.Logging;
using CUETools.CDImage;
using CUETools.Codecs;
using System.Threading;

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
		const int CB_AUDIO = 4 * 588 + 2 + 294 + 16;
		const int MAXSCANS = 20;
		const int NSECTORS = 16;
		//const int MSECTORS = 5*1024*1024 / (4 * 588);
		const int MSECTORS = 2400;
		int _currentTrack = -1, _currentIndex = -1, _currentTrackActualStart = -1;
		Logger m_logger;
		CDImageLayout _toc;
		char m_device_letter;
		InquiryResult m_inqury_result;
		int m_max_sectors;
		int _timeout = 10;
		Crc16Ccitt _crc;
		int _currentScan;
		public byte[,,] UserData;
		public byte[,,] C2Data;
		public byte[,,] QData;
		public long[] Quality;
		BitArray _errors;
		int _errorsCount;
		int _crcErrorsCount = 0;
		byte[] _currentData = new byte[MSECTORS * 4 * 588];
		short[] _valueScore = new short[256];
		bool _debugMessages = false;
		ReadCDCommand _readCDCommand = ReadCDCommand.Unknown;
		ReadCDCommand _forceReadCommand = ReadCDCommand.Unknown;
		Device.MainChannelSelection _mainChannelMode = Device.MainChannelSelection.UserData;
		Device.SubChannelMode _subChannelMode = Device.SubChannelMode.QOnly;
		Device.C2ErrorMode _c2ErrorMode = Device.C2ErrorMode.Mode296;
		string _autodetectResult;
		byte[] _readBuffer = new byte[NSECTORS * CB_AUDIO];
		byte[] _subchannelBuffer = new byte[NSECTORS * 16];
		bool _qChannelInBCD = true;

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

		public int Timeout
		{
			get
			{
				return _timeout;
			}
			set
			{
				_timeout = value;
			}
		}

		public bool DebugMessages
		{
			get
			{
				return _debugMessages;
			}
			set
			{
				_debugMessages = value;
			}
		}

		public string AutoDetectReadCommand
		{
			get
			{
				if (_autodetectResult != null || TestReadCommand())
					return _autodetectResult;
				string ret = _autodetectResult;
				_autodetectResult = null;
				return ret;
			}
		}

		public bool ForceD8
		{
			get
			{
				return _forceReadCommand == ReadCDCommand.ReadCdD8h;
			}
			set
			{
				_forceReadCommand = value ? ReadCDCommand.ReadCdD8h : ReadCDCommand.Unknown;
			}
		}


		public bool ForceBE
		{
			get
			{
				return _forceReadCommand == ReadCDCommand.ReadCdBEh;
			}
			set
			{
				_forceReadCommand = value ? ReadCDCommand.ReadCdBEh : ReadCDCommand.Unknown;
			}
		}

		public string CurrentReadCommand
		{
			get
			{
				return _readCDCommand == ReadCDCommand.Unknown ? "unknown" :
					string.Format("{0}, {1:X2}h, {2}{3}, {4} blocks at a time", 
					(_readCDCommand == ReadCDCommand.ReadCdBEh ? "BEh" : "D8h"),
					(_mainChannelMode == Device.MainChannelSelection.UserData ? 0x10 : 0xF8) +
					(_c2ErrorMode == Device.C2ErrorMode.None ? 0 : _c2ErrorMode == Device.C2ErrorMode.Mode294 ? 2 : 4),
					(_subChannelMode == Device.SubChannelMode.None ? "00h" : _subChannelMode == Device.SubChannelMode.QOnly ? "02h" : "04h"),
					_qChannelInBCD ? "" : "nonBCD",
					m_max_sectors);
			}
		}

		public CDDriveReader()
		{
			m_logger = new Logger();
			_crc = new Crc16Ccitt(InitialCrcValue.Zeros);
			UserData = new byte[MAXSCANS, MSECTORS, 4 * 588];
			C2Data = new byte[MAXSCANS, MSECTORS, 294];
			QData = new byte[MAXSCANS, MSECTORS, 16];
			Quality = new long[MAXSCANS];
		}

		public bool Open(char Drive)
		{
			Device.CommandStatus st;

			// Open the base device
			m_device_letter = Drive;
			if (m_device != null)
				Close();

			m_device = new Device(m_logger);
			if (!m_device.Open(m_device_letter))
				throw new Exception("Open failed: " + WinDev.Win32ErrorToString(m_device.LastError));

			// Get device info
			st = m_device.Inquiry(out m_inqury_result);
			if (st != Device.CommandStatus.Success || !m_inqury_result.Valid)
				throw new SCSIException("Inquiry", m_device, st);
			if (m_inqury_result.PeripheralQualifier != 0 || m_inqury_result.PeripheralDeviceType != Device.MMCDeviceType)
				throw new Exception(Path + " is not an MMC device");

			m_max_sectors = Math.Min(NSECTORS, m_device.MaximumTransferLength / CB_AUDIO - 1);
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

			//m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, (ushort)(0x7fff), (ushort)(0x7fff));
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
				throw new SCSIException("ReadTOC", m_device, st);
				//throw new Exception("ReadTOC: " + (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));

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
			if (_toc.AudioLength > 0)
			{
				if (_toc[1].IsAudio)
					_toc[1][0].Start = 0;
				Position = 0;
			}
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
				return m_max_sectors * 588;
			}
		}

		private int ProcessSubchannel(int sector, int Sectors2Read, bool updateMap)
		{
			int posCount = 0;
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int q_pos = (sector - _currentStart + iSector);
				int ctl = QData[_currentScan, q_pos, 0] >> 4;
				int adr = QData[_currentScan, q_pos,  0] & 7;
				bool preemph = (ctl & 1) == 1;
				bool dcp = (ctl & 2) == 2;

				for (int i = 0; i < 10; i++)
					_subchannelBuffer[i] = QData[_currentScan, q_pos, i];
				if (!_qChannelInBCD && adr == 1)
				{
					_subchannelBuffer[3] = toBCD(_subchannelBuffer[3]);
					_subchannelBuffer[4] = toBCD(_subchannelBuffer[4]);
					_subchannelBuffer[5] = toBCD(_subchannelBuffer[5]);
					_subchannelBuffer[7] = toBCD(_subchannelBuffer[7]);
					_subchannelBuffer[8] = toBCD(_subchannelBuffer[8]);
					_subchannelBuffer[9] = toBCD(_subchannelBuffer[9]);
				}
				ushort crc = _crc.ComputeChecksum(_subchannelBuffer, 0, 10);
				crc ^= 0xffff;
				if ((QData[_currentScan, q_pos, 10] != 0 || QData[_currentScan, q_pos, 11] != 0) &&
					((byte)(crc & 0xff) != QData[_currentScan, q_pos, 11] || (byte)(crc >> 8) != QData[_currentScan, q_pos, 10])
					)
				{
					if (!updateMap)
						continue;
					_crcErrorsCount++;
					if (_debugMessages && _crcErrorsCount < 4)
					{
						StringBuilder st = new StringBuilder();
						for (int i = 0; i < 12; i++)
							st.AppendFormat(",0x{0:X2}", QData[_currentScan, q_pos, i]);
						System.Console.WriteLine("\rCRC error@{0}{1};", CDImageLayout.TimeToString((uint)(sector + iSector)), st.ToString());
					}
					continue;
				}
				switch (adr)
				{
					case 1: // current position
						{
							int iTrack = fromBCD(QData[_currentScan, q_pos, 1]);
							int iIndex = fromBCD(QData[_currentScan, q_pos, 2]);
							int mm = _qChannelInBCD ? fromBCD(QData[_currentScan, q_pos, 7]) : QData[_currentScan, q_pos, 7];
							int ss = _qChannelInBCD ? fromBCD(QData[_currentScan, q_pos, 8]) : QData[_currentScan, q_pos, 8];
							int ff = _qChannelInBCD ? fromBCD(QData[_currentScan, q_pos, 9]) : QData[_currentScan, q_pos, 9];
							//if (sec != sector + iSector)
							//    System.Console.WriteLine("\rLost sync: {0} vs {1} ({2:X} vs {3:X})", CDImageLayout.TimeToString((uint)(sector + iSector)), CDImageLayout.TimeToString((uint)sec), sector + iSector, sec);
							if (iTrack == 110)
							{
								if (sector + iSector + 75 < _toc.AudioLength)
									throw new Exception("lead out area encountred");
								// make sure that data is zero?
								return posCount;
							}
							if (iTrack == 0)
								throw new Exception("lead in area encountred");
							posCount++;
							if (!updateMap)
								break;
							int sec = ff + 75 * (ss + 60 * mm) - 150; // sector + iSector;
							if (iTrack >= _toc.FirstAudio + _toc.AudioTracks)
								throw new Exception("strange track number encountred");
							if (iTrack != _currentTrack)
							{
								if (_currentTrack != -1 && iTrack != _currentTrack + 1)
								{
									if (_debugMessages)
										System.Console.WriteLine("\nNon-consequent track at {0}: {1} after {2}", CDImageLayout.TimeToString((uint)(sector + iSector)), iTrack, _currentTrack);
									//throw new Exception("invalid track");
									continue;
								}
								if (iIndex != 1 && iIndex != 0)
								{
									if (_debugMessages)
										System.Console.WriteLine("\nInvalid track start index at {0}: {1}.{2}", CDImageLayout.TimeToString((uint)(sector + iSector)), iTrack, iIndex);
									//throw new Exception("invalid index");
									continue;
								}
								_currentTrack = iTrack;
								_currentTrackActualStart = sec;
								_currentIndex = iIndex;
							}
							else if (iIndex != _currentIndex)
							{
								if (iIndex != _currentIndex + 1)
								{
									if (_debugMessages)
										System.Console.WriteLine("\nNon-consequent index at {0}: {1} after {2}", CDImageLayout.TimeToString((uint)(sector + iSector)), iIndex, _currentIndex);
									//throw new Exception("invalid index");
									continue;
								}
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
							if (dcp)
								_toc[iTrack].DCP = true;
							break;
						}
					case 2: // catalog
						if (updateMap && _toc.Catalog == null)
						{
							StringBuilder catalog = new StringBuilder();
							for (int i = 1; i < 8; i++)
								catalog.AppendFormat("{0:x2}", QData[_currentScan, q_pos, i]);
							_toc.Catalog = catalog.ToString(0, 13);
						}
						break;
					case 3: //isrc
						if (updateMap && _toc[_currentTrack].ISRC == null)
						{
							StringBuilder isrc = new StringBuilder();
							isrc.Append(from6bit(QData[_currentScan, q_pos, 1] >> 2));
							isrc.Append(from6bit(((QData[_currentScan, q_pos, 1] & 0x3) << 4) + (0x0f & (QData[_currentScan, q_pos, 2] >> 4))));
							isrc.Append(from6bit(((QData[_currentScan, q_pos, 2] & 0xf) << 2) + (0x03 & (QData[_currentScan, q_pos, 3] >> 6))));
							isrc.Append(from6bit((QData[_currentScan, q_pos, 3] & 0x3f)));
							isrc.Append(from6bit(QData[_currentScan, q_pos, 4] >> 2));
							isrc.Append(from6bit(((QData[_currentScan, q_pos, 4] & 0x3) << 4) + (0x0f & (QData[_currentScan, q_pos, 5] >> 4))));
							isrc.AppendFormat("{0:x}", QData[_currentScan, q_pos, 5] & 0xf);
							isrc.AppendFormat("{0:x2}", QData[_currentScan, q_pos, 6]);
							isrc.AppendFormat("{0:x2}", QData[_currentScan, q_pos, 7]);
							isrc.AppendFormat("{0:x}", QData[_currentScan, q_pos, 8] >> 4);
							if (!isrc.ToString().Contains("#") && isrc.ToString() != "0000000000")
								_toc[_currentTrack].ISRC = isrc.ToString();
						}
						break;
				}
			}
			return posCount;
		}

		public unsafe bool TestReadCommand()
		{
			//ReadCDCommand[] readmode = { ReadCDCommand.ReadCdBEh, ReadCDCommand.ReadCdD8h };
			ReadCDCommand[] readmode = { ReadCDCommand.ReadCdD8h, ReadCDCommand.ReadCdBEh };
			Device.SubChannelMode[] submode = { Device.SubChannelMode.QOnly, Device.SubChannelMode.None, Device.SubChannelMode.RWMode };
			Device.C2ErrorMode[] c2mode = { Device.C2ErrorMode.Mode296, Device.C2ErrorMode.Mode294, Device.C2ErrorMode.None };
			Device.MainChannelSelection[] mainmode = { Device.MainChannelSelection.UserData, Device.MainChannelSelection.F8h };
			bool found = false;
			_currentStart = 0;
			_currentScan = 0;
			_currentTrack = -1;
			_currentIndex = -1;
			m_max_sectors = Math.Min(NSECTORS, m_device.MaximumTransferLength / CB_AUDIO - 1);
			int sector = 3;
			for (int q = 0; q <= 1 && !found; q++)
				for (int c = 0; c <= 2 && !found; c++)
					for (int r = 0; r <= 1 && !found; r++)
						for (int m = 0; m <= 1 && !found; m++)
						{
							_readCDCommand = readmode[r];
							_subChannelMode = submode[q];
							_c2ErrorMode = c2mode[c];
							_mainChannelMode = mainmode[m];
							if (_forceReadCommand != ReadCDCommand.Unknown && _readCDCommand != _forceReadCommand)
								continue;
							if (_readCDCommand == ReadCDCommand.ReadCdD8h) // && (_c2ErrorMode != Device.C2ErrorMode.None || _mainChannelMode != Device.MainChannelSelection.UserData))
								continue;
							Array.Clear(_readBuffer, 0, _readBuffer.Length); // fill with something nasty instead?
							DateTime tm = DateTime.Now;
							Device.CommandStatus st = FetchSectors(sector, m_max_sectors, false, false);
							TimeSpan delay = DateTime.Now - tm;
							if (st == Device.CommandStatus.Success && _subChannelMode == Device.SubChannelMode.QOnly)
							{
								_qChannelInBCD = false;
								int sub1 = ProcessSubchannel(sector, m_max_sectors, false);
								_qChannelInBCD = true;
								int sub2 = ProcessSubchannel(sector, m_max_sectors, false);
								_qChannelInBCD = sub2 >= sub1;
								if (sub1 == 0 && sub2 == 0)
								{
									_autodetectResult += string.Format("{0}: {1}\n", CurrentReadCommand, "Got no subchannel information");
									continue;
								}
							}
							_autodetectResult += string.Format("{0}: {1} ({2}ms)\n", CurrentReadCommand, (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()), delay.TotalMilliseconds);
							found = st == Device.CommandStatus.Success && _subChannelMode != Device.SubChannelMode.RWMode;// && _subChannelMode != Device.SubChannelMode.QOnly;
							//sector += m_max_sectors;
						}
			//if (found)
			//    for (int n = 1; n <= m_max_sectors; n++)
			//    {
			//        Device.CommandStatus st = FetchSectors(0, n, false, false);
			//        if (st != Device.CommandStatus.Success)
			//        {
			//            _autodetectResult += string.Format("Maximum sectors: {0}, else {1}; max length {2}\n", n - 1, (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()), m_device.MaximumTransferLength);
			//            m_max_sectors = n - 1;
			//            break;
			//        }
			//    }
			if (found)
				_autodetectResult += "Chosen " + CurrentReadCommand + "\n";
			else
				_readCDCommand = ReadCDCommand.Unknown;
			_currentStart = -1;
			return found;
		}

		private unsafe void  ReorganiseSectors(int sector, int Sectors2Read)
		{
			int c2Size = _c2ErrorMode == Device.C2ErrorMode.None ? 0 : _c2ErrorMode == Device.C2ErrorMode.Mode294 ? 294 : 296;
			int oldSize = 4 * 588 +	c2Size + (_subChannelMode == Device.SubChannelMode.None ? 0 : 16);
			fixed (byte* readBuf = _readBuffer, qBuf = _subchannelBuffer, userData = UserData, c2Data = C2Data, qData = QData)
			{
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					byte* sectorPtr = readBuf + iSector * oldSize;
					byte* userDataPtr = userData + (_currentScan * MSECTORS + sector - _currentStart + iSector) * 4 * 588;
					byte* c2DataPtr = c2Data + (_currentScan * MSECTORS + sector - _currentStart + iSector) * 294;
					byte* qDataPtr = qData + (_currentScan * MSECTORS + sector - _currentStart + iSector) * 16;

					for (int sample = 0; sample < 4 * 588; sample++)
						userDataPtr[sample] = sectorPtr[sample];
					if (_c2ErrorMode != Device.C2ErrorMode.None)
						for (int c2 = 0; c2 < 294; c2++)
							c2DataPtr[c2] = sectorPtr[4 * 588 + c2Size - 294 + c2];
					else
						for (int c2 = 0; c2 < 294; c2++)
							c2DataPtr[c2] = 0; // 0xff??
					if (_subChannelMode != Device.SubChannelMode.None)
						for (int qi = 0; qi < 16; qi++)
							qDataPtr[qi] = sectorPtr[4 * 588 + c2Size + qi];
					else
						for (int qi = 0; qi < 16; qi++)
							qDataPtr[qi] = qBuf[iSector * 16 + qi];
				}
			}
		}

		private unsafe Device.CommandStatus FetchSectors(int sector, int Sectors2Read, bool abort, bool subchannel)
		{
			Device.CommandStatus st;
			fixed (byte* data = _readBuffer)
			{
				if (_debugMessages)
				{
					int size = (4 * 588 +
						(_subChannelMode == Device.SubChannelMode.QOnly ? 16 : _subChannelMode == Device.SubChannelMode.RWMode ? 96 : 0) +
						(_c2ErrorMode == Device.C2ErrorMode.Mode294 ? 294 : _c2ErrorMode == Device.C2ErrorMode.Mode296 ? 296 : 0)) * (int)Sectors2Read;
					MemSet(data, size, 0xff);
				}
				if (_readCDCommand == ReadCDCommand.ReadCdBEh)
					st = m_device.ReadCDAndSubChannel(_mainChannelMode, _subChannelMode, _c2ErrorMode, 1, false, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)Sectors2Read, (IntPtr)((void*)data), _timeout);
				else
					st = m_device.ReadCDDA(_subChannelMode, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)Sectors2Read, (IntPtr)((void*)data), _timeout);
			}
			
			if (st == Device.CommandStatus.Success && _subChannelMode == Device.SubChannelMode.None && subchannel)
				st = m_device.ReadSubChannel(2, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)Sectors2Read, ref _subchannelBuffer, _timeout);

			if (st == Device.CommandStatus.Success)
			{
				ReorganiseSectors(sector, Sectors2Read);
				return st;
			}

			if (!abort)
				return st;
			SCSIException ex = new SCSIException("ReadCD", m_device, st);
			if (sector != 0 && Sectors2Read > 1 && st == Device.CommandStatus.DeviceFailed && m_device.GetSenseAsc() == 0x64 && m_device.GetSenseAscq() == 0x00)
			{
				if (_debugMessages)
					System.Console.WriteLine("\n{0}: retrying one sector at a time", ex.Message);
				int iErrors = 0;
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					if (FetchSectors(sector + iSector, 1, false, subchannel) != Device.CommandStatus.Success)
					{
						iErrors ++;
						for (int i = 0; i < 4 * 588; i++)
							UserData[_currentScan, sector + iSector - _currentStart, i] = 0;
						for (int i = 0; i < 294; i++)
							C2Data[_currentScan, sector + iSector - _currentStart, i] = 0xff;
						for (int i = 0; i < 16; i++)
							QData[_currentScan, sector + iSector - _currentStart, i] = 0;
						if (_debugMessages)
							System.Console.WriteLine("\nSector lost");
					}
				}
				if (iErrors < Sectors2Read)
					return Device.CommandStatus.Success;
			}
			throw ex;
		}

		private unsafe void ZeroMemory(short *buf, int count)
		{
			if (IntPtr.Size == 4)
			{
				Int32* start = (Int32*)buf;
				Int32* end = (Int32*)(buf + count);
				while (start < end)
					*(start++) = 0;
			}
			else if (IntPtr.Size == 8)
			{
				Int64* start = (Int64*)buf;
				Int64* end = (Int64*)(buf + count);
				while (start < end)
					*(start++) = 0;
			}
			else 
				throw new Exception("wierd IntPtr.Size");
		}

		private unsafe void ZeroMemory(byte* buf, int count)
		{
			if (IntPtr.Size == 4)
			{
				Int32* start = (Int32*)buf;
				Int32* end = (Int32*)(buf + count);
				while (start < end)
					*(start++) = 0;
				for (int i = 0; i < (count & 3); i++)
					buf[count - i - 1] = 0;
			}
			else if (IntPtr.Size == 8)
			{
				Int64* start = (Int64*)buf;
				Int64* end = (Int64*)(buf + count);
				while (start < end)
					*(start++) = 0;
				for (int i = 0; i < (count & 7); i++)
					buf[count - i - 1] = 0;
			}
			else
				throw new Exception("wierd IntPtr.Size");
		}

		private unsafe void MemSet(byte* buf, int count, byte val)
		{
			Int32 intVal = (((((val << 8) + val) << 8) + val) << 8) + val;
			if (IntPtr.Size == 4)
			{
				Int32* start = (Int32*)buf;
				Int32* end = (Int32*)(buf + count);
				while (start < end)
					*(start++) = intVal;
				for (int i = 0; i < (count & 3); i++)
					buf[count - i - 1] = val;
			}
			else if (IntPtr.Size == 8)
			{
				Int64 int64Val = ((Int64)intVal << 32) + intVal;
				Int64* start = (Int64*)buf;
				Int64* end = (Int64*)(buf + count);
				while (start < end)
					*(start++) = int64Val;
				for (int i = 0; i < (count & 7); i++)
					buf[count - i - 1] = val;
			}
			else
				throw new Exception("wierd IntPtr.Size");
		}

		private void PrintErrors(int pass, int sector, int Sectors2Read, byte[] realData)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				if (_debugMessages)
				{
					StringBuilder st = new StringBuilder();
					for (int i = 0; i < 294; i++)
						if (C2Data[_currentScan, pos, i] != 0)
						{
							for (int j = i; j < i + 23; j++)
								if (j < 294)
									st.AppendFormat("{0:X2}", C2Data[_currentScan, pos, j]);
								else
									st.Append("  ");
							System.Console.WriteLine("\rC2 error @{0}[{1:000}]{2};", CDImageLayout.TimeToString((uint)(sector + iSector)), i, st.ToString());
							return;
						}
					//for (int i = 0; i < 4 * 588; i++)
					//    if (_currentData[pos * 4 * 588 + i] != realData[pos * 4 * 588 + i])
					//    {
					//        StringBuilder st = new StringBuilder();
					//        for (int j = i; j < i + 25; j++)
					//            if (j < 4 * 588)
					//                st.AppendFormat("{0:X2}", realData[pos * 4 * 588 + j]);
					//            else
					//                st.Append("  ");
					//        System.Console.WriteLine("\r{0}[--][{1:X3}]{2};", CDImageLayout.TimeToString((uint)(sector + iSector)), i, st.ToString());
					//        st.Length = 0;
					//        for (int result = 0; result <= pass; result++)
					//        {
					//            for (int j = i; j < i + 25; j++)
					//                if (j < 4 * 588)
					//                    st.AppendFormat("{0:X2}", UserData[result, pos, j]);
					//                else
					//                    st.Append("  ");
					//            System.Console.WriteLine("\r{0}[{3:X2}][{1:X3}]{2};", CDImageLayout.TimeToString((uint)(sector + iSector)), i, st.ToString(), result);
					//            st.Length = 0;
					//            //int c2Bit = 0x80 >> (i % 8);
					//            //byte value = UserData[result, pos, i];
					//            //short score = (short)(1 + (((C2Data[result, pos, i >> 3] & c2Bit) == 0) ? (short) 10 : (short)0));
					//            //st.AppendFormat("{0:X2}[{1:X2}]", value, score);
					//        }
					//        i += 25;
					//        //return;
					//        //while (st.Length < 46)
					//        //    st.Append(' ');
					//        //System.Console.WriteLine("\rReal error @{0}[{1:000}]{2};", CDImageLayout.TimeToString((uint)(sector + iSector)), i, st.ToString());
					//    }
				}
			}
		}

		private unsafe void CorrectSectors0(int sector, int Sectors2Read)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					byte bestValue = UserData[0, pos, iPar];
					_currentErrorsCount += (C2Data[0, pos, iPar >> 3] >> (iPar & 7)) & 1;
					_currentData[pos * 4 * 588 + iPar] = bestValue;
				}
			}
		}

		private unsafe void CorrectSectors1(int sector, int Sectors2Read)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					byte val1 = UserData[0, pos, iPar];
					byte val2 = UserData[1, pos, iPar];
					int err1 = (C2Data[0, pos, iPar >> 3] >> (iPar & 7)) & 1;
					int err2 = (C2Data[1, pos, iPar >> 3] >> (iPar & 7)) & 1;
					_currentErrorsCount += err1 | err2 | (val1 != val2 ? 1 : 0);
					_currentData[pos * 4 * 588 + iPar] = err1 != 0 ? val2 : val1;
					//if (_debugMessages && ((C2Data[_currentScan, pos, iPar >> 3] >> (iPar & 7)) & 1) != 0 && _currentErrorsCount < 10)
					//    System.Console.WriteLine("\rC2 error @{0}, byte {1:0000}                                    ", CDImageLayout.TimeToString((uint)(sector + iSector)), iPar);
				}
			}
		}

		private unsafe void CorrectSectors(int pass, int sector, int Sectors2Read, bool findWorst, bool markErrors)
		{
			if (pass == 0)
			{
				CorrectSectors0(sector, Sectors2Read);
				return;
			}
			if (pass == 1)
			{
				CorrectSectors1(sector, Sectors2Read);
				return;
			}
			short c2Score = 10;
			fixed (short* valueScore = _valueScore)
			{
				fixed (byte* userData = UserData, c2Data = C2Data, qData = QData)
				{
					for (int iSector = 0; iSector < Sectors2Read; iSector++)
					{
						int pos = sector - _currentStart + iSector;
						for (int iPar = 0; iPar < 4 * 588; iPar++)
						{
							int c2Bit = 0x80 >> (iPar % 8);

							ZeroMemory(valueScore, 256);
							byte bestValue = 0;
							short totalScore = 0;
							for (int result = 0; result <= pass; result++)
							{
								int offs = (result * MSECTORS + pos) * 4 * 588 + iPar;
								byte value = userData[offs];
								short score = (short)(1 + (((c2Data[offs >> 3] & c2Bit) == 0) ? c2Score : (short)0));
								valueScore[value] += score;
								totalScore += score;
								if (valueScore[value] > valueScore[bestValue])
									bestValue = value;
							}
//							if (_debugMessages && ((C2Data[_currentScan, pos, iPar >> 3] >> (iPar & 7)) & 1) != 0 && _currentErrorsCount < 10)
//								System.Console.WriteLine("\rC2 error @{0}, byte {1:0000}                                    ", CDImageLayout.TimeToString((uint)(sector + iSector)), iPar);
							bool fError = valueScore[bestValue] <= _correctionQuality + c2Score + totalScore / 2;
							if (fError)
								_currentErrorsCount++;
							_currentData[(sector - _currentStart + iSector) * 4 * 588 + iPar] = bestValue;
							if (findWorst)
							{
								for (int result = 0; result <= pass; result++)
									Quality[result] += Math.Min(0, valueScore[userData[(result * MSECTORS + pos) * 4 * 588 + iPar]] - c2Score - 2);
							}
							if (markErrors)
							{
								_errors[sector + iSector] |= fError;
								_errorsCount += fError ? 1 : 0;
							}
						}
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
		//            int c2Pos = iSector * CB_AUDIO + 2 + 4 * 588 + iPar / 8;
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

		public unsafe void PrefetchSector(int iSector)
		{
			int nExtraPasses = MAXSCANS / 2 + _correctionQuality;

			if (_currentStart == MSECTORS * (iSector / MSECTORS))
				return;

			if (_readCDCommand == ReadCDCommand.Unknown && !TestReadCommand())
				throw new Exception("failed to autodetect read command: " + _autodetectResult);

			_currentStart = MSECTORS * (iSector / MSECTORS);
			_currentEnd = Math.Min(_currentStart + MSECTORS, (int)_toc.AudioLength);

			//FileStream correctFile = new FileStream("correct.wav", FileMode.Open);
			//byte[] realData = new byte[MSECTORS * 4 * 588];
			//correctFile.Seek(0x2C + _currentStart * 588 * 4, SeekOrigin.Begin);
			//if (correctFile.Read(realData, _driveOffset * 4, MSECTORS * 4 * 588 - _driveOffset * 4) != MSECTORS * 4 * 588 - _driveOffset * 4)
			//    throw new Exception("read");
			//correctFile.Close();

			fixed (byte* userData = UserData, c2Data = C2Data, qData = QData)
			{
				for (int pass = 0; pass < MAXSCANS + nExtraPasses; pass++)
				{
					DateTime PassTime = DateTime.Now, LastFetch = DateTime.Now;
					if (pass < MAXSCANS)
						_currentScan = pass;
					else
					{
						_currentScan = 0;
						for (int result = 1; result < MAXSCANS; result++)
							if (Quality[result] < Quality[_currentScan])
								_currentScan = result;
						//if (worstPass < 0)
						//    System.Console.WriteLine("bad scan");
						//else
						//    System.Console.WriteLine("{0}->{1}, {2}->{3}", _scanResults[worstPass].Quality, _currentScan.Quality, CorrectSectorsTest(_currentStart, _currentEnd, 10, realData, -1), CorrectSectorsTest(_currentStart, _currentEnd, 10, realData, worstPass));
					}
					for (int result = 0; result < MAXSCANS; result++)
						Quality[result] = 0;
					_currentErrorsCount = 0;

					for (int sector = _currentStart; sector < _currentEnd; sector += m_max_sectors)
					{
						int Sectors2Read = Math.Min(m_max_sectors, _currentEnd - sector);
						int speed = pass == 5 ? 300 : pass == 6 ? 150 : pass == 7 ? 75 : 32500; // sectors per second
						int msToSleep = 1000 * Sectors2Read / speed - (int)((DateTime.Now - LastFetch).TotalMilliseconds);

						//if (msToSleep > 0) Thread.Sleep(msToSleep);
						LastFetch = DateTime.Now;
						FetchSectors(sector, Sectors2Read, true, pass == 0);
						//TimeSpan delay1 = DateTime.Now - LastFetch;
						if (pass == 0)
							ProcessSubchannel(sector, Sectors2Read, true);
						//DateTime LastFetched = DateTime.Now;
						CorrectSectors(Math.Min(pass, MAXSCANS - 1), sector, Sectors2Read, pass >= MAXSCANS - 1, pass == MAXSCANS - 1 + nExtraPasses);
						PrintErrors(Math.Min(pass, MAXSCANS - 1), sector, Sectors2Read, /*realData*/null);
						//TimeSpan delay2 = DateTime.Now - LastFetched;
						//if (sector == _currentStart)
							//System.Console.WriteLine("\n{0},{1}", delay1.TotalMilliseconds, delay2.TotalMilliseconds);
						if (ReadProgress != null)
							ReadProgress(this, new ReadProgressArgs(sector + Sectors2Read, pass, _currentStart, _currentEnd, _currentErrorsCount, PassTime));
					}
					//System.Console.WriteLine();
					//if (CorrectSectorsTest(start, _currentEnd, 10, realData) == 0)
					//    break;
					//if (pass == MAXSCANS - 1 + nExtraPasses)
					//    break;
					if (_currentErrorsCount == 0 && pass >= _correctionQuality)
						break;
					//if (_currentErrorsCount == 0 && pass >= _correctionQuality)
					//{
					//    bool syncOk = true;
					//    //if (pass == 0)
					//    //{
					//    //    ScanResults saved = _currentScan;
					//    //    _currentScan = new ScanResults(_currentEnd - _currentStart, CB_AUDIO);
					//    //    for (int sector = _currentStart; sector < _currentEnd && syncOk; sector += 7)
					//    //    {
					//    //        FetchSectors(sector, 2);
					//    //        for (int i = 0; i < 2 * CB_AUDIO; i++)
					//    //            if (_currentScan.Data[(sector - _currentStart) * CB_AUDIO + i] != saved.Data[(sector - _currentStart) * CB_AUDIO + i])
					//    //            {
					//    //                System.Console.WriteLine("Lost Sync");
					//    //                syncOk = false;
					//    //                break;
					//    //            }
					//    //    }
					//    //    _currentScan = saved;
					//    //}
					//    if (syncOk)
					//        break;
					//}
				}
			}
		}

		public int[,] Read(int[,] buff)
		{
			if (_toc == null)
				throw new Exception("Read: invalid TOC");
			if (_sampleOffset - _driveOffset >= (uint)Length)
				return null;
			if (_sampleOffset >= (int)Length)
				return new int[_driveOffset - (_sampleOffset - (int)Length), ChannelCount];
			if (_sampleOffset < 0)
				return new int[-_sampleOffset, ChannelCount];
			PrefetchSector((int)_sampleOffset / 588);
			int samplesRead = Math.Min(_currentEnd * 588, (int)Length + _driveOffset) - _sampleOffset;
			buff = new int[samplesRead, ChannelCount];
			AudioSamples.BytesToFLACSamples_16(_currentData, (_sampleOffset - _currentStart * 588) * 4, buff, 0, (uint) samplesRead, 2);
			_sampleOffset = _currentEnd * 588;
			return buff;
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			if (_toc == null)
				throw new Exception("Read: invalid TOC");
			if (_sampleOffset - _driveOffset >= (uint)Length)
			    return 0;
			if (_sampleOffset >= (uint)Length)
			{
				int samplesRead = _driveOffset - (_sampleOffset - (int)Length);
			    for (int i = 0; i < samplesRead; i++)
			        for (int c = 0; c < ChannelCount; c++)
			            buff[i, c] = 0;
			    _sampleOffset += samplesRead;
			    return (uint) samplesRead;
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

		public string Path
		{
			get
			{
				string result = m_device_letter + ": ";
				result += "[" + m_inqury_result.VendorIdentification + " " +
					m_inqury_result.ProductIdentification + " " +
					m_inqury_result.FirmwareVersion + "]";
				return result;
			}
		}

		public string ARName
		{
			get
			{
				return m_inqury_result.VendorIdentification.TrimEnd(' ', '\0').PadRight(8, ' ') + " - " + m_inqury_result.ProductIdentification.TrimEnd(' ', '\0');
			}
		}

		public string EACName
		{
			get
			{
				return m_inqury_result.VendorIdentification.TrimEnd(' ', '\0') + " " + m_inqury_result.ProductIdentification.TrimEnd(' ', '\0');
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
				if (_toc.AudioLength <= 0)
					throw new Exception("no audio");
				_currentTrack = -1;
				_currentIndex = -1;
				_crcErrorsCount = 0;
				_errorsCount = 0;				
				_errors = new BitArray((int)_toc.AudioLength); // !!!
				_sampleOffset = (int)value + _driveOffset;
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
			return "CUERipper v1.9.4 Copyright (C) 2008 Gregory S. Chudov";
			// ripper.GetName().Name + " " + ripper.GetName().Version;
		}

		private int fromBCD(byte hex)
		{
			return (hex >> 4) * 10 + (hex & 15);
		}

		private byte toBCD(int val)
		{
			return (byte)(((val / 10) << 4) + (val % 10));
		}

		private char from6bit(int bcd)
		{
			char[] ISRC6 = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#', '#', '#', '#', '#', '#', '#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
			bcd &= 0x3f;
			return bcd >= ISRC6.Length ? '#' : ISRC6[bcd];
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

	enum ReadCDCommand
	{
		ReadCdBEh,
		ReadCdD8h,
		Unknown
	};

	public sealed class SCSIException : Exception
	{
		public SCSIException(string args, Device device, Device.CommandStatus st)
			: base(args + ": " + (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(device.GetSenseAsc(), device.GetSenseAscq()) : st.ToString()))
		{
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
