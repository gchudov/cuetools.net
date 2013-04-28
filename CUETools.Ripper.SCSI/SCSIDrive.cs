// ****************************************************************************
// 
// CUERipper
// Copyright (C) 2008-13 Grigory Chudov (gchudov@gmail.com)
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
using CUETools.Ripper;
using System.Threading;

namespace CUETools.Ripper.SCSI
{
	/// <summary>
	/// 
	/// </summary>
	public class CDDriveReader : ICDRipper
	{
		byte[] cdtext = null;
		private Device m_device;
		int _sampleOffset = 0;
		int _driveOffset = 0;
		int _correctionQuality = 1;
		int _currentStart = -1, _currentEnd = -1, _currentErrorsCount = 0;
		const int CB_AUDIO = 4 * 588 + 2 + 294 + 16;
		const int NSECTORS = 16;
		//const int MSECTORS = 5*1024*1024 / (4 * 588);
		const int MSECTORS = 2400;
		Logger m_logger;
		CDImageLayout _toc;
		CDImageLayout _toc2;
		char m_device_letter;
		InquiryResult m_inqury_result;
		int m_max_sectors;
		int _timeout = 10;
		Crc16Ccitt _crc;
		public long[,,] UserData;
		public byte[,] C2Count;
		public long[] byte2long;
		BitArray _errors;
		int _errorsCount;
		int _crcErrorsCount = 0;
		AudioBuffer currentData = new AudioBuffer(AudioPCMConfig.RedBook, MSECTORS * 588);
		short[] _valueScore = new short[256];
		bool _debugMessages = false;
		ReadCDCommand _readCDCommand = ReadCDCommand.Unknown;
		ReadCDCommand _forceReadCommand = ReadCDCommand.Unknown;
		Device.MainChannelSelection _mainChannelMode = Device.MainChannelSelection.UserData;
		Device.C2ErrorMode _c2ErrorMode = Device.C2ErrorMode.Mode296;
		string _autodetectResult;
		byte[] _readBuffer = new byte[NSECTORS * CB_AUDIO];
        byte[] _subchannelBuffer = new byte[NSECTORS * CB_AUDIO];
		bool _qChannelInBCD = true;

		private ReadProgressArgs progressArgs = new ReadProgressArgs();
		public event EventHandler<ReadProgressArgs> ReadProgress;

		public CDImageLayout TOC
		{
			get
			{
				return gapsDetected && _toc2 != null ? _toc2 : _toc;
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
				TestReadCommand();
				return _autodetectResult;
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
					(_gapDetection == GapDetectionMethod.ReadCD ? "BEh" : _gapDetection == GapDetectionMethod.ReadSubchannel ? "42h" : ""),
					_qChannelInBCD ? "" : "nonBCD",
					m_max_sectors);
			}
		}

		public CDDriveReader()
		{
			m_logger = new Logger();
			_crc = new Crc16Ccitt(InitialCrcValue.Zeros);
			byte2long = new long[256];
			for (long i = 0; i < 256; i++)
			{
				long bl = 0;
				for (int b = 0; b < 8; b++)
					bl += ((i >> b) & 1) << (b << 3);
				byte2long[i] = bl;
			}
		}

		public bool Open(char Drive)
		{
			Device.CommandStatus st;

			m_inqury_result = null;

			// Open the base device
			m_device_letter = Drive;
			if (m_device != null)
				Close();

			m_device = new Device(m_logger);
			if (!m_device.Open(m_device_letter))
				throw new ReadCDException(Resource1.DeviceOpenError, Marshal.GetExceptionForHR(Marshal.GetHRForLastWin32Error()));
			//throw new ReadCDException(Resource1.DeviceOpenError + ": " + WinDev.Win32ErrorToString(m_device.LastError));

			// Get device info
			st = m_device.Inquiry(out m_inqury_result);
			if (st != Device.CommandStatus.Success)
				throw new SCSIException(Resource1.DeviceInquiryError, m_device, st);
			if (!m_inqury_result.Valid || m_inqury_result.PeripheralQualifier != 0 || m_inqury_result.PeripheralDeviceType != Device.MMCDeviceType)
				throw new ReadCDException(Resource1.DeviceNotMMC);

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
				throw new SCSIException(Resource1.ReadTOCError, m_device, st);
				//throw new Exception("ReadTOC: " + (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));

			//byte[] qdata = null;
			//st = m_device.ReadPMA(out qdata);
			//if (st != Device.CommandStatus.Success)
			//    throw new SCSIException("ReadPMA", m_device, st);

			//st = m_device.ReadCDText(out cdtext, _timeout);
			// new CDTextEncoderDecoder

			_toc2 = null;
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
			} else
				throw new ReadCDException(Resource1.NoAudio);

            UserData = new long[MSECTORS, 2, 4 * 588];
            C2Count = new byte[MSECTORS, 294];

			return true;
		}

		public void Close()
		{
			UserData = null;
			C2Count = null;
			if (m_device != null)
				m_device.Close();
			m_device = null;
			_toc = null;
			_toc2 = null;
			gapsDetected = false;
			readCommandFound = false;
			_currentStart = -1;
			_currentEnd = -1;
		}

		public void Dispose()
		{
			Close();
		}

		public int BestBlockSize
		{
			get
			{
				return m_max_sectors * 588;
			}
		}

		private enum GapDetectionMethod
		{
			ReadCD,
			ReadSubchannel,
			None
		}

		private GapDetectionMethod _gapDetection = GapDetectionMethod.None;

		private unsafe void LocateLastSector(int sec0, int sec1, int iTrack, int iIndex, ref int maxIndex, out int pos)
		{
			if (sec1 <= sec0)
			{
				pos = Math.Min(sec0, sec1);
				return;
			}
			int msector = (sec0 + sec1) / 2;
			int fsector = Math.Max(sec0, msector - 1);
			int lsector = Math.Min(sec1, msector + 3);

			if (lsector >= fsector && _gapDetection == GapDetectionMethod.ReadCD)
			{
				Device.CommandStatus st = m_device.ReadSubChannel(2, (uint)fsector, (uint)(lsector - fsector + 1), ref _subchannelBuffer, _timeout);
				if (st != Device.CommandStatus.Success)
					lsector = fsector - 1;
			}

			fixed (byte * data = _subchannelBuffer)
				for (int sector = fsector; sector <= lsector; sector++)
				{
					Device.CommandStatus st = Device.CommandStatus.Success;
					int sTrack, sIndex, sPos, ctl;
					switch (_gapDetection)
					{
						case GapDetectionMethod.ReadSubchannel:
							{
								// seek to given sector
								if (_readCDCommand == ReadCDCommand.ReadCdBEh)
									st = m_device.ReadCDAndSubChannel(_mainChannelMode, Device.SubChannelMode.None, _c2ErrorMode, 1, false, (uint)sector, 1, (IntPtr)((void*)data), _timeout);
								else
									st = m_device.ReadCDDA(Device.SubChannelMode.None, (uint)sector, 1, (IntPtr)((void*)data), _timeout);
								if (st != Device.CommandStatus.Success)
									continue;
								st = m_device.ReadSubChannel42(1, 0, ref _subchannelBuffer, 0, _timeout);
								// x x x x 01 adrctl tr ind abs abs abs rel rel rel
								if (st != Device.CommandStatus.Success)
									continue;
								if (_subchannelBuffer[0] != 0 || _subchannelBuffer[2] != 0 || _subchannelBuffer[3] != 12 || _subchannelBuffer[4] != 1)
									continue;

								ctl = _subchannelBuffer[5] & 0xf;
								int adr = (_subchannelBuffer[5] >> 4) & 0xf;
								sTrack = _subchannelBuffer[6];
								sIndex = _subchannelBuffer[7];
								sPos = (_subchannelBuffer[8] << 24) | (_subchannelBuffer[9] << 16) | (_subchannelBuffer[10] << 8) | (_subchannelBuffer[11]);
								//int sRel = (_subchannelBuffer[12] << 24) | (_subchannelBuffer[13] << 16) | (_subchannelBuffer[14] << 8) | (_subchannelBuffer[15]);
								if (adr != 1)
									continue;

								if (sTrack < _toc2.FirstAudio || sTrack >= _toc2.FirstAudio + _toc2.AudioTracks)
									continue;

								break;
							}
						case GapDetectionMethod.ReadCD:
							{
								int offs = 16 * (sector - fsector);
								ctl = _subchannelBuffer[offs + 0] >> 4;
								int adr = _subchannelBuffer[offs + 0] & 7;
								if (!_qChannelInBCD && adr == 1)
								{
									_subchannelBuffer[offs + 3] = toBCD(_subchannelBuffer[offs + 3]);
									_subchannelBuffer[offs + 4] = toBCD(_subchannelBuffer[offs + 4]);
									_subchannelBuffer[offs + 5] = toBCD(_subchannelBuffer[offs + 5]);
									_subchannelBuffer[offs + 7] = toBCD(_subchannelBuffer[offs + 7]);
									_subchannelBuffer[offs + 8] = toBCD(_subchannelBuffer[offs + 8]);
									_subchannelBuffer[offs + 9] = toBCD(_subchannelBuffer[offs + 9]);
								}

								ushort crc = _crc.ComputeChecksum(_subchannelBuffer, offs, 10);
								crc ^= 0xffff;
								ushort scrc = (ushort)((_subchannelBuffer[offs + 10] << 8) | _subchannelBuffer[offs + 11]);
								if (scrc != 0 && scrc != crc)
									continue;
								if (adr != 1)
									continue;

								sTrack = fromBCD(_subchannelBuffer[offs + 1]);
								sIndex = fromBCD(_subchannelBuffer[offs + 2]);

								if (sTrack < _toc2.FirstAudio || sTrack >= _toc2.FirstAudio + _toc2.AudioTracks)
									continue;

								int mm = fromBCD(_subchannelBuffer[offs + 7]);
								int ss = fromBCD(_subchannelBuffer[offs + 8]);
								int ff = fromBCD(_subchannelBuffer[offs + 9]);
								sPos = ff + 75 * (ss + 60 * mm) - 150;
								break;
							}
						default:
							continue;
					}

					bool preemph = (ctl & 1) == 1;
					bool dcp = (ctl & 2) == 2;
					if (preemph)
						_toc2[sTrack].PreEmphasis = true;
					if (dcp)
						_toc2[sTrack].DCP = true;

					if (sPos <= sec0 || sPos > sec1)
						continue;
					if (sTrack > iTrack || (sTrack == iTrack && iIndex >= 0 && sIndex > iIndex))
					{
						LocateLastSector(sec0, sPos - 1, iTrack, iIndex, ref maxIndex, out pos);
						return;
					}
					if (sTrack < iTrack || (sTrack == iTrack && (iIndex < 0 || sIndex <= iIndex)))
					{
						if (sTrack == iTrack && iIndex < 0)
							maxIndex = sIndex;
						LocateLastSector(sPos, sec1, iTrack, iIndex, ref maxIndex, out pos);
						return;
					}
				}
			if (sec1 <= sec0 + 16)
			{
				pos = Math.Min(sec0, sec1);
				return;
			}

			// TODO: catch?
			throw new Exception("gap detection failed");
		}

		private unsafe void TestGaps()
		{
			_gapDetection = GapDetectionMethod.None;

			//st = m_device.Seek((uint)(sector + i * 33) + _toc[_toc.FirstAudio][0].Start);
			//if (st != Device.CommandStatus.Success)
			//    break;
			//bool ready;
			//st = m_device.TestUnitReady(out ready);
			//if (st != Device.CommandStatus.Success)
			//    break;
			//if (!ready)
			//{
			//    st = Device.CommandStatus.NotSupported;
			//    break;
			//}

			// try ReadCD:
			Device.CommandStatus st;
			int sector = 3;

			if (_readCDCommand == ReadCDCommand.ReadCdBEh)
			{
                // PLEXTOR PX-W1210A always returns data, even if asked only for subchannel.
                // So we fill the buffer with magic data, give extra space for command so it won't hang the drive,
                // request subchannel data and check if magic data was overwritten.
                bool overwritten = false;
                for (int i = 0; i < 16; i++)
                {
                    _subchannelBuffer[m_max_sectors * (588 * 4 + 16) - 16 + i] = (byte)(13 + i);
                }
                fixed (byte* data = _subchannelBuffer)
                {
                    st = m_device.ReadCDAndSubChannel(Device.MainChannelSelection.None, Device.SubChannelMode.QOnly, Device.C2ErrorMode.None, 1, false, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)m_max_sectors, (IntPtr)((void*)data), _timeout);
                }
                for (int i = 0; i < 16; i++)
                {
                    if (_subchannelBuffer[m_max_sectors * (588 * 4 + 16) - 16 + i] != (byte)(13 + i))
                        overwritten = true;
                }
                if (overwritten)
                    st = Device.CommandStatus.NotSupported;
                //else
                //    st = m_device.ReadSubChannel(2, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)m_max_sectors, ref _subchannelBuffer, _timeout);
				if (st == Device.CommandStatus.Success)
				{
					int[] goodsecs = new int[2];
					for (int bcd = 1; bcd >= 0; bcd--)
					{
						for (int i = 0; i < m_max_sectors; i++)
						{
							int adr = _subchannelBuffer[i * 16 + 0] & 7;
							if (bcd == 0 && adr == 1)
							{
								_subchannelBuffer[i * 16 + 3] = toBCD(_subchannelBuffer[i * 16 + 3]);
								_subchannelBuffer[i * 16 + 4] = toBCD(_subchannelBuffer[i * 16 + 4]);
								_subchannelBuffer[i * 16 + 5] = toBCD(_subchannelBuffer[i * 16 + 5]);
								_subchannelBuffer[i * 16 + 7] = toBCD(_subchannelBuffer[i * 16 + 7]);
								_subchannelBuffer[i * 16 + 8] = toBCD(_subchannelBuffer[i * 16 + 8]);
								_subchannelBuffer[i * 16 + 9] = toBCD(_subchannelBuffer[i * 16 + 9]);
							}

							ushort crc = _crc.ComputeChecksum(_subchannelBuffer, i * 16, 10);
							crc ^= 0xffff;
							ushort scrc = (ushort)((_subchannelBuffer[i * 16 + 10] << 8) | _subchannelBuffer[i * 16 + 11]);
							if (scrc != 0 && scrc != crc)
								continue;
							if (adr != 1)
								continue;

							int sTrack = fromBCD(_subchannelBuffer[i * 16 + 1]);
							int sIndex = fromBCD(_subchannelBuffer[i * 16 + 2]);

							if (sTrack == 0 || sTrack == 110)
								continue;

							int mm = fromBCD(_subchannelBuffer[i * 16 + 7]);
							int ss = fromBCD(_subchannelBuffer[i * 16 + 8]);
							int ff = fromBCD(_subchannelBuffer[i * 16 + 9]);
							int sPos = ff + 75 * (ss + 60 * mm) - 150;

							if (sPos < sector + i - 8 || sPos > sector + i + 8)
								continue;

							goodsecs[bcd]++;
						}
					}

					if (goodsecs[0] > 0 || goodsecs[1] > 0)
					{
						_qChannelInBCD = goodsecs[1] >= goodsecs[0];
						_gapDetection = GapDetectionMethod.ReadCD;
					}
				}
			}

			if (_gapDetection == GapDetectionMethod.None)
			{
				fixed (byte* data = _subchannelBuffer)
				{
					// seek to given sector
					if (_readCDCommand == ReadCDCommand.ReadCdBEh)
						st = m_device.ReadCDAndSubChannel(_mainChannelMode, Device.SubChannelMode.None, _c2ErrorMode, 1, false, (uint)sector + _toc[_toc.FirstAudio][0].Start, 1, (IntPtr)((void*)data), _timeout);
					else
						st = m_device.ReadCDDA(Device.SubChannelMode.None, (uint)sector + _toc[_toc.FirstAudio][0].Start, 1, (IntPtr)((void*)data), _timeout);
				}
				if (st == Device.CommandStatus.Success)
				{
					st = m_device.ReadSubChannel42(1, 0, ref _subchannelBuffer, 0, _timeout);
					if (st == Device.CommandStatus.Success)
					{
						if (_subchannelBuffer[0] == 0 && _subchannelBuffer[2] == 0 && _subchannelBuffer[3] == 12 && _subchannelBuffer[4] == 1)
						{
							int ctl = _subchannelBuffer[5] & 0xf;
							int adr = (_subchannelBuffer[5] >> 4) & 0xf;
							if (adr == 1)
							{
								_gapDetection = GapDetectionMethod.ReadSubchannel;
							}
						}
					}
				}
			}
		}

		public bool GapsDetected
		{
			get
			{
				return gapsDetected;
			}
		}

        public unsafe void EjectDisk()
        {
            if (m_device != null)
            {
                m_device.StartStopUnit(true, Device.PowerControl.NoChange, Device.StartState.EjectDisk);
            }
            else
            {
                try
                {
                    m_device = new Device(m_logger);
                    if (m_device.Open(m_device_letter))
                    {
                        try
                        {
                            m_device.StartStopUnit(true, Device.PowerControl.NoChange, Device.StartState.LoadDisk);
                        }
                        finally
                        {
                            m_device.Close();
                        }
                    }
                }
                finally
                {
                    m_device = null;
                }
            }
        }

		bool gapsDetected = false;

		public unsafe bool DetectGaps()
		{
			if (!TestReadCommand())
				throw new ReadCDException(Resource1.AutodetectReadCommandFailed + ":\n" + _autodetectResult);

			if (_gapDetection == GapDetectionMethod.None)
			{
				gapsDetected = false;
				return false;
			}

			if (gapsDetected)
				return true;

			_toc2 = (CDImageLayout)_toc.Clone();

			if (_gapDetection == GapDetectionMethod.ReadSubchannel)
			{
				Device.CommandStatus st = m_device.ReadSubChannel42(2, 0, ref _subchannelBuffer, 0, _timeout);
				if (st == Device.CommandStatus.Success)
					if (_subchannelBuffer[0] == 0 && _subchannelBuffer[2] == 0 && _subchannelBuffer[3] == 20
						&& _subchannelBuffer[4] == 2 && _subchannelBuffer[8] == 0x80)
					{
						string catalog = Encoding.ASCII.GetString(_subchannelBuffer, 9, 13);
						if (catalog.ToString() != "0000000000000")
							_toc2.Barcode = catalog.ToString();
					}
			}

			int sec0 = (int)_toc2[_toc2.FirstAudio][0].Start, disc1 = (int)(_toc2[_toc2.FirstAudio][0].Start + _toc2.AudioLength) - 1;
			for (int iTrack = _toc2.FirstAudio; iTrack < _toc2.FirstAudio + _toc2.AudioTracks; iTrack++)
			{
				if (ReadProgress != null)
				{
					progressArgs.Action = Resource1.StatusDetectingGaps;
					progressArgs.Pass = -1;
					progressArgs.Position = (iTrack - _toc2.FirstAudio) * 3;
					progressArgs.PassStart = 0;
					progressArgs.PassEnd = _toc2.TrackCount * 3 - 1;
					progressArgs.ErrorsCount = 0;
					progressArgs.PassTime = DateTime.Now;
					ReadProgress(this, progressArgs);
				}
				int sec1, idx1 = 1;
				LocateLastSector(sec0, Math.Min(disc1, (int)_toc[iTrack].End + 16), iTrack, -1, ref idx1, out sec1);
				int isec0 = sec0;
				for (int idx = 0; idx <= idx1; idx++)
				{
					int isec1 = sec1, iidx1 = 1;
					if (idx < idx1)
					{
						if (ReadProgress != null)
						{
							progressArgs.Position = (iTrack - _toc2.FirstAudio) * 3 + 1;
							progressArgs.PassTime = DateTime.Now;
							ReadProgress(this, progressArgs);
						}
						LocateLastSector(isec0, sec1, iTrack, idx, ref iidx1, out isec1);
					}
					if (isec1 > isec0)
					{
						if (idx == 0 && iTrack > 1)
							_toc2[iTrack][0].Start = _toc2[iTrack].Start - (uint)(isec1 - isec0 + 1);
						if (idx > 1)
							_toc2[iTrack].AddIndex(new CDTrackIndex((uint)idx, (uint)(_toc2[iTrack][0].Start + isec0 - sec0)));
					}
					isec0 = isec1 + 1;
				}

				if (ReadProgress != null)
				{
					progressArgs.Position = (iTrack - _toc2.FirstAudio) * 3 + 2;
					progressArgs.PassTime = DateTime.Now;
					ReadProgress(this, progressArgs);
				}

				if (_gapDetection == GapDetectionMethod.ReadSubchannel)
				{
					Device.CommandStatus st = m_device.ReadSubChannel42(3, iTrack, ref _subchannelBuffer, 0, _timeout);
					if (st == Device.CommandStatus.Success)
						if (_subchannelBuffer[0] == 0 && _subchannelBuffer[2] == 0 && _subchannelBuffer[3] == 20
							&& _subchannelBuffer[4] == 3 && _subchannelBuffer[8] == 0x80) //&& _subchannelBuffer[6] == iTrack)
						{
							string isrc = Encoding.ASCII.GetString(_subchannelBuffer, 9, 12);
							if (!isrc.ToString().Contains("#") && isrc.ToString() != "000000000000")
								_toc2[iTrack].ISRC = isrc.ToString();
						}
				}
				if (_gapDetection == GapDetectionMethod.ReadCD)
				{
					Device.CommandStatus st = m_device.ReadSubChannel(2, _toc2[iTrack].Start + 16, 100, ref _subchannelBuffer, _timeout);
					if (st == Device.CommandStatus.Success)
					{
						for (int offs = 0; offs < 100 * 16; offs += 16)
						{
							int ctl = _subchannelBuffer[offs + 0] >> 4;
							int adr = _subchannelBuffer[offs + 0] & 7;
							if (adr != 2 && adr != 3)
								continue;
							ushort crc = _crc.ComputeChecksum(_subchannelBuffer, offs, 10);
							crc ^= 0xffff;
							ushort scrc = (ushort)((_subchannelBuffer[offs + 10] << 8) | _subchannelBuffer[offs + 11]);
							if (scrc != 0 && scrc != crc)
								continue;
							if (adr == 3 && _toc2[iTrack].ISRC == null)
							{
								StringBuilder isrc = new StringBuilder();
								isrc.Append(from6bit(_subchannelBuffer[offs + 1] >> 2));
								isrc.Append(from6bit(((_subchannelBuffer[offs + 1] & 0x3) << 4) + (0x0f & (_subchannelBuffer[offs + 2] >> 4))));
								isrc.Append(from6bit(((_subchannelBuffer[offs + 2] & 0xf) << 2) + (0x03 & (_subchannelBuffer[offs + 3] >> 6))));
								isrc.Append(from6bit((_subchannelBuffer[offs + 3] & 0x3f)));
								isrc.Append(from6bit(_subchannelBuffer[offs + 4] >> 2));
								isrc.Append(from6bit(((_subchannelBuffer[offs + 4] & 0x3) << 4) + (0x0f & (_subchannelBuffer[offs + 5] >> 4))));
								isrc.AppendFormat("{0:x}", _subchannelBuffer[offs + 5] & 0xf);
								isrc.AppendFormat("{0:x2}", _subchannelBuffer[offs + 6]);
								isrc.AppendFormat("{0:x2}", _subchannelBuffer[offs + 7]);
								isrc.AppendFormat("{0:x}", _subchannelBuffer[offs + 8] >> 4);
								if (!isrc.ToString().Contains("#") && isrc.ToString() != "000000000000")
									_toc2[iTrack].ISRC = isrc.ToString();
							}
							if (adr == 2 && _toc2.Barcode == null)
							{
								StringBuilder barcode = new StringBuilder();
								for (int i = 1; i < 8; i++)
									barcode.AppendFormat("{0:x2}", _subchannelBuffer[offs + i]);
								if (barcode.ToString(0, 13) != "0000000000000")
									_toc2.Barcode = barcode.ToString(0, 13);
							}
						}
					}
				}
				sec0 = sec1 + 1;
			}

			gapsDetected = true;
			return true;
		}

		bool readCommandFound = false;

		public unsafe bool TestReadCommand()
		{
			if (readCommandFound)
				return true;

			//ReadCDCommand[] readmode = { ReadCDCommand.ReadCdBEh, ReadCDCommand.ReadCdD8h };
            ReadCDCommand[] readmode = { ReadCDCommand.ReadCdBEh, ReadCDCommand.ReadCdD8h };
			Device.C2ErrorMode[] c2mode = { Device.C2ErrorMode.Mode294, Device.C2ErrorMode.Mode296, Device.C2ErrorMode.None };
			Device.MainChannelSelection[] mainmode = { Device.MainChannelSelection.UserData, Device.MainChannelSelection.F8h };
			bool found = false;
			_autodetectResult = "";
			_currentStart = 0;
			m_max_sectors = Math.Min(NSECTORS, m_device.MaximumTransferLength / CB_AUDIO - 1);
			int sector = 3;
			int pass = 0;

			for (int c = 0; c <= 2 && !found; c++)
				for (int r = 0; r <= 1 && !found; r++)
					for (int m = 0; m <= 1 && !found; m++)
					{
						_readCDCommand = readmode[r];
						_c2ErrorMode = c2mode[c];
						_mainChannelMode = mainmode[m];
						if (_forceReadCommand != ReadCDCommand.Unknown && _readCDCommand != _forceReadCommand)
							continue;
                        if (_readCDCommand == ReadCDCommand.ReadCdD8h && (_c2ErrorMode != Device.C2ErrorMode.None || _mainChannelMode != Device.MainChannelSelection.UserData))
							continue;
						Array.Clear(_readBuffer, 0, _readBuffer.Length); // fill with something nasty instead?
						DateTime tm = DateTime.Now;
						if (ReadProgress != null)
						{
							progressArgs.Action = Resource1.StatusDetectingDriveFeatures;
							progressArgs.Pass = -1;
							progressArgs.Position = pass++;
							progressArgs.PassStart = 0;
							progressArgs.PassEnd = 2 * 3 * 2 - 1;
							progressArgs.ErrorsCount = 0;
							progressArgs.PassTime = tm;
							ReadProgress(this, progressArgs);
						}
						System.Diagnostics.Trace.WriteLine("Trying " + CurrentReadCommand);
						Device.CommandStatus st = FetchSectors(sector, m_max_sectors, false);
						TimeSpan delay = DateTime.Now - tm;
						_autodetectResult += string.Format("{0}: {1} ({2}ms)\n", CurrentReadCommand, (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()), delay.TotalMilliseconds);
						found = st == Device.CommandStatus.Success;
						
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
            {
                TestGaps();
                _autodetectResult += "Chosen " + CurrentReadCommand + "\n";
            }
            else
            {
                _gapDetection = GapDetectionMethod.None;
                _readCDCommand = ReadCDCommand.Unknown;
            }

			_currentStart = -1;
			_currentEnd = -1;
			readCommandFound = found;
			return found;
		}

		//int dbg_pass;
		//FileStream fs_d, fs_c;

		private unsafe void ReorganiseSectors(int sector, int Sectors2Read)
		{
			int c2Size = _c2ErrorMode == Device.C2ErrorMode.None ? 0 : _c2ErrorMode == Device.C2ErrorMode.Mode294 ? 294 : 296;
			int oldSize = 4 * 588 + c2Size;
			fixed (byte* readBuf = _readBuffer, c2Count = C2Count)
			fixed (long* userData = UserData)
			{
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					byte* sectorPtr = readBuf + iSector * oldSize;
					long* userDataPtr = userData + (sector - _currentStart + iSector) * 8 * 588;
					byte* c2CountPtr = c2Count + (sector - _currentStart + iSector) * 294;

					//if (_currentStart > 0)
					//{
					//    string nm_d = string.Format("Y:\\Temp\\dbg\\{0:x}-{1:00}.bin", _currentStart, dbg_pass);
					//    string nm_c = string.Format("Y:\\Temp\\dbg\\{0:x}-{1:00}.c2", _currentStart, dbg_pass);
					//    if (fs_d != null && fs_d.Name != nm_d) { fs_d.Close(); fs_d = null; }
					//    if (fs_c != null && fs_c.Name != nm_c) { fs_c.Close(); fs_c = null; }
					//    if (fs_d == null) fs_d = new FileStream(nm_d, FileMode.Create);
					//    if (fs_c == null) fs_c = new FileStream(nm_c, FileMode.Create);
					//    fs_d.Seek((sector - _currentStart + iSector) * 4 * 588, SeekOrigin.Begin);
					//    fs_d.Write(_readBuffer, iSector * oldSize, 4 * 588);
					//    fs_c.Seek((sector - _currentStart + iSector) * 296, SeekOrigin.Begin);
					//    fs_c.Write(_readBuffer, iSector * oldSize + 4 * 588, 296);
					//}

					if (_c2ErrorMode != Device.C2ErrorMode.None)
					{
						int offs = 0;
						if (c2Size == 296)
						{
							// TODO: sometimes (e.g on PIONEER 215D) sector C2 byte is placed after C2 info,
							// not before, like it says in mmc6r02g.pdf !!
							int c2 = 0;
							for (int pos = 2; pos < 294; pos++)
								c2 |= sectorPtr[4 * 588 + pos];
							if (sectorPtr[4 * 588 + 294] == (c2 | sectorPtr[4 * 588 + 0] | sectorPtr[4 * 588 + 1]))
								offs = 0;
							else
								offs = 2;
							// TSSTcorp CDDVDW SH-S203B SB03
							// TSSTcorpCD/DVDW SH-W162C TS12
							//	Both produced Exception("invalid C2 pointers");
							//else if (sectorPtr[4 * 588] == (c2 | sectorPtr[4 * 588 + 294] | sectorPtr[4 * 588 + 295]))
							//    offs = 2;
							//else 
							//    throw new Exception("invalid C2 pointers");
						}
						for (int pos = 0; pos < 294; pos++)
						{
							int c2d = sectorPtr[4 * 588 + pos + offs]; 
							int c2 = ((-c2d) >> 31) & 1;
							c2CountPtr[pos] += (byte)c2;
							int sample = pos << 3;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]]; sample++;
							userDataPtr[sample + c2 * 4 * 588] += byte2long[sectorPtr[sample]];
						}
					}
					else
					{
						for (int sample = 0; sample < 4 * 588; sample++)
							userDataPtr[sample] += byte2long[sectorPtr[sample]];
					}
				}
			}
		}

		private unsafe void ClearSectors(int sector, int Sectors2Read)
		{
			fixed (long* userData = &UserData[sector - _currentStart, 0, 0])
			fixed (byte* c2Count = &C2Count[sector - _currentStart, 0])
			{
				AudioSamples.MemSet((byte*)userData, 0, 8 * 2 * 4 * 588 * Sectors2Read);
				AudioSamples.MemSet(c2Count, 0, 294 * Sectors2Read);
			}
		}

		private unsafe Device.CommandStatus FetchSectors(int sector, int Sectors2Read, bool abort)
		{
			Device.CommandStatus st;
			fixed (byte* data = _readBuffer)
			{
				if (_debugMessages)
				{
					int size = (4 * 588 + (_c2ErrorMode == Device.C2ErrorMode.Mode294 ? 294 : _c2ErrorMode == Device.C2ErrorMode.Mode296 ? 296 : 0)) 
						* (int)Sectors2Read;
					AudioSamples.MemSet(data, 0xff, size);
				}
				if (_readCDCommand == ReadCDCommand.ReadCdBEh)
					st = m_device.ReadCDAndSubChannel(_mainChannelMode, Device.SubChannelMode.None, _c2ErrorMode, 1, false, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)Sectors2Read, (IntPtr)((void*)data), _timeout);
				else
					st = m_device.ReadCDDA(Device.SubChannelMode.None, (uint)sector + _toc[_toc.FirstAudio][0].Start, (uint)Sectors2Read, (IntPtr)((void*)data), _timeout);
			}

			if (st == Device.CommandStatus.Success)
			{
				ReorganiseSectors(sector, Sectors2Read);
				return st;
			}

			if (!abort)
				return st;
			SCSIException ex = new SCSIException(Resource1.ReadCDError, m_device, st);
			if (sector != 0 && Sectors2Read > 1 && st == Device.CommandStatus.DeviceFailed && m_device.GetSenseAsc() == 0x64 && m_device.GetSenseAscq() == 0x00)
			{
				if (_debugMessages)
					System.Console.WriteLine("\n{0}: retrying one sector at a time", ex.Message);
				int iErrors = 0;
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					if (FetchSectors(sector + iSector, 1, false) != Device.CommandStatus.Success)
					{
						iErrors ++;
						for (int i = 0; i < 294; i++)
							C2Count[sector + iSector - _currentStart, i] ++;
						if (_debugMessages)
							System.Console.WriteLine("\nSector lost");
					}
				}
				if (iErrors < Sectors2Read)
					return Device.CommandStatus.Success;
			}
			throw ex;
		}

		private unsafe void CorrectSectors(int pass, int sector, int Sectors2Read, bool markErrors)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				// avg - pass + 1
				// p  a  l  o
				// 0  1  1  2
				// 2  4  3  3
				// 4  7  2  4
				// 6 10     5
				//16 25    10
				bool fError = false;
				const byte c2div = 128;
				int er_limit = c2div * (1 + _correctionQuality) - 1;
				// need at least 1 + _correctionQuality good passes
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					long val = UserData[pos, 0, iPar];
					long val1 = UserData[pos, 1, iPar];
					byte c2 = C2Count[pos, iPar >> 3];
					int ave = (pass + 1 - c2) * c2div + c2;
					int bestValue = 0;
					for (int i = 0; i < 8; i++)
					{
						int sum = ave - 2 * (int)((val & 0xff) * c2div + (val1 & 0xff));
						int sig = sum >> 31; // bit value
						fError |= (sum ^ sig) < er_limit;
						bestValue += sig & (1 << i);
						val >>= 8;
						val1 >>= 8;
					}
					currentData.Bytes[pos * 4 * 588 + iPar] = (byte)bestValue;
				}
				int newerr = (fError ? 1 : 0);
				//_currentErrorsCount += newerr;
				_currentErrorsCount += newerr - errtmp[pos];
				errtmp[pos] = newerr;
				if (markErrors)
				{
					_errors[sector + iSector] |= fError;
					_errorsCount += fError ? 1 : 0;
				}
			}

		}

		int[] errtmp = new int[MSECTORS];

		public unsafe void PrefetchSector(int iSector)
		{
			if (iSector >= _currentStart && iSector < _currentEnd)
				return;

			if (!TestReadCommand())
				throw new ReadCDException(Resource1.AutodetectReadCommandFailed + "\n" + _autodetectResult);

			_currentStart = iSector;
			_currentEnd = _currentStart + MSECTORS;
			if (_currentEnd > (int)_toc.AudioLength)
			{
				_currentEnd = (int)_toc.AudioLength;
				_currentStart = Math.Max(0, _currentEnd - MSECTORS);
			}

			int neededSize = (_currentEnd - _currentStart) * 588;
			if (currentData.Size < neededSize)
				currentData.Prepare(new byte[neededSize * 4], neededSize);
			currentData.Length = neededSize;

			//FileStream correctFile = new FileStream("correct.wav", FileMode.Open);
			//byte[] realData = new byte[MSECTORS * 4 * 588];
			//correctFile.Seek(0x2C + _currentStart * 588 * 4, SeekOrigin.Begin);
			//if (correctFile.Read(realData, _driveOffset * 4, MSECTORS * 4 * 588 - _driveOffset * 4) != MSECTORS * 4 * 588 - _driveOffset * 4)
			//    throw new Exception("read");
			//correctFile.Close();

			_currentErrorsCount = 0;
			for (int i = 0; i < MSECTORS; i++)
				errtmp[i] = 0;

			//Device.CommandStatus st = m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, (ushort)(176 * 4), 65535);
			//if (st != Device.CommandStatus.Success)
			//    System.Console.WriteLine("SetCdSpeed: {0}", (st == Device.CommandStatus.DeviceFailed ? Device.LookupSenseError(m_device.GetSenseAsc(), m_device.GetSenseAscq()) : st.ToString()));

			// TODO:
			//int max_scans = 1 << _correctionQuality;
			int max_scans = 16 << _correctionQuality;
			for (int pass = 0; pass < max_scans; pass++)
			{
//				dbg_pass = pass;
				DateTime PassTime = DateTime.Now, LastFetch = DateTime.Now;

				for (int sector = _currentStart; sector < _currentEnd; sector += m_max_sectors)
				{
					int Sectors2Read = Math.Min(m_max_sectors, _currentEnd - sector);
					int speed = pass == 5 ? 300 : pass == 6 ? 150 : pass == 7 ? 75 : 32500; // sectors per second
					int msToSleep = 1000 * Sectors2Read / speed - (int)((DateTime.Now - LastFetch).TotalMilliseconds);

					//if (msToSleep > 0) Thread.Sleep(msToSleep);

					LastFetch = DateTime.Now;
					if (pass == 0) 
						ClearSectors(sector, Sectors2Read);
					FetchSectors(sector, Sectors2Read, true);
					//TimeSpan delay1 = DateTime.Now - LastFetch;
					//DateTime LastFetched = DateTime.Now;
					if (pass >= _correctionQuality)
						CorrectSectors(pass, sector, Sectors2Read, pass == max_scans - 1);
					//TimeSpan delay2 = DateTime.Now - LastFetched;
					//if (sector == _currentStart)
					//System.Console.WriteLine("\n{0},{1}", delay1.TotalMilliseconds, delay2.TotalMilliseconds);
					if (ReadProgress != null)
					{
						progressArgs.Action = Resource1.StatusRipping;
						progressArgs.Position = sector + Sectors2Read;
						progressArgs.Pass = pass;
						progressArgs.PassStart = _currentStart;
						progressArgs.PassEnd = _currentEnd;
						progressArgs.ErrorsCount = _currentErrorsCount;
						progressArgs.PassTime = PassTime;
						ReadProgress(this, progressArgs);
					}
				}
				if (pass >= _correctionQuality && _currentErrorsCount == 0)
					break;
			}
		}

		public unsafe int Read(AudioBuffer buff, int maxLength)
		{
			if (_toc == null)
				throw new ReadCDException("Read: invalid TOC");
			buff.Prepare(this, maxLength);
			if (Position >= Length)
				return 0;
			if (_sampleOffset >= Length)
			{
				for (int i = 0; i < buff.ByteLength; i++)
					buff.Bytes[i] = 0;
				_sampleOffset += buff.Length;
				return buff.Length; // == Remaining
			}
			if (_sampleOffset < 0)
			{
				buff.Length = Math.Min(buff.Length, -_sampleOffset);
				for (int i = 0; i < buff.ByteLength; i++)
					buff.Bytes[i] = 0;
				_sampleOffset += buff.Length;
				return buff.Length;
			}
			PrefetchSector(/*(int)_toc[_toc.FirstAudio][0].Start +*/ (_sampleOffset / 588));
			buff.Length = Math.Min(buff.Length, (int)Length - _sampleOffset);
			buff.Length = Math.Min(buff.Length, _currentEnd * 588 - _sampleOffset);
			if ((_sampleOffset - _currentStart * 588) == 0 && (maxLength < 0 || (_currentEnd - _currentStart) * 588 <= buff.Length))
			{
				buff.Swap(currentData);
				_currentStart = -1;
				_currentEnd = -1;
			} 
			else
				fixed (byte* dest = buff.Bytes, src = &currentData.Bytes[(_sampleOffset - _currentStart * 588) * 4])
					AudioSamples.MemCpy(dest, src, buff.ByteLength);
			_sampleOffset += buff.Length;
			return buff.Length;
		}

		public long Length
		{
			get
			{
				if (_toc == null)
					throw new ReadCDException("invalid TOC");
				return 588 * (int)_toc.AudioLength;
			}
		}

		public AudioPCMConfig PCM
		{
			get
			{
				return AudioPCMConfig.RedBook;
			}
		}

		public string Path
		{
			get
			{
				string result = m_device_letter + ": ";
				if (m_inqury_result != null && m_inqury_result.Valid)
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
				return m_inqury_result == null || !m_inqury_result.Valid ? null :
					m_inqury_result.VendorIdentification.TrimEnd(' ', '\0').PadRight(8, ' ') + " - " + 
					m_inqury_result.ProductIdentification.TrimEnd(' ', '\0');
			}
		}

		public string EACName
		{
			get
			{
				return m_inqury_result.VendorIdentification.TrimEnd(' ', '\0') + " " + m_inqury_result.ProductIdentification.TrimEnd(' ', '\0');
			}
		}

		public long Position
		{
			get
			{
				return _sampleOffset - _driveOffset;
			}
			set
			{
				if (_toc == null || _toc.AudioLength <= 0)
					throw new ReadCDException(Resource1.NoAudio);
				_crcErrorsCount = 0;
				_errorsCount = 0;
				_currentStart = -1;
				_currentEnd = -1;
				_errors = new BitArray((int)_toc.AudioLength); // !!!
				_sampleOffset = (int)value + _driveOffset;
			}
		}

		public long Remaining
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
				if (value < 0 || value > 3)
					throw new Exception("invalid CorrectionQuality");
				_correctionQuality = value;
			}
		}		

		public string RipperVersion
		{
			get
			{
				return "CUERipper v2.1.5 Copyright (C) 2008-13 Grigory Chudov";
				// ripper.GetName().Name + " " + ripper.GetName().Version;
			}
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
			: base(args + ": " + (st == Device.CommandStatus.DeviceFailed ? device.GetErrorString() : st.ToString()))
		{
		}
	}

	public sealed class ReadCDException : Exception
	{
		public ReadCDException(string args, Exception inner)
			: base(args + ": " + inner.Message, inner) { }
		public ReadCDException(string args)
			: base(args) { }
	}
}
