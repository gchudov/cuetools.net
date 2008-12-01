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
		uint _samplesInBuffer = 0;
		uint _samplesBufferOffset = 0;
		uint _samplesBufferSector = 0;
		int _driveOffset = 0;
		const int CB_AUDIO = 588 * 4 + 16;
		const int NSECTORS = 32;
		int _currentTrack = -1, _currentIndex = -1, _currentTrackActualStart = -1;
		Logger m_logger;
		CDImageLayout _toc;
		DeviceInfo m_info;

		public CDImageLayout TOC
		{
			get
			{
				return _toc;
			}
		}

		public CDDriveReader()
		{
			m_logger = new Logger();
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

			SpeedDescriptorList speed_list;
			st = m_device.GetSpeed(out speed_list);
			if (st != Device.CommandStatus.Success)
				throw new Exception("GetSpeed failed: SCSI error");

			st = m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, 32767/*Device.OptimumSpeed*/, Device.OptimumSpeed);
			if (st != Device.CommandStatus.Success)
				throw new Exception("SetCdSpeed failed: SCSI error");

			IList<TocEntry> toc;
			st = m_device.ReadToc((byte)0, false, out toc);
			if (st != Device.CommandStatus.Success)
				throw new Exception("ReadToc failed: SCSI error");

			st = m_device.ReadCDText(out cdtext);
			// new CDTextEncoderDecoder

			_toc = new CDImageLayout();
			for (int iTrack = 0; iTrack < toc.Count - 1; iTrack++)
				_toc.AddTrack(new CDTrack((uint)iTrack + 1, 
					toc[iTrack].StartSector,
					toc[iTrack + 1].StartSector - toc[iTrack].StartSector - 
					    ((toc[iTrack + 1].Control == 0 || iTrack + 1 == toc.Count - 1) ? 0U : 152U * 75U), 
					toc[iTrack].Control == 0));
			if (_toc[1].IsAudio)
				_toc[1][0].Start = 0;
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
				return Math.Min(m_device.MaximumTransferLength / CB_AUDIO, NSECTORS) * 588;
			}
		}

		private void ProcessSubchannel(int sector, int Sectors2Read)
		{
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int q_pos = (iSector + 1) * (588 * 4 + 16) - 16;
				int ctl = _sectorBuffer[q_pos + 0] >> 4;
				int adr = _sectorBuffer[q_pos + 0] & 7;
				bool preemph = (ctl == 1);
				switch (adr)
				{
					case 1: // current position
						{
							int iTrack = fromBCD(_sectorBuffer[q_pos + 1]);
							int iIndex = fromBCD(_sectorBuffer[q_pos + 2]);
							if (iTrack == 110)
								throw new Exception("lead out area encountred");
							if (iTrack == 0)
								throw new Exception("lead in area encountred");
							if (iTrack != _currentTrack)
							{
								_currentTrack = iTrack;
								_currentTrackActualStart = sector + iSector;
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
									uint pregap = (uint) (sector + iSector - _currentTrackActualStart);
									_toc[iTrack][0].Start = _toc[iTrack].Start - pregap;
									_currentTrackActualStart = sector + iSector;
								} else
									_toc[iTrack].AddIndex(new CDTrackIndex((uint)iIndex, (uint)(_toc[iTrack].Start + sector + iSector - _currentTrackActualStart)));
								_currentIndex = iIndex;
							}
							break;
						}
					case 2: // catalog
						if (_toc.Catalog == null)
						{
							StringBuilder catalog = new StringBuilder();
							for (int i = 1; i < 8; i++)
								catalog.AppendFormat("{0:x2}", _sectorBuffer[q_pos + i]);
							_toc.Catalog = catalog.ToString(0, 13);
						}
						break;
					case 3: //isrc
						if (_toc[_currentTrack].ISRC == null)
						{
							StringBuilder isrc = new StringBuilder();
							char[] ISRC6 = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#', '#', '#', '#', '#', '#', '#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
							isrc.Append(ISRC6[_sectorBuffer[q_pos + 1] >> 2]);
							isrc.Append(ISRC6[((_sectorBuffer[q_pos + 1] & 0x3) << 4) + (_sectorBuffer[q_pos + 2] >> 4)]);
							isrc.Append(ISRC6[((_sectorBuffer[q_pos + 2] & 0xf) << 2) + (_sectorBuffer[q_pos + 3] >> 6)]);
							isrc.Append(ISRC6[(_sectorBuffer[q_pos + 3] & 0x3f)]);
							isrc.Append(ISRC6[_sectorBuffer[q_pos + 4] >> 2]);
							isrc.Append(ISRC6[((_sectorBuffer[q_pos + 4] & 0x3) << 4) + (_sectorBuffer[q_pos + 5] >> 4)]);
							isrc.AppendFormat("{0:x}", _sectorBuffer[q_pos + 5] & 0xf);
							isrc.AppendFormat("{0:x2}", _sectorBuffer[q_pos + 6]);
							isrc.AppendFormat("{0:x2}", _sectorBuffer[q_pos + 7]);
							isrc.AppendFormat("{0:x}", _sectorBuffer[q_pos + 8] >> 4);
							_toc[_currentTrack].ISRC = isrc.ToString();
						}
						break;
				}
			}
		}

		private unsafe void FetchSectors(int sector, int Sectors2Read)
		{
			fixed (byte* data = _sectorBuffer)
			{
				Device.CommandStatus st = m_device.ReadCDAndSubChannel(2, 1, true, (uint)sector, (uint)Sectors2Read, (IntPtr)((void*)data), Sectors2Read * (2352 + 16));
				if (st != Device.CommandStatus.Success)
					throw new Exception("ReadCDAndSubChannel failed: SCSI error");
			}
			ProcessSubchannel(sector, Sectors2Read);
		}

		public unsafe uint Read(int[,] buff, uint sampleCount)
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
			if (_samplesInBuffer > 0)
			{
				uint samplesRead = Math.Min(_samplesInBuffer, sampleCount);
				AudioSamples.BytesToFLACSamples_16(_sectorBuffer, (int)(_samplesBufferSector * (588 * 4 + 16) + _samplesBufferOffset * 4), buff, (int)pos, samplesRead, 2);
				pos += samplesRead;
				sampleCount -= samplesRead;
				_sampleOffset += (int) samplesRead;
				if (sampleCount == 0)
				{
					_samplesInBuffer -= samplesRead;
					_samplesBufferOffset += samplesRead;
					if (silenceCount > 0)
					{
						uint nullSamplesRead = (uint) silenceCount;
						for (int i = 0; i < nullSamplesRead; i++)
							for (int c = 0; c < ChannelCount; c++)
								buff[pos + i, c] = 0;
						pos += nullSamplesRead;
						_sampleOffset += (int)nullSamplesRead;
					}
					return pos;
				}
				_samplesInBuffer = 0;
				_samplesBufferOffset = 0;
				_samplesBufferSector = 0;
			}
			// if (_sampleOffset < PreGapLength && !_overreadIntoPreGap ... ?
			int firstSector = (int)_sampleOffset / 588;
			int lastSector = (int)(_sampleOffset + sampleCount + 587) / 588;
			for (int sector = firstSector; sector < lastSector; sector += NSECTORS)
			{
				int Sectors2Read = ((sector + NSECTORS) < lastSector) ? NSECTORS : (lastSector - sector);
				FetchSectors(sector, Sectors2Read);
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					uint samplesRead = (uint) (Math.Min((int)sampleCount, 588) - (_sampleOffset % 588));
					AudioSamples.BytesToFLACSamples_16(_sectorBuffer, iSector * (588 * 4 + 16) + ((int)_sampleOffset % 588) * 4, buff, (int)pos, samplesRead, 2);
					pos += samplesRead;
					sampleCount -= samplesRead;
					_sampleOffset += (int) samplesRead;
					if (sampleCount == 0)
					{
						_samplesBufferSector = (uint)iSector;
						_samplesBufferOffset = samplesRead;
						_samplesInBuffer = 588U - samplesRead;
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
				}
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
				return (ulong)588 * (_toc[_toc.TrackCount].IsAudio ? _toc[_toc.TrackCount].End + 1 : _toc[_toc.TrackCount - 1].End + 1);
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

		public string Path
		{
			get
			{
				return m_info.LongDesc;
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

		byte[] _sectorBuffer = new byte[CB_AUDIO * NSECTORS];

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
}
