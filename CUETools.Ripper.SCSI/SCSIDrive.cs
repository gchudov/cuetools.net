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
using Bwg.Scsi;
using Bwg.Logging;
using AudioCodecsDotNet;

namespace CUETools.Ripper.SCSI
{
	/// <summary>
	/// 
	/// </summary>
	public class CDDriveReader : IAudioSource
	{
		byte[] cdtext = null;
		private Device m_device;
		uint _sampleOffset = 0;
		uint _samplesInBuffer = 0;
		uint _samplesBufferOffset = 0;
		uint _samplesBufferSector = 0;
		const int CB_AUDIO = 588 * 4 + 16;
		const int NSECTORS = 32;
		int _currentTrack = -1, _currentIndex = -1, _currentTrackActualStart = -1;
		Logger m_logger = null;
		CDImage _toc;

		public CDImage TOC
		{
			get
			{
				return _toc;
			}
		}

		public CDDriveReader()
		{
		}

		public bool Open(char Drive)
		{
			Device.CommandStatus st;

			// Open the base device
			m_device = new Device(m_logger);
			if (!m_device.Open(Drive))
				throw new Exception("SCSI error");

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
				throw new Exception("SCSI error");

			st = m_device.SetCdSpeed(Device.RotationalControl.CLVandNonPureCav, Device.OptimumSpeed, Device.OptimumSpeed);
			if (st != Device.CommandStatus.Success)
				throw new Exception("SCSI error");

			IList<TocEntry> toc;
			st = m_device.ReadToc((byte)0, false, out toc);
			if (st != Device.CommandStatus.Success)
				throw new Exception("SCSI error");

			st = m_device.ReadCDText(out cdtext);
			// new CDTextEncoderDecoder

			_toc = new CDImage(toc[toc.Count - 1].StartSector);
			uint cddbDiscId = 0;
			uint discId1 = 0;
			uint discId2 = 0;
			for (int iTrack = 0; iTrack < toc.Count - 1; iTrack++)
			{
				_toc.tracks.Add(new CDTrack((uint)iTrack + 1, toc[iTrack].StartSector,
					toc[iTrack + 1].StartSector - toc[iTrack].StartSector));
				discId1 += toc[iTrack].StartSector;
				discId2 += (toc[iTrack].StartSector == 0 ? 1 : toc[iTrack].StartSector) * ((uint)iTrack + 1);
				cddbDiscId += sumDigits((uint)(toc[iTrack].StartSector / 75) + 2);
			}
			discId1 += toc[toc.Count - 1].StartSector;
			discId2 += (toc[toc.Count - 1].StartSector == 0 ? 1 : toc[toc.Count - 1].StartSector) * ((uint)toc.Count);
			discId1 &= 0xFFFFFFFF;
			discId2 &= 0xFFFFFFFF;
			cddbDiscId = (((cddbDiscId % 255) << 24) +
				(((uint)(toc[toc.Count - 1].StartSector / 75) - (uint)(toc[0].StartSector / 75)) << 8) +
				(uint)(toc.Count - 1)) & 0xFFFFFFFF;
			_toc._cddbId = string.Format("{0:X8}", cddbDiscId);
			_toc._ArId = string.Format("{0:x8}-{1:x8}-{2:x8}", discId1, discId2, cddbDiscId);
			return true;
		}

		public void Close()
		{
			_toc = null;
		}

		public int BestBlockSize
		{
			get
			{
				return Math.Min(m_device.MaximumTransferLength / CB_AUDIO, NSECTORS) * 588;
			}
		}

		public unsafe uint Read(int[,] buff, uint sampleCount)
		{
			if (_toc == null)
				throw new Exception("invalid TOC");
			if (_sampleOffset >= (uint)Length)
				return 0;
			if (_sampleOffset + sampleCount > Length)
				sampleCount = (uint)Length - _sampleOffset;
			uint pos = 0;
			if (_samplesInBuffer > 0)
			{
				uint samplesRead = Math.Min(_samplesInBuffer, sampleCount);
				AudioSamples.BytesToFLACSamples_16(_sectorBuffer, (int)(_samplesBufferSector * (588 * 4 + 16) + _samplesBufferOffset * 4), buff, (int)pos, samplesRead, 2);
				pos += samplesRead;
				sampleCount -= samplesRead;
				_sampleOffset += samplesRead;
				if (sampleCount == 0)
				{
					_samplesInBuffer -= samplesRead;
					_samplesBufferOffset += samplesRead;
					return pos;
				}
				_samplesInBuffer = 0;
				_samplesBufferOffset = 0;
				_samplesBufferSector = 0;
			}
			// if (_sampleOffset < PreGapLength && !_overreadIntoPreGap ... ?
			int firstSector = (int)_sampleOffset / 588;
			int lastSector = (int)(_sampleOffset + sampleCount + 577) / 588;
			for (int sector = firstSector; sector < lastSector; sector += NSECTORS)
			{
				int Sectors2Read = ((sector + NSECTORS) < lastSector) ? NSECTORS : (lastSector - sector);
				fixed (byte* data = _sectorBuffer)
				{
					Device.CommandStatus st = m_device.ReadCDAndSubChannel(2, 1, true, (uint)sector, (uint)Sectors2Read, (IntPtr)((void*)data), Sectors2Read * (2352 + 16));
					if (st != Device.CommandStatus.Success)
						throw new Exception("SCSI error");
				}
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
				{
					uint samplesRead = Math.Min(sampleCount, 588U) - (_sampleOffset % 588);
					AudioSamples.BytesToFLACSamples_16(_sectorBuffer, iSector * (588 * 4 + 16) + ((int)_sampleOffset % 588) * 4, buff, (int)pos, samplesRead, 2);
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
									if (iTrack != _currentTrack)
									{
										_currentTrack = iTrack;
										_currentTrackActualStart = sector + iSector;
										_currentIndex = iIndex;
										if (_currentIndex == 1)
											_toc.tracks[iTrack - 1].indexes.Add(new CDTrackIndex(1, _toc.tracks[iTrack - 1].Start.Sector));
										else if (_currentIndex != 0)
											throw new Exception("invalid index");
									}
									else
										if (iIndex != _currentIndex)
										{
											if (iIndex != _currentIndex + 1)
												throw new Exception("invalid index");
											_currentIndex = iIndex;
											if (_currentIndex == 1)
											{
												int pregap = sector + iSector - _currentTrackActualStart;
												_toc.tracks[iTrack - 1].indexes.Add(new CDTrackIndex(0, (uint)(_toc.tracks[iTrack - 1].Start.Sector - pregap)));
												_currentTrackActualStart = sector + iSector;
											}
											_toc.tracks[iTrack - 1].indexes.Add(new CDTrackIndex((uint)iIndex, (uint)(_toc.tracks[iTrack - 1].Start.Sector + sector + iSector - _currentTrackActualStart)));
											_currentIndex = iIndex;
										}
									break;
								}
							case 2: // catalog
								if (_toc._catalog == null)
								{
									StringBuilder catalog = new StringBuilder();
									for (int i = 1; i < 8; i++)
										catalog.AppendFormat("{0:x2}", _sectorBuffer[q_pos + i]);
									_toc._catalog = catalog.ToString(0, 13);
								}
								break;
							case 3: //isrc
								break;
						}
					}
					pos += samplesRead;
					sampleCount -= samplesRead;
					_sampleOffset += samplesRead;
					if (sampleCount == 0)
					{
						_samplesBufferSector = (uint)iSector;
						_samplesBufferOffset = samplesRead;
						_samplesInBuffer = 588U - samplesRead;
						return pos;
					}
				}
			}
			return pos;
		}

		public ulong Length
		{
			get
			{
				if (_toc == null)
					throw new Exception("invalid TOC");
				return (ulong)588 * _toc.Length.Sector;
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
				return m_device.Name;
			}
		}

		public ulong Position
		{
			get
			{
				return _sampleOffset;
			}
			set
			{
				_sampleOffset = (uint)value;
			}
		}

		public ulong Remaining
		{
			get
			{
				return Length - Position;
			}
		}

		byte[] _sectorBuffer = new byte[CB_AUDIO * NSECTORS];

		private int fromBCD(byte hex)
		{
			return (hex >> 4) * 10 + (hex & 15);
		}

		private uint sumDigits(uint n)
		{
			uint r = 0;
			while (n > 0)
			{
				r = r + (n % 10);
				n = n / 10;
			}
			return r;
		}
	}

	public class CDTrackIndex
	{
		public CDTrackIndex(uint index, uint sector)
		{
			_sector = sector;
			_index = index;
		}

		public uint Sector
		{
			get
			{
				return _sector;
			}
		}

		public uint Index
		{
			get
			{
				return _index;
			}
		}

		public string MSF
		{
			get
			{
				return new MinuteSecondFrame(_sector).ToString("M:S:F");
			}
		}

		uint _sector;
		uint _index;
	}

	public class CDTrack
	{
		public CDTrack(uint number, uint start, uint length)
		{
			_number = number;
			_start = start;
			_length = length;
			indexes = new List<CDTrackIndex>();
		}

		public CDTrackIndex Start
		{
			get
			{
				return new CDTrackIndex(0, _start);
			}
		}

		public CDTrackIndex Length
		{
			get
			{
				return new CDTrackIndex(0, _length);
			}
		}

		public CDTrackIndex End
		{
			get
			{
				return new CDTrackIndex(0, _start + _length - 1);
			}
		}

		public uint Number
		{
			get
			{
				return _number;
			}
		}

		public IList<CDTrackIndex> indexes;

		uint _start;
		uint _length;
		uint _number;
	}

	public class CDImage
	{
		public CDImage(uint length)
		{
			tracks = new List<CDTrack>();
			_length = length;
		}

		public IList<CDTrack> tracks;

		public CDTrackIndex Length
		{
			get
			{
				return new CDTrackIndex(0, _length);
			}
		}

		public string _catalog;
		public string _cddbId;
		public string _ArId;
		uint _length;
	}
}
