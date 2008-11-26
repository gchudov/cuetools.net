//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for DotNet
// 
// Copyright (C) 2006 by Jack W. Griffin (butchg@comcast.net)
//
// This program is free software; you can redistribute it and/or modify 
// it under the terms of the GNU General Public License as published by 
// the Free Software Foundation; either version 2 of the License, or 
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
// for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program; if not, write to the 
//
// Free Software Foundation, Inc., 
// 59 Temple Place, Suite 330, 
// Boston, MA 02111-1307 USA
//

namespace Bwg.Scsi
{
    enum ScsiCommandCode
    {
        TestUnitReady = 0x00,
        RequestSense = 0x03,
        FormatUnit = 0x04,
        Inquiry = 0x12,
        StartStopUnit = 0x1B,
        PreventAllowMediumRemoval = 0x1E,
        ReadFormatCapacities = 0x23,
        ReadCapacity = 0x25,
        Read = 0x28,
        Seek = 0x2B,
        Write = 0x2A,
        Erase = 0x2C,
        WriteAndVerify = 0x2E,
        Verify = 0x2F,
        SyncronizeCache = 0x35,
        WriteBuffer = 0x3B,
        ReadBuffer = 0x3C,
        ReadSubChannel = 0x42,
        ReadTocPmaAtip = 0x43,
        PlayAudio10 = 0x45,
        GetConfiguration = 0x46,
        PlayAudioMSF = 0x47,
        GetEventStatusNotification = 0x4A,
        PauseResume = 0x4B,
        StopPlayScan = 0x4E,
        ReadDiskInformation = 0x51,
        ReadTrackInformation = 0x52,
        ReserveTrack = 0x53,
        SendOpcInformation = 0x54,
        ModeSelect = 0x55,
        RepairTrack = 0x58,
        ModeSense = 0x5A,
        CloseTrackSession = 0x5B,
        ReadBufferCapacity = 0x5C,
        SendCueSheet = 0x5D,
        Blank = 0xA1,
        SendKey = 0xA3,
        ReportKey = 0xA4,
        PlayAudio12 = 0xA5,
        LoadUnloadMedium = 0xA6,
        SetReadAhead = 0xA7,
        Read12 = 0xA8,
        Write12 = 0xAA,
        GetPerformance = 0xAC,
        ReadDvdStructure = 0xAD,
        SetStreaming = 0xB6,
        ReadCdMSF = 0xB9,
        Scan = 0xBA,
        SetCdSpeed = 0xBB,
        MechanismStatus = 0xBD,
        ReadCd = 0xBE,
        SendDvdStructure = 0xBF
    } ;
}
