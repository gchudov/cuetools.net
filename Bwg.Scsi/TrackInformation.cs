//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for Windows XP
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

using System;
using System.Collections.Generic;
using System.Text;

namespace Bwg.Scsi
{
    /// <summary>
    /// This data structure contains the results from call ReadTrackInformation() to read
    /// data about a track
    /// </summary>
    public class TrackInformation : Result
    {
        #region public types
        /// <summary>
        /// This type represents the data mode for the track
        /// </summary>
        public enum DataModeType
        {
            /// <summary>
            /// Mode 1 user data (digital data, 2048 bytes per sector)
            /// </summary>
            Mode1 = 0x01,

            /// <summary>
            /// Mode 2 user data (digital data, multiple forms)
            /// </summary>
            Mode2 = 0x02,

            /// <summary>
            /// Audio data, 2352 bytes of audio data per sector
            /// </summary>
            Audio = 0x0f
        } ;
        #endregion

        #region public Track Information Members
        /// <summary>
        /// 
        /// </summary>
        public readonly ushort TrackNumber;

        /// <summary>
        /// 
        /// </summary>
        public readonly ushort SessionNumber;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool Copy;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool Damage;

        /// <summary>
        /// 
        /// </summary>
        public readonly byte TrackMode;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool RT;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool Blank;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool Packet;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool FP;

        /// <summary>
        /// 
        /// </summary>
        public readonly DataModeType DataMode;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool LRA_V;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool NWA_V;

        /// <summary>
        /// 
        /// </summary>
        public readonly int TrackStartAddress;

        /// <summary>
        /// 
        /// </summary>
        public readonly int NextWritableAddress;

        /// <summary>
        /// 
        /// </summary>
        public readonly int FreeBlocks;

        /// <summary>
        /// 
        /// </summary>
        public readonly int FixedPacketSize;

        /// <summary>
        /// 
        /// </summary>
        public readonly int TrackSize;

        /// <summary>
        /// 
        /// </summary>
        public readonly int LastRecordedAddress;

        /// <summary>
        /// 
        /// </summary>
        public readonly int ReadCompatLBA;

        #endregion

        /// <summary>
        /// Contruct a read track result object from the raw data.
        /// </summary>
        /// <param name="buf">the raw data buffer</param>
        /// <param name="size">the size of the raw buffer in bytes</param>
        public TrackInformation(IntPtr buf, int size) : base(buf, size)
        {
            TrackNumber = Get16(32, 2) ;
            SessionNumber = Get16(33, 3) ;

            Copy = GetBit(5,4) ;
            Damage = GetBit(5,5) ;
            TrackMode = (byte)(Get8(5) & 0x0f) ;

            RT = GetBit(6,7) ;
            Blank = GetBit(6,6) ;
            Packet = GetBit(6,5) ;
            FP = GetBit(6,4) ;
            DataMode = (DataModeType)(Get8(6) & 0x0f);

            LRA_V = GetBit(7, 1) ;
            NWA_V = GetBit(7, 0) ;

            TrackStartAddress = (int)Get32(8) ;
            NextWritableAddress = (int)Get32(12) ;
            FreeBlocks = Get32Int(16) ;
            FixedPacketSize = Get32Int(20) ;
            TrackSize = Get32Int(24) ;
            LastRecordedAddress = (int)Get32(28) ;
            ReadCompatLBA = (int)Get32(36) ;
        }

        #region public properties

        /// <summary>
        /// An alias for fixed packet size depending on the mode of the track.
        /// </summary>
        public int BlockingFactor
        {
            get
            {
                return FixedPacketSize;
            }
        }
        #endregion
    }
}

