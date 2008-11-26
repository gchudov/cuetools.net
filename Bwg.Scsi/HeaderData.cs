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
    /// This class contains the header data read from a sector
    /// </summary>
    public class HeaderData : Result
    {
        #region public types
        /// <summary>
        /// The block type, see SCSI MMC spec
        /// </summary>
        public enum BlockTypeType : byte
        {

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            UserDataBlock = 0x00,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            FourthRunInBlock = 0x01,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            ThirdRunInBlock = 0x02,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            SecondRunInBlock = 0x03,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            FirstRunInBlock = 0x04,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            LinkBlock = 0x05,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            SecondRunOutBlock = 0x06,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            FirstRunOutBlock = 0x07
        } ;

        /// <summary>
        /// The data mode for the sector (only applies if track mode is 0x04)
        /// </summary>
        public enum DataTypeType : byte
        {
            /// <summary>
            /// Mode 0 data, 2336 bytes of zero
            /// </summary>
            Mode0Data = 0x00,

            /// <summary>
            /// Mode 1 data, 2048 bytes of user data
            /// </summary>
            Mode1Data = 0x01,

            /// <summary>
            /// Mode 2 data, multiple forms of data
            /// </summary>
            Mode2Data = 0x02,

            /// <summary>
            /// Reserved, should not be returned by a drive
            /// </summary>
            Reserved = 0x03
        } ;
        #endregion

        #region public data members
        /// <summary>
        /// The time data read from the sector
        /// </summary>
        public readonly MinuteSecondFrame MSF;

        /// <summary>
        /// The block type, see SCSI MMC spec
        /// </summary>
        public readonly BlockTypeType BlockType;

        /// <summary>
        /// The data type, see SCSI MMC spec
        /// </summary>
        public readonly DataTypeType DataType;
        #endregion

        /// <summary>
        /// Create the header data object from a buffer containing the header data
        /// </summary>
        /// <param name="buf">buffer containing the data</param>
        /// <param name="size">size of the data</param>
        public HeaderData(IntPtr buf, int size)
            : base(buf, size)
        {
            byte b ;

            MSF = new MinuteSecondFrame(Get8(0), Get8(1), Get8(2));
            b = Get8(3);
            BlockType = (BlockTypeType)((b >> 5) & 0x07);
            DataType = (DataTypeType)(b & 0x03);
        }
    }
}
