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
    /// This class represents subheader data for a given sector
    /// </summary>
    public class SubheaderData : Result
    {
        #region public types
        /// <summary>
        /// The submode type
        /// </summary>
        [Flags]
        public enum SubmodeType : byte
        {

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            EndOfFile = 0x80,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            RealTimeBlock = 0x40,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            Form2 = 0x20,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            TriggerBlock = 0x10,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            DataBlock = 0x08,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            AudioBlock = 0x04,          // Not traditional CD-DA

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            VideoBlock = 0x02,

            /// <summary>
            /// See SCSI MMC spec
            /// </summary>
            EndOfRecord = 0x01
        } ;
        #endregion

        #region public data members
        /// <summary>
        /// File number, see SCSI MMC spec
        /// </summary>
        public readonly byte FileNumber;

        /// <summary>
        /// Channel number, see SCSI MMC spec
        /// </summary>
        public readonly byte ChannelNumber;

        /// <summary>
        /// Submode, see SCSI MMC spec
        /// </summary>
        public readonly SubmodeType Submode;

        /// <summary>
        /// Coding information, see SCSI MMC spec
        /// </summary>
        public readonly byte CodingInformation;

        #endregion

        /// <summary>
        /// Create subheader data information from subheader data read
        /// </summary>
        /// <param name="buf">the buffer containing the subheader data</param>
        /// <param name="size">the size of the data</param>
        public SubheaderData(IntPtr buf, int size)
            : base(buf, size)
        {
            FileNumber = Get8(0);
            ChannelNumber = Get8(1);
            Submode = (SubmodeType)Get8(2);
            CodingInformation = Get8(3);
        }
    }
}
