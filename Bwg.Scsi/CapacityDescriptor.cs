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

using System;
using System.Collections.Generic;
using System.Text;

namespace Bwg.Scsi
{
    /// <summary>
    /// 
    /// </summary>
    public class CapacityDescriptor : Result
    {
        #region public types
        /// <summary>
        /// 
        /// </summary>
        public enum DescriptorType : byte
        {
            /// <summary>
            /// 
            /// </summary>
            Reserved = 0,

            /// <summary>
            /// 
            /// </summary>
            Unformatted = 1,

            /// <summary>
            /// 
            /// </summary>
            Formatted = 2,

            /// <summary>
            /// 
            /// </summary>
            NoMedia = 3
        } ;
        #endregion

        #region public variables

        /// <summary>
        /// The number of blocks on this media.
        /// </summary>
        public readonly uint NumberOfBlocks;

        /// <summary>
        /// This length of a single block for this capacity
        /// </summary>
        public readonly uint BlockLength;

        /// <summary>
        /// This is the descriptor type, see the SCSI-3 MMC specification
        /// </summary>
        public readonly DescriptorType DescType;

        /// <summary>
        /// This is the type of format for the capacity descriptor, see the SCSI-3 MMC spec
        /// </summary>
        public readonly byte FormatType;

        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="offset"></param>
        /// <param name="size"></param>
        public CapacityDescriptor(IntPtr buffer, int offset, int size)
            : base(buffer, size)
        {
            byte b;

            NumberOfBlocks = Get32(offset);

            b = Get8(offset + 4);
            DescType = (DescriptorType)(b & 0x03);
            FormatType = (byte)(b >> 2);

            BlockLength = Get24(offset + 5);
        }
        #endregion
    }
}
