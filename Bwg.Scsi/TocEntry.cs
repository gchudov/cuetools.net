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
    /// A class representing a single TOC entry
    /// </summary>
    public class TocEntry : Result
    {
        /// <summary>
        /// The ADR value for this track
        /// </summary>
        public byte Adr;

        /// <summary>
        /// The control value for this track
        /// </summary>
        public byte Control;

        /// <summary>
        /// The track number for this track
        /// </summary>
        public byte Number;

        /// <summary>
        /// The start address (in sector) for this track
        /// </summary>
        public uint StartSector ;

        /// <summary>
        /// The starting MSF for this track
        /// </summary>
        public MinuteSecondFrame StartMSF;

        /// <summary>
        /// Create a new TOC entry
        /// </summary>
        /// <param name="buffer">the memory buffer holding the TOC</param>
        /// <param name="offset">the offset to the TOC entry of interest</param>
        /// <param name="size">the overall size of the buffer</param>
        /// <param name="mode">time versus lba mode</param>
        public TocEntry(IntPtr buffer, int offset, int size, bool mode) : base(buffer, size)
        {
            byte b = Get8(offset + 1) ;
            Adr = (byte)((b >> 4) & 0x0f) ;
            Control = (byte)(b & 0x0f) ;
            Number = Get8(offset + 2) ;

            if (mode)
                StartMSF = new MinuteSecondFrame(Get8(offset + 5), Get8(offset + 6), Get8(offset + 7));
            else
            {
                StartSector = Get32(offset + 4);
                if (StartSector > 16777216 - 150)
                {
                    // Fix incorrect TOC, where the first track is reported to start at a frame smaller than 150
                    StartSector = 0;
                }
            }
        }
    }
}
