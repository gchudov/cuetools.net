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
    /// This class represents a single entry in the OPC table
    /// </summary>
    public class OpcTableEntry : Result
    {
        /// <summary>
        /// The speed to which this entry applies
        /// </summary>
        public readonly ushort Speed;

        /// <summary>
        /// The OPC table entries
        /// </summary>
        public readonly byte[] Table;

        /// <summary>
        /// Constructor for an OPC table entry
        /// </summary>
        /// <param name="buffer">The raw buffer being processed</param>
        /// <param name="size">The size of the buffer</param>
        /// <param name="offset">The offset into the buffer that contains the OPC entry</param>
        public OpcTableEntry(IntPtr buffer, int size, int offset) : base(buffer, size)
        {
            Speed = Get16(offset);

            Table = new byte [6] ;
            for (int i = 0; i < 6; i++)
                Table[i] = Get8(offset + i + 2);
        }
    }
}
