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
    /// This class represents a performance entry from the device's performance list
    /// </summary>
    public class Performance : Result
    {
        /// <summary>
        /// The start LBA for this entry
        /// </summary>
        public readonly uint StartLBA;

        /// <summary>
        /// The starting performance for this entry
        /// </summary>
        public readonly uint StartPerformance;

        /// <summary>
        /// The end LBA for this entry
        /// </summary>
        public readonly uint EndLBA;

        /// <summary>
        /// The end performance for this entry
        /// </summary>
        public readonly uint EndPerformance;

        /// <summary>
        /// The constructor to create a performance entry
        /// </summary>
        /// <param name="buffer">the buffer</param>
        /// <param name="offset">the offset into the buffer</param>
        /// <param name="size">the size of the buffer</param>
        public Performance(IntPtr buffer, int offset, int size)
            : base(buffer, size)
        {
            StartLBA = Get32(offset);
            StartPerformance = Get32(offset + 4);
            EndLBA = Get32(offset + 8);
            EndPerformance = Get32(offset + 12);
        }

        /// <summary>
        /// Return a human readable string representing this object.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string str = string.Empty;

            str += "Start LBA=" + StartLBA.ToString();
            str += ", Start Perf=" + StartPerformance.ToString() + " Kb/sec";
            str += ", End LBA=" + EndLBA.ToString();
            str += ", End Perf=" + EndPerformance.ToString() + " Kb/sec";
            return str;
        }
    }
}
