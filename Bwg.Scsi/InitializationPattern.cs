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
    /// The class representing a pattern to use in initialization the surface of a
    /// DVD.
    /// </summary>
    public class InitializationPattern
    {
        /// <summary>
        /// The type of modifier for the IP pattern
        /// </summary>
        public enum IPModifierType : byte
        {
            /// <summary>
            /// 
            /// </summary>
            NoHeader = 0,

            /// <summary>
            /// 
            /// </summary>
            WriteLBA = 1,

            /// <summary>
            /// 
            /// </summary>
            WriteLBAPhysical = 2,

            /// <summary>
            /// 
            /// </summary>
            Reserved = 3
        }

        /// <summary>
        /// The modifier
        /// </summary>
        public IPModifierType IPModifier;

        /// <summary>
        /// 
        /// </summary>
        public byte PatternType;

        /// <summary>
        /// 
        /// </summary>
        public byte[] Pattern;
    }
}
