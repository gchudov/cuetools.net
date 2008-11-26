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
    /// This class represents a list of features returned from the SCSI device.
    /// </summary>
    public class FeatureList : Result
    {
        /// <summary>
        /// The profile associated with this feature list.
        /// </summary>
        public readonly ushort Profile;

        /// <summary>
        /// The list of features returned from the device
        /// </summary>
        public readonly IList<Feature> Features;

        /// <summary>
        /// The constructure for the ScsiFeatureList object
        /// </summary>
        /// <param name="buffer">The pointer to the memory area containing the SCSI reply</param>
        /// <param name="size">The size of the SCSI reply</param>
        public FeatureList(IntPtr buffer, int size) : base(buffer, size)
        {
            Profile = Get16(6);
            Features = new List<Feature>() ;

            uint len = Get32(0);
            int offset = 8;
            while (offset < len + 8 && offset < size)
            {
                Feature f = new Feature(buffer, size, ref offset);
                Features.Add(f);
            }
        }
    }
}
