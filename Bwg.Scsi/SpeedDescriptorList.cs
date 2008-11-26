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
using System.Collections;
using System.Text;
using System.Diagnostics;

namespace Bwg.Scsi
{
    /// <summary>
    /// 
    /// </summary>
    public class SpeedDescriptorList : Result, IEnumerable<SpeedDescriptor>, IEnumerable
    {
        private IList<SpeedDescriptor> m_list;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="size"></param>
        public SpeedDescriptorList(IntPtr buffer, int size)
            : base(buffer, size)
        {
            m_list = new List<SpeedDescriptor>();

            uint len = Get32(0) + 4;
            int index = 8;
            Debug.Assert(size >= 8);

            while (index < len && index < size)
            {
                SpeedDescriptor p = new SpeedDescriptor(buffer, index, size);
                m_list.Add(p);

                index += 16;
            }
        }

        /// <summary>
        /// This method is an indexor into the list of descriptors
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public SpeedDescriptor this[int index]
        {
            get
            {
                return m_list[index];
            }
        }

        /// <summary>
        /// Return the number of speed descriptors in the list
        /// </summary>
        public int Count
        {
            get { return m_list.Count; }
        }

        /// <summary>
        /// Return an enumerator for this list
        /// </summary>
        /// <returns></returns>
        public IEnumerator<SpeedDescriptor> GetEnumerator()
        {
            return m_list.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_list.GetEnumerator();
        }
    }
}
