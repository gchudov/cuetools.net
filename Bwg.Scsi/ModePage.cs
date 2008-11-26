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
using System.Runtime.InteropServices;

namespace Bwg.Scsi
{
    /// <summary>
    /// This object represents a single mode page in a mode table.
    /// </summary>
    public class ModePage : Result
    {
        #region private member variables
        /// <summary>
        /// The data associated with the page table.  This data also includes the page code and length
        /// data elements from the mode page data.
        /// </summary>
        protected byte[] m_page_data;

        #endregion

        #region constructor
        /// <summary>
        /// This is the constructor for the mode page object.  It builds a mode page object from
        /// the raw reply area from the SCSI device.
        /// </summary>
        /// <param name="buffer">A pointer to the raw reply area</param>
        /// <param name="size">The size of the raw reply area in bytes</param>
        /// <param name="offset">The offset to the mode page data</param>
        public ModePage(IntPtr buffer, int size, ref ushort offset) : base(buffer, size)
        {
            byte b = Get8(offset);
            offset++;
            byte len = Get8(offset);
            offset++;

            m_page_data = new byte[len + 2];
            m_page_data[0] = b;
            m_page_data[1] = len;
            for(int i = 0 ; i < len ; i++)
                m_page_data[i + 2] = Get8(offset++);
        }
        #endregion

        #region public properties

        /// <summary>
        /// This is the data that makes up this mode page
        /// </summary>
        public byte[] PageData
        {
            get
            {
                return m_page_data;
            }
        }

        /// <summary>
        /// The length of this mode page in bytes
        /// </summary>
        public ushort Length
        {
            get
            {
                return (ushort)m_page_data.GetLength(0);
            }
        }

        /// <summary>
        /// The page code for this page
        /// </summary>
        public byte PageCode
        {
            get
            {
                return (byte)(m_page_data[0] & 0x3f);
            }
        }

        /// <summary>
        /// If true, these parameters are saved permanently
        /// </summary>
        public bool ParametersSavable
        {
            get
            {
                return ((m_page_data[0] & 0x80) != 0) ? true : false;
            }

            set
            {
                if (value)
                    m_page_data[0] |= 0x80;
                else
                    m_page_data[0] &= 0x7f;
            }
        }

        #endregion

        #region public methods
        /// <summary>
        /// Retreive a 32 bit value from a mode page
        /// </summary>
        /// <param name="offset">offset in the page to the value</param>
        /// <returns>a 32 bit value retreived from the mode page data</returns>
        public int ModeGet32(int offset)
        {
            return m_page_data[offset + 0] << 24 |
                   m_page_data[offset + 1] << 16 |
                   m_page_data[offset + 2] << 8 |
                   m_page_data[offset + 3] << 0;
        }

        /// <summary>
        /// Set a 32 bit value in a mode page
        /// </summary>
        /// <param name="offset">offset in the page to where the value will be placed</param>
        /// <param name="value">the value to write</param>
        public void ModeSet32(int offset, int value)
        {
            m_page_data[offset + 0] = (byte)((value >> 24) & 0xff);
            m_page_data[offset + 1] = (byte)((value >> 16) & 0xff);
            m_page_data[offset + 2] = (byte)((value >> 8) & 0xff);
            m_page_data[offset + 3] = (byte)((value >> 0) & 0xff);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        public ushort ModeGet16(int offset)
        {
            return (ushort)(m_page_data[offset + 0] << 8 |
                   m_page_data[offset + 1] << 0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <param name="value"></param>
        public void ModeSet16(int offset, ushort value)
        {
            m_page_data[offset + 0] = (byte)((value >> 8) & 0xff);
            m_page_data[offset + 1] = (byte)((value >> 0) & 0xff);
        }
        #endregion
    }
}
