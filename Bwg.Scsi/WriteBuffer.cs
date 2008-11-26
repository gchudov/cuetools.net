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
using System.Runtime.InteropServices ;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class is used to buffer data to the SCSI device when
    /// writing large amounts of data to the SCSI device.
    /// </summary>
    public class WriteBuffer : IDisposable
    {
        [DllImport("ntdll.dll")]
        internal static extern void RtlZeroMemory(IntPtr dest, int size);

        [DllImport("ntdll.dll")]
        internal static extern void RtlFillMemory(IntPtr dest, int size, byte value);

        #region public data members

        /// <summary>
        /// The size of the buffer associated with this object (in sectors)
        /// </summary>
        public int BufferSize;

        /// <summary>
        /// The amount of data stored in the buffer (in sectors)
        /// </summary>
        public int DataSize;

        /// <summary>
        /// The size of a sector of data stored in this buffer page
        /// </summary>
        public int SectorSize;

        /// <summary>
        /// This contains the logical block address for this block of data
        /// </summary>
        public long LogicalBlockAddress;

        /// <summary>
        /// The source of this buffer, used only for debugging
        /// </summary>
        public string SourceString;

        #endregion

        #region private data members

        /// <summary>
        /// The actual buffer memory
        /// </summary>
        private IntPtr m_buffer_ptr;

        #endregion

        #region public properties

        /// <summary>
        /// Returns a buffer pointer to the internal buffer. 
        /// </summary>
        public IntPtr BufferPtr
        {
            get
            {
                return m_buffer_ptr;
            }
        }
        #endregion

        #region constructors

        /// <summary>
        /// Create a new buffer object
        /// </summary>
        /// <param name="size">the size of the buffer</param>
        public WriteBuffer(int size)
        {
            m_buffer_ptr = Marshal.AllocHGlobal((int)size);
            BufferSize = size;
            LogicalBlockAddress = long.MaxValue;
        }
        #endregion

        #region public methods
        /// <summary>
        /// Free the memory associated with this buffer object
        /// </summary>
        public void Dispose()
        {
            Marshal.FreeHGlobal(m_buffer_ptr);
        }

        /// <summary>
        /// Fill the buffer with all zeros
        /// </summary>
        public void Zero()
        {
            RtlZeroMemory(m_buffer_ptr, BufferSize) ;
        }

        /// <summary>
        /// Fill the buffer with data of a specific byte.
        /// </summary>
        /// <param name="b"></param>
        public void Fill(byte b)
        {
            RtlFillMemory(m_buffer_ptr, BufferSize, b);
        }

        #endregion
    }
}
