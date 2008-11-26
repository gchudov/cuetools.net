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
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class is a base class for classes that contain results from SCSI commands.  This
    /// class contains information about whether or not that result is valid and provides functions
    /// for extract data from the SCSI response.
    /// </summary>
    public abstract class Result
    {
        #region private data members
        /// <summary>
        /// The data buffer containing the data that makes up the result
        /// </summary>
        private IntPtr m_buffer ;

        /// <summary>
        /// The size of the data in the buffer
        /// </summary>
        private int m_size ;

        /// <summary>
        /// This member is true if the result is valid.  It is false if an error occurred while
        /// parsing the SCSI result.
        /// </summary>
        protected bool m_valid;

        #endregion

        #region constructor
        /// <summary>
        /// This is the constructor for the ScsiResult class.
        /// </summary>
        /// <param name="buffer">A pointer to the memory area containing the SCSI reply</param>
        /// <param name="size">The size of the memory area containing the SCSI reply</param>
        public Result(IntPtr buffer, int size)
        {
            m_valid = true;
            m_buffer = buffer;
            m_size = size;
        }
        #endregion

        #region public properties
        /// <summary>
        /// This property is true if the result is valid, and false otherwise.
        /// </summary>
        public bool Valid
        {
            get
            {
                return m_valid;
            }
        }

        /// <summary>
        /// This property returns the size of the buffer
        /// </summary>
        public int BufferSize
        {
            get { return m_size; }
        }

        /// <summary>
        /// This property returns the buffer
        /// </summary>
        public IntPtr Buffer
        {
            get { return m_buffer; }
        }
        #endregion

        #region public member functions
        /// <summary>
        /// This method returns a byte of data from the memory area associated with the SCSI reply.
        /// </summary>
        /// <param name="offset">The location of the byte to return within the reply</param>
        /// <returns>a byte of data from the reply</returns>
        public byte Get8(int offset)
        {
            if (offset >= m_size)
                throw new Exception("offset " + offset.ToString() + " is out side the range of the buffer");

            return Marshal.ReadByte(m_buffer, offset);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        public ushort Get16(int offset)
        {
            return (ushort)((Get8(offset) << 8) | Get8(offset + 1));
        }

        /// <summary>
        /// Get a 16 bit number located in two different places in the buffer
        /// </summary>
        /// <param name="msb">offset to the MSB of the number</param>
        /// <param name="lsb">offset to the LSB of the number</param>
        /// <returns></returns>
        public ushort Get16(int msb, int lsb)
        {
            return (ushort)((Get8(msb) << 8) | Get8(lsb));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        public uint Get24(int offset)
        {
            return (UInt32)((Get8(offset) << 16) | (Get8(offset + 1) << 8) | Get8(offset + 2));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        public uint Get32(int offset)
        {
            return (UInt32)((Get8(offset) << 24) | (Get8(offset + 1) << 16) | (Get8(offset + 2) << 8) | Get8(offset + 3));
        }

        /// <summary>
        /// This method returns a 32 bit integer from an offset in the data buffer.  The data is
        /// ordered as big endian data per the SCSI-3 specification.
        /// </summary>
        /// <param name="offset">the offset into the buffer</param>
        /// <returns></returns>
        public int Get32Int(int offset)
        {
            uint b = Get32(offset);
            Debug.Assert(b < Int32.MaxValue);
            return (int)b;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="start"></param>
        /// <param name="stop"></param>
        /// <returns></returns>
        public string GetString(int start, int stop)
        {
            string s = "";

            for (int i = start; i <= stop; i++)
            {
                byte b = Get8(i);
                s += (char)b;
            }

            return s;
        }

        /// <summary>
        /// Return a true or false value based on a bit in the buffer
        /// </summary>
        /// <param name="offset">offset to the byte containing the bit</param>
        /// <param name="bit">the specific bit number</param>
        /// <returns>boolean reflecting the value of the bit</returns>
        public bool GetBit(int offset, int bit)
        {
            byte b = Get8(offset);

            if ((b & (1 << bit)) != 0)
                return true;

            return false;
        }
        #endregion
    }
}
