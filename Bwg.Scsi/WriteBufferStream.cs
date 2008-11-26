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
using System.IO ;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Bwg.Scsi
{
    /// <summary>
    /// A stream that pushes the output data through the buffer pool to feed
    /// the CD/DVD burner.
    /// </summary>
    public class WriteBufferStream: Stream
    {
        #region private data members

        /// <summary>
        /// The current length of the stream in bytes
        /// </summary>
        private long m_length ;

        /// <summary>
        /// The buffer pool, used to get buffers to write into and used to send
        /// buffers to the device.
        /// </summary>
        private WriteBufferPool m_pool;

        /// <summary>
        /// The current buffer we are writing into
        /// </summary>
        private WriteBuffer m_buffer;

        /// <summary>
        /// The size of the sectors that will be sent to the device, in bytes
        /// </summary>
        private int m_sector_size;

        /// <summary>
        /// The number of sectors that fit into a single write buffer
        /// </summary>
        private int m_sector_count;

        /// <summary>
        /// The index into the current write buffer
        /// </summary>
        private int m_index;

        /// <summary>
        /// If true, we are closing
        /// </summary>
        private bool m_closing;

        /// <summary>
        /// The logical block address for the stream data
        /// </summary>
        private long m_lba;

        #endregion

        #region constructors
        /// <summary>
        /// This is a stream class that writes the data to the write buffer pool to be sent
        /// down to the CD/DVD device.
        /// </summary>
        /// <param name="pool">the buffer pool to get pages from and send data to</param>
        /// <param name="sector_size">the size of the sectors to buffer</param>
        /// <param name="lba">the logical block address for the first block from this stream</param>
        public WriteBufferStream(WriteBufferPool pool, int sector_size, long lba)
        {
            m_length = 0;
            m_pool = pool;
            m_buffer = null;
            m_index = 0;
            m_sector_size = sector_size;
            m_sector_count = m_pool.PageSize / m_sector_size;
            m_closing = false;
            m_lba = lba;
        }
        #endregion

        #region public properties
        /// <summary>
        /// 
        /// </summary>
        public override bool CanRead { get { return false; } }

        /// <summary>
        /// 
        /// </summary>
        public override bool CanSeek { get { return false; } }

        /// <summary>
        /// 
        /// </summary>
        public override bool CanWrite { get { return true ; } }

        /// <summary>
        /// 
        /// </summary>
        public override long Length { get { return m_length ; } }

        /// <summary>
        /// 
        /// </summary>
        public override long Position 
        { 
            get 
            { 
                return m_length; 
            }
            set
            {
                throw new Exception("The method or operation is not implemented.");
            }
        }

        #endregion

        #region public methods

        /// <summary>
        /// This method flushes any existing data to the device.
        /// </summary>
        public override void Flush()
        {
            if (m_buffer != null)
            {
                //
                // In general using flush, except when closing the device can be very
                // dangerous.  This method catches those cases where the result sent to
                // the device would be incorrect and asserts accordingly.  Basically we
                // always send data down to the device in sectors.  When we flush we send
                // down whatever data is left in the buffer.  If we are closing, this has
                // the effect of rounding the last bit of data to a sector boundary.  If
                // we are not closing, it can have the effect of inserting data in the 
                // middle of the data stream.  Flusing only works correctly if we are sitting
                // on the boundary of a sector.  Therefore, if this assert has fired it means
                // we are not closing and not sitting on the boundary of a sector.
                //
                Debug.Assert(m_closing || (m_index % m_sector_size) == 0);

                m_buffer.DataSize = m_index / m_sector_size;
                if ((m_index % m_sector_size) != 0)
                    m_buffer.DataSize++;
                m_buffer.SectorSize = m_sector_size;

                m_buffer.SourceString = "WriteBufferStream, Flush method";
                m_buffer.LogicalBlockAddress = m_lba;
                m_lba = long.MaxValue;
                m_pool.SendBufferToDevice(m_buffer);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="offset"></param>
        /// <param name="origin"></param>
        /// <returns></returns>
        public override long Seek(long offset, SeekOrigin origin)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="offset"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public override int Read(byte[] buffer, int offset, int count)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="offset"></param>
        /// <param name="count"></param>
        public override void Write(byte[] buffer, int offset, int count)
        {
            while (count > 0)
            {
                if (m_buffer == null)
                {
                    m_buffer = m_pool.GetFreeBuffer();
                    m_index = 0;
                }

                //
                // Write up to the end of this buffer
                //
                int remaining = m_sector_count * m_sector_size - m_index;

                if (remaining > count)
                    remaining = count;

                IntPtr dest = new IntPtr(m_buffer.BufferPtr.ToInt32() + m_index);
                Marshal.Copy(buffer, offset, dest, remaining);
                m_index += remaining;
                m_length += remaining;
                count -= remaining ;
                offset += remaining ;

                if (m_index == m_sector_count * m_sector_size)
                {
                    m_buffer.DataSize = m_index / m_sector_size;
                    m_buffer.SectorSize = m_sector_size;
                    m_buffer.LogicalBlockAddress = m_lba;
                    m_buffer.SourceString = "WriteBufferStream, Write method";
                    m_lba = long.MaxValue;
                    m_pool.SendBufferToDevice(m_buffer);
                    m_buffer = null;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public override void SetLength(long value)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        /// <summary>
        /// 
        /// </summary>
        public override void Close()
        {
            m_closing = true;

            Flush();
            base.Close();
        }

        #endregion
    }
}
