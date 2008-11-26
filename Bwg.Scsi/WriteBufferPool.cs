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
using System.Threading;
using System.Diagnostics;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class manages a series of buffers that are used to provide data
    /// for the burning process.
    /// </summary>
    public class WriteBufferPool
    {
        #region private data members
        int m_count ;

        private Object m_buffer_list_lock;
        private IList<WriteBuffer> m_buffer_list;

        private Object m_free_list_lock;
        private IList<WriteBuffer> m_free_list;

        private bool m_end_of_data;

        private int m_produced;
        private int m_consumed;

        private int m_pagesize;

        private int m_mult;
        private bool m_padded;

        #endregion

        #region public properties
        /// <summary>
        /// If this property is true, all data has been pushed into the
        /// buffer stream.
        /// </summary>
        public bool EndOfData
        {
            get
            {
                return m_end_of_data;
            }
            set
            {
                m_end_of_data = value;
            }
        }

        /// <summary>
        /// This property returns the number of 2k pages that have been produced by
        /// the generator thread.
        /// </summary>
        public int Produced
        {
            get
            {
                return m_produced;
            }
        }

        /// <summary>
        /// This property returns the number of 2k pages that have been consumed
        /// by the writer thread.
        /// </summary>
        public int Consumed
        {
            get
            {
                return m_consumed;
            }
        }

        /// <summary>
        /// This is the size of the pages stored in this buffer pool, in bytes.
        /// </summary>
        public int PageSize
        {
            get
            {
                return m_pagesize;
            }
        }

        /// <summary>
        /// Return the percent of the buffers used
        /// </summary>
        public double PercentUsed
        {
            get
            {
                double v;

                lock (m_buffer_list_lock)
                {
                    v = (double)m_buffer_list.Count / (double)m_count * 100.0;
                }
                return v;
            }
        }

        #endregion

        #region constructors

        /// <summary>
        /// Initialize the buffer stream by creating the buffers
        /// </summary>
        /// <param name="cnt">the number of buffers to create</param>
        /// <param name="size">the size of each buffer</param>
        /// <param name="mult">the multiple that each buffer must meet for the device</param>
        /// <param name="secsize">the size of the sectors in the buffer in bytes</param>
        public WriteBufferPool(int cnt, int size, int mult, int secsize)
        {
            m_count = cnt ;
            m_end_of_data = false;
            m_pagesize = size;
            m_mult = mult;

            //
            // If the multiple setting is non-zero, we may have to adjust the page size
            //
            if (m_mult != 0)
            {
                int blocks = size / (mult * secsize);
                m_pagesize = blocks * mult * secsize;
            }

            // Used to lock the data buffer list
            m_buffer_list_lock = new Object();

            // Used to lock the free buffer list
            m_free_list_lock = new Object();

            // The data buffer list
            m_buffer_list = new List<WriteBuffer>();

            // THe free buffer list
            m_free_list = new List<WriteBuffer>() ;

            // Create the buffers and place them all in the
            // free list.
            for(int i = 0 ; i < cnt ; i++)
                m_free_list.Add(new WriteBuffer(size)) ;

            m_produced = 0;
            m_consumed = 0;

            //
            // This starts out false and is only set to true if we have to pad
            // a buffer to reach an appropriate multiple of the multiple count
            // for this device.  We should only have to pad the last buffer sent
            // to the device.  This value is used to detect data being supplied
            // after having to pad a data buffer, implying that the data put into
            // the queue did not full the buffer and was not at the end of the 
            // queue.
            //
            m_padded = false;
        }
        #endregion

        #region public methods
        /// <summary>
        /// Get a free write buffer to fill with data
        /// </summary>
        /// <returns>a write buffer</returns>
        public WriteBuffer GetFreeBuffer()
        {
            while (true)
            {
                lock (m_free_list_lock)
                {
                    if (m_free_list.Count > 0)
                    {
                        Debug.Assert(m_padded == false);
                        WriteBuffer buf = m_free_list[0] ;
                        m_free_list.RemoveAt(0) ;
                        return buf;
                    }

                    if (m_end_of_data == true)
                        return null;
                }

                // Sleep for 10 msec while we wait for buffers to become
                // availsble.
                Thread.Sleep(1) ;
            }
        }

        /// <summary>
        /// Add a buffer to the list
        /// </summary>
        /// <param name="buf"></param>
        public void SendBufferToDevice(WriteBuffer buf)
        {
            lock (m_buffer_list_lock)
            {
                if (m_mult != 0 && ((buf.DataSize % m_mult) != 0))
                    System.Diagnostics.Debug.WriteLine("Added a block of " + buf.DataSize.ToString() + " blocks, not an event multiple") ;

                m_buffer_list.Add(buf);
                m_produced += (int)buf.DataSize;
            }
        }

        /// <summary>
        /// Get the next buffer of data from the steam
        /// </summary>
        /// <returns>the next buffer stream, or null if all data has been processed</returns>
        public WriteBuffer GetNextDataBuffer()
        {
            while (true)
            {
                lock (m_buffer_list_lock)
                {
                    if (m_buffer_list.Count > 0)
                    {
                        WriteBuffer buf = m_buffer_list[0];
                        m_buffer_list.RemoveAt(0);
                        m_consumed += (int)buf.DataSize;

                        if (m_mult != 0 && (buf.DataSize % m_mult) != 0)
                        {
                            //
                            // The buffer is not an even multiple of the 
                            //
                            m_padded = true;
                            buf.DataSize = ((buf.DataSize / m_mult) + 1) * m_mult;
                        }

                        return buf;
                    }

                    if (m_end_of_data)
                        return null;
                }

                // Wait for the reader thread to put data into the buffer
                // stream
                // Thread.Sleep(1);
            }
        }

        /// <summary>
        /// Add a buffer back to the free list to be used again
        /// </summary>
        /// <param name="buf">the buffer to add to the list</param>
        public void FreeWriteBuffer(WriteBuffer buf)
        {
            lock (m_free_list_lock)
            {
                m_free_list.Add(buf);
            }
        }
        #endregion
    }
}
