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
//using Bwg.Utils;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class reads data from a file and sends this data to the buffered
    /// write stream.
    /// </summary>
    public unsafe class FileReader
    {
        #region private member variables
        /// <summary>
        /// The OS file for reading data from a file into a memory buffer (IntPtr)
        /// </summary>
        private OsFileReader m_reader;

        /// <summary>
        /// The buffer pool to receive the data
        /// </summary>
        private WriteBufferPool m_buffer_stream;
        #endregion


        /// <summary>
        /// Create a new file reader that sends the data read to a buffer stream
        /// object
        /// </summary>
        /// <param name="buffer">the buffer to receive the data</param>
        public FileReader(WriteBufferPool buffer)
        {
            m_buffer_stream = buffer;
            m_reader = new OsFileReader();
        }

        /// <summary>
        /// Open a file to read data from
        /// </summary>
        /// <param name="filename">the name of the file</param>
        /// <returns>true if the file is opened, false otherwise</returns>
        public bool Open(string filename)
        {
            return m_reader.Open(filename);
        }

        /// <summary>
        /// This method closes an open file.
        /// </summary>
        /// <returns>true if the file is closed sucessfully, false otherwise</returns>
        public bool Close()
        {
            return m_reader.Close();
        }

        /// <summary>
        /// This function reads data from the file and sends it to the
        /// buffer stream to be sent to the burner.  This function does not
        /// return until all of the data from the file is read.
        /// </summary>
        /// <returns></returns>
        public bool ReadData(int sector_size, long lba)
        {
            bool shortbuf = false;
            int NumberRead ;
            int num_sectors ;
            bool first = true;
            
            // This is the number of whole sectors that will fit into the buffer.
            num_sectors = m_buffer_stream.PageSize / sector_size;

            while (true)
            {
                WriteBuffer buf ;
                
                buf = m_buffer_stream.GetFreeBuffer();
                if (!m_reader.ReadData(buf.BufferPtr, num_sectors * sector_size, out NumberRead))
                    return false ;

                // If the read succeeded, but read zero bytes, we have reached the
                // end of the file.
                if (NumberRead == 0)
                    return true;

                if (shortbuf)
                    throw new Exception("read two short packets");

                // Mark the buffer with the amount of data that has
                // been read from the file
                int sectors = NumberRead / sector_size ;
                if ((NumberRead % sector_size) != 0)
                {
                    shortbuf = true;
                    sectors++;
                }

                if (first)
                {
                    buf.SourceString = "First block from FileReader";
                    buf.LogicalBlockAddress = lba;
                    first = false;
                }
                else
                {
                    buf.SourceString = "Not first block from FileReader";
                    buf.LogicalBlockAddress = long.MaxValue;
                }

                buf.SectorSize = sector_size;
                buf.DataSize = sectors;

                // Send the buffer to the device
                m_buffer_stream.SendBufferToDevice(buf);
            }
        }
    }
}
