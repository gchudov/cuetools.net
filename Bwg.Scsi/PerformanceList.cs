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
    /// A list of performance entries from the drive
    /// </summary>
    public class PerformanceList : Result, IEnumerable<Performance>, IEnumerable
    {
        #region public types
        /// <summary>
        /// The type of the data (read or write)
        /// </summary>
        public enum DataType
        {
            /// <summary>
            /// This means read data
            /// </summary>
            ReadData,

            /// <summary>
            /// This means write data
            /// </summary>
            WriteData
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum ExceptType : byte
        {
            /// <summary>
            /// 
            /// </summary>
            Nominal = 0,

            /// <summary>
            /// 
            /// </summary>
            Entire = 1,

            /// <summary>
            /// 
            /// </summary>
            Exceptions = 2
        }

        #endregion

        #region private data members
        private DataType m_type;
        private IList<Performance> m_list;
        private bool m_except;
        #endregion

        /// <summary>
        /// The constructor for the class
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="size"></param>
        public PerformanceList(IntPtr buffer, int size)
            : base(buffer, size)
        {
            m_list = new List<Performance>();

            uint len = Get32(0);
            int index = 8;
            Debug.Assert(size >= 8);

            byte b = Get8(4);
            if ((b & 0x02) != 0)
                m_type = DataType.WriteData;
            else
                m_type = DataType.ReadData;

            if ((b & 0x01) != 0)
                m_except = true;
            else
                m_except = false;

            while (index < len && index + 16 <= len && index + 16 <= size)
            {
                Performance p = new Performance(buffer, index, size);
                m_list.Add(p);

                index += 16;
            }
        }

        /// <summary>
        /// Return an indexed entry tino the performance list
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Performance this[int index]
        {
            get
            {
                return m_list[index];
            }
        }

        #region properties
        /// <summary>
        /// The type of the data
        /// </summary>
        public DataType Type
        {
            get
            {
                return m_type;
            }
        }

        /// <summary>
        /// If true, this is exception data
        /// </summary>
        public bool IsExceptionData
        {
            get
            {
                return m_except;
            }
        }
        #endregion

        #region public members
        /// <summary>
        /// Return an enumerator for this list
        /// </summary>
        /// <returns></returns>
        public IEnumerator<Performance> GetEnumerator()
        {
            return m_list.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_list.GetEnumerator();
        }

        #endregion
    }
}
