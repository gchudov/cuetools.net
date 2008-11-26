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
    /// This class represents the list of format parameters that are presented to
    /// the device when performing a FormatUnit() operation.
    /// </summary>
    public class FormatParameterList
    {
        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public enum UnitFormatType : byte
        {
            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            FullFormat = 0,
            
            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            SpareAreaExpansion = 1,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            ZoneReformat = 4,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            ZoneFormat = 5,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            CD_DVD_RW_FullFormat = 0x10,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            CD_DVD_RW_GrowSession = 0x11,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            CD_DVD_RW_AddSession = 0x12,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            DVD_RW_QuickGrowLastSession = 0x13,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            DVD_RW_QuickAddSession = 0x14,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            DVD_RW_QuickFormat = 0x15,

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            MRW_FullFormat = 0x24,          // CD-RW NumberOfBlocks = 0xffffffff
                                            // DVD+RW NumberOfBlocks = 0xffffffff

            /// <summary>
            /// See SCSI MMC-4 specification
            /// </summary>
            DVD_Plus_RW_BasicFormat = 0x25, // NumberOfBlock = 0xffffffff

            /// <summary>
            /// Basic format for DVD+RW (either SL or DL), see SCSI MMC-6
            /// </summary>
            DVD_Plus_RW_BasicFormat_DL_SL = 0x26,

            /// <summary>
            /// Format with spare areas for BluRay-RE
            /// </summary>
            BDRE_Format_With_Spare_Areas = 0x30,

            /// <summary>
            /// Format without spare areas for BlueRay-RE
            /// </summary>
            BDRE_Format_Without_Spare_Areas = 0x31,

            /// <summary>
            /// Format a BD-R disk with spare area
            /// </summary>
            BDR_Format_With_Spare_Areas = 0x32

        } ;

        private byte m_flags;

        /// <summary>
        /// The initializaion pattern for this format operation (not yet supported)
        /// </summary>
        public InitializationPattern IP;

        /// <summary>
        /// The format type for this format operation
        /// </summary>
        public UnitFormatType FormatType;

        /// <summary>
        /// The subtype for the format
        /// </summary>
        public byte FormatSubType;

        /// <summary>
        /// The number of blocks for the format operation
        /// </summary>
        public uint NumberOfBlocks;

        /// <summary>
        /// The format type dependent parameter
        /// </summary>
        public uint Parameter;

        /// <summary>
        /// Constructor for the parameter list object
        /// </summary>
        public FormatParameterList()
        {
            m_flags = 0;
            IP = null;
        }

        /// <summary>
        /// The size of the format parameters list in bytes, as it will exist in
        /// the command to the SCSI unit.
        /// </summary>
        public uint Size
        {
            get
            {
                return 4 + 8;
            }
        }

        #region public flags

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte FOV
        {
            get
            {
                return (byte)((m_flags >> 7) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x80;
                else
                    m_flags &= 0x7f;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte DPRY
        {
            get
            {
                return (byte)((m_flags >> 6) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x40;
                else
                    m_flags &= 0xbf;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte DCRT
        {
            get
            {
                return (byte)((m_flags >> 5) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x20;
                else
                    m_flags &= 0xdf;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte STPF
        {
            get
            {
                return (byte)((m_flags >> 4) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x10;
                else
                    m_flags &= 0xEF;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte TRYOUT
        {
            get
            {
                return (byte)((m_flags >> 2) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x04;
                else
                    m_flags &= 0xFB;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte IMMED
        {
            get
            {
                return (byte)((m_flags >> 1) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x02;
                else
                    m_flags &= 0xFD;
            }
        }

        /// <summary>
        /// See SCSI MMC-4 specification
        /// </summary>
        public byte VS
        {
            get
            {
                return (byte)((m_flags >> 0) & 0x01);
            }
            set
            {
                if (value != 0)
                    m_flags |= 0x01;
                else
                    m_flags &= 0xFE;
            }
        }
        #endregion

        /// <summary>
        /// Format the parameter list to the memory block to be sent down to the device.
        /// The result should be a format parameter block as seen in the SCSI MMC specs.
        /// </summary>
        /// <param name="dest"></param>
        public void FormatToMemory(IntPtr dest)
        {
            byte v;
            int offset;

            Marshal.WriteByte(dest, 0, 0);              // Byte 0 - reserved

            v = m_flags;
            if (IP != null)
                v |= 0x08;
            Marshal.WriteByte(dest, 1, v);              // Byte 1 - flags

            Marshal.WriteByte(dest, 2, 0);
            Marshal.WriteByte(dest, 3, 8);              // Format descriptor length = 8

            offset = 4 ;
            if (IP != null)
            {
                // If the initialization pattern exists, it goes here
                throw new Exception("Not supported yet");
            }

            // Write the # of blocks
            Marshal.WriteByte(dest, offset++, (byte)((NumberOfBlocks >> 24) & 0xff));
            Marshal.WriteByte(dest, offset++, (byte)((NumberOfBlocks >> 16) & 0xff));
            Marshal.WriteByte(dest, offset++, (byte)((NumberOfBlocks >> 8) & 0xff));
            Marshal.WriteByte(dest, offset++, (byte)((NumberOfBlocks >> 0) & 0xff));

            byte b = (byte)(((byte)FormatType << 2) | FormatSubType) ;
            Marshal.WriteByte(dest, offset++, b) ;

            Marshal.WriteByte(dest, offset++, (byte)((Parameter >> 16) & 0xff));
            Marshal.WriteByte(dest, offset++, (byte)((Parameter >> 8) & 0xff));
            Marshal.WriteByte(dest, offset++, (byte)((Parameter >> 0) & 0xff));
        }
    }
}
