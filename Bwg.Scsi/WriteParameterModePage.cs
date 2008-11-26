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

namespace Bwg.Scsi
{
    /// <summary>
    /// This class represents mode page 5, Write Parameters Mode Page
    /// </summary>
    public class WriteParameterModePage : ModePage
    {
        #region public types
        /// <summary>
        /// The type of write (packet/incremental, track, session)
        /// </summary>
        public enum WriteTypeType : byte
        {
            /// <summary>
            /// Packet or incremental
            /// </summary>
            PacketIncremental = 0,

            /// <summary>
            /// Track at once
            /// </summary>
            TrackAtOnce = 1,

            /// <summary>
            /// Session at once
            /// </summary>
            SessionAtOnce = 2,

            /// <summary>
            /// Raw data recording
            /// </summary>
            Raw = 3,

            /// <summary>
            /// Layer jump recording
            /// </summary>
            LayerJumpRecording = 4
        }

        /// <summary>
        /// The multisession state of the write page
        /// </summary>
        public enum MultiSessionType : byte
        {
            /// <summary>
            /// No B0 pointer, therefore no next session allowed
            /// </summary>
            NoNextSession = 0,

            /// <summary>
            /// B0 pointer equal to 0xFFFFFF, therefore not next session
            /// allowed
            /// </summary>
            CDNoNextSessionFFFF = 1,

            /// <summary>
            /// Reserved, do not use
            /// </summary>
            Reserved = 2,

            /// <summary>
            /// B0 pointer points to next session
            /// </summary>
            NextSessionAllowed = 3
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum TrackModeType : byte
        {
            /// <summary>
            /// 
            /// </summary>
            TwoChannelAudio = 0x00,

            /// <summary>
            /// 
            /// </summary>
            TwoChannelAudioWithPreemphasis = 0x01,

            /// <summary>
            /// 
            /// </summary>
            DataUninterrupted = 0x04,

            /// <summary>
            /// 
            /// </summary>
            DataIncremental = 0x05,

            /// <summary>
            /// 
            /// </summary>
            FourChannelAudio = 0x08,

            /// <summary>
            /// 
            /// </summary>
            FourChannelAudioWithPreemphasis = 0x09,
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum DataBlockTypeType : byte
        {
            /// <summary>
            /// Raw data, 2352 bytes
            /// </summary>
            RawData = 0,

            /// <summary>
            /// Raw data with P and Q subchannels, 2368 bytes
            /// </summary>
            RawDataWithPAndQ = 1,

            /// <summary>
            /// Raw data with P - W subchannels packed, 2448 bytes
            /// </summary>
            RawDataWithPToWPacked = 2,

            /// <summary>
            /// Raw data with P - W subchannels raw, 2448 bytes
            /// </summary>
            RawDataWithPToWRaw = 3,

            /// <summary>
            /// Data mode 1, 2048 bytes
            /// </summary>
            DataMode1 = 8,

            /// <summary>
            /// Data mode 2, 2336 bytes
            /// </summary>
            DataMode2 = 9,

            /// <summary>
            /// Data mode 2, sub-header from write params, 2048 bytes
            /// </summary>
            DataMode2Form1 = 10,

            /// <summary>
            /// Data mode 2, sub-header included, 2056 bytes
            /// </summary>
            DataMode2Form1Subheader = 11,

            /// <summary>
            /// Data mode 2, form 2, sub-header from write params, 2324 bytes
            /// </summary>
            DataMode2Form2 = 12,

            /// <summary>
            /// Data mode 2, form 2, sub-header included, 2332 bytes
            /// </summary>
            DataMode2Form2Subheader = 13,
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum SessionFormatType : byte
        {
            /// <summary>
            /// CD DA, CD ROM or other data disk
            /// </summary>
            CDDA_CDROM = 0x00,

            /// <summary>
            /// CD-I disk
            /// </summary>
            CD_I = 0x10,

            /// <summary>
            /// CD-ROM XA Disk
            /// </summary>
            CDROM_XA = 0x20,
        } ;

        #endregion

        #region constructor
        /// <summary>
        /// Construct the mode page
        /// </summary>
        /// <param name="buffer">pointer to a buffer contains the mode page data</param>
        /// <param name="size">the size of the buffer area</param>
        /// <param name="offset">the offset to the mode page</param>
        public WriteParameterModePage(IntPtr buffer, int size, ref ushort offset)
            : base(buffer, size, ref offset)
        {
        }
        #endregion

        #region public properties
        /// <summary>
        /// This property is the burn proof settings on this mode page
        /// </summary>
        public bool BurnProof
        {
            get
            {
                return (m_page_data[2] & 0x40) != 0;
            }
            set
            {
                if (value)
                    m_page_data[2] |= 0x40;
                else
                    m_page_data[2] &= 0xbf;
            }
        }

        /// <summary>
        /// If true, the link size field is valid.  If not true, the link size is assumed to be
        /// sever (per the SCSI MMC specification).
        /// </summary>
        public bool LinkSizeValid
        {
            get
            {
                return (m_page_data[2] & 0x20) != 0;
            }
            set
            {
                if (value)
                    m_page_data[2] |= 0x20;
                else
                    m_page_data[2] &= 0xdf;
            }
        }

        /// <summary>
        /// If true, any write will not effect the disk, and the write operation will only be a test.  If
        /// false the writes will go to the disk.  This is also known as simulation.
        /// </summary>
        public bool TestWrite
        {
            get
            {
                return (m_page_data[2] & 0x10) != 0;
            }
            set
            {
                if (value)
                    m_page_data[2] |= 0x10;
                else
                    m_page_data[2] &= 0xef;
            }
        }

        /// <summary>
        /// This property sets the write type
        /// </summary>
        public WriteTypeType WriteType
        {
            get
            {
                return (WriteTypeType)(m_page_data[2] & 0x0f);
            }
            set
            {
                m_page_data[2] &= 0xf0;
                m_page_data[2] |= (byte)value;
            }
        }

        /// <summary>
        /// This property controls the multi-session mode of the session or track to
        /// be burned.
        /// </summary>
        public MultiSessionType MultiSession
        {
            get
            {
                return (MultiSessionType)((m_page_data[3] >> 6) & 0x03);
            }
            set
            {
                Debug.Assert(value != MultiSessionType.Reserved);

                m_page_data[3] &= 0x3F;
                m_page_data[3] |= (byte)((byte)value << 6);
            }
        }

        /// <summary>
        /// This property controls whether the write mode is fixed packet.  This only applies
        /// if the write type is set to Packet/Incremental.
        /// </summary>
        public bool FixedPacket
        {
            get
            {
                return (m_page_data[3] & 0x20) != 0;
            }
            set
            {
                if (value)
                    m_page_data[3] |= 0x20;
                else
                    m_page_data[3] &= 0xdf;
            }
        }

        /// <summary>
        /// If true and the media is CD, SCMS copy protection is enabled.
        /// </summary>
        public bool Copy
        {
            get
            {
                return (m_page_data[3] & 0x10) != 0;
            }
            set
            {
                if (value)
                    m_page_data[3] |= 0x10;
                else
                    m_page_data[3] &= 0xef;
            }
        }

        /// <summary>
        /// The track mode.  This should be 5 for DVD media, and is the 
        /// </summary>
        public TrackModeType TrackMode
        {
            get
            {
                return (TrackModeType)(m_page_data[3] & 0x0d);
            }
            set
            {
                m_page_data[3] &= 0xf2;
                m_page_data[3] |= (byte)value;
            }
        }

        /// <summary>
        /// If true, digital copy if premitted of the content, otherwise it is not.
        /// </summary>
        public bool DigitalCopyPermitted
        {
            get
            {
                return (m_page_data[3] & 0x02) != 0;
            }
            set
            {
                if (value)
                    m_page_data[3] |= 0x02;
                else
                    m_page_data[3] &= 0xfd;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public DataBlockTypeType DataBlockType
        {
            get
            {
                return (DataBlockTypeType)(m_page_data[4] & 0x0f);
            }
            set
            {
                m_page_data[4] &= 0xf0;
                m_page_data[4] |= (byte)value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte LinkSize
        {
            get
            {
                return m_page_data[5];
            }
            set
            {
                m_page_data[5] = value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte HostApplicationCode
        {
            get
            {
                return (byte)(m_page_data[7] & 0x3f);
            }
            set
            {
                m_page_data[7] &= 0xc0;
                m_page_data[7] |= (byte)(value & 0x3f);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public SessionFormatType SessionFormat
        {
            get
            {
                return (SessionFormatType)m_page_data[8];
            }
            set
            {
                m_page_data[8] = (byte)value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int PacketSize
        {
            get
            {
                return ModeGet32(10);
            }
            set
            {
                ModeSet32(10, value);
            }
        }
        
        /// <summary>
        /// 
        /// </summary>
        public ushort AudioPauseLength
        {
            get
            {
                return ModeGet16(14);
            }
            set
            {
                ModeSet16(14, value);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte[] MediaCatalogNumber
        {
            get
            {
                byte[] num = new byte[16];

                for (int i = 0; i < 16; i++)
                    num[i] = m_page_data[16 + i];

                return num;
            }
            set
            {
                Debug.Assert(value.GetLength(0) == 16);

                for (int i = 0; i < 16; i++)
                    m_page_data[16 + i] = value[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte[] InternationalStandardRecordingCode
        {
            get
            {
                byte[] num = new byte[16];

                for (int i = 0; i < 16; i++)
                    num[i] = m_page_data[32 + i];

                return num;
            }
            set
            {
                Debug.Assert(value.GetLength(0) == 16);

                for (int i = 0; i < 16; i++)
                    m_page_data[32 + i] = value[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte SubHeaderByte0
        {
            get
            {
                return m_page_data[48];
            }
            set
            {
                m_page_data[48] = value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte SubHeaderByte1
        {
            get
            {
                return m_page_data[49];
            }
            set
            {
                m_page_data[49] = value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte SubHeaderByte2
        {
            get
            {
                return m_page_data[50];
            }
            set
            {
                m_page_data[50] = value;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public byte SubHeaderByte3
        {
            get
            {
                return m_page_data[51];
            }
            set
            {
                m_page_data[51] = value;
            }
        }
        #endregion
    }
}
