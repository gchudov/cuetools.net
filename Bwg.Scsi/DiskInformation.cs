//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for DotNet
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
    /// This class contains the results of the ReadDiskInformation SCSI command
    /// </summary>
    public class DiskInformation : Result
    {
        #region Public types
        /// <summary>
        /// Type defining the state of the last session on the disk
        /// </summary>
        public enum SessionStateType
        {
            /// <summary>
            /// The session is empty
            /// </summary>
            EmptySession = 0,

            /// <summary>
            /// The session is incomplete
            /// </summary>
            IncompleteSession = 1,

            /// <summary>
            /// The session is damaged
            /// </summary>
            DamagedSession = 2,

            /// <summary>
            /// The session is complete
            /// </summary>
            CompleteSession = 3
        } ;

        /// <summary>
        /// Type defining the status of the disk
        /// </summary>
        public enum DiskStatusType
        {
            /// <summary>
            /// The disk is empty
            /// </summary>
            EmptyDisk = 0,

            /// <summary>
            /// The disk is incomplete
            /// </summary>
            IncompleteDisk = 1,

            /// <summary>
            /// The disk is finalized
            /// </summary>
            FinalizedDisk = 2,

            /// <summary>
            /// The disk is in an other state which is dependent on the media type
            /// </summary>
            RandomlyRecordable = 3
        } ;

        /// <summary>
        /// The state of a background format
        /// </summary>
        public enum BackgroundFormatStatusType
        {
            /// <summary>
            /// No background format was started, or it does not apply
            /// </summary>
            NotStartedOrDoesNotApply = 0,

            /// <summary>
            /// A background format was started, did not complete, and is not running
            /// </summary>
            StartedButNotRunning = 1,

            /// <summary>
            /// A background format is in progress and is not complete
            /// </summary>
            InProgress = 2,

            /// <summary>
            /// A background format was started and is complete
            /// </summary>
            Complete = 3
        } ;
        #endregion

        #region Public Data Members

        private readonly bool m_erasable;

        /// <summary>
        /// If true, the disk is erasable.  If false the disk cannot be erased
        /// </summary>
        public bool Erasable
        {
            get { return m_erasable; }
        } 

        private readonly SessionStateType m_session_state;

        /// <summary>
        /// The state of the last session on the disk
        /// </summary>        
        public SessionStateType SessionState
        {
            get { return m_session_state; }
        } 

        private readonly DiskStatusType m_disk_status;


        /// <summary>
        /// The status of the disk
        /// </summary>
        public DiskStatusType DiskStatus
        {
            get { return m_disk_status; }
        } 


        private readonly byte m_first_track;

        /// <summary>
        /// Number of the first track on the disk
        /// </summary>
        public byte FirstTrack
        {
            get { return m_first_track; }
        } 


        private readonly ushort m_session_count;

        /// <summary>
        /// Number of sessions on the disk
        /// </summary>
        public ushort SessionCount
        {
            get { return m_session_count; }
        } 


        private readonly ushort m_first_track_in_last_session;

        /// <summary>
        /// The first track number in the last session
        /// </summary>
        public ushort FirstTrackInLastSession
        {
            get { return m_first_track_in_last_session; }
        } 


        private readonly ushort m_last_track_in_last_session;

        /// <summary>
        /// The last track number in the last session
        /// </summary>
        public ushort LastTrackInLastSession
        {
            get { return m_last_track_in_last_session; }
        } 


        private readonly bool m_disk_id_valid;

        /// <summary>
        /// If true, the disk ID was valid
        /// </summary>
        public bool DiskIdValid
        {
            get { return m_disk_id_valid; }
        } 


        private readonly uint m_disk_identification;

        /// <summary>
        /// The four byte disk id
        /// </summary>
        public uint DiskIdentification
        {
            get { return m_disk_identification; }
        } 


        private readonly bool m_disk_bar_code_valid;

        /// <summary>
        /// If true, the disk bar code was valid
        /// </summary>
        public bool DiskBarCodeValid
        {
            get { return m_disk_bar_code_valid; }
        } 


        private readonly byte[] m_disk_bar_code;

        /// <summary>
        /// The eight byte disk bar code
        /// </summary>
        public byte[] DiskBarCode
        {
            get { return m_disk_bar_code; }
        } 


        private readonly bool m_unrestricted_disk_use;

        /// <summary>
        /// If true, this disk is unrestriced it its use
        /// </summary>
        public bool UnrestrictedDiskUse
        {
            get { return m_unrestricted_disk_use; }
        } 


        private readonly bool m_disk_application_code_valid;

        /// <summary>
        /// If true, the disk application code is valid
        /// </summary>
        public bool DiskApplicationCodeValid
        {
            get { return m_disk_application_code_valid; }
        } 


        private readonly byte m_disk_application_code;

        /// <summary>
        /// The disk application code
        /// </summary>
        public byte DiskApplicationCode
        {
            get { return m_disk_application_code; }
        } 


        private readonly bool m_dirty_bit;

        /// <summary>
        /// The Dbit (dirty bit) for MRW media
        /// </summary>
        public bool DirtyBit
        {
            get { return m_dirty_bit; }
        } 


        private readonly BackgroundFormatStatusType m_background_format_status;

        /// <summary>
        /// The status of a background format
        /// </summary>
        public BackgroundFormatStatusType BackgroundFormatStatus
        {
            get { return m_background_format_status; }
        } 


        private readonly MinuteSecondFrame m_last_session_lead_in_start_address;

        /// <summary>
        /// The last session leadin start address, media dependent, see SCSI-3 MMC spec
        /// </summary>
        public MinuteSecondFrame LastSessionLeadInStartAddress
        {
            get { return m_last_session_lead_in_start_address; }
        } 


        private readonly MinuteSecondFrame m_last_possible_leadout_start_address;

        /// <summary>
        /// The last possible leadout start address, media dependent, see SCSI-3 MMC spec
        /// </summary>
        public MinuteSecondFrame LastPossibleLeadoutStartAddress
        {
            get { return m_last_possible_leadout_start_address; }
        } 


        /// <summary>
        /// The OPC table
        /// </summary>
        public IList<OpcTableEntry> OpcTable;

        #endregion

        #region constructor
        /// <summary>
        /// The constructor for the DiskInformation class.  It parses the information from the
        /// memory buffer that contains the raw SCSI result.
        /// </summary>
        /// <param name="buffer">The buffer containing the raw SCSI result</param>
        /// <param name="size">The size of the raw SCSI result buffer</param>
        public DiskInformation(IntPtr buffer, int size) : base(buffer, size)
        {
            ushort len = Get16(0);
            byte b = Get8(2);

            if ((b & 0x10) != 0)
                m_erasable = true;
            else
                m_erasable = false;

            m_session_state = (SessionStateType)((b >> 2) & 0x03);
            m_disk_status = (DiskStatusType)(b & 0x03);

            m_first_track = Get8(3);
            m_session_count = (ushort)(Get8(4) | (Get8(9) << 8));
            m_first_track_in_last_session = (ushort)(Get8(5) | (Get8(10) << 8));
            m_last_track_in_last_session = (ushort)(Get8(6) | (Get8(11) << 8));

            b = Get8(7);
            if ((b & 0x80) != 0)
            {
                m_disk_id_valid = true;
                m_disk_identification = Get32(12);
            }
            else
                m_disk_id_valid = false;

            if ((b & 0x40) != 0)
            {
                m_disk_bar_code_valid = true;
                m_disk_bar_code = new byte[8];
                for (int i = 24; i <= 31; i++)
                    m_disk_bar_code[i - 24] = Get8(i);
            }
            else
            {
                m_disk_bar_code_valid = false;
                m_disk_bar_code = null;
            }

            if ((b & 0x20) != 0)
                m_unrestricted_disk_use = true;
            else
                m_unrestricted_disk_use = false;

            if ((b & 0x10) != 0)
            {
                m_disk_application_code_valid = true;
                m_disk_application_code = Get8(32);
            }
            else
                m_disk_application_code_valid = false;

            if ((b & 0x04) != 0)
                m_dirty_bit = true;
            else
                m_dirty_bit = false;

            m_background_format_status = (BackgroundFormatStatusType)(b & 0x03);

            m_last_session_lead_in_start_address = new MinuteSecondFrame(Get8(16) * 60 + Get8(17), Get8(18), Get8(19));
            m_last_possible_leadout_start_address = new MinuteSecondFrame(Get8(20) * 60 + Get8(21), Get8(22), Get8(23));

            OpcTable = new List<OpcTableEntry>();

            if (BufferSize > 33)
            {
                byte cnt = Get8(33);
                int offset = 34;

                for(byte i = 0 ; i < cnt && offset < BufferSize ; i++)
                {
                    OpcTableEntry entry = new OpcTableEntry(Buffer, BufferSize, offset);
                    OpcTable.Add(entry);
                    offset += 8;
                }
            }
        }
        #endregion
    }
}
