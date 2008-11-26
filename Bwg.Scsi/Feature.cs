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

namespace Bwg.Scsi
{
    /// <summary>
    /// This class describes a single SCSI feature.  The data associated with the feature is
    /// stored as a series of bytes that must be interpreted by the calling code.
    /// </summary>
    public class Feature : Result
    {
        #region Feature Numbers
        /// <summary>
        /// This enumerated type indicates the type of the feature.
        /// </summary>
        public enum FeatureType : ushort
        {
            /// <summary>
            /// The ProfileList feature (see the SCSI-3 MMC specification)
            /// </summary>
            ProfileList = 0x0000,

            /// <summary>
            /// The Core feature (see the SCSI-3 MMC specification)
            /// </summary>
            Core = 0x0001,

            /// <summary>
            /// The Morphing feature (see the SCSI-3 MMC specification)
            /// </summary>
            Morphing = 0x0002,

            /// <summary>
            /// The RemovableMedia feature (see the SCSI-3 MMC specification)
            /// </summary>
            RemovableMedia = 0x0003,

            /// <summary>
            /// The WriteProtect feature (see the SCSI-3 MMC specification)
            /// </summary>
            WriteProtect = 0x0004,

            /// <summary>
            /// The RandomReadable feature (see the SCSI-3 MMC specification)
            /// </summary>
            RandomReadable = 0x0010,

            /// <summary>
            /// The MultiRead feature (see the SCSI-3 MMC specification)
            /// </summary>
            MultiRead = 0x001d,

            /// <summary>
            /// The CDRead feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDRead = 0x001e,

            /// <summary>
            /// The DVDRead feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVDRead = 0x001f,

            /// <summary>
            /// The RandomWritable feature (see the SCSI-3 MMC specification)
            /// </summary>
            RandomWritable = 0x0020,

            /// <summary>
            /// The IncrementalStreamWritable feature (see the SCSI-3 MMC specification)
            /// </summary>
            IncrementalStreamWritable = 0x0021,

            /// <summary>
            /// The SectorErasable feature (see the SCSI-3 MMC specification)
            /// </summary>
            SectorErasable = 0x0022,

            /// <summary>
            /// The Formattable feature (see the SCSI-3 MMC specification)
            /// </summary>
            Formattable = 0x0023,

            /// <summary>
            /// The HardwareDefectManagement feature (see the SCSI-3 MMC specification)
            /// </summary>
            HardwareDefectManagement = 0x0024,

            /// <summary>
            /// The WriteOnce feature (see the SCSI-3 MMC specification)
            /// </summary>
            WriteOnce = 0x0025,

            /// <summary>
            /// The RestrictedOverwrite feature (see the SCSI-3 MMC specification)
            /// </summary>
            RestrictedOverwrite = 0x0026,

            /// <summary>
            /// The CDRWCavWrite feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDRWCavWrite = 0x0027,

            /// <summary>
            /// The MRW feature (see the SCSI-3 MMC specification)
            /// </summary>
            MRW = 0x0028,

            /// <summary>
            /// The EnhancedDefectReporting feature (see the SCSI-3 MMC specification)
            /// </summary>
            EnhancedDefectReporting = 0x0029,

            /// <summary>
            /// The DVDPlusRW feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVDPlusRW = 0x002a,

            /// <summary>
            /// The DVDPlusR feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVDPlusR = 0x002b,

            /// <summary>
            /// The RigidRestrictedOverwrite feature (see the SCSI-3 MMC specification)
            /// </summary>
            RigidRestrictedOverwrite = 0x002c,

            /// <summary>
            /// The CDTrackAtOnce feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDTrackAtOnce = 0x002d,

            /// <summary>
            /// The CDSessionAtOnce feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDSessionAtOnce = 0x002e,

            /// <summary>
            /// The DVDMinusR_DVDMinusRW feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVDMinusR_DVDMinusRW = 0x002f,

            /// <summary>
            /// The DDCDRead feature (see the SCSI-3 MMC specification)
            /// </summary>
            DDCDRead = 0x0030,

            /// <summary>
            /// The DDCDMinusR feature (see the SCSI-3 MMC specification)
            /// </summary>
            DDCDMinusR = 0x0031,

            /// <summary>
            /// The DDCDMinusRW feature (see the SCSI-3 MMC specification)
            /// </summary>
            DDCDMinusRW = 0x0032,

            /// <summary>
            /// The layer jump recording feature (see the SCSI MMC-5 specification)
            /// </summary>
            LayerJumpRecording = 0x0033,

            /// <summary>
            /// The CDRWMediaWriteSupport feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDRWMediaWriteSupport = 0x0037,

            /// <summary>
            /// 
            /// </summary>
            BD_RPseudoOverWrite = 0x0038,

            /// <summary>
            /// The dual layer feature for DVD+RW
            /// </summary>
            DvdPlusRWDualLayer = 0x003a,

            /// <summary>
            /// The dual layer featuer for DVD+R
            /// </summary>
            DvdPlusRDualLayer = 0x003b,

            /// <summary>
            /// 
            /// </summary>
            BDRead = 0x0040,

            /// <summary>
            /// 
            /// </summary>
            BDWrite = 0x0041,

            /// <summary>
            /// 
            /// </summary>
            TSR = 0x0042,

            /// <summary>
            /// 
            /// </summary>
            HdDvdRead = 0x0050,

            /// <summary>
            /// 
            /// </summary>
            HdDvdWrite = 0x0051,

            /// <summary>
            /// 
            /// </summary>
            OldHybridDisk = 0x0052,

            /// <summary>
            /// 
            /// </summary>
            HybridDisk = 0x0080,

            /// <summary>
            /// The PowerMangaementFeature feature (see the SCSI-3 MMC specification)
            /// </summary>
            PowerMangaementFeature = 0x0100,

            /// <summary>
            /// The S_M_A_R_T feature (see the SCSI-3 MMC specification)
            /// </summary>
            S_M_A_R_T = 0x0101,

            /// <summary>
            /// The EmbeddedChanger feature (see the SCSI-3 MMC specification)
            /// </summary>
            EmbeddedChanger = 0x0102,

            /// <summary>
            /// The CDAudioExternalPlay feature (see the SCSI-3 MMC specification)
            /// </summary>
            CDAudioExternalPlay = 0x0103,

            /// <summary>
            /// The MicrocodeUpgrade feature (see the SCSI-3 MMC specification)
            /// </summary>
            MicrocodeUpgrade = 0x0104,

            /// <summary>
            /// The Timeout feature (see the SCSI-3 MMC specification)
            /// </summary>
            Timeout = 0x0105,

            /// <summary>
            /// The DVDCSS feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVDCSS = 0x0106,

            /// <summary>
            /// The RealTimeStreaming feature (see the SCSI-3 MMC specification)
            /// </summary>
            RealTimeStreaming = 0x0107,

            /// <summary>
            /// The LogicalUnitSerialNumber feature (see the SCSI-3 MMC specification)
            /// </summary>
            LogicalUnitSerialNumber = 0x0108,

            /// <summary>
            /// The MediaSerialNumber feature (see the SCSI-3 MMC specification)
            /// </summary>
            MediaSerialNumber = 0x0109,

            /// <summary>
            /// The DiskControlBlock feature (see the SCSI-3 MMC specification)
            /// </summary>
            DiskControlBlock = 0x010A,

            /// <summary>
            /// The DVD_CPRM feature (see the SCSI-3 MMC specification)
            /// </summary>
            DVD_CPRM = 0x010B,

            /// <summary>
            /// The FirmwareInformation feature (see the SCSI-3 MMC specification)
            /// </summary>
            FirmwareInformation = 0x010C,

            /// <summary>
            /// 
            /// </summary>
            AACS = 0x010d,

            /// <summary>
            /// 
            /// </summary>
            VCPS = 0x0110
        } ;
        #endregion

        #region private variables

        /// <summary>
        /// The feature code for this feature
        /// </summary>
        private FeatureType m_code;

        /// <summary>
        /// The verion for this feature
        /// </summary>
        private byte m_version;

        /// <summary>
        /// If true, this feature is persistent (see SCSI MMC spec for more details)
        /// </summary>
        private bool m_persistent;

        /// <summary>
        /// If true, this feature is current (see SCSI MMC spec for more details)
        /// </summary>
        private bool m_current;

        /// <summary>
        /// The raw data assocaited with the feature.  This data may be invalid if the
        /// feature is invalid.
        /// </summary>
        private byte[] m_data;

        #endregion

        #region constructor
        /// <summary>
        /// This is a constructor for the SCSI feature.  It takes data from a raw buffer and
        /// creates the SCSI featuer object
        /// </summary>
        /// <param name="buffer">Pointer to the start of the raw reply buffer</param>
        /// <param name="size">Size of the raw reply buffer</param>
        /// <param name="offset">The offset to start parsing a SCSI feature from the reply buffer</param>
        public Feature(IntPtr buffer, int size, ref int offset) : base(buffer, size)
        {
            m_code = (FeatureType)Get16(offset);
            offset += 2;

            byte b = Get8(offset);
            offset++;

            if ((b & 0x01) != 0)
                m_current = true;
            else
                m_current = false;

            if ((b & 0x02) != 0)
                m_persistent = true;
            else
                m_persistent = false;

            m_version = (byte)((b >> 2) & 0x0f);

            byte len = Get8(offset);
            offset++;

            m_data = new byte[len];
            for(int i = 0 ; i < len && offset < BufferSize ; i++)
            {
                m_data[i] = Get8(offset);
                offset++;
            }
        }
        #endregion

        #region public methods
        /// <summary>
        /// Return a string representing a human readable form of this object.
        /// </summary>
        /// <returns>string representing this object</returns>
        public override string ToString()
        {
            return "Feature(type=" + m_code.ToString() + ")";
        }
        #endregion

        #region public propreties
        /// <summary>
        /// Return the code associated with this feature
        /// </summary>
        public FeatureType Code { get { return m_code; } }

        /// <summary>
        /// Return the version of the current feature
        /// </summary>
        public byte Version { get { return m_version; } }

        /// <summary>
        /// Return the presistence state of the feature.  If true, this feature is persistant
        /// </summary>
        public bool Persistent { get { return m_persistent; } }

        /// <summary>
        /// Return the current flag associated with the feature.  If true, this feature is currently active.
        /// </summary>
        public bool Current { get { return m_current; } }

        /// <summary>
        /// This item contains the data from the feature
        /// </summary>
        public byte[] Data { get { return m_data; } }

        /// <summary>
        /// This property returns the name of the feature.
        /// </summary>
        public string Name
        {
            get
            {
                return m_code.ToString();
            }
        }
        #endregion
    }
}
