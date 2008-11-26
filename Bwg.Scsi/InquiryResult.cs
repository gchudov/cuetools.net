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
    /// This class contains the information returned from a SCSI Inquiry request
    /// </summary>
    public sealed class InquiryResult : Result
    {
        /// <summary>
        /// The peripheral qualifier field from the SCSI inquiry reply, should be zero for MMC devices.
        /// </summary>
        public readonly byte PeripheralQualifier ;

        /// <summary>
        /// The peripheral device type field from the SCSI inquiry reply, should be five for MMC devices.
        /// </summary>
        public readonly byte PeripheralDeviceType;

        /// <summary>
        /// This gives the version number of the Inquiry data returned
        /// </summary>
        public readonly byte ResponseDataFormat;

        /// <summary>
        /// The RMB field from the SCSI inquiry reply, should be true for MMC devices.
        /// </summary>
        public readonly bool RMB;

        /// <summary>
        /// 
        /// </summary>
        public readonly byte Version;

        /// <summary>
        /// 
        /// </summary>
        public readonly string VendorIdentification;

        /// <summary>
        /// 
        /// </summary>
        public readonly string ProductIdentification;

        /// <summary>
        /// The version of the firmware
        /// </summary>
        public readonly string FirmwareVersion;

        /// <summary>
        /// 
        /// </summary>
        public readonly string ProductRevision;

        /// <summary>
        /// This property returns true if the device supports removable medium.
        /// </summary>
        public bool RemovableMediumBit { get { return RMB; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="size"></param>
        public InquiryResult(IntPtr buffer, int size) : base(buffer, size)
        {
            ResponseDataFormat = (byte)(Get8(3) & 0x0f);

            //
            // Make sure the inquiry data is in the right format
            // A data format of 2 is good up through MMC-3.  A data format of 3
            // is used for MMC-4 and MMC-5.  Some drives (especially from LiteOn)
            // use a value of 1, but still behave as expected.
            //
            if (ResponseDataFormat != 2 && ResponseDataFormat != 3 && ResponseDataFormat != 1)
            {
                m_valid = false;
                return;
            }

            PeripheralQualifier = (byte)((Get8(0) >> 5) & 0x07);
            PeripheralDeviceType = (byte)(Get8(0) & 0x1f);
            RMB = false;
            if ((Get8(1) & 0x80) != 0)
                RMB = true;

            Version = Get8(0x02);

            VendorIdentification = GetString(8, 15);
            ProductIdentification = GetString(16, 31);

            int len = 35;
            if (len >= size)
                len = size - 1;
            FirmwareVersion = GetString(32, len);

            if (size >= 56)
                ProductRevision = GetString(36, 55);
            else
                ProductRevision = string.Empty;
        }
    }
}
