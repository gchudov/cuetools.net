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
    /// This class represents the ATIP information read from the disk
    /// </summary>
    public class AtipInfo : Result
    {
        /// <summary>
        /// The target write power for the disk
        /// </summary>
        public readonly byte IndicativeTargetWritingPower;

        /// <summary>
        /// 
        /// </summary>
        public readonly byte ReferenceSpeed;

        /// <summary>
        /// If true, this is a double density CD
        /// </summary>
        public readonly bool IsDDCD;

        /// <summary>
        /// If true, this disk is for unrestricted use
        /// </summary>
        public readonly bool UnrestrictedUse;

        /// <summary>
        /// If true, the media is CDRW, otherwise it is CDR
        /// </summary>
        public readonly bool MediaCDRW;

        /// <summary>
        /// The CD/RW media subtype
        /// </summary>
        public readonly byte MediaSubtype;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool A1Valid;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool A2Valid;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool A3Valid;

        /// <summary>
        /// The ATIP value for the starting leadin time
        /// </summary>
        public readonly MinuteSecondFrame StartTimeOfLeadin;

        /// <summary>
        /// This is the address for the last possible leadout
        /// </summary>
        public readonly MinuteSecondFrame LastPossibleStartOfLeadout;

        /// <summary>
        /// The data from the A1 field
        /// </summary>
        public readonly byte[] A1Values;

        /// <summary>
        /// The data from the A1 field
        /// </summary>
        public readonly byte[] A2Values;

        /// <summary>
        /// The data from the A3 field
        /// </summary>
        public readonly byte[] A3Values;

        /// <summary>
        /// This constructs the ATIP infomation
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="size"></param>
        public AtipInfo(IntPtr buffer, int size) : base(buffer, size)
        {
            byte b;

            if (size < 4)
                return;

            b = Get8(4);
            IndicativeTargetWritingPower = (byte)((b >> 4) & 0x0f);
            IsDDCD = GetBit(4, 3);
            ReferenceSpeed = (byte)(b & 0x07);

            b = Get8(5);
            UnrestrictedUse = GetBit(5, 6);

            b = Get8(6);
            MediaCDRW = GetBit(6, 6);
            MediaSubtype = (byte)((b >> 3) & 0x07);

            A1Valid = GetBit(6, 2);
            A2Valid = GetBit(6, 1);
            A3Valid = GetBit(6, 0);

            StartTimeOfLeadin = new MinuteSecondFrame(Get8(8), Get8(9), Get8(10));
            LastPossibleStartOfLeadout = new MinuteSecondFrame(Get8(12), Get8(13), Get8(14));

            A1Values = new byte[3];
            for (int i = 0; i < 3; i++)
                A1Values[i] = Get8(i + 16);

            A2Values = new byte[3];
            for (int i = 0; i < 3; i++)
                A2Values[i] = Get8(i + 20);

            A3Values = new byte[3];
            for (int i = 0; i < 3; i++)
                A3Values[i] = Get8(i + 24);
        }
    }
}
