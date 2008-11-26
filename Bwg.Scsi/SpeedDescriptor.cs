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
    /// 
    /// </summary>
    public class SpeedDescriptor : Result
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly Device.RotationalControl WRC;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool Exact;

        /// <summary>
        /// 
        /// </summary>
        public readonly bool MRW;

        /// <summary>
        /// 
        /// </summary>
        public readonly uint EndLBA;

        /// <summary>
        /// 
        /// </summary>
        public readonly int ReadSpeed;

        /// <summary>
        /// 
        /// </summary>
        public readonly int WriteSpeed;

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="offset"></param>
        /// <param name="size"></param>
        public SpeedDescriptor(IntPtr buffer, int offset, int size)
            : base(buffer, size)
        {
            byte b = Get8(offset);

            if (((b & 0x03) >> 3 == 0))
                WRC = Device.RotationalControl.CLVandNonPureCav ;
            else if (((b & 0x03) >> 3) == 1)
                WRC = Device.RotationalControl.PureCav ;

            if ((b & 0x02) == 0)
                Exact = false ;
            else
                Exact = true ;

            if ((b & 0x01) == 0)
                MRW = false ;
            else
                MRW = true ;

            EndLBA = Get32(offset + 4);
            ReadSpeed = Get32Int(offset + 8);
            WriteSpeed = Get32Int(offset + 12);
        }

        /// <summary>
        /// Return a human readable string representing this object
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string str = string.Empty;

            str += "WRC=" + WRC.ToString();
            str += ", Exact=" + Exact.ToString();
            str += ", MRW=" + MRW.ToString();
            str += ", EndLba=" + EndLBA.ToString();
            str += ", Read=" + ReadSpeed.ToString();
            str += ", Write=" + WriteSpeed.ToString();

            return str;
        }
        #endregion
    }
}
