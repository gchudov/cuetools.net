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
    /// This class represents a mode table from the scsi device.
    /// </summary>
    public class ModeTable : Result
    {
        /// <summary>
        /// This is the size of the header associated with a mode table.
        /// </summary>
        public const int ModeTableHeaderSize = 8;

        /// <summary>
        /// The set of mode pages in this mode table.
        /// </summary>
        public IList<ModePage> Pages;

        /// <summary>
        /// The constructor for the mode table object.  It builds the mode table from
        /// the raw reply buffer from the SCSI device.
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="size"></param>
        public ModeTable(IntPtr buffer, int size): base(buffer, size)
        {
            ushort len = Get16(0);
            ushort index = ModeTableHeaderSize;

            Pages = new List<ModePage>();

            while (index < len && index < size)
            {
                int b0 = Get8(index) & 0x3f;

                ModePage page ;
                if (b0 == 0x05)
                    page = new WriteParameterModePage(buffer, size, ref index) ;
                else
                    page = new ModePage(buffer, size, ref index);

                Pages.Add(page);
            }
        }

        /// <summary>
        /// This property returns the total size of the raw mode table in bytes
        /// </summary>
        public ushort Size
        {
            get
            {
                ushort len = ModeTableHeaderSize;

                foreach (ModePage p in Pages)
                    len += p.Length;

                return len;
            }
        }

        /// <summary>
        /// Format a mode table back into raw buffer for shipment down to a SCSI device.
        /// </summary>
        /// <param name="buffer">The buffer to receive the mode table</param>
        public void Format(IntPtr buffer)
        {
            ushort s = (ushort)(Size - ModeTableHeaderSize);
            Marshal.WriteByte(buffer, 0, (byte)((s >> 8) & 0xff));
            Marshal.WriteByte(buffer, 1, (byte)(s & 0xff));

            int offset = ModeTableHeaderSize;
            foreach (ModePage p in Pages)
            {
                IntPtr dest = new IntPtr(buffer.ToInt32() + offset);
                Marshal.Copy(p.PageData, 0, dest, p.Length);
                offset += p.Length;
            }
        }
    }
}
