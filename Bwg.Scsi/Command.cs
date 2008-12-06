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
using System.Diagnostics;

namespace Bwg.Scsi
{
	public class Command : IDisposable
    {
        [DllImport("ntdll.dll")]
        internal static extern void RtlZeroMemory(IntPtr dest, int size);

        public enum CmdDirection
        {
            In,
            Out,
            None
        } ;

        /// <summary>
        /// The data buffer for the SCSI command
        /// </summary>
        private int m_buffer_size;
        private IntPtr m_buffer;
        private bool m_delete_buffer;

        /// <summary>
        /// The CDB buffer for the command.  This gets copied into the SCSI_PASS_THROUGH_DIRECT
        /// structure when this command is sent to the SCSI target.
        /// </summary>
        private byte[] m_cdb;

        /// <summary>
        /// The direction of data transfer via the m_buffer field above
        /// </summary>
        private CmdDirection m_dir;

        /// <summary>
        /// The amount of time to pass in seconds before a timeout
        /// </summary>
        private int m_timeout;

        public Command(ScsiCommandCode code, byte cdbsize, int bufsize, CmdDirection dir, int timeout)
        {
            Debug.Assert(bufsize < UInt16.MaxValue);

            m_cdb = new byte[cdbsize];
            m_cdb[0] = (byte)code;

            m_buffer_size = bufsize;
            if (bufsize == 0)
                m_buffer = IntPtr.Zero;
            else
            {
                m_delete_buffer = true;
                m_buffer = Marshal.AllocHGlobal(bufsize);
                RtlZeroMemory(m_buffer, bufsize);
            }

            m_dir = dir;
            m_timeout = timeout;
        }

        public Command(ScsiCommandCode code, byte cdbsize, IntPtr buffer, int bufsize, CmdDirection dir, int timeout)
        {
            m_cdb = new byte[cdbsize];
            m_cdb[0] = (byte)code;

            m_buffer_size = bufsize;
            if (bufsize == 0)
                m_buffer = IntPtr.Zero;
            else
            {
                m_delete_buffer = false;
                m_buffer = buffer;
            }

            m_dir = dir;
            m_timeout = timeout;
        }

        public int TimeOut
        { 
            get 
            { 
                return m_timeout; 
            } 
        }

        public ushort GetCDBLength()
        {
            return (ushort)m_cdb.GetLength(0);
        }

        public CmdDirection Direction { get { return m_dir; } }
        public int BufferSize { get { return m_buffer_size; } }
        public IntPtr GetBuffer() { return m_buffer ; }
        public byte GetCDB(byte addr) { return m_cdb[addr] ; }

        public void SetCDB8(byte addr, byte value)
        {
            m_cdb[addr] = value;
        }

        public void SetCDB16(byte addr, ushort value)
        {
            m_cdb[addr] = (byte)((value >> 8) & 0xff);
            m_cdb[addr + 1] = (byte)(value & 0xff);
        }

        public void SetCDB24(byte addr, uint value)
        {
            m_cdb[addr] = (byte)((value >> 16) & 0xff);
            m_cdb[addr + 1] = (byte)((value >> 8) & 0xff);
            m_cdb[addr + 2] = (byte)(value & 0xff);
        }

        public void SetCDB24(byte addr, int value)
        {
            m_cdb[addr] = (byte)((value >> 16) & 0xff);
            m_cdb[addr + 1] = (byte)((value >> 8) & 0xff);
            m_cdb[addr + 2] = (byte)(value & 0xff);
        }

        public void SetCDB32(byte addr, uint value)
        {
            m_cdb[addr] = (byte)((value >> 24) & 0xff);
            m_cdb[addr + 1] = (byte)((value >> 16) & 0xff);
            m_cdb[addr + 2] = (byte)((value >> 8) & 0xff);
            m_cdb[addr + 3] = (byte)(value & 0xff);
        }

        public void SetCDB32(byte addr, int value)
        {
            m_cdb[addr] = (byte)((value >> 24) & 0xff);
            m_cdb[addr + 1] = (byte)((value >> 16) & 0xff);
            m_cdb[addr + 2] = (byte)((value >> 8) & 0xff);
            m_cdb[addr + 3] = (byte)(value & 0xff);
        }

        public void SetBuffer8(ushort addr, byte value)
        {
            Marshal.WriteByte(m_buffer, addr, value);
        }

        public void SetBuffer16(ushort addr, ushort value)
        {
            Marshal.WriteByte(m_buffer, addr + 0, (byte)((value >> 8) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 1, (byte)((value >> 0) & 0xff));
        }

        public void SetBuffer24(ushort addr, uint value)
        {
            Marshal.WriteByte(m_buffer, addr + 0, (byte)((value >> 16) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 1, (byte)((value >> 8) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 2, (byte)((value >> 0) & 0xff));
        }

        public void SetBuffer32(ushort addr, uint value)
        {
            Marshal.WriteByte(m_buffer, addr + 0, (byte)((value >> 24) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 1, (byte)((value >> 16) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 2, (byte)((value >> 8) & 0xff));
            Marshal.WriteByte(m_buffer, addr + 3, (byte)((value >> 0) & 0xff));
        }

        public byte GetBuffer8(ushort addr)
        {
            return Marshal.ReadByte(m_buffer, addr);
        }

        public ushort GetBuffer16(ushort addr)
        {
            ushort v = 0;

            v |= (ushort)(Marshal.ReadByte(m_buffer, addr) << 8);
            v |= (ushort)(Marshal.ReadByte(m_buffer, addr + 1) << 0);

            return v;
        }

        public uint GetBuffer24(ushort addr)
        {
            uint v = 0;

            v |= (uint)(Marshal.ReadByte(m_buffer, addr) << 16);
            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 1) << 8);
            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 2) << 0);

            return v;
        }

        public uint GetBuffer32(ushort addr)
        {
            uint v = 0;

            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 0) << 24);
            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 1) << 16);
            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 2) <<  8);
            v |= (uint)(Marshal.ReadByte(m_buffer, addr + 3) <<  0);

            return v;
        }

        public void Dispose()
        {
            if (m_buffer_size > 0 && m_delete_buffer)
                Marshal.FreeHGlobal(m_buffer);
            m_cdb = null;
        }
    }
}
