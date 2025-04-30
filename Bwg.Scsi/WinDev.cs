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
using System.Runtime.InteropServices;

namespace Bwg.Scsi
{
    /// <summary>
    /// 
    /// </summary>
    public unsafe class WinDev : ISysDev
    {
        //
        // External functions required to interface to th
        //
        [DllImport("Kernel32.dll", SetLastError = true)]
        private static extern IntPtr CreateFile(string h, uint acc, uint share, IntPtr sec, uint disp, uint flash, uint temp);

        [DllImport("Kernel32.dll", SetLastError = true)]
        private static extern bool DeviceIoControl(IntPtr h, uint code, IntPtr inbuf, uint insize, IntPtr outbuf, uint outsize, ref uint returned, IntPtr Overlapped);

        [DllImport("Kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr h);

        [DllImport("Kernel32.dll", SetLastError = false, CharSet = CharSet.Unicode)]
        private static extern uint FormatMessage(uint flags, IntPtr src, uint errcode, uint langid, IntPtr str, uint size, IntPtr vargs) ;

        /// <summary>
        /// The name of the device
        /// </summary>
        public string Name
        {
            get { CheckOpen() ; return m_name; }
        }

        /// <summary>
        /// Returns TRUE if the device is open, FALSE otherwise
        /// </summary>
        public bool IsOpen { get { return m_handle.ToInt32() != -1; } }

        private string m_name;
        private IntPtr m_handle;
        private int m_last_error;

        /// <summary>
        /// 
        /// </summary>
        public WinDev()
        {
            m_handle = new IntPtr(-1);
        }
        /// <summary>
        /// 
        /// </summary>
        public int LastError 
        { 
            get 
            { 
                return m_last_error; 
            } 
        }

        /// <summary>
        /// 
        /// </summary>
        public void Close()
        {
            CloseHandle(m_handle);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public bool Open(string name)
        {
            string dname = name;

            //
            // CreateFile
            //      name
            //      accessmode = GenericRead | GenericWrite
            UInt32 acc = 0x80000000 | 0x40000000;
            m_handle = CreateFile(dname, acc, 0x01, (IntPtr)0, 3, 0x00000080, (uint)0);
            if (m_handle.ToInt32() == -1)
            {
                m_last_error = Marshal.GetLastWin32Error();
                return false;
            }

            m_name = name;
            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="letter"></param>
        /// <returns></returns>
        public bool Open(char letter)
        {
            string dname = "\\\\.\\" + letter + ":";
            return Open(dname);
        }

        /// <summary>
        /// 
        /// </summary>
        protected void CheckOpen()
        {
            if (m_handle.ToInt32() == -1)
                throw new Exception("device is not open") ;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="code"></param>
        /// <param name="inbuf"></param>
        /// <param name="insize"></param>
        /// <param name="outbuf"></param>
        /// <param name="outsize"></param>
        /// <param name="ret"></param>
        /// <param name="overlapped"></param>
        /// <returns></returns>
        public bool Control(uint code, IntPtr inbuf, uint insize, IntPtr outbuf, uint outsize, ref uint ret, IntPtr overlapped)
        {
            bool b;

            CheckOpen() ;
            b = DeviceIoControl(m_handle, code, inbuf, insize, outbuf, outsize, ref ret, overlapped);
            if (!b)
                m_last_error = Marshal.GetLastWin32Error();

            return b;
        }

        /// <summary>
        /// Convert a WIN32 error code to an error string.
        /// </summary>
        /// <param name="error">the error code to convert</param>
        /// <returns>the string for the error code</returns>
        public string ErrorCodeToString(int error)
        {
            IntPtr buffer = Marshal.AllocHGlobal(1024) ;
            uint ret = FormatMessage(0x1000, IntPtr.Zero, (uint)error, 0, buffer, 1024, IntPtr.Zero);
            if (ret == 0)
                return "cannot find win32 error string for this code (" + error.ToString() + ")";

            string str = Marshal.PtrToStringUni(buffer);
            str = str.Trim();
            Marshal.FreeHGlobal(buffer);

            return str;
        }

        private bool _disposed;
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                if (IsOpen) Close();

                _disposed = true;
            }
        }

        ~WinDev() => Dispose(disposing: false);

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
