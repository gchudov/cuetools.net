// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace Bwg.Scsi
{
    // Retains Bwg's base-class contract; the Windows build still uses WinDev.cs.
    public class WinDev : IDisposable
    {
        private SafeFileHandle handle;
        private string name;
        private uint lastError;

        [DllImport("libc", EntryPoint = "open", SetLastError = true)]
        private static extern int OpenFile(string path, int flags);

        [DllImport("libc", EntryPoint = "ioctl", SetLastError = true)]
        private static extern int Ioctl(SafeFileHandle fd, nuint request, IntPtr data);

        public string Name { get { CheckOpen(); return name; } }
        public bool IsOpen { get { return handle != null && !handle.IsInvalid && !handle.IsClosed; } }
        public uint LastError { get { return lastError; } }

        public virtual bool Open(string path)
        {
            Close();
            // Do not allow the optical helper to send commands to arbitrary disks.
            path = CUETools.Ripper.OpticalDrives.Resolve(path);
            int fd = OpenFile(path, 2 | 0x800 | 0x80000); // O_RDWR | O_NONBLOCK | O_CLOEXEC
            if (fd < 0)
            {
                lastError = (uint)Marshal.GetLastPInvokeError();
                return false;
            }
            handle = new SafeFileHandle(new IntPtr(fd), true);
            name = path;
            lastError = 0;
            return true;
        }

        public virtual bool Open(char drive)
        {
            throw new PlatformNotSupportedException("Use an optical device path, such as /dev/sr0.");
        }

        public void Close()
        {
            handle?.Dispose();
            handle = null;
        }

        public void Dispose() { Close(); }

        protected void CheckOpen()
        {
            if (!IsOpen) throw new InvalidOperationException("The optical drive is not open.");
        }

        protected virtual bool ScsiControl(IntPtr request)
        {
            CheckOpen();
            int result = Ioctl(handle, 0x2285, request); // SG_IO
            lastError = result < 0 ? (uint)Marshal.GetLastPInvokeError() : 0;
            return result >= 0;
        }

        // Unreachable Windows transport methods remain compiled in Device.cs.
        protected bool Control(uint code, IntPtr input, uint inputSize, IntPtr output,
            uint outputSize, ref uint returned, IntPtr overlapped)
        {
            throw new PlatformNotSupportedException("Windows device control is unavailable on Linux.");
        }

        public static string Win32ErrorToString(uint error)
        {
            return new Win32Exception((int)error).Message;
        }
    }
}
