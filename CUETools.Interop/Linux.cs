#region Copyright (C) 2025 Max Visser
/*
    Copyright (C) 2025 Max Visser

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <https://www.gnu.org/licenses/>.
*/
#endregion
/*
 * Warning: This is a highly experimental implementation for Linux.
 * The code in this file requires thorough review and improvement.  
 * 
 * 64-bit is the only supported architecture at this time. 
 * Backporting to 32-bit is possible, but not planned.
 */
using System.Runtime.InteropServices;
using System;

namespace CUETools.Interop
{
    public class Linux
    {
        #region Constants

        public const uint CDROM_DRIVE_STATUS = 0x5326;
        public const int CDROM_LOCKDOOR = 0x5329;

        public const int CDS_DISC_OK = 4;
        public const int O_RDONLY = 0;

        public const int SG_DXFER_FROM_DEV = -3;

        public const int SG_GET_RESERVED_SIZE = 0x2272;
        public const int SG_IO = 0x2285;

        public const string CDROM_DEVICE_PATH = "/dev/sr";

        #endregion

        #region Structs

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        struct Stat
        {
            public ulong st_dev;
            public ulong st_ino;
            public ulong st_nlink;
            public uint st_mode;
            public uint st_uid;
            public uint st_gid;
            public int padding0;
            public ulong st_rdev;
            public long st_size;
            public long st_blksize;
            public long st_blocks;
            public long st_atime;
            public long st_atimensec;
            public long st_mtime;
            public long st_mtimensec;
            public long st_ctime;
            public long st_ctimensec;
            public long reserved0;
            public long reserved1;
            public long reserved2;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct SG_IO_HDR
        {
            public int interface_id;
            public int dxfer_direction;
            public byte cmd_len;
            public byte mx_sb_len;
            public ushort iovec_count;
            public uint dxfer_len;
            public IntPtr dxferp;
            public IntPtr cmdp;
            public IntPtr sbp;
            public uint timeout;
            public uint flags;
            public int pack_id;
            public IntPtr usr_ptr;
            public byte status;
            public byte masked_status;
            public byte msg_status;
            public byte sb_len_wr;
            public ushort host_status;
            public ushort driver_status;
            public int resid;
            public uint duration;
            public uint info;
        }

        #endregion

        #region Functions

        private const string LIBC = "libc";

        [DllImport(LIBC, SetLastError = true, CharSet = CharSet.Ansi)]
        public static extern IntPtr strerror(int errnum);

        [DllImport(LIBC, SetLastError = true, CharSet = CharSet.Ansi)]
        public static extern int open(string pathname, int flags);

        [DllImport(LIBC, SetLastError = true)]
        public static extern int close(int fd);

        [DllImport(LIBC, SetLastError = true)]
        public static extern int ioctl(int fd, uint request);

        [DllImport(LIBC, SetLastError = true)]
        public static extern int ioctl(int fd, uint request, IntPtr arg);

        [DllImport(LIBC, SetLastError = true, CharSet = CharSet.Ansi)]
        public static extern int lstat(string path, IntPtr stat);

        #endregion

        #region Helper Functions

        public static int GetErrorCode()
            => Marshal.GetLastWin32Error();

        public static string GetErrorString(int errorCode)
            => Marshal.PtrToStringAnsi(strerror(errorCode));

        public static string GetErrorString()
            => GetErrorString(GetErrorCode());

        public static unsafe bool PathExists(string path)
        {
            if (sizeof(void*) != 8) throw Non64BitException();

            Stat stat;
            var pStat = new IntPtr(&stat);
            return lstat(path, pStat) == 0;
        }

        public static PlatformNotSupportedException Non64BitException()
            => new PlatformNotSupportedException("This application only supports 64-bit systems on Linux.");

        #endregion
    }
}
