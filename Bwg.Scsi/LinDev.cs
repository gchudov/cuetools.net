#region Copyright (C) 2026 Max Visser
/*
    Copyright (C) 2026 Max Visser

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
 * Linux reimplementation of WinDev.cs
 * Tested on Arch and Mint in combination with a USB DVD-drive
 * Various cases are still untested, consider it experimental
 * 
 * Output files validated against CTDB
 * CUE file and gaps validated against Windows rip and cdrdao under Linux
 */
#if NETSTANDARD2_0
using CUETools.Interop;
using System;
using System.Runtime.InteropServices;

namespace Bwg.Scsi
{
    internal unsafe class LinDev : ISysDev
    {
        private string _name;
        public string Name
        {
            get 
            { 
                CheckOpen(); 
                return _name;
            }
        }

        private int _fd = -1;
        public bool IsOpen => _fd != -1;

        public LinDev()
        {
            _fd = -1;
        }

        public int LastError { get; private set; }

        public void Close()
        {
            if (IsOpen)
            {
                Linux.close(_fd);
                _fd = -1;
                _name = null;
            }
        }

        public bool Open(string name)
            => throw new NotImplementedException($"{System.Reflection.MethodBase.GetCurrentMethod().Name} is not implemented.");

        public bool Open(char number)
        {
            string name = $"{Linux.CDROM_DEVICE_PATH}{number}";
            _fd = Linux.open(name, Linux.O_RDONLY);

            if (_fd == -1)
            {
                LastError = Marshal.GetLastWin32Error();
                return false;
            }

            _name = name;
            return true;
        }

        protected void CheckOpen()
        {
            if (!IsOpen) throw new Exception("device is not open");
        }

        public bool Control(uint code, IntPtr inbuf, uint insize, IntPtr outbuf, uint outsize, ref uint ret, IntPtr overlapped)
        {
            CheckOpen();
            ret = 0;

            if (inbuf != outbuf) throw new NotImplementedException("Unexpected state");

            switch (code)
            {
                case Device.IOCTL_SCSI_GET_CAPABILITIES:
                    {
                        var caps = (Device.IO_SCSI_CAPABILITIES*)outbuf;

                        uint maxTransferLength = 0;
                        var result = Linux.ioctl(_fd, Linux.SG_GET_RESERVED_SIZE, new IntPtr(&maxTransferLength));
                        if (result < 0)
                        {
                            LastError = Marshal.GetLastWin32Error();
                            return false;
                        }

                        caps->MaximumTransferLength = maxTransferLength;
                        return true;
                    }
                case Device.IOCTL_SCSI_PASS_THROUGH_DIRECT:
                    {
                        var length = *(ushort*)outbuf;
                        if (length == Device.m_scsi_request_size_32)
                        {
                            var winScsi = (Device.SCSI_PASS_THROUGH_DIRECT32*)outbuf;
                            var linScsi = new Linux.SG_IO_HDR
                            {
                                interface_id = 'S',
                                dxfer_direction = TransferDirection(winScsi->DataIn, winScsi->DataTransferLength),
                                cmdp = new IntPtr(winScsi->CdbData),
                                cmd_len = winScsi->CdbLength,
                                dxfer_len = winScsi->DataTransferLength,
                                dxferp = winScsi->DataBuffer,
                                mx_sb_len = winScsi->SenseInfoLength,
                                sbp = new IntPtr(winScsi->SenseInfo),
                                timeout = winScsi->TimeOutValue * 1000
                            };

                            if (!SendScsiCommand(ref linScsi)) return false;

                            winScsi->ScsiStatus = linScsi.status;
                            return true;
                        }
                        else if (length == Device.m_scsi_request_size_64)
                        {
                            var winScsi = (Device.SCSI_PASS_THROUGH_DIRECT64*)outbuf;
                            var linScsi = new Linux.SG_IO_HDR
                            {
                                interface_id = 'S',
                                dxfer_direction = TransferDirection(winScsi->DataIn, winScsi->DataTransferLength),
                                cmdp = new IntPtr(winScsi->CdbData),
                                cmd_len = winScsi->CdbLength,
                                dxfer_len = winScsi->DataTransferLength,
                                dxferp = winScsi->DataBuffer,
                                mx_sb_len = winScsi->SenseInfoLength,
                                sbp = new IntPtr(winScsi->SenseInfo),
                                timeout = winScsi->TimeOutValue * 1000
                            };

                            if (!SendScsiCommand(ref linScsi)) return false;

                            winScsi->ScsiStatus = linScsi.status;
                            return true;
                        }

                        return false;
                    }
                case Device.IOCTL_STORAGE_MEDIA_REMOVAL:
                    {
                        var mediaRemoval = (Device.PREVENT_MEDIA_REMOVAL*)outbuf;

                        var lockDoor = new IntPtr(mediaRemoval->PreventMediaRemoval == 1 ? 1 : 0);

                        var result = Linux.ioctl(_fd, Linux.CDROM_LOCKDOOR, lockDoor);
                        if (result < 0)
                        {
                            LastError = Marshal.GetLastWin32Error();
                            return false;
                        }

                        return true;
                    }
                default:
                    throw new NotImplementedException($"Unknown SCSI instruction {code}");
            }
        }

        private static int TransferDirection(byte readDataFromDevice, uint transferLength)
        {
            if (transferLength == 0) return Linux.SG_DXFER_NONE;

            switch (readDataFromDevice)
            {
                case 0: return Linux.SG_DXFER_TO_DEV;
                case 1: return Linux.SG_DXFER_FROM_DEV;
                default: throw new NotImplementedException(
                    $"Unsupported TransferDirection value {readDataFromDevice}");
            }
        }

        private bool SendScsiCommand(ref Linux.SG_IO_HDR header)
        {
            int result;
            fixed (Linux.SG_IO_HDR* headerPtr = &header)
            {
                result = Linux.ioctl(_fd, Linux.SG_IO, new IntPtr(headerPtr));
            }

            if (result < 0)
            {
                LastError = Marshal.GetLastWin32Error();
                return false;
            }

            if (header.host_status != Linux.DID_OK)
            {
                LastError = Linux.EIO;
                return false;
            }

            return true;
        }

        public string ErrorCodeToString(int error)
            => Linux.GetErrorString(error);

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

        ~LinDev() => Dispose(disposing: false);

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

#endif // NETSTANDARD2_0