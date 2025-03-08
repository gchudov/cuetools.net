#region Copyright (C) 2025 Gregory S. Chudov, Max Visser
/*
    Copyright (C) 2025 Gregory S. Chudov, Max Visser

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
// This file contains modified code from frmCUERipper.cs.
#endregion
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace CUERipper.Avalonia.Services
{
    public class WindowsDriveNotificationService : IDriveNotificationService, IDisposable
    {
        private delegate IntPtr WndProcDelegate(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
        private readonly WndProcDelegate _wndProcDelegate;

        private IntPtr? _hwnd;

        private Action? _onDriveRefresh;
        private Action<char>? _onDriveUnmounted;
        private Action<char>? _onDriveMounted;

        private List<char> _driveList = [];

        public void SetCallbacks(Action onDriveRefresh
            , Action<char> onDriveUnmounted
            , Action<char> onDriveMounted)
        {
            _onDriveRefresh = onDriveRefresh;
            _onDriveUnmounted = onDriveUnmounted;
            _onDriveMounted = onDriveMounted;
        }

        private readonly ILogger _logger;
        public WindowsDriveNotificationService(ILogger<WindowsDriveNotificationService> logger)
        {
            _logger = logger;
            _wndProcDelegate = CustomWndProc;

            Init();
        }

        private static List<char> GetCDDrives()
            => DriveInfo.GetDrives()
                .Where(d => d.DriveType == DriveType.CDRom)
                .Select(d => d.Name[0])
                .ToList();

        const string SCANNING_WINDOW = "CUERipperDriveScanningWindow";
        const string CLASS_NAME = "CUERipperDriveScanningClass";
        private void Init()
        {
#if !NET47
            if (!OperatingSystem.IsWindows())
            {
                throw new InvalidOperationException("Windows-specific code was executed on a non-Windows platform.");
            }
#endif

            _driveList = GetCDDrives();

            var wndClass = new WNDCLASS
            {
                lpszClassName = CLASS_NAME,
                lpfnWndProc = Marshal.GetFunctionPointerForDelegate(_wndProcDelegate),
                hInstance = GetModuleHandle(null)
            };
            RegisterClass(ref wndClass);

            _hwnd = CreateWindowEx(0, CLASS_NAME, SCANNING_WINDOW, 0, 0, 0, 0, 0, IntPtr.Zero, IntPtr.Zero, wndClass.hInstance, IntPtr.Zero);
            if (_hwnd == IntPtr.Zero) _logger.LogError("Failed to create drive scanning window.");
        }

        #region private constants
        /// <summary>
        /// The window message of interest, device change
        /// </summary>
        const int WM_DEVICECHANGE = 0x0219;
        const ushort DBT_DEVICEARRIVAL = 0x8000; // Called when a disc is inserted
        const ushort DBT_DEVICEREMOVECOMPLETE = 0x8004; // Called when a disc is removed
        const ushort DBT_DEVNODES_CHANGED = 0x0007;
        #endregion

        [StructLayout(LayoutKind.Sequential)]
        internal class DEV_BROADCAST_HDR
        {
            internal Int32 dbch_size;
            internal Int32 dbch_devicetype;
            internal Int32 dbch_reserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct DEV_BROADCAST_VOLUME
        {
            public DEV_BROADCAST_HDR Header;
            public int UnitMask;
            public short Flags;
        }

        private static int FirstBitSet(int iIn)
        {
            for (int i = 0; i < 32; i++)
            {
                if ((iIn & 1) != 0)
                    return i;

                iIn >>= 1;
            }

            return -1;
        }

        private const int DBT_DEVTYPE_VOLUME = 2;
        private static char ConvertToDriveLetter(IntPtr lParam)
        {
            if (Marshal.PtrToStructure(lParam, typeof(DEV_BROADCAST_HDR)) is DEV_BROADCAST_HDR hdr 
                    && hdr.dbch_devicetype == DBT_DEVTYPE_VOLUME)
            {
                var obj = Marshal.PtrToStructure(lParam, typeof(DEV_BROADCAST_VOLUME));
                if (obj == null) return (char)0;

                var vol = (DEV_BROADCAST_VOLUME)obj;
                return (char)(FirstBitSet(vol.UnitMask) + ('A'));
            }

            return (char)0;
        }
        /// <summary>
        /// This method is called when a window message is processed by the dotnet application
        /// framework.  We override this method and look for the WM_DEVICECHANGE message. All
        /// messages are delivered to the base class for processing, but if the WM_DEVICECHANGE
        /// method is seen, we also alert any BWGBURN programs that the media in the drive may
        /// have changed.
        /// </summary>
        /// <param name="m">the windows message being processed</param>
        private IntPtr CustomWndProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam)
        {
            switch (msg)
            {
                case WM_DEVICECHANGE:
                    int val = wParam.ToInt32();
                    switch (val)
                    {
                        case DBT_DEVICEREMOVECOMPLETE:
                            {
                                var driveLetter = ConvertToDriveLetter(lParam);
                                var driveInfo = DriveInfo.GetDrives();
                                var filteredDevice = driveInfo
                                    .Where(d => d.Name[0] == driveLetter && d.DriveType == DriveType.CDRom)
                                    .FirstOrDefault();

                                if (filteredDevice != null) _onDriveUnmounted?.Invoke(driveLetter);
                            }
                            break;
                        case DBT_DEVICEARRIVAL:
                            {
                                var driveLetter = ConvertToDriveLetter(lParam);
                                var driveInfo = DriveInfo.GetDrives();
                                var filteredDevice = driveInfo
                                    .Where(d => d.Name[0] == driveLetter && d.DriveType == DriveType.CDRom)
                                    .FirstOrDefault();

                                if (filteredDevice != null) _onDriveMounted?.Invoke(driveLetter);
                            }
                            break;
                        case DBT_DEVNODES_CHANGED:
                            {
                                var currentDrives = GetCDDrives();
                                var difference = currentDrives.Except(_driveList)
                                    .Concat(_driveList.Except(currentDrives));

                                if (difference.Any())
                                {
                                    _onDriveRefresh?.Invoke();
                                    _driveList = currentDrives;
                                }
                            }
                            break;
                    }
                    break;
            }

            return DefWindowProc(hWnd, msg, wParam, lParam);
        }

        const string USER32_DLL = "user32.dll";
        const string KERNEL32_DLL = "kernel32.dll";

        [DllImport(USER32_DLL, SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateWindowEx(
            int dwExStyle, string lpClassName, string lpWindowName,
            int dwStyle, int x, int y, int nWidth, int nHeight,
            IntPtr hWndParent, IntPtr hMenu, IntPtr hInstance, IntPtr lpParam);

        [DllImport(USER32_DLL, SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern bool DestroyWindow(IntPtr hWnd);

        [DllImport(USER32_DLL, SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr DefWindowProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);

        [DllImport(USER32_DLL, SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern ushort RegisterClass([In] ref WNDCLASS lpWndClass);

        [DllImport(KERNEL32_DLL, CharSet = CharSet.Unicode)]
        private static extern IntPtr GetModuleHandle(string? lpModuleName);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct WNDCLASS
        {
            public uint style;
            public IntPtr lpfnWndProc;
            public int cbClsExtra;
            public int cbWndExtra;
            public IntPtr hInstance;
            public IntPtr hIcon;
            public IntPtr hCursor;
            public IntPtr hbrBackground;
            [MarshalAs(UnmanagedType.LPWStr)] public string lpszMenuName;
            [MarshalAs(UnmanagedType.LPWStr)] public string lpszClassName;
        }

        private bool disposedValue;
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                /*
                if (disposing)
                {
                }
                */

                if (_hwnd != null)
                {
                    DestroyWindow(_hwnd.Value);
                    _hwnd = null;
                }

                disposedValue = true;
            }
        }
        
        ~WindowsDriveNotificationService()
            => Dispose(disposing: false);

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
