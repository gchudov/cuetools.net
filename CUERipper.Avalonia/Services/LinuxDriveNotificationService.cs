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

// The implementation in NullDriveNotificationService doesn't function correctly on .NET 8 under Linux.
// As a temporary solution, I've implemented a quick workaround to ensure the drive notification works.
// Further investigation needed...

using CUETools.Interop;
using CUETools.Ripper;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace CUERipper.Avalonia.Services
{
    public class LinuxDriveNotificationService : IDriveNotificationService, IDisposable
    {
        private readonly Thread _thread;
        private volatile bool _requestExit;

        private Action? _onDriveRefresh;
        private Action<char>? _onDriveUnmounted;
        private Action<char>? _onDriveMounted;

        public void SetCallbacks(Action onDriveRefresh
            , Action<char> onDriveUnmounted
            , Action<char> onDriveMounted)
        {
            _onDriveRefresh = onDriveRefresh;
            _onDriveUnmounted = onDriveUnmounted;
            _onDriveMounted = onDriveMounted;
        }

        private readonly ILogger _logger;

        public LinuxDriveNotificationService(ILogger<NullDriveNotificationService> logger)
        {
            _logger = logger;
            _thread = new Thread(ScanDrives)
            {
                IsBackground = true
            };
        }

        private void ScanDrives()
        {
            _logger.LogInformation("Started scanning for drives.");

            Dictionary<char, bool> knownDrives = [];

            while (!_requestExit)
            {
                Dictionary<char, bool> currentDrives = [];
                try
                {
                    currentDrives = CDDrivesList.DrivesAvailable()
                        .Select(d => new { Drive = d, IsReady = IsDriveReady(d) })
                        .ToDictionary(item => item.Drive, item => item.IsReady);
                }
                catch(Exception ex)
                {
                    _logger.LogError(ex, "Failed to retrieve the available drives.");
                }

                var mountedDrives = currentDrives.Where(c => !knownDrives
                    .Any(k => c.Key == k.Key));

                var unmountedDrives = knownDrives.Where(k => !currentDrives
                    .Any(c => c.Key == k.Key));

                if (mountedDrives.Any() || unmountedDrives.Any())
                {
                    _onDriveRefresh?.Invoke();
                }

                var driveStateChange = currentDrives.Where(
                    c => knownDrives.TryGetValue(c.Key, out var knownDrive) && knownDrive != c.Value
                );

                foreach (var drive in driveStateChange)
                {
                    if (drive.Value) _onDriveMounted?.Invoke(drive.Key);
                    else _onDriveUnmounted?.Invoke(drive.Key);
                }

                knownDrives = currentDrives;

                Thread.Sleep(500);
            }

            _logger.LogInformation("Drive scanning has been stopped.");
        }

        private bool IsDriveReady(char drive)
        {
            var fullPath = $"{Linux.CDROM_DEVICE_PATH}{drive}";
            var fd = Linux.open(fullPath, Linux.O_RDONLY);

            if (fd == -1)
            {
                _logger.LogWarning("Drive scanning failed for '{fullPath}' with {errorCode} - {errorMessage}"
                    , fullPath
                    , Linux.GetErrorCode()
                    , Linux.GetErrorString());

                return false;
            }

            var result = Linux.ioctl(fd, Linux.CDROM_DRIVE_STATUS);
            if (result < 0)
            {
                _logger.LogWarning("Drive scanning failed for '{fullPath}' with {errorCode} - {errorMessage}"
                    , fullPath
                    , Linux.GetErrorCode()
                    , Linux.GetErrorString());
            }

            return result == Linux.CDS_DISC_OK;
        }

        private bool _disposed;

        /// <summary>
        /// Class is sealed, so no need for inheritance concerns including a complex dispose pattern.
        /// </summary>
        public void Dispose()
        {
            if (_disposed == true) return;
            _disposed = true;

            _requestExit = true;
            _thread.Join(1000);

            GC.SuppressFinalize(this);
        }
    }
}