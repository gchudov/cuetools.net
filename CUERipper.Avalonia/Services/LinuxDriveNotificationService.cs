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

using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Interop;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;

namespace CUERipper.Avalonia.Services
{
    public sealed class LinuxDriveNotificationService : IDriveNotificationService, IDisposable
    {
        private const int PollInterval = 500;

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

        private readonly ICDDriveEnumerator _driveEnumerator;
        private readonly ILogger _logger;

        public LinuxDriveNotificationService(ICDDriveEnumerator driveEnumerator
            , ILogger<LinuxDriveNotificationService> logger)
        {
            _driveEnumerator = driveEnumerator;
            _logger = logger;
            _thread = new Thread(ScanDrives)
            {
                IsBackground = true
            };
            _thread.Start();
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
                    currentDrives = _driveEnumerator.DrivesAvailable()
                        .Select(d => new { Drive = d, IsReady = IsDriveReady(d) })
                        .ToDictionary(item => item.Drive, item => item.IsReady);
                }
                catch(Exception ex)
                {
                    _logger.LogError(ex, "Failed to retrieve the available drives.");
                    Thread.Sleep(PollInterval);
                    continue;
                }

                var mountedDrives = currentDrives.Where(c => !knownDrives
                    .Any(k => c.Key == k.Key))
                    .ToImmutableList();

                var unmountedDrives = knownDrives.Where(k => !currentDrives
                    .Any(c => c.Key == k.Key))
                    .ToImmutableList();

                if (!mountedDrives.IsEmpty || !unmountedDrives.IsEmpty)
                {
                    mountedDrives.ForEach(d => _logger.LogInformation("Drive {DriveKey} mounted.", d.Key));
                    unmountedDrives.ForEach(d => _logger.LogInformation("Drive {DriveKey} unmounted.", d.Key));

                    _onDriveRefresh?.Invoke();
                }

                var driveStateChange = currentDrives.Where(
                    c => knownDrives.TryGetValue(c.Key, out var knownDrive) && knownDrive != c.Value
                );

                foreach (var drive in driveStateChange)
                {
                    _logger.LogInformation("Drive state has changed for drive {DriveKey}.", drive.Key);

                    if (drive.Value) _onDriveMounted?.Invoke(drive.Key);
                    else _onDriveUnmounted?.Invoke(drive.Key);
                }

                knownDrives = currentDrives;

                Thread.Sleep(PollInterval);
            }

            _logger.LogInformation("Drive scanning has been stopped.");
        }

        private bool IsDriveReady(char drive)
        {
            var fullPath = $"{Linux.CDROM_DEVICE_PATH}{drive}";

            var fd = Linux.open(fullPath, Linux.O_RDONLY | Linux.O_NONBLOCK);
            if (fd == -1)
            {
                _logger.LogWarning("Drive scanning failed for '{fullPath}' with {errorCode} - {errorMessage}"
                    , fullPath
                    , Linux.GetErrorCode()
                    , Linux.GetErrorString());

                return false;
            }

            try
            {
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
            finally
            {
                Linux.close(fd);
            }
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