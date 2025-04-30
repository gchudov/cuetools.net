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
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

namespace CUERipper.Avalonia.Services
{
    public class NullDriveNotificationService : IDriveNotificationService, IDisposable
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

        public NullDriveNotificationService(ILogger<NullDriveNotificationService> logger)
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

            Dictionary<string, (bool IsReady, DriveInfo DriveInfo)> knownDrives = [];

            while (!_requestExit)
            {
                var currentDrives = DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.CDRom)
                                        .Select(drive => (drive.IsReady, drive))
                                        .ToDictionary(d => d.drive.Name);

                var mountedDrives = currentDrives.Where(c => !knownDrives
                    .Any(k => c.Key == k.Key));

                var unmountedDrives = knownDrives.Where(k => !currentDrives
                    .Any(c => c.Key == k.Key));

                if (mountedDrives.Any() || unmountedDrives.Any())
                {
                    _onDriveRefresh?.Invoke();
                }

                var driveStateChange = currentDrives.Where(
                    c => knownDrives.TryGetValue(c.Key, out var knownDrive) && knownDrive.IsReady != c.Value.IsReady);

                foreach (var drive in driveStateChange)
                {
                    if(drive.Value.IsReady) _onDriveMounted?.Invoke(drive.Key[0]);
                    else _onDriveUnmounted?.Invoke(drive.Key[0]);
                }

                knownDrives = currentDrives;

                Thread.Sleep(500);
            }

            _logger.LogInformation("Drive scanning has been stopped.");
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
