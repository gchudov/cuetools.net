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
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using CUETools.CDImage;
using CUETools.Ripper;
using System;
using System.Collections.Immutable;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services.Abstractions
{
    public interface IDiscRippingService
    {
        public char SelectedDrive { get; set; }
        
        /// <summary>
        /// Fired from UI thread
        /// </summary>
        public event EventHandler<DriveChangedEventArgs>? OnSelectedDriveChanged;
        /// <summary>
        /// Fired from non UI thread
        /// </summary>
        public event EventHandler<ReadProgressArgs>? OnRippingProgress;
        /// <summary>
        /// Fired from non UI thread
        /// </summary>
        public event EventHandler<DirectoryConflictEventArgs>? OnDirectoryConflict;

        IImmutableDictionary<char, DriveInformation> QueryAvailableDriveInformation();
        bool IsDriveAccessible();
        string GetDriveName();
        string GetDriveARName();

        CDImageLayout? GetDiscTOC(); 

        void EjectTray();
        int GetDriveOffset();

        Task<CUEResult> RipAsync(RipSettings settings, CancellationToken token);
    }
}
