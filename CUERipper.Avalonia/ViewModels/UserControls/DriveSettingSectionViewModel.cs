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
using CommunityToolkit.Mvvm.ComponentModel;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Ripper;
using Microsoft.Extensions.Localization;
using System;
using System.Collections.ObjectModel;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class DriveSettingSectionViewModel : ViewModelBase
    {
        [ObservableProperty]
        private int selectedSecureMode = Constants.SecureModeDefault;
        partial void OnSelectedSecureModeChanged(int oldValue, int newValue)
        {
            if (oldValue == newValue) return;

            SelectedSecureModeText = Constants.SecureModeValues[Math.Max(0, Math.Min(Constants.SecureModeValues.Length - 1, newValue))];
            _config.SecureModeIndex = newValue;
        }

        [ObservableProperty]
        private string selectedSecureModeText = Constants.SecureModeValues[Constants.SecureModeDefault];

        public ObservableCollection<string> C2ErrorMode { get; set; }
            = [.. Enum.GetNames(typeof(DriveC2ErrorModeSetting))];

        [ObservableProperty]
        private string selectedC2ErrorMode = string.Empty;

        partial void OnSelectedC2ErrorModeChanged(string? oldValue, string newValue)
        {
            if (string.Compare(oldValue, newValue) == 0) return;
            if (!_ripperService.IsDriveAccessible()) return; 

            int index = C2ErrorMode.IndexOf(newValue);

            if (_config.DriveC2ErrorModes.ContainsKey(_ripperService.GetDriveARName()))
            {
                _config.DriveC2ErrorModes[_ripperService.GetDriveARName()] = index;
            }
            else
            {
                _config.DriveC2ErrorModes.Add(_ripperService.GetDriveARName(), index);
            }
        }

        [ObservableProperty]
        private bool testAndCopyEnabled = false;

        partial void OnTestAndCopyEnabledChanged(bool oldValue, bool newValue)
        {
            if (oldValue == newValue) return;

            _config.TestAndCopyEnabled = newValue;
        }

        [ObservableProperty]
        private int driveOffset = 0;

        partial void OnDriveOffsetChanged(int oldValue, int newValue)
        {
            if (oldValue == newValue) return;
            if (!_ripperService.IsDriveAccessible()) return;

            if (_config.DriveOffsets.ContainsKey(_ripperService.GetDriveARName()))
            {
                _config.DriveOffsets[_ripperService.GetDriveARName()] = newValue;
            }
            else
            {
                _config.DriveOffsets.Add(_ripperService.GetDriveARName(), newValue);
            }
        }

        private readonly ICUEConfigFacade _config;
        private readonly ICUERipperService _ripperService;
        private readonly IStringLocalizer _localizer;

        public DriveSettingSectionViewModel(ICUEConfigFacade config
            , ICUERipperService ripperService
            , IStringLocalizer<Language> localizer)
        {
            _config = config;
            _ripperService = ripperService;
            _localizer = localizer;

            _ripperService.OnSelectedDriveChanged += (object? sender, DriveChangedEventArgs e) => {
                if (_ripperService.IsDriveAccessible())
                {
                    SelectedC2ErrorMode = _config.DriveC2ErrorModes.TryGetValue(_ripperService.GetDriveARName(), out int c2Value)
                        ? C2ErrorMode[(c2Value >= 0 && c2Value <= 3 ? c2Value : C2ErrorMode.Count - 1)]
                        : C2ErrorMode[C2ErrorMode.Count - 1];

                    DriveOffset = _config.DriveOffsets.TryGetValue(_ripperService.GetDriveARName(), out int offsetValue)
                        ? offsetValue
                        : _ripperService.GetDriveOffset();
                }
                else
                {
                    SelectedC2ErrorMode = C2ErrorMode[C2ErrorMode.Count - 1];
                    DriveOffset = 0;
                }
            };
        }

        internal void SetInitState()
        {
            SelectedSecureMode = _config.SecureModeIndex >= 0 && _config.SecureModeIndex < Constants.SecureModeValues.Length
                ? _config.SecureModeIndex
                : Constants.SecureModeDefault;

            TestAndCopyEnabled = _config.TestAndCopyEnabled;
        }
    }
}
