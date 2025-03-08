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
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using System.Collections.ObjectModel;

namespace CUERipper.Avalonia.ViewModels
{
    public partial class PathFormatDialogViewModel : ViewModelBase
    {
        [ObservableProperty]
        private bool readOnly;

        [ObservableProperty]
        private bool maximumReached;

        public ObservableCollection<string> Formats { get; } = [];

        [ObservableProperty]
        private int formatIndex = -1;
        partial void OnFormatIndexChanged(int oldValue, int newValue)
        {
            if (oldValue == newValue) return;
            if (newValue == -1)
            {
                // Workaround, as editing removes existing string in the observable collection.
                FormatIndex = oldValue < Formats.Count ? oldValue : 0;
                return;
            }

            ReadOnly = newValue < Constants.DefaultPathFormats.Length;
            FormatText = Formats[newValue];

            // TODO find a better place, maybe... :)
            MaximumReached = Formats.Count - Constants.DefaultPathFormats.Length >= Constants.MaxPathFormats;
        }

        [ObservableProperty]
        private string outputPreview = string.Empty;

        [ObservableProperty]
        private string formatText = string.Empty;
        partial void OnFormatTextChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;
            if (string.Compare(oldValue, newValue) == 0) return;

            Formats[FormatIndex] = newValue;

            try
            {
                OutputPreview = _meta.PathStringFromFormat(newValue, _config);
            }
            catch
            {
                OutputPreview = "Couldn't parse the current format.";
            }
        }

        private readonly ICUEConfigFacade _config;
        private readonly AlbumMetadata? _meta;
        public PathFormatDialogViewModel(ICUEConfigFacade config, AlbumMetadata? meta)
        {
            _config = config;
            _meta = meta;
        }
    }
}
