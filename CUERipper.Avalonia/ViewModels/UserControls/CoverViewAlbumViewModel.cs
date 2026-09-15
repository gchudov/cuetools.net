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
using CommunityToolkit.Mvvm.ComponentModel;
using CUERipper.Avalonia.Models.Abstractions;
using System;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class CoverViewAlbumViewModel : ObservableObject, IEquatable<CoverViewAlbumViewModel>
    {
        public string Uri { get; init; }
        public string Uri150 { get; init; }
        public bool IsPrimary { get; init; }

        public IBitmap? Bitmap150 { get; set; }

        private bool _isSelected;
        public bool IsSelected
        {
            get => _isSelected;
            set
            {
                _isSelected = value;
                BorderColor = value ? "#0078D4" : "Transparent";
            }
        }

        [ObservableProperty]
        private string borderColor = "Transparent";

        public CoverViewAlbumViewModel(string uri, string uri150, bool isPrimary)
        {
            Uri = uri;
            Uri150 = uri150;
            IsPrimary = isPrimary;
        }

        public bool Equals(CoverViewAlbumViewModel? other)
        {
            if (other == null) return false;
            return Uri == other.Uri && Uri150 == other.Uri150;
        }

        public override bool Equals(object? obj)
            => Equals(obj as CoverViewAlbumViewModel);

        public override int GetHashCode()
            => (Uri?.GetHashCode() ?? 0) * 31 ^ (Uri150?.GetHashCode() ?? 0);
    }
}
