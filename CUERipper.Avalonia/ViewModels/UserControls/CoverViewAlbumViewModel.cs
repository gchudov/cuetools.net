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
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using System;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class CoverViewAlbumViewModel : ObservableObject, IEquatable<CoverViewAlbumViewModel>
    {
        public string Uri { get; set; }
        public string Uri150 { get; set; }
        public Bitmap? Bitmap150 { get; set; }

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

        public CoverViewAlbumViewModel(string uri, string uri150)
        {
            Uri = uri;
            Uri150 = uri150;
        }

        public bool Equals(CoverViewAlbumViewModel? other)
        {
            if (other == null) return false;
            return Uri == other.Uri && Uri150 == other.Uri150;
        }
    }
}
