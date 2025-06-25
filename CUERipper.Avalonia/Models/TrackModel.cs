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
using System;
namespace CUERipper.Avalonia.Models
{
    public partial class TrackModel : ObservableObject
    {
        public int TrackNo { get; init; }

        private string title = string.Empty;
        public string Title {
            get => title;
            set { title = value; OnUpdate?.Invoke(this); }
        }

        public string Length { get; init; } = string.Empty;

        private string artist = string.Empty;
        public string Artist {
            get => artist;
            set { artist = value; OnUpdate?.Invoke(this); }
        }

        public Action<TrackModel>? OnUpdate { private get; init; }

        [ObservableProperty]
        private int progress;
    }
}
