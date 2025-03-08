﻿#region Copyright (C) 2025 Max Visser
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

namespace CUERipper.Avalonia.ViewModels
{
    public partial class MessageBoxViewModel : ViewModelBase
    {
        [ObservableProperty]
        private string message = string.Empty;

        [ObservableProperty]
        private string affirm = string.Empty;

        [ObservableProperty]
        private string negate = string.Empty;

        [ObservableProperty]
        private bool showNegate;
    }
}
