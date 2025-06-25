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
using System;

namespace CUERipper.Avalonia.Events
{
    public class DirectoryConflictEventArgs : EventArgs
    {
        public string Directory { get; set; }
        public bool CanModifyContent { get; set; }

        public DirectoryConflictEventArgs(string directory, bool canModifyContent)
        {
            Directory = directory;
            CanModifyContent = canModifyContent;
        }
    }
}