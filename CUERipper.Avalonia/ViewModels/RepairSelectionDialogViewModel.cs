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
using CommunityToolkit.Mvvm.Input;
using CUERipper.Avalonia.Extensions;
using CUETools.Processor;
using System.Collections.ObjectModel;
using System.Linq;

namespace CUERipper.Avalonia.ViewModels
{
    public sealed partial class RepairSelectionDialogViewModel : ViewModelBase
    {
        public ObservableCollection<string> Paths { get; } = [];

        [ObservableProperty]
        private int index = 0;
        partial void OnIndexChanged(int value)
        {
            Description = value >= 0 && value < _sourceFiles.Length
                ? _sourceFiles[value].contents
                : string.Empty;
        }

        [ObservableProperty]
        private string description = string.Empty;

        [ObservableProperty]
        // Default is null, triggers on true or false
        private bool? affirmative;

        public RepairSelectionDialogViewModel()
        {

        }

        private CUEToolsSourceFile[] _sourceFiles = [];
        public void SetSourceFiles(CUEToolsSourceFile[] sourceFiles)
        {
            _sourceFiles = sourceFiles;

            new ObservableCollection<string>(
                sourceFiles.Select(s => s.path)
            ).MoveAll(Paths);

            Index = sourceFiles.Length > 0 ? 0 : -1;
            Description = Index >= 0 ? _sourceFiles[Index].contents : string.Empty;
        }

        [RelayCommand]
        private void Confirm() => Affirmative = true;

        [RelayCommand]
        private void Cancel() => Affirmative = false;
    }
}
