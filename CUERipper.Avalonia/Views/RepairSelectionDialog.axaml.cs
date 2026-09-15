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
using Avalonia.Controls;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.ViewModels;
using CUETools.Processor;
using System.ComponentModel;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class RepairSelectionDialog : Window
{
    public RepairSelectionDialogViewModel ViewModel => DataContext as RepairSelectionDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(RepairSelectionDialogViewModel), DataContext?.GetType());

    public RepairSelectionDialog()
    {
        InitializeComponent();
    }

    public RepairSelectionDialog(RepairSelectionDialogViewModel viewModel)
    {
        InitializeComponent();

        viewModel.PropertyChanged += OnViewModelPropertyChanged;

        DataContext = viewModel;
    }

    public async Task<int> CreateDialogAsync(Window owner,
        CUEToolsSourceFile[] sourceFiles)
    {
        Owner = owner;

        ViewModel.SetSourceFiles(sourceFiles);

        await this.ShowDialog(owner, lockParent: true);

        return ViewModel.Affirmative == true
            ? ViewModel.Index
            : -1;
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(RepairSelectionDialogViewModel.Affirmative)) Close();
    }
}
