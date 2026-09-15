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
using Avalonia.Controls;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.ViewModels;
using System.ComponentModel;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class PathFormatDialog : Window
{
    public PathFormatDialogViewModel ViewModel => DataContext as PathFormatDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(PathFormatDialogViewModel), DataContext?.GetType());
    
    public PathFormatDialog()
    {
        InitializeComponent();
    }

    public PathFormatDialog(PathFormatDialogViewModel viewModel)
    {
        InitializeComponent();

        viewModel.PropertyChanged += OnViewModelPropertyChanged;

        DataContext = viewModel;
    }

    public async Task<bool> CreateDialogAsync(Window owner,
        AlbumMetadata? meta)
    {
        Owner = owner;
        ViewModel.SetMetadata(meta);

        await this.ShowDialog(owner, lockParent: true);

        return ViewModel.Affirmative ?? false;
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(PathFormatDialogViewModel.Affirmative)) Close();
    }
}