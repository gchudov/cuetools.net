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
using Avalonia.Threading;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.ViewModels;
using System.ComponentModel;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class UpdateDialog : Window
{
    public UpdateDialogViewModel ViewModel => DataContext as UpdateDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(UpdateDialogViewModel), DataContext?.GetType());

    public UpdateDialog()
    {
        InitializeComponent();
    }

    public UpdateDialog(UpdateDialogViewModel viewModel)
    {
        InitializeComponent();

        viewModel.PropertyChanged += OnViewModelPropertyChanged;
        Closing += OnWindowClosing;

        DataContext = viewModel;
    }

    public async Task<bool> CreateDialogAsync(Window owner)
    {
        Owner = owner;

        await this.ShowDialog(owner, lockParent: true);

        return ViewModel.Affirmative ?? false;
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(UpdateDialogViewModel.Affirmative))
        {
            Dispatcher.UIThread.Post(Close);
        }
    }

    private void OnWindowClosing(object? sender, WindowClosingEventArgs e)
    {
        ViewModel.CancelDownload();
    }
}
