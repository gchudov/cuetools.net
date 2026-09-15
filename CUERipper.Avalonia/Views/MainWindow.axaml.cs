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
using Avalonia.Interactivity;
using Avalonia.Threading;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.ViewModels;
using System;
using System.ComponentModel;

namespace CUERipper.Avalonia.Views
{
    public sealed partial class MainWindow : Window
    {
        public MainWindowViewModel ViewModel => DataContext as MainWindowViewModel
            ?? throw new ViewModelMismatchException(typeof(MainWindowViewModel), DataContext?.GetType());

        public MainWindow()
        {
            InitializeComponent();
        }

        public MainWindow(MainWindowViewModel viewModel)
        {
            InitializeComponent();
            DataContextChanged += OnDataContextChanged;
            Closing += OnWindowClosing;
            viewModel.PropertyChanged += OnViewModelPropertyChanged;

            DataContext = viewModel;
        }

        private async void OnDataContextChanged(object? sender, EventArgs e)
        {
            ViewModel.Initialize();
            await ViewModel.RefreshSessionAsync();
            await ViewModel.CheckForUpdateAsync();
        }

        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainWindowViewModel.InstallPending)
                && ViewModel.InstallPending)
            {
                Dispatcher.UIThread.Post(Close);
            }
        }

        private void OnSplitViewPaneClosing(object? sender, CancelRoutedEventArgs args)
        {
            if (ViewModel.SplitPaneOpen) args.Cancel = true;
        }

        private bool _closeApproved = false;
        private async void OnWindowClosing(object? sender, WindowClosingEventArgs e)
        {
            if (_closeApproved) return;

            e.Cancel = true;

            if (await ViewModel.TryShutdownAsync())
            {
                _closeApproved = true;
                Close();
            }
        }
    }
}