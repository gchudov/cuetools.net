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
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.ViewModels;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class PathFormatDialog : Window
{
    public PathFormatDialogViewModel ViewModel => DataContext as PathFormatDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(PathFormatDialogViewModel), DataContext?.GetType());

    public required ICUEConfigFacade Config;
    
    public PathFormatDialog()
    {
        InitializeComponent();

        buttonNew.Click += OnNewClicked;
        buttonCopy.Click += OnCopyClicked;
        buttonDelete.Click += OnDeleteClicked;

        buttonOk.Click += OnOkClicked;
        buttonCancel.Click += OnCancelClicked;
    }

    public void SetButtonIcons(IIconService iconService)
    {
        Image BindImage(AppIcon icon) => new() { Source = iconService.GetIcon(icon), Width = 18, Height = 18 };

        buttonNew.Content = BindImage(AppIcon.Add);
        buttonCopy.Content = BindImage(AppIcon.Multiply);
        buttonDelete.Content = BindImage(AppIcon.Subtract);
    }

    private void OnNewClicked(object? sender, EventArgs e)
    {
        ViewModel.Formats.Add(string.Empty);
        ViewModel.FormatIndex = ViewModel.Formats.Count - 1;
    }

    private void OnCopyClicked(object? sender, EventArgs e)
    {
        ViewModel.Formats.Add(ViewModel.Formats[ViewModel.FormatIndex]);
        ViewModel.FormatIndex = ViewModel.Formats.Count - 1;
    }

    private void OnDeleteClicked(object? sender, EventArgs e)
    {
        ViewModel.FormatIndex -= 1;
        ViewModel.Formats.RemoveAt(ViewModel.FormatIndex + 1);
    }

    private void OnOkClicked(object? sender, EventArgs e)
    {
        Config.PathFormat = ViewModel.Formats[ViewModel.FormatIndex];
        Config.PathFormatTemplates = ViewModel.Formats
            .Skip(Constants.DefaultPathFormats.Length)
            .Select(s => s)
            .ToList();

        Close();
    }

    private void OnCancelClicked(object? sender, EventArgs e)
    {
        Close();
    }

    public static async Task CreateAsync(Window owner, AlbumMetadata? meta, ICUEConfigFacade config, IIconService iconService)
    {
        var pathFormatWindow = new PathFormatDialog()
        {
            Owner = owner
            , DataContext = new PathFormatDialogViewModel(config, meta)
            , Config = config
        };

        pathFormatWindow.SetButtonIcons(iconService);

        if (pathFormatWindow.DataContext is PathFormatDialogViewModel viewModel)
        {
            new ObservableCollection<string>(
                Constants.DefaultPathFormats
                    .Concat(config.PathFormatTemplates)
            ).MoveAll(viewModel.Formats);

            var index = viewModel.Formats.IndexOf(config.PathFormat);
            if (index == -1)
            {
                var newIndex = Constants.DefaultPathFormats.Length;
                viewModel.Formats.Insert(newIndex, config.PathFormat);
                viewModel.FormatIndex = newIndex;
            }

            viewModel.FormatIndex = index;

            await pathFormatWindow.ShowDialog(owner, lockParent: true);
        }
    }
}