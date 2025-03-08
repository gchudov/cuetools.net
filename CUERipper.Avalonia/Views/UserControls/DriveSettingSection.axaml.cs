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
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.ViewModels.UserControls;
using Microsoft.Extensions.Localization;
using System;

namespace CUERipper.Avalonia.Views.UserControls;

public partial class DriveSettingSection : UserControl
{
    public DriveSettingSectionViewModel ViewModel => DataContext as DriveSettingSectionViewModel
        ?? throw new ViewModelMismatchException(typeof(DriveSettingSectionViewModel), DataContext?.GetType());

    private ICUERipperService? _ripperService;
    private IIconService? _iconService;

    public DriveSettingSection()
    {
        InitializeComponent();
    }

    public void Init(ICUEConfigFacade config
        , ICUERipperService ripperService
        , IStringLocalizer<Language> localizer
        , IIconService iconService)
    {
        _ripperService = ripperService;
        _iconService = iconService;

        var viewModel = new DriveSettingSectionViewModel(config, ripperService, localizer);
        viewModel.SetInitState();
        DataContext = viewModel;

        Image BindImage(AppIcon icon) => new() { Source = _iconService.GetIcon(icon), Width = 18, Height = 18 };

        buttonResetDriveSettings.Click += OnResetDriveSettingsClicked;
        buttonResetDriveSettings.Content = BindImage(AppIcon.Cross);
    }

    private void OnResetDriveSettingsClicked(object? sender, EventArgs e)
    {
        ViewModel.DriveOffset = _ripperService?.GetDriveOffset()
            ?? throw new NotInitializedException(nameof(_ripperService));

        ViewModel.SelectedSecureMode = Constants.SecureModeDefault;
        ViewModel.SelectedC2ErrorMode = ViewModel.C2ErrorMode[ViewModel.C2ErrorMode.Count - 1];
        ViewModel.TestAndCopyEnabled = false;
    }
}