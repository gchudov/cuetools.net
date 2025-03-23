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
using System.Linq;

namespace CUERipper.Avalonia.Views.UserControls;

public partial class EncodingSection : UserControl
{
    public EncodingSectionViewModel ViewModel => DataContext as EncodingSectionViewModel
        ?? throw new ViewModelMismatchException(typeof(EncodingSectionViewModel), DataContext?.GetType());

    private ICUEConfigFacade? _config;
    private IIconService? _iconService;
    public EncodingSection()
    {
        InitializeComponent();
    }

    public void Init(ICUEConfigFacade config
        , IStringLocalizer<Language> localizer
        , IIconService iconService)
    {
        _config = config;
        _iconService = iconService;

        var viewModel = new EncodingSectionViewModel(config, localizer);
        viewModel.SetInitState();
        DataContext = viewModel;
        
        Image BindImage(AppIcon icon) => new() { Source = _iconService.GetIcon(icon), Width = 18, Height = 18 };

        buttonEncoderSettings.Click += OnEncoderSettingsClicked;
        buttonEncoderSettings.Content = BindImage(AppIcon.Bolt);
        buttonSettings.Click += OnSettingsClicked;
        buttonSettings.Content = BindImage(AppIcon.Cog);
    }

    private async void OnEncoderSettingsClicked(object? sender, EventArgs e)
    {
        if (_config == null) return;

        var encoderSettings = _config.Encoders
            .Where(e => string.Compare(e.Name, ViewModel.SelectedEncoder, true) == 0
                && string.Compare(e.Extension, ViewModel.SelectedEncoding, true) == 0)
            .Select(e => e.Settings)
            .FirstOrDefault();

        if (encoderSettings != null)
        {
            var parent = TopLevel.GetTopLevel(this);

            await EncoderOptionsDialog.CreateAsync(
                parent as Window ?? throw new UnexpectedParentException(typeof(Window), parent?.GetType())
                , encoderSettings
            );
        }
    }

    private async void OnSettingsClicked(object? sender, EventArgs e)
    {
        var parent = TopLevel.GetTopLevel(this);

        await OptionsDialog.CreateAsync(
            parent as Window ?? throw new UnexpectedParentException(typeof(Window), parent?.GetType())
            , _config ?? throw new NotInitializedException(nameof(_config))
        );
    }
}