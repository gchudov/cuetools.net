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
using System.Threading.Tasks;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Services.Abstractions;
using System;
using Avalonia.Interactivity;
using CUERipper.Avalonia.Events;
using Avalonia.Threading;
using CUERipper.Avalonia.Views;
using Microsoft.Extensions.Localization;
using System.IO;
using System.Diagnostics;

namespace CUERipper.Avalonia;

public partial class UpdateDialog : Window
{
    public required IUpdateService UpdateService { get; init; }
    public required IStringLocalizer Localizer { get; init; }
    public UpdateDialog()
    {
        InitializeComponent();

        buttonInstall.Click += OnInstallClicked;
        buttonCancel.Click += OnCancelClicked;
    }

    public void Init()
    {
        var data = UpdateService.UpdateMetadata
            ?? throw new ArgumentNullException(nameof(UpdateService.UpdateMetadata));

        textVersion.Text = $"Version: {data.CurrentVersion} -> {data.Version}";
        textSize.Text = $"Size: {(double)data.Size / (1024 * 1024):F2} MiB";
        textAuthor.Text = $"Author: {data.Author}";
        textDate.Text = $"Date: {data.Date.ToString("yyyy-MM-dd HH:mm")}";

        textDescription.Text = data.Description;
    }

    private async void OnInstallClicked(object? sender, RoutedEventArgs e)
    {
        buttonInstall.IsEnabled = false;
        buttonCancel.IsEnabled = false;

        progressBarDownload.IsVisible = true;

        var success = await UpdateService.DownloadAsync((object? sender, GenericProgressEventArgs e) =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                progressBarDownload.Value = Math.Min(Math.Ceiling(e.Progress), 100);
            });
        });

        if (success)
        {
            var result = await MessageBox.CreateDialogAsync(title: "Update downloaded"
                , message: "CUERipper must close to apply the update."
                , Owner as Window ?? throw new InvalidCastException("Failed to cast property Owner to type Window")
                , Localizer
                , MessageBox.MessageBoxType.OkCancel
            );

            if (!result) return;

            if (File.Exists(Constants.UpdaterExecutable))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = Constants.UpdaterExecutable,
                    Arguments = $"Apply Update-{UpdateService.UpdateMetadata!.Version}",
                    UseShellExecute = true,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false,
                    CreateNoWindow = false
                });

                Environment.Exit(0);
            }
            else
            {
                await MessageBox.CreateDialogAsync(title: "Update failed"
                    , message: "Couldn't find the CUETools updater."
                    , Owner as Window ?? throw new InvalidCastException("Failed to cast property Owner to type Window")
                    , Localizer
                );
            }
        }
        else
        {
            await MessageBox.CreateDialogAsync(title: "Update failed"
                , message: "Failed to download update, check error log."
                , Owner as Window ?? throw new InvalidCastException("Failed to cast property Owner to type Window")
                , Localizer
            );
        }

        Close();
    }

    private void OnCancelClicked(object? sender, RoutedEventArgs e)
    {
        Close();
    }

    public static async Task CreateAsync(Window owner, IUpdateService updateService, IStringLocalizer localizer)
    {
        var updateWindow = new UpdateDialog()
        {
            Owner = owner
            , UpdateService = updateService
            , Localizer = localizer
        };

        updateWindow.Init();
        await updateWindow.ShowDialog(owner, lockParent: true);
    }
}