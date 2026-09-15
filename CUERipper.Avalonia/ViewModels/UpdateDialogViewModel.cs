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
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using Microsoft.Extensions.Localization;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.ViewModels
{
    public partial class UpdateDialogViewModel : ViewModelBase, IDisposable
    {
        [ObservableProperty]
        private string version = string.Empty;

        [ObservableProperty]
        private string filesize = string.Empty;

        [ObservableProperty]
        private string author = string.Empty;

        [ObservableProperty]
        private string date = string.Empty;

        [ObservableProperty]
        private string description = string.Empty;

        [ObservableProperty]
        [NotifyCanExecuteChangedFor(nameof(InstallCommand))]
        private bool canInstall;

        [ObservableProperty]
        [NotifyCanExecuteChangedFor(nameof(InstallCommand))]
        private bool isDownloading;

        [ObservableProperty]
        private double downloadProgress;

        [ObservableProperty]
        // Default is null, triggers on true or false
        private bool? affirmative;

        public string TextInstall { get => _localizer["Update:Install"]; }
        public string TextCancel { get => _localizer["Generic:Cancel"]; }
        public string ToolTipDownloadProgress { get => _localizer["Update:ToolTipProgress"]; }

        private readonly CancellationTokenSource _downloadCts = new();

        private readonly IDialogService _dialogService;
        private readonly IUpdateService _updateService;
        private readonly IStringLocalizer _localizer;

        public UpdateDialogViewModel(IDialogService dialogService
            , IUpdateService updateService
            , IStringLocalizer<Language> localizer)
        {
            _dialogService = dialogService;
            _updateService = updateService;
            _localizer = localizer;

            var data = _updateService.UpdateMetadata;
            if (data == null) return;

            Version = $"{_localizer["Update:Version"]}: {data.CurrentVersion} -> {data.Version}";
            Filesize = $"{_localizer["Update:Size"]}: {(double)data.Size / (1024 * 1024):F2} MiB";
            Author = $"{_localizer["Update:Author"]}: {data.Author}";
            Date = $"{_localizer["Update:Date"]}: {data.Date:yyyy-MM-dd HH:mm}";
            Description = data.Description;

            CanInstall = OS.IsWindows();
        }

        private bool CanExecuteInstall() => CanInstall && !IsDownloading;

        [RelayCommand(CanExecute = nameof(CanExecuteInstall))]
        private async Task InstallAsync()
        {
            IsDownloading = true;
            DownloadProgress = 0;

            try
            {
                var success = await _updateService.DownloadAsync(OnDownloadProgress, _downloadCts.Token);
                if (!success)
                {
                    await _dialogService.ShowMessageAsync(new MessageBoxDefinition(
                        _localizer["Update:Failed"]
                        , _localizer["Update:FailedMessage"]
                        , MessageBoxType.Ok
                    ));

                    return;
                }

                var agreedToUpdate = await _dialogService.ShowMessageAsync(new MessageBoxDefinition(
                    _localizer["Update:Downloaded"]
                    , _localizer["Update:QuestionClose"]
                    , MessageBoxType.OkCancel
                ));

                if (agreedToUpdate)
                {
                    _updateService.Install();
                }

                Affirmative = agreedToUpdate;
            }
            catch (OperationCanceledException)
            {
                // Ok
            }
            finally
            {
                IsDownloading = false;
            }
        }

        public void CancelDownload() => _downloadCts.Cancel();

        [RelayCommand]
        private void Cancel()
        {
            CancelDownload();
            Affirmative = false;
        }

        private void OnDownloadProgress(object? sender, GenericProgressEventArgs e)
        {
            var progress = Math.Min(Math.Ceiling(e.Progress), 100d);
            if (progress == DownloadProgress) return;

            DownloadProgress = progress;
        }

        public void Dispose() => _downloadCts.Dispose();
    }
}
