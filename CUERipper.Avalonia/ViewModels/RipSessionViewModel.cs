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
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Processor;
using CUETools.Ripper;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.ViewModels
{
    public sealed partial class RipSessionViewModel : ViewModelBase, IDisposable
    {
        public delegate Task<RipSettings> RipSettingsFactory(CancellationToken ct);

        public event EventHandler<TrackProgressEventArgs>? OnTrackProgress;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(IsRipping))]
        [NotifyPropertyChangedFor(nameof(IsDiscIdle))]
        [NotifyCanExecuteChangedFor(nameof(StartCommand))]
        [NotifyCanExecuteChangedFor(nameof(AbortCommand))]
        private SessionState mode = SessionState.Init;

        public bool IsRipping { get => Mode == SessionState.Ripping; }
        public bool IsDiscIdle { get => Mode == SessionState.Ready || Mode == SessionState.Done; }

        [ObservableProperty]
        private string status = string.Empty;

        [ObservableProperty]
        private int readingProgress;

        [ObservableProperty]
        private int totalProgress;

        [ObservableProperty]
        private int errorProgress;

        public bool HasRunningTask { get => _rippingTask != null && !_rippingTask.IsCompleted; }

        private volatile bool _isUsingDrive;
        public bool IsUsingDrive { get => _isUsingDrive; }

        [ObservableProperty]
        private bool isStopping;

        private RipSettingsFactory? _buildSettings;
        private Task<CUEResult>? _rippingTask;
        private CancellationTokenSource _rippingCts = new();

        private readonly ICUEConfigFacade _config;
        private readonly IDiscRippingService _rippingService;
        private readonly IDiscImageService _imageService;
        private readonly IAlbumMetadataService _metaService;
        private readonly IDialogService _dialogService;
        private readonly IUIDispatcher _dispatcher;
        private readonly IStringLocalizer _localizer;
        private readonly ILogger _logger;

        public RipSessionViewModel(ICUEConfigFacade config
            , IDiscRippingService rippingService
            , IDiscImageService imageService
            , IAlbumMetadataService metaService
            , IDialogService dialogService
            , IUIDispatcher dispatcher
            , IStringLocalizer<Language> localizer
            , ILogger<RipSessionViewModel> logger)
        {
            _config = config;
            _rippingService = rippingService;
            _imageService = imageService;
            _metaService = metaService;
            _dialogService = dialogService;
            _dispatcher = dispatcher;
            _localizer = localizer;
            _logger = logger;

            _rippingService.OnDirectoryConflict += DirectoryConflictCallback;
            _rippingService.OnRippingProgress += RipperStatusCallback;

            _imageService.OnProgress += ImageProgressCallback;
            _imageService.OnRepairSelection += RepairSelectionCallback;
        }

        public void UseSettingsFactory(RipSettingsFactory factory)
            => _buildSettings = factory;

        private bool CanStart => IsDiscIdle;
        private bool CanAbort => IsRipping;

        [RelayCommand(CanExecute = nameof(CanStart))]
        private async Task StartAsync()
        {
            if (_buildSettings == null) throw new NotInitializedException(nameof(RipSettingsFactory));

            if (HasRunningTask)
            {
                _logger.LogError("Ripping already in progress, start shouldn't be reachable.");
                return;
            }

            Mode = SessionState.Ripping;

            _metaService.Save();

            if (!_rippingCts.TryReset())
            {
                _rippingCts.Dispose();
                _rippingCts = new CancellationTokenSource();
            }

            Status = _localizer["Status:DownloadingAlbumCover"];

            try
            {
                var settings = await _buildSettings(_rippingCts.Token);

                _rippingTask = RunAsync(settings, _rippingCts.Token);

                ReportFinished(await _rippingTask);
            }
            catch (OperationCanceledException)
            {
                Mode = SessionState.Done;
                return;
            }
        }

        private async Task<CUEResult> RunAsync(RipSettings settings, CancellationToken ct)
        {
            CUEResult result;

            _isUsingDrive = true;
            try
            {
                result = await _rippingService.RipAsync(settings, ct)
                    .ConfigureAwait(false);
            }
            finally
            {
                _isUsingDrive = false;
            }

            if (!result.IsSuccess) return result;

            var processed = await _imageService.ProcessAsync(result.CUEPath
                , settings.EncodingConfiguration
                , repairable: result.Status == RipStatus.Repairable
                , ct).ConfigureAwait(false);

            return processed ?? result;
        }

        [RelayCommand(CanExecute = nameof(CanAbort))]
        private async Task AbortAsync()
        {
            _rippingCts.Cancel();
            Status = _localizer["Status:RipperStop"];

            await WaitForCompletionAsync();

            Status = _localizer["Status:RipperStopped"];
            Mode = SessionState.Done;
        }

        public async Task CancelAsync()
        {
            if (!HasRunningTask) return;

            _rippingCts.Cancel();
            await WaitForCompletionAsync();
        }

        private async Task WaitForCompletionAsync()
        {
            if (_rippingTask == null) return;

            IsStopping = true;
            try
            {
                await _rippingTask;
            }
            catch (OperationCanceledException)
            {
                // Ok
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Ripping task faulted while waiting for it to finish.");
            }
            finally
            {
                IsStopping = false;
            }
        }

        private void ReportProgress(int reading, int total, int error)
        {
            ReadingProgress = reading;
            TotalProgress = total;
            ErrorProgress = error;
        }

        public void ResetProgress() => ReportProgress(0, 0, 0);

        #region Ripping Callbacks

        private void ImageProgressCallback(object? sender, CUEToolsProgressEventArgs args)
        {
            string status = args.status;
            double percent = args.percent;

            _dispatcher.Post(() =>
            {
                Status = status;
                TotalProgress = MathClamp.Clamp((int)Math.Round(percent * 100), 0, 100);
            });
        }

        private void RepairSelectionCallback(object? sender, CUEToolsSelectionEventArgs args)
        {
            if (args.choices is CUEToolsSourceFile[] sourceFiles)
            {
                // TODO figure out how to NOT do it like this
                // https://github.com/davidfowl/AspNetCoreDiagnosticScenarios/blob/master/AsyncGuidance.md#avoid-using-taskresult-and-taskwait
                var result = Task.Run(() => _dispatcher.InvokeAsync(
                    () => _dialogService.ShowRepairSelectionAsync(sourceFiles)
                )).GetAwaiter().GetResult();

                args.selection = result;
            }
        }

        private void ReportFinished(CUEResult result)
        {
            Mode = SessionState.Done;
            Status = result.StatusText;

            if (result.IsSuccess && _config.AutomaticRip) return;

            if (string.IsNullOrWhiteSpace(result.PopupContent)) return;

            var messageBox = new MessageBoxDefinition(result.StatusText
                , result.PopupContent
                , MessageBoxType.Ok);

            // Fire and forget
            _ = _dialogService.ShowMessageAsync(messageBox)
                .ContinueWith(task
                    => _logger.LogError(task.Exception, "Failed to report the rip result.")
                , CancellationToken.None
                , TaskContinuationOptions.OnlyOnFaulted
                , TaskScheduler.Default);
        }

        private void DirectoryConflictCallback(object? sender, DirectoryConflictEventArgs e)
        {
            // TODO figure out how to NOT do it like this
            // https://github.com/davidfowl/AspNetCoreDiagnosticScenarios/blob/master/AsyncGuidance.md#avoid-using-taskresult-and-taskwait            
            var result = Task.Run(() => _dispatcher.InvokeAsync(
                () =>
                {
                    var messageBox = new MessageBoxDefinition(_localizer["Warning:DirectoryExists"]
                        , _localizer["Warning:QuestionOverwriteDestination"]
                        , MessageBoxType.YesNo
                    );

                    return _dialogService.ShowMessageAsync(messageBox);
                }
            )).GetAwaiter().GetResult();

            e.CanModifyContent = result;
        }

        private void RipperStatusCallback(object? sender, ReadProgressArgs args)
        {
            if (sender is not ICDRipper audioSource) return;

            int audioLength = (int)audioSource.TOC.AudioLength;
            int correctionQuality = audioSource.CorrectionQuality;
            int audioTrackCount = audioSource.TOC.TrackCount;
            var trackLength = new List<int>();
            for (int i = 0; i < audioTrackCount; ++i)
            {
                trackLength.Add((int)audioSource.TOC[i + 1].Length);
            }

            int processed = args.Position - args.PassStart;
            TimeSpan elapsed = DateTime.Now - args.PassTime;
            double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;

            double trackPercentage = (double)(args.Position - args.PassStart) / (args.PassEnd - args.PassStart);
            string retry = args.Pass > 0 ? $" ({_localizer["Status:Retry"]} {args.Pass})" : "";
            string status = (elapsed.TotalSeconds > 0 && args.Pass >= 0) ?
                string.Format("{0} @{1:00.00}x{2}...", args.Action, speed, retry) :
                string.Format("{0}{1}...", args.Action, retry);

            _dispatcher.Post(() =>
            {
                int passTotalLength = args.PassEnd - args.PassStart;
                double correctionLength = (double)passTotalLength / (correctionQuality + 1);
                double correctionProcessed = (double)processed / (correctionQuality + 1) + correctionLength * Math.Min(args.Pass, correctionQuality);
                double currentProgress = args.PassStart + correctionProcessed;

                double errorRatio = Math.Log(args.ErrorsCount / 10.0 + 1);
                double passRatio = Math.Log((args.PassEnd - args.PassStart) / 10.0 + 1);
                double errorPercentage = (errorRatio / passRatio) * 100;

                Status = currentProgress >= audioLength
                    ? _localizer["Status:Finalizing"]
                    : status;

                ReportProgress(
                    reading: MathClamp.Clamp((int)(trackPercentage * 100), 0, 100)
                    , total: (int)Math.Round((MathClamp.Clamp(currentProgress, 0, audioLength) / audioLength * 100))
                    , error: MathClamp.Clamp((int)errorPercentage, 0, 100));

                var trackProgress = new List<int>(audioTrackCount);
                for (int i = 0; i < audioTrackCount; ++i)
                {
                    var progressFraction = Math.Min(currentProgress / trackLength[i], 1f);
                    trackProgress.Add(Convert.ToInt32(Math.Round(progressFraction * 100f)));

                    if (trackLength[i] >= currentProgress) break;
                    else currentProgress -= trackLength[i];
                }

                OnTrackProgress?.Invoke(this, new TrackProgressEventArgs(trackProgress));
            });
        }

        #endregion

        private bool _disposed = false;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (!_rippingCts.IsCancellationRequested) _rippingCts.Cancel();

            if (HasRunningTask)
            {
                try
                {
                    _rippingTask!.Wait();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Ripping task threw an exception while waiting for disposal.");
                }
            }

            _rippingCts.Dispose();

            _rippingService.OnDirectoryConflict -= DirectoryConflictCallback;
            _rippingService.OnRippingProgress -= RipperStatusCallback;

            _imageService.OnProgress -= ImageProgressCallback;
            _imageService.OnRepairSelection -= RepairSelectionCallback;

            GC.SuppressFinalize(this);
        }
    }
}
