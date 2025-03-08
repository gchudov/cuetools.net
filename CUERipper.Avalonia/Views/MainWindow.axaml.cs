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
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Configuration;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.ViewModels;
using CUERipper.Avalonia.Views.UserControls;
using CUETools.Processor;
using CUETools.Ripper;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Views
{
    public sealed partial class MainWindow : Window, IDisposable
    {
        public MainWindowViewModel ViewModel => DataContext as MainWindowViewModel
            ?? throw new ViewModelMismatchException(typeof(MainWindowViewModel), DataContext?.GetType());

        private readonly ICUERipperService _ripperService;
        private readonly ICUEMetaService _metaService;
        private readonly IDriveNotificationService _driveNotificationService;
        private readonly ICUEConfigFacade _config;
        private readonly IStringLocalizer<Language> _localizer;
        private readonly IIconService _iconService;
        private readonly IUpdateService _updateService;
        private readonly ILogger _logger;

        private Task? _rippingTask;
        private CancellationTokenSource _rippingCts = new();

        /// <summary>
        /// Added for readability.
        /// </summary>
        private enum UIMode
        {
            Init
            , Ready
            , Ripping
            , Done
        }

// TODO not clean, refactor
#if DEBUG
#pragma warning disable 8618
        /// <summary>
        /// This constructor should only be used by Avalonia in design mode.
        /// </summary>
        /// <exception cref="Exception"></exception>
        public MainWindow()
        {
            if (!Design.IsDesignMode) throw new NotInAvaloniaDesignModeException();

            _config = CUEConfigFacade.Create();

            InitializeComponent();
            SetUI(UIMode.Init);
        }
#pragma warning restore 8618
#endif

        public MainWindow(ICUERipperService ripperService
            , ICUEMetaService metaService
            , IDriveNotificationService driveNotificationService
            , ICUEConfigFacade config
            , IStringLocalizer<Language> localizer
            , IIconService iconService
            , IUpdateService updateService
            , ILogger<MainWindow> logger)
        {
            _ripperService = ripperService;
            _metaService = metaService;
            _driveNotificationService = driveNotificationService;
            _config = config;
            _localizer = localizer;
            _iconService = iconService;
            _updateService = updateService;
            _logger = logger;

            InitializeComponent();
            DataContextChanged += OnDataContextChanged;
            Closing += OnWindowClosing;

            driveSettingSection.Init(config, ripperService, localizer, iconService);
            coverViewer.Init(metaService);

            Image BindImage(AppIcon icon) => new() { Source = _iconService.GetIcon(icon), Width = 18, Height = 18 };

            buttonRefreshDrives.Click += OnRefreshDrivesClicked;
            buttonRefreshDrives.Content = BindImage(AppIcon.Disc);
            buttonAdvancedSearch.Click += OnAdvancedSearchClicked;
            buttonAdvancedSearch.Content = BindImage(AppIcon.Search);
            buttonResetSearch.Click += OnResetSearchClicked;
            buttonResetSearch.Content = BindImage(AppIcon.Cross);
            buttonEject.Click += OnEjectClicked;
            buttonEject.Content = BindImage(AppIcon.Eject);
            buttonGo.Click += OnGoClicked;
            buttonAbort.Click += OnAbortClicked;
            buttonTogglePane.Click += OnTogglePane;
            buttonGoSidePane.Click += OnGoClicked;
            buttonAbortSidePane.Click += OnAbortClicked;
            buttonPathFormat.Click += OnPathFormatClicked;
            buttonUpdate.Click += OnUpdateClicked;
            buttonUpdate.Content = BindImage(AppIcon.New);
            buttonUpdate.IsVisible = false;

            tabControlEncoding.SelectionChanged += OnEncodingTabChanged;
            coverViewer.ViewModel.PropertyChanged += OnCoverViewerPropertyChanged;

            _driveNotificationService.SetCallbacks(OnDriveListRefreshRequestedCallback
                , OnDriveUnmountedCallback
                , OnDriveMountedCallback);

            _ripperService.OnSecondaryProgress += RepairStatusCallback;
            _ripperService.OnRepairSelection += RepairSelectionCallback;
            _ripperService.OnRippingProgress += RipperStatusCallback;
            _ripperService.OnFinish += RipperFinishedCallback;
            _ripperService.OnDirectoryConflict += DirectoryConflictCallback;

            SetUI(UIMode.Init);
        }

        private async Task InitializeApplicationAsync()
        {
            InitializeEncodingTab();

            ViewModel.SetInitState(coverViewer.ViewModel.CurrentCover);

            if (ViewModel.CDDriveAvailable && ViewModel.AlbumReleases.Any())
            {
                coverViewer.Feed();

                SetUI(UIMode.Ready);

                if (_config.AutomaticRip && ViewModel.GetSelectedAlbumMeta() != null)
                {
                    await StartRippingAsync();
                }
            }
            else
            {
                SetUI(UIMode.Init);
                _logger.LogInformation(Constants.NoCDDriveFound);
            }

            var fetched = await _updateService.FetchAsync();
            buttonUpdate.IsVisible = fetched && _updateService.UpdateMetadata.UpdateAvailable();
        }

        private async void OnDataContextChanged(object? sender, EventArgs e)
            => await InitializeApplicationAsync();

        private async void OnRefreshDrivesClicked(object? sender, EventArgs e)
            => await InitializeApplicationAsync();

        private void OnAdvancedSearchClicked(object? sender, EventArgs e)
        {
            // Advanced search ignores the cache
            _metaService.GetAlbumMetaInformation(true);
            ViewModel.RefreshAlbums();
        }

        private void OnResetSearchClicked(object? sender, EventArgs e)
        {
            _metaService.ResetAlbumMetaInformation();
            _metaService.GetAlbumMetaInformation(false);
            ViewModel.RefreshAlbums();
        }

        private void OnEjectClicked(object? sender, EventArgs e)
            => _ripperService.EjectTray();

        private async void OnGoClicked(object? sender, EventArgs e)
            => await StartRippingAsync();

        private async Task StartRippingAsync()
        {
            if (_rippingTask != null && !_rippingTask.IsCompleted)
            {
                _logger.LogError("Ripping already in progress, how can this button be clicked?");
                return;
            }

            SetUI(UIMode.Ripping);

            if (!_rippingCts.TryReset())
            {
                _rippingCts.Dispose();
                _rippingCts = new();
            }

            var metadata = ViewModel.GetSelectedAlbumMeta();
            if(metadata != null) _metaService.FinalizeMetadata(metadata);

            lblStatus.Text = _localizer["Status:DownloadingAlbumCover"];
            var albumCoverUri = await coverViewer.GetCurrentCoverAsync(_rippingCts.Token);

            var ripSettings = new RipSettings
            {
                DriveOffset = driveSettingSection.ViewModel.DriveOffset
                , C2ErrorModeSetting = (DriveC2ErrorModeSetting)Enum.Parse(typeof(DriveC2ErrorModeSetting), driveSettingSection.ViewModel.SelectedC2ErrorMode, true)
                , CorrectionQuality = driveSettingSection.ViewModel.SelectedSecureMode
                , TestAndCopy = driveSettingSection.ViewModel.TestAndCopyEnabled
                , AlbumCoverUri = albumCoverUri
                , EncodingConfiguration = GetEncodingConfigurationFromTabControl()
            };

            _rippingTask = _ripperService.RipAudioTracks(ripSettings, _rippingCts.Token);
        }

        private async void OnAbortClicked(object? sender, EventArgs e)
        {
            if (_rippingTask == null || _rippingTask.IsCompleted)
            {
                _logger.LogError("No rip in progress, how can this button be clicked?");
                return;
            }

            _rippingCts.Cancel();

            lblStatus.Text = _localizer["Status:RipperStop"];

            await WaitForTaskToFinishAsync();

            lblStatus.Text = _localizer["Status:RipperStopped"];

            SetUI(UIMode.Done);
        }

        private void OnTogglePane(object? sender, EventArgs e)
        {
            ViewModel.SplitPaneOpen = !ViewModel.SplitPaneOpen;
            buttonTogglePane.Content = ViewModel.SplitPaneOpen ? ">" : "<"; 
        }

        private async void OnPathFormatClicked(object? sender, EventArgs e)
        {
            var meta = ViewModel.GetSelectedAlbumMeta();

            await PathFormatDialog.CreateAsync(this, meta, _config, _iconService);

            ViewModel.OutputPath = meta.PathStringFromFormat(_config.PathFormat, _config) ?? string.Empty;
        }

        private async void OnUpdateClicked(object? sender, EventArgs e)
        {            
            await UpdateDialog.CreateAsync(this, _updateService, _localizer);
        }

        private void SetUI(UIMode uiMode)
        {            
            buttonGo.IsVisible = uiMode != UIMode.Ripping;
            buttonGo.IsEnabled = uiMode == UIMode.Ready || uiMode == UIMode.Done;
            buttonAbort.IsVisible = !buttonGo.IsVisible;

            buttonGoSidePane.IsVisible = buttonGo.IsVisible;
            buttonGoSidePane.IsEnabled = buttonGo.IsEnabled;
            buttonAbortSidePane.IsVisible = buttonAbort.IsVisible;

            buttonPathFormat.IsEnabled = uiMode != UIMode.Ripping;

            GetDiscDriveControls()
                .ForEach(c => c.IsEnabled = uiMode != UIMode.Ripping);

            GetAlbumReleaseControls()
                .ForEach(c => c.IsEnabled = uiMode != UIMode.Ripping && uiMode != UIMode.Init);

            tabControlEncoding.IsEnabled = uiMode != UIMode.Ripping;

            driveSettingSection.IsEnabled = uiMode != UIMode.Ripping && uiMode != UIMode.Init;

            GetContentGridControls()
                .ForEach(c => c.IsReadOnly = uiMode == UIMode.Ripping || uiMode == UIMode.Init);

            // BUG causes flickering on tab header when moving mouse over images
            // coverViewer.IsEnabled = uiMode != UIMode.Ripping;
            coverViewer.ViewModel.IsReadOnly = uiMode == UIMode.Ripping;

            buttonTogglePane.Content = _config.DetailPaneOpened ? ">" : "<";

            lblStatus.Text = uiMode switch
            {
                UIMode.Init => ""
                , UIMode.Ready => Design.IsDesignMode ? "Ready" : _localizer["Status:Ready"]
                , _ => lblStatus.Text
            };

            buttonUpdate.IsEnabled = uiMode != UIMode.Ripping;
        }

        private void RepairSelectionCallback(object? sender, CUEToolsSelectionEventArgs args)
        {
            if (args.choices is CUEToolsSourceFile[] sourceFiles)
            {
                // TODO figure out how to NOT do it like this
                // https://github.com/davidfowl/AspNetCoreDiagnosticScenarios/blob/master/AsyncGuidance.md#avoid-using-taskresult-and-taskwait
                var result = Task.Run(() => Dispatcher.UIThread.InvokeAsync(
                    () => RepairSelectionDialog.CreateAsync(this, sourceFiles)
                )).GetAwaiter().GetResult();

                args.selection = result;
            }
        }

        private void RepairStatusCallback(object? sender, CUEToolsProgressEventArgs args)
        {
            string status = args.status;
            Dispatcher.UIThread.Post(() =>
            {
                lblStatus.Text = status;
            });
        }

        private void RipperStatusCallback(object? sender, ReadProgressArgs args)
        {
            if (sender is not ICDRipper audioSource) return;

            int audioLength = (int)audioSource.TOC.AudioLength;
            int correctionQuality = audioSource.CorrectionQuality;
            int audioTrackCount = audioSource.TOC.TrackCount;
            var trackLength = new List<int>();
            for(int i = 0; i < audioTrackCount; ++i)
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

            Dispatcher.UIThread.Post(() =>
            {
                int passTotalLength = args.PassEnd - args.PassStart;
                double correctionLength = (double)passTotalLength / (correctionQuality + 1);
                double correctionProcessed = (double)processed / (correctionQuality + 1) + correctionLength * args.Pass;
                double currentProgress = args.PassStart + correctionProcessed;

                lblStatus.Text = currentProgress >= audioLength ? _localizer["Status:Finalizing"] : status;

                double errorRatio = Math.Log(args.ErrorsCount / 10.0 + 1);
                double passRatio = Math.Log((args.PassEnd - args.PassStart) / 10.0 + 1);
                double errorPercentage = (errorRatio / passRatio) * 100;

                if (DataContext is MainWindowViewModel viewModel)
                {
                    viewModel.ReadingProgress = MathClamp.Clamp((int)(trackPercentage * 100), 0, 100);
                    viewModel.ErrorProgress = MathClamp.Clamp((int)errorPercentage, 0, 100);
                    viewModel.TotalProgress = (int)Math.Round((MathClamp.Clamp(currentProgress, 0, audioLength) / audioLength * 100));

                    for (int i = 0; i < audioTrackCount && i < viewModel.Tracks.Count; ++i)
                    {
                        var progressFraction = Math.Min(currentProgress / trackLength[i], 1f);
                        viewModel.Tracks[i].Progress = Convert.ToInt32(Math.Round(progressFraction * 100f));

                        if (trackLength[i] >= currentProgress) break;
                        else currentProgress -= trackLength[i];
                    }
                }
            });
        }

        private void RipperFinishedCallback(object? sender, RipperFinishedEventArgs e)
        {
            Dispatcher.UIThread.Post(async () =>
            {
                lblStatus.Text = e.Status;

                if (!string.IsNullOrWhiteSpace(e.PopupContent))
                {
                    await MessageBox.CreateDialogAsync(e.Status, e.PopupContent, this, _localizer);
                }

                SetUI(UIMode.Done);
            });
        }

        private void DirectoryConflictCallback(object? sender, DirectoryConflictEventArgs e)
        {
            var message = "The destination folder is not empty. This program will modify its contents by adding and removing files. Do you want to continue?";
            
            // TODO figure out how to NOT do it like this
            // https://github.com/davidfowl/AspNetCoreDiagnosticScenarios/blob/master/AsyncGuidance.md#avoid-using-taskresult-and-taskwait            
            var result = Task.Run(() => Dispatcher.UIThread.InvokeAsync(
                () => MessageBox.CreateDialogAsync($"Conflict: {e.Directory}", message, this, _localizer, MessageBox.MessageBoxType.YesNo)
            )).GetAwaiter().GetResult();

            e.CanModifyContent = result;
        }

        private void OnDriveListRefreshRequestedCallback()
        {
            Dispatcher.UIThread.Post(async () =>
            {
                if (_rippingTask != null && !_rippingTask.IsCompleted)
                {
                    var drives = _ripperService.QueryDrivesAvailable().Select(d => d.Key);
                    if (drives.Contains(_ripperService.SelectedDrive)) return;

                    _rippingCts.Cancel();
                    await WaitForTaskToFinishAsync();
                }

                await InitializeApplicationAsync();
            });
        }

        private void OnDriveUnmountedCallback(char driveLetter)
        {
            Dispatcher.UIThread.Post(async () =>
            {
                if (driveLetter == _ripperService.SelectedDrive)
                {
                    if (_rippingTask != null && !_rippingTask.IsCompleted)
                    {
                        _rippingCts.Cancel();
                        lblStatus.Text = _localizer["Status:DiscUnexpectedRemove"];
                    }
                    else
                    {
                        lblStatus.Text = _localizer["Status:DiscRemoved"];
                    }

                    await WaitForTaskToFinishAsync();

                    SetUI(UIMode.Init);
                }
            });
        }

        private void OnDriveMountedCallback(char driveLetter)
        {
            Dispatcher.UIThread.Post(async() =>
            {
                if (driveLetter == _ripperService.SelectedDrive) await InitializeApplicationAsync();
            });
        }

        private async Task WaitForTaskToFinishAsync()
        {
            if (_rippingTask == null) return;

            var previousCursor = Cursor;
            using (Cursor = new Cursor(StandardCursorType.Wait))
            {
                while (!_rippingTask.IsCompleted)
                {
                    await Task.Delay(100);
                }
            }

            Cursor = previousCursor;
        }

        private void OnSplitViewPaneClosing(object? sender, CancelRoutedEventArgs args)
        {
            if (ViewModel.SplitPaneOpen) args.Cancel = true;
        }

        private TabItem CreateEncodingTabItem(EncodingConfiguration? encodingConfig)
        {
            var control = new EncodingSection();
            control.Init(_config, _localizer, _iconService);

            if (encodingConfig != null)
            {
                control.ViewModel.SetConfiguration(encodingConfig);
            }

            var tabItem = new TabItem
            {
                Header = tabControlEncoding.Items.Count - 2
                , Content = control
                , FontSize = 20
            };

            if ((int)tabItem.Header == 0) tabItem.Header = "*";

            return tabItem;
        }

        private void OnEncodingTabCollectionChanged()
        {
            tabItemEncodingAdd.IsEnabled = tabControlEncoding.Items.Count < 9;
            tabItemEncodingRemove.IsEnabled = tabControlEncoding.Items.Count > 3;
        }

        private void OnEncodingTabChanged(object? sender, SelectionChangedEventArgs e)
        {
            if (tabControlEncoding.SelectedIndex == tabControlEncoding.Items.Count - 2)
            {
                var tabItem = CreateEncodingTabItem(null);
                tabControlEncoding.Items.Insert(tabControlEncoding.Items.Count - 2, tabItem);
                tabControlEncoding.SelectedIndex = tabControlEncoding.Items.Count - 3;

                OnEncodingTabCollectionChanged();

                e.Handled = true;
            }
            else if(tabControlEncoding.SelectedIndex == tabControlEncoding.Items.Count - 1)
            {
                tabControlEncoding.SelectedIndex = tabControlEncoding.Items.Count - 4;
                tabControlEncoding.Items.RemoveAt(tabControlEncoding.Items.Count - 3);

                OnEncodingTabCollectionChanged();

                e.Handled = true;
            }
        }

        private void InitializeEncodingTab()
        {
            // Check if already initialized. Skip if it is.
            if (tabControlEncoding.Items.Count != 2) return;

            EncodingConfiguration[] encodingConfig = [];
            try
            {
                if (!string.IsNullOrWhiteSpace(_config.EncodingConfiguration))
                {
                    encodingConfig = JsonConvert.DeserializeObject<EncodingConfiguration[]>(_config.EncodingConfiguration)
                        ?? encodingConfig;
                }
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to parse encoding configuration: {Config}", _config.EncodingConfiguration);
            }

            if (encodingConfig.Length == 0)
            {
                var tabItem = CreateEncodingTabItem(null);
                tabControlEncoding.Items.Insert(0, tabItem);
                tabControlEncoding.SelectedIndex = 0;
            }
            else
            {
                for (int i = 0; i < encodingConfig.Length; ++i)
                {
                    // Skip the first one, read it from the shared settings (CUERipper old)
                    var tabItem = CreateEncodingTabItem(i == 0 ? null : encodingConfig[i]);
                    tabControlEncoding.Items.Insert(i, tabItem);
                }

                tabControlEncoding.SelectedIndex = 0;
            }

            OnEncodingTabCollectionChanged();
        }

        private EncodingConfiguration[] GetEncodingConfigurationFromTabControl()
            => tabControlEncoding.Items
                .Select(i => (i as TabItem)?.FindLogicalDescendantOfType<EncodingSection>())
                .Select(i => i?.ViewModel.GetConfiguration())
                .OfType<EncodingConfiguration>()
                .ToArray();

        private void OnWindowClosing(object? sender, WindowClosingEventArgs e)
            => _config.EncodingConfiguration = JsonConvert.SerializeObject(GetEncodingConfigurationFromTabControl());

        private void OnCoverViewerPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName != nameof(CoverViewer.ViewModel.CurrentCover)) return;

            if (coverViewer.ViewModel.CurrentCover != null)
            {
                ViewModel.AlbumCoverImage = coverViewer.ViewModel.CurrentCover;
            }
        }

        private List<InputElement> GetDiscDriveControls()
            => [
                comboBoxDiscDrives
                , comboBoxDiscDrivesSidePane
                , buttonRefreshDrives
                , buttonEject
            ];

        private List<InputElement> GetAlbumReleaseControls()
            => [
                comboBoxAlbumReleases
                , comboBoxAlbumReleasesSidePane
                , buttonAdvancedSearch
                , buttonResetSearch
            ];

        private List<DataGrid> GetContentGridControls()
            => [
                gridTrackList
                , gridMetadata
            ];

        private bool _disposed;

        /// <summary>
        /// Class is sealed, so no need for inheritance concerns including a complex dispose pattern.
        /// </summary>
        public void Dispose()
        {
            if (_disposed == true) return;
            _disposed = true;

            coverViewer.Dispose();

            if (!_rippingCts.IsCancellationRequested) _rippingCts.Cancel();

            if (_rippingTask != null)
            {
                if (!_rippingTask.IsCompleted) _rippingTask.Wait();
                _rippingTask.Dispose();
            }

            _rippingCts.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}