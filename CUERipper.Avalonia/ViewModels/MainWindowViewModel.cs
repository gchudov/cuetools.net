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
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.ViewModels.UserControls;
using CUETools.Ripper;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.ViewModels
{
    public partial class MainWindowViewModel : ViewModelBase, IDisposable
    {
        public ObservableCollection<string> DiscDrives { get; set; } = [];

        [ObservableProperty]
        private string selectedDrive = string.Empty;

        public bool CDDriveAvailable => string.Compare(DiscDrives[0], Constants.NoCDDriveFound) != 0;

        partial void OnSelectedDriveChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;
            if (string.Compare(oldValue, newValue) == 0) return;
            if (string.Compare(newValue, Constants.NoCDDriveFound) == 0) return;

            _rippingService.SelectedDrive = newValue[0];
            _config.DefaultDrive = newValue;
        }

        public ObservableCollection<AlbumRelease> AlbumReleases { get; set; } = [];

        [ObservableProperty]
        private AlbumRelease? selectedAlbum;

        partial void OnSelectedAlbumChanged(AlbumRelease? oldValue, AlbumRelease? newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue?.Name)) return;
            if (oldValue == newValue) return;

            var meta = GetSelectedAlbumMeta();
            _metaService.SelectedMetadata = meta;

            OutputPath = meta?.PathStringFromFormat(_config.PathFormat, _config) ?? "Output Path";
        }

        [ObservableProperty]
        private string outputPath = "Output Path";

        [ObservableProperty]
        private bool updateAvailable;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(TogglePaneGlyph))]
        private bool splitPaneOpen;
        partial void OnSplitPaneOpenChanged(bool oldValue, bool newValue)
        {
            _config.DetailPaneOpened = newValue;
        }

        [ObservableProperty]
        private bool installPending;

        public string TogglePaneGlyph { get => SplitPaneOpen ? ">" : "<"; }

        public string HeaderTracks { get => _localizer["Main:Tracks"]; }
        public string HeaderMetadata { get => _localizer["Main:Metadata"]; }

        public IBitmap? IconRefreshDrives { get; }
        public IBitmap? IconAdvancedSearch { get; }
        public IBitmap? IconResetSearch { get; }
        public IBitmap? IconEject { get; }
        public IBitmap? IconUpdate { get; }

        public DriveSettingSectionViewModel DriveSettings { get; }
        public EncodingTabContainerViewModel EncodingTabs { get; }
        public MetaGridViewModel MetaGrid { get; }
        public TrackGridViewModel TrackGrid { get; }
        public CoverViewerViewModel CoverViewer { get; }
        public RipSessionViewModel RipSession { get; }

        private readonly ICUEConfigFacade _config;
        private readonly IDiscRippingService _rippingService;
        private readonly IAlbumMetadataService _metaService;
        private readonly IStringLocalizer _localizer;
        private readonly IIconService _iconService;
        private readonly IDialogService _dialogService;
        private readonly IUpdateService _updateService;
        private readonly IUIDispatcher _dispatcher;
        private readonly ILogger _logger;
        public MainWindowViewModel(ICUEConfigFacade config
            , IDiscRippingService rippingService
            , IAlbumMetadataService metaService
            , IStringLocalizer<Language> stringLocalizer
            , IIconService iconService
            , IDialogService dialogService
            , IUpdateService updateService
            , IUIDispatcher dispatcher
            , IDriveNotificationService driveNotificationService
            , ILogger<MainWindowViewModel> logger
            , DriveSettingSectionViewModel driveSettings
            , EncodingTabContainerViewModel encodingTabs
            , MetaGridViewModel metaGrid
            , TrackGridViewModel trackGrid
            , CoverViewerViewModel coverViewer
            , RipSessionViewModel ripSession)
        {
            _config = config;
            _rippingService = rippingService;
            _metaService = metaService;
            _localizer = stringLocalizer;
            _iconService = iconService;
            _dialogService = dialogService;
            _updateService = updateService;
            _dispatcher = dispatcher;
            _logger = logger;

            DriveSettings = driveSettings;
            EncodingTabs = encodingTabs;
            MetaGrid = metaGrid;
            TrackGrid = trackGrid;
            CoverViewer = coverViewer;
            RipSession = ripSession;

            driveNotificationService.SetCallbacks(OnDriveListRefreshRequestedCallback
                , OnDriveUnmountedCallback
                , OnDriveMountedCallback);

            _rippingService.OnSelectedDriveChanged += (object? _, DriveChangedEventArgs e)
                => _dispatcher.Post(async () => {
                    // Prevent double initializing and only re-initialize when a new drive has been selected
                    if (e.PreviousDrive != Constants.NullDrive && e.PreviousDrive != e.NextDrive)
                    {
                        await RefreshSessionAsync();
                    }
                });

            RipSession.UseSettingsFactory(BuildRipSettingsAsync);
            RipSession.PropertyChanged += OnSessionPropertyChanged;
            RipSession.OnTrackProgress += OnTrackProgress;

            IconRefreshDrives = iconService.GetIcon(AppIcon.Disc);
            IconAdvancedSearch = iconService.GetIcon(AppIcon.Search);
            IconResetSearch = iconService.GetIcon(AppIcon.Cross);
            IconEject = iconService.GetIcon(AppIcon.Eject);
            IconUpdate = iconService.GetIcon(AppIcon.New);
        }

        [RelayCommand]
        private void AdvancedSearch()
        {
            // Advanced search ignores the cache
            _metaService.Search(advancedSearch: true);
            RefreshAlbums();
        }

        [RelayCommand]
        private void ResetSearch()
        {
            _metaService.Reset();
            _metaService.Search(advancedSearch: false);
            RefreshAlbums();
        }

        [RelayCommand]
        private void EjectTray() => _rippingService.EjectTray();

        [RelayCommand]
        private void TogglePane() => SplitPaneOpen = !SplitPaneOpen;

        private async Task<RipSettings> BuildRipSettingsAsync(CancellationToken ct)
        {
            var albumCoverUri = await CoverViewer.GetCurrentCoverAsync(ct);

            return new RipSettings
            {
                DriveOffset = DriveSettings.DriveOffset
                , C2ErrorModeSetting = (DriveC2ErrorModeSetting)Enum.Parse(typeof(DriveC2ErrorModeSetting), DriveSettings.SelectedC2ErrorMode, true)
                , CorrectionQuality = DriveSettings.SelectedSecureMode
                , TestAndCopy = DriveSettings.TestAndCopyEnabled
                , AlbumCoverUri = albumCoverUri
                , EncodingConfiguration = EncodingTabs.GetEncodingConfigurations()
            };
        }

        [RelayCommand]
        private async Task ShowPathFormatAsync()
        {
            var meta = _metaService.SelectedMetadata;
            await _dialogService.ShowPathFormatAsync(meta);

            OutputPath = meta.PathStringFromFormat(_config.PathFormat, _config) ?? string.Empty;
        }

        [RelayCommand]
        private async Task ShowUpdateAsync()
        {
            InstallPending = await _dialogService.ShowUpdateAsync();
        }

        private bool RefreshAlbums()
        {
            AlbumReleases.Clear();

            var metaInfo = _metaService.Search(advancedSearch: false);
            new ObservableCollection<AlbumRelease>(
                metaInfo.Select((meta, index) =>
                {
                    const string YEAR_SEPERATOR = ": ";

                    string year = meta.Data.Year;
                    string artist = meta.Data.Artist ?? Constants.UnknownArtist;
                    string title = meta.Data.Title ?? Constants.UnknownTitle;
                    string country = meta.Data.Country ?? string.Empty;
                    string labelName = meta.Data.Label ?? string.Empty;
                    string barcode = meta.Data.Barcode ?? string.Empty;
                    string releaseDate = meta.Data.ReleaseDate ?? string.Empty;

                    if (string.IsNullOrWhiteSpace(country)
                        && string.IsNullOrWhiteSpace(labelName)
                        && string.IsNullOrWhiteSpace(barcode)
                        && string.IsNullOrWhiteSpace(releaseDate))
                    {
                        return new AlbumRelease($"{(string.IsNullOrWhiteSpace(year) ? string.Empty : year + YEAR_SEPERATOR)}{artist} - {title}"
                            , Icon: _iconService.GetIcon(meta.Source)
                            , Index: index);
                    }

                return new AlbumRelease($"{(string.IsNullOrWhiteSpace(year) ? string.Empty : year + YEAR_SEPERATOR)}{artist} - {title} ({country} - {labelName} {barcode} - {releaseDate})"
                    , Icon: _iconService.GetIcon(meta.Source)
                    , Index: index);
                })
            ).MoveAll(AlbumReleases);

            SelectedAlbum = AlbumReleases.Any() ? AlbumReleases[0] : null;
            return SelectedAlbum != null;
        }

        private AlbumMetadata? GetSelectedAlbumMeta()
        {
            if (!CDDriveAvailable || SelectedAlbum == null) return null;

            var index = Math.Min(Math.Max(0, SelectedAlbum.Index), AlbumReleases.Count - 1);
            var albumMetaInformation = _metaService.Search(advancedSearch: false);
            return index < albumMetaInformation.Count ? albumMetaInformation.ElementAt(index) : null;
        }

        private void ClearSession()
        {
            AlbumReleases.Clear();
            DiscDrives.Clear();

            CoverViewer.Clear();
            TrackGrid.Clear();
            MetaGrid.Clear();

            RipSession.ResetProgress();
        }

        private void InitializeSession()
        {
            if (DiscDrives.Count != 0)
            {
                throw new InvalidOperationException($"{nameof(InitializeSession)} requires a cleared state, call {nameof(ClearSession)} first.");
            }

            foreach (var driveName in _rippingService.QueryAvailableDriveInformation())
            {
                DiscDrives.Add(driveName.Value.Name);
            }

            if (DiscDrives.Count == 0)
            {
                DiscDrives.Add(Constants.NoCDDriveFound);
                _logger.LogInformation(Constants.NoCDDriveFound);
            }

            SelectedDrive = !string.IsNullOrWhiteSpace(_config.DefaultDrive)
                    && DiscDrives.Contains(_config.DefaultDrive)
                ? _config.DefaultDrive
                : DiscDrives[0];

            RipSession.Mode = SessionState.Init;

            if (DiscDrives[0] == Constants.NoCDDriveFound) return;

            if(RefreshAlbums())
            {
                CoverViewer.Feed();
                RipSession.Mode = SessionState.Ready;
            }
        }

        private async Task StartAutomaticRipAsync()
        {
            if (RipSession.Mode != SessionState.Ready) return;
            if (_metaService.SelectedMetadata == null) return;

            await RipSession.StartCommand.ExecuteAsync(null);
        }

        [RelayCommand]
        public async Task RefreshSessionAsync()
        {
            ClearSession();
            InitializeSession();

            if (_config.AutomaticRip) await StartAutomaticRipAsync();
        }

        public async Task CheckForUpdateAsync()
        {
            var fetched = await _updateService.FetchAsync();
            UpdateAvailable = fetched && _updateService.UpdateMetadata.UpdateAvailable();
        }

        public void Initialize()
        {
            EncodingTabs.InitializeTabs();
            SplitPaneOpen = _config.DetailPaneOpened;
        }

        public async Task<bool> TryShutdownAsync()
        {
            if (RipSession.HasRunningTask)
            {
                var messageBox = new MessageBoxDefinition(_localizer["Warning:CantClose"]
                    , _localizer["Warning:RipInProgress"]
                    , MessageBoxType.YesNo
                );

                var result = await _dialogService.ShowMessageAsync(messageBox);
                if (!result) return false;

                await RipSession.CancelAsync();
            }

            EncodingTabs.PersistTabs();

            return true;
        }

        #region Drive Callbacks

        private void OnDriveListRefreshRequestedCallback()
        {
            _dispatcher.Post(async () =>
            {
                if (RipSession.HasRunningTask)
                {
                    var drives = _rippingService.QueryAvailableDriveInformation().Select(d => d.Key);
                    if (drives.Contains(_rippingService.SelectedDrive)) return;
                    if (!RipSession.IsUsingDrive) return;

                    await RipSession.CancelAsync();
                }

                await RefreshSessionAsync();
            });
        }

        private void OnDriveUnmountedCallback(char driveLetter)
        {
            _dispatcher.Post(async () =>
            {
                if (driveLetter == _rippingService.SelectedDrive)
                {
                    if (RipSession.HasRunningTask)
                    {
                        if (!RipSession.IsUsingDrive) return;

                        RipSession.Status = _localizer["Status:DiscUnexpectedRemove"];
                        await RipSession.CancelAsync();
                    }
                    else
                    {
                        RipSession.Status = _localizer["Status:DiscRemoved"];
                    }

                    RipSession.Mode = SessionState.Init;
                }
            });
        }

        private void OnDriveMountedCallback(char driveLetter)
        {
            _dispatcher.Post(async () =>
            {
                if (RipSession.HasRunningTask && !RipSession.IsUsingDrive) return;
                if (driveLetter == _rippingService.SelectedDrive) await RefreshSessionAsync();
            });
        }

        #endregion

        private void OnSessionPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName != nameof(RipSessionViewModel.Mode)) return;

            TrackGrid.IsReadOnly = !RipSession.IsDiscIdle;
            MetaGrid.IsReadOnly = !RipSession.IsDiscIdle;
            DriveSettings.IsReadOnly = !RipSession.IsDiscIdle;

            CoverViewer.IsReadOnly = RipSession.IsRipping;
            EncodingTabs.IsReadOnly = RipSession.IsRipping;

            if (RipSession.IsRipping)
            {
                TrackGrid.ResetProgress();
                RipSession.ResetProgress();
            }

            // Init, Ripping and Done keep whatever the ripper last reported
            if (RipSession.Mode == SessionState.Ready) RipSession.Status = _localizer["Status:Ready"];
        }

        private void OnTrackProgress(object? sender, TrackProgressEventArgs e)
        {
            int boundary = Math.Min(e.TrackProgress.Count, TrackGrid.Tracks.Count);
            for(int i = 0; i < boundary; ++i)
            {
                TrackGrid.Tracks[i].Progress = e.TrackProgress[i];
            }
        }

        private bool _disposed = false;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            RipSession.PropertyChanged -= OnSessionPropertyChanged;
            RipSession.OnTrackProgress -= OnTrackProgress;
        }
    }
}
