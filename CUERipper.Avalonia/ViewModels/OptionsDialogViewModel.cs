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
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions;
using CUETools.CTDB;
using CUETools.Processor;
using System.Collections.ObjectModel;

namespace CUERipper.Avalonia.ViewModels
{
    public partial class OptionsDialogViewModel : ViewModelBase
    {
        public ObservableCollection<IOptionProxy> CTDBOptions { get; } = [];
        public ObservableCollection<IOptionProxy> ExtractionOptions { get; } = [];
        public ObservableCollection<IOptionProxy> ProxyOptions { get; } = [];
        public ObservableCollection<IOptionProxy> VariousOptions { get; } = [];

        private readonly ICUEConfigFacade _config;
        public OptionsDialogViewModel(ICUEConfigFacade config)
        {
            _config = config;

            BindConfig();
        }

        private void BindConfig()
        {
            new ObservableCollection<IOptionProxy> {
                new StringOptionProxy("CTDB Server", "db.cuetools.net"
                    , new(() => _config.CTDBServer))
                , new EnumOptionProxy<CTDBMetadataSearch>("Metadata search", CTDBMetadataSearch.Default
                    , new(() => _config.MetadataSearch))
                , new EnumOptionProxy<CUEConfigAdvanced.CTDBCoversSize>("Album art size", CUEConfigAdvanced.CTDBCoversSize.Large
                    , new(() => _config.CoversSize))
                , new EnumOptionProxy<CUEConfigAdvanced.CTDBCoversSearch>("Album art search", CUEConfigAdvanced.CTDBCoversSearch.Primary
                    , new(() => _config.CoversSearch))
                , new BoolOptionProxy("Detailed log", false
                    , new(() => _config.DetailedCTDBLog))
            }.MoveAll(CTDBOptions);

            new ObservableCollection<IOptionProxy> {
                new BoolOptionProxy("Preserve HTOA", true
                    , new(() => _config.PreserveHTOA))
                , new BoolOptionProxy("Detect Indexes", true
                    , new(() => _config.DetectGaps))
                , new BoolOptionProxy("EAC log style", true
                    , new(() => _config.CreateEACLog))
                , new BoolOptionProxy("Create M3U playlist", false
                    , new(() => _config.CreateM3U))
                , new BoolOptionProxy("Embed album art", true
                    , new(() => _config.EmbedAlbumArt))
                , new IntOptionProxy("Max album art size"
                    , defaultValue: CUEConfig.Constants.MaxAlbumArtSize
                    , minValue: CUEConfig.Constants.MaxAlbumArtSizeLowerBound
                    , maxValue: CUEConfig.Constants.MaxAlbumArtSizeUpperBound
                    , new(() => _config.MaxAlbumArtSize))
                , new BoolOptionProxy("Eject after rip", false
                    , new(() => _config.EjectAfterRip))
                , new BoolOptionProxy("Disable eject disc", true
                    , new(() => _config.DisableEjectDisc))
                , new StringOptionProxy("Track filename", "%tracknumber%. %title%"
                    , new(() => _config.TrackFilenameFormat))
                , new BoolOptionProxy("Automatic rip", false
                    , new(() => _config.AutomaticRip))
                , new BoolOptionProxy("Skip repair", false
                    , new(() => _config.SkipRepair))
            }.MoveAll(ExtractionOptions);

            new ObservableCollection<IOptionProxy> {
                new EnumOptionProxy<CUEConfigAdvanced.ProxyMode>("Proxy mode", CUEConfigAdvanced.ProxyMode.System
                    , new(() => _config.UseProxyMode))
                , new StringOptionProxy("Host", "127.0.0.1"
                    , new(() => _config.ProxyServer))
                , new IntOptionProxy("Port"
                    , defaultValue: 8080
                    , minValue: 0
                    , maxValue: 65535
                    , new(() => _config.ProxyPort))
                , new StringOptionProxy("Auth user", string.Empty
                    , new(() => _config.ProxyUser))
                , new StringOptionProxy("Auth password", string.Empty
                    , new(() => _config.ProxyPassword))

            }.MoveAll(ProxyOptions);

            new ObservableCollection<IOptionProxy> {
                new StringOptionProxy("Freedb site address", "gnudb.gnudb.org"
                    , new(() => _config.FreedbSiteAddress))
                , new BoolOptionProxy("Check for updates", true
                    , new(() => _config.CheckForUpdates))
            }.MoveAll(VariousOptions);
        }
    }
}
