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
using CUERipper.Avalonia.ViewModels;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions;
using CUETools.CTDB;
using CUETools.Processor;
using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class OptionsDialog : Window
{
    public OptionsDialogViewModel ViewModel => DataContext as OptionsDialogViewModel
        ?? throw new ViewModelMismatchException(typeof(OptionsDialogViewModel), DataContext?.GetType());

    public required ICUEConfigFacade Config { get; init; }
    public OptionsDialog()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        new ObservableCollection<IOptionProxy> {
            new StringOptionProxy("CTDB Server", "db.cuetools.net"
                , new(() => Config.CTDBServer))
            , new EnumOptionProxy<CTDBMetadataSearch>("Metadata search", CTDBMetadataSearch.Default
                , new(() => Config.MetadataSearch))
            , new EnumOptionProxy<CUEConfigAdvanced.CTDBCoversSize>("Album art size", CUEConfigAdvanced.CTDBCoversSize.Large
                , new(() => Config.CoversSize))
            , new EnumOptionProxy<CUEConfigAdvanced.CTDBCoversSearch>("Album art search", CUEConfigAdvanced.CTDBCoversSearch.Primary
                , new(() => Config.CoversSearch))
            , new BoolOptionProxy("Detailed log", false
                , new(() => Config.DetailedCTDBLog))
        }.MoveAll(ViewModel.CTDBOptions);

        new ObservableCollection<IOptionProxy> {
            new BoolOptionProxy("Preserve HTOA", true
                , new(() => Config.PreserveHTOA))
            , new BoolOptionProxy("Detect Indexes", true
                , new(() => Config.DetectGaps))
            , new BoolOptionProxy("EAC log style", true
                , new(() => Config.CreateEACLog))
            , new BoolOptionProxy("Create M3U playlist", false
                , new(() => Config.CreateM3U))
            , new BoolOptionProxy("Embed album art", true
                , new(() => Config.EmbedAlbumArt))
            , new BoolOptionProxy("Eject after rip", false
                , new(() => Config.EjectAfterRip))
            , new BoolOptionProxy("Disable eject disc", true
                , new(() => Config.DisableEjectDisc))
            , new StringOptionProxy("Track filename", "%tracknumber%. %title%"
                , new(() => Config.TrackFilenameFormat))
            , new BoolOptionProxy("Automatic rip", false
                , new(() => Config.AutomaticRip))
            , new BoolOptionProxy("Skip repair", false
                , new(() => Config.SkipRepair))
        }.MoveAll(ViewModel.ExtractionOptions);

        new ObservableCollection<IOptionProxy> {
            new EnumOptionProxy<CUEConfigAdvanced.ProxyMode>("Proxy mode", CUEConfigAdvanced.ProxyMode.System
                , new(() => Config.UseProxyMode))
            , new StringOptionProxy("Host", "127.0.0.1"
                , new(() => Config.ProxyServer))
            , new IntOptionProxy("Port", 8080
                , new(() => Config.ProxyPort))
            , new StringOptionProxy("Auth user", string.Empty
                , new(() => Config.ProxyUser))
            , new StringOptionProxy("Auth password", string.Empty
                , new(() => Config.ProxyPassword))

        }.MoveAll(ViewModel.ProxyOptions);

        new ObservableCollection<IOptionProxy> {
            new StringOptionProxy("Freedb site address", "gnudb.gnudb.org"
                , new(() => Config.FreedbSiteAddress))
        }.MoveAll(ViewModel.VariousOptions);
    }

    public static async Task CreateAsync(Window owner, ICUEConfigFacade config)
    {
        var optionsWindow = new OptionsDialog()
        {
            Owner = owner
            , Config = config
            , DataContext = new OptionsDialogViewModel()
        };

        await optionsWindow.ShowDialog(owner, lockParent: true);
    }
}