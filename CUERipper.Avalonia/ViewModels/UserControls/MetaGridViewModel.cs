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
using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.ViewModels.Bindings;
using CUETools.Processor;
using Microsoft.Extensions.Localization;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class MetaGridViewModel : ViewModelBase, IDisposable
    {
        public ObservableCollection<EditableFieldProxy> Metadata { get; } = [];

        [ObservableProperty]
        private bool isReadOnly = true;

        [ObservableProperty]
        private string albumTitle = string.Empty;

        [ObservableProperty]
        private string albumArtist = string.Empty;

        [ObservableProperty]
        private string albumYear = string.Empty;

        [ObservableProperty]
        private string albumDisc = string.Empty;


        private readonly IAlbumMetadataService _metaService;
        private readonly IStringLocalizer _localizer;
        public MetaGridViewModel(IAlbumMetadataService metaService
            , IStringLocalizer<Language> stringLocalizer)
        {
            _metaService = metaService;
            _localizer = stringLocalizer;

            _metaService.OnSelectedMetadataChanged += OnSelectedMetadataChanged;
        }

        public void Clear()
        {
            Metadata.Clear();
        }

        private string FormatAlbumDisc(CUEMetadata data)
            => $"{_localizer["Main:Disc"]} {data.DiscNumber ?? "1"} {_localizer["Main:DiscSeperator"]} {data.TotalDiscs ?? "1"}";

        public void OnSelectedMetadataChanged(object? sender, SelectedMetadataChangedEventArgs e)
        {
            Clear();

            var meta = e.AlbumMetadata;
            if (meta == null) return;

            AlbumTitle = meta.Data.Title;
            AlbumArtist = meta.Data.Artist;
            AlbumYear = meta.Data.Year;
            AlbumDisc = FormatAlbumDisc(meta.Data);

            meta.Data.Title = string.IsNullOrWhiteSpace(meta.Data.Title) ? Constants.UnknownTitle : meta.Data.Title;
            meta.Data.Artist = string.IsNullOrWhiteSpace(meta.Data.Artist) ? Constants.UnknownArtist : meta.Data.Artist;
                
            new ObservableCollection<EditableFieldProxy> {
                new (_localizer["Meta:Artist"], () => meta.Data.Artist, x => { 
                    meta.Data.Artist = x; 
                    AlbumArtist = x; 
                })
                , new (_localizer["Meta:Title"], () => meta.Data.Title, x => {
                    meta.Data.Title = x; 
                    AlbumTitle = x; 
                })
                , new (_localizer["Meta:Genre"], () => meta.Data.Genre, x => meta.Data.Genre = x)
                , new (_localizer["Meta:Year"], () => meta.Data.Year, x => {
                    meta.Data.Year = x;
                    AlbumYear = x;
                })
                , new (_localizer["Meta:CurrentDisc"], () => meta.Data.DiscNumber, x => { 
                    meta.Data.DiscNumber = x; 
                    AlbumDisc = FormatAlbumDisc(meta.Data);
                })
                , new (_localizer["Meta:TotalDiscs"], () => meta.Data.TotalDiscs, x => {
                    meta.Data.TotalDiscs = x;
                    AlbumDisc = FormatAlbumDisc(meta.Data);
                })
                , new (_localizer["Meta:DiscName"], () => meta.Data.DiscName, x => meta.Data.DiscName = x)
                , new (_localizer["Meta:Label"], () => meta.Data.Label, x => meta.Data.Label = x)
                , new (_localizer["Meta:LabelNo"], () => meta.Data.LabelNo, x => meta.Data.LabelNo = x)
                , new (_localizer["Meta:ReleaseDate"], () => meta.Data.ReleaseDate, x => meta.Data.ReleaseDate = x)
                , new (_localizer["Meta:Barcode"], () => meta.Data.Barcode, x => meta.Data.Barcode = x)
                , new (_localizer["Meta:Country"], () => meta.Data.Country, x => meta.Data.Country = x)
                , new (_localizer["Meta:Comment"], () => meta.Data.Comment, x => meta.Data.Comment = x)
            }.MoveAll(Metadata);
        }

        private bool _disposed = false;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _metaService.OnSelectedMetadataChanged -= OnSelectedMetadataChanged;
        }
    }
}
