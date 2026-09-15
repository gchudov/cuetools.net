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
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.ViewModels.UserControls;
using CUERipper.Avalonia.Views.UserControls.Abstractions;

namespace CUERipper.Avalonia.Views.UserControls;

public sealed partial class TrackGrid : CUEGrid<TrackGridColumnKey
        , GridColumnDefinition<TrackGridColumnKey>
        , TrackViewModel
    >
{
    public TrackGrid()
    {
        InitializeComponent();

        DefineColumns();
        InitGrid(trackGrid);
    }
    
    private void DefineColumns()
    {
        Columns.Add(TrackGridColumnKey.TrackNo, new GridColumnDefinition<TrackGridColumnKey> {
            Header = "#"
            , HeaderBinding = false
            , Binding = nameof(TrackViewModel.TrackNo)
            , ReadOnly = true
            , Clipboard = true
            , Create = CreateTextColumn
        });

        Columns.Add(TrackGridColumnKey.Title, new GridColumnDefinition<TrackGridColumnKey> {
            Header = "HeaderTitle"
            , HeaderBinding = true
            , Binding = nameof(TrackViewModel.Title)
            , ReadOnly = false
            , Clipboard = true
            , Create = CreateTextColumn
        });

        Columns.Add(TrackGridColumnKey.Length, new GridColumnDefinition<TrackGridColumnKey> {
            Header = "HeaderLength"
            , HeaderBinding = true
            , Binding = nameof(TrackViewModel.Length)
            , ReadOnly = true
            , Clipboard = true
            , Create = CreateTextColumn
        });

        Columns.Add(TrackGridColumnKey.Progress, new GridColumnDefinition<TrackGridColumnKey> {
            Header = "HeaderProgress"
            , HeaderBinding = true
            , Binding = nameof(TrackViewModel.Progress)
            , ReadOnly = true
            , Clipboard = false
            , Create = CreateProgressColumn
        });

        Columns.Add(TrackGridColumnKey.Artist, new GridColumnDefinition<TrackGridColumnKey> {
            Header = "HeaderArtist"
            ,HeaderBinding = true
            , Binding = nameof(TrackViewModel.Artist)
            , ReadOnly = false
            , Clipboard = true
            , Create = CreateTextColumn
        });
    }
}
