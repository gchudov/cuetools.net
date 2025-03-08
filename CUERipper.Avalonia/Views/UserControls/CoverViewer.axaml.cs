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
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels.UserControls;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Views.UserControls;

public sealed partial class CoverViewer : UserControl, IDisposable
{
    public CoverViewerViewModel ViewModel => DataContext as CoverViewerViewModel
        ?? throw new ViewModelMismatchException(typeof(CoverViewerViewModel), DataContext?.GetType());

    public Bitmap? PlaceholderCover { get; set; }

    private readonly InterruptibleJob _thumbnailJob = new();

    private ICUEMetaService? _metaService;
    public CoverViewer()
    {
        InitializeComponent();

        DataContext = new CoverViewerViewModel();
    }

    public void Init(ICUEMetaService metaService)
    {
        _metaService = metaService;

        PlaceholderCover = GetPlaceholderAlbumCover();
        ViewModel.CurrentCover = PlaceholderCover;
    }

    public void Feed()
    {
        if (_metaService == null) throw new NotInitializedException(nameof(_metaService));
        
        ClearCovers();

        var albumCovers = _metaService.GetAlbumMetaInformation(false)
            .SelectMany(x => x.Data.AlbumArt)
            .Where(x => !string.IsNullOrWhiteSpace(x.uri) || !string.IsNullOrWhiteSpace(x.uri150))
            .Select(x => new CoverViewAlbumViewModel(!string.IsNullOrWhiteSpace(x.uri) ? x.uri : x.uri150
                , !string.IsNullOrWhiteSpace(x.uri150) ? x.uri150 : x.uri))
            .Distinct()
            .ToArray();

        _thumbnailJob.Run(async (CancellationToken ct) =>
        {
            foreach (var cover in albumCovers)
            {
                var bitmap = await _metaService.FetchImageAsync(cover.Uri150, ct);
                if (ct.IsCancellationRequested) break;

                if (bitmap != null)
                {
                    cover.Bitmap150 = bitmap;
                    Dispatcher.UIThread.Post(() =>
                    {
                        if (ViewModel.AlbumCovers.None())
                        {
                            cover.IsSelected = true;
                            ViewModel.CurrentCover = cover.Bitmap150;
                        }

                        ViewModel.AlbumCovers.Add(cover);
                    });
                }
            }
        });
    }

    private void CoverClicked(object? sender, RoutedEventArgs e)
    {
        if (ViewModel.IsReadOnly)
        {
            e.Handled = true;
            return;
        }

        if (sender is Image image && image.DataContext is CoverViewAlbumViewModel cover)
        {
            var previous = ViewModel.AlbumCovers.Where(x => x.IsSelected).FirstOrDefault();
            if (previous != null) previous.IsSelected = false;

            cover.IsSelected = true;

            ViewModel.CurrentCover = cover.Bitmap150;
        }
    }

    public async Task<string> GetCurrentCoverAsync(CancellationToken ct)
    {
        if (_metaService == null) throw new NotInitializedException(nameof(_metaService));

        await TryWaitForAtLeastOneThumbnail(ct);

        var cover = ViewModel.AlbumCovers.Where(x => x.IsSelected).FirstOrDefault();
        if (cover == null) return string.Empty;

        await _metaService.FetchImageAsync(cover.Uri, ct);
        return cover.Uri;
    }

    /// <summary>
    /// Attempts to retrieve at least one album thumbnail before continuing execution.
    /// </summary>
    /// <returns>A task that completes when an album thumbnail is available or the retry limit is reached.</returns>
    public async Task TryWaitForAtLeastOneThumbnail(CancellationToken ct)
    {
        const int MAX_RETRIES = 10;
        const int MAX_TIME_MS = 5000;
        const int MAX_RETRY_MS = MAX_TIME_MS / MAX_RETRIES;

        int retry = 0;
        while (_thumbnailJob.IsExecuting && ViewModel.AlbumCovers.None() && retry < MAX_RETRIES)
        {
            await Task.Delay(MAX_RETRY_MS, ct);
            ++retry;
        }
    }

    private void ClearCovers()
    {
        ViewModel.CurrentCover = PlaceholderCover;

        foreach(var cover in ViewModel.AlbumCovers)
        {
            cover.Bitmap150?.Dispose();
        }

        ViewModel.AlbumCovers.Clear();
    }

    private static Bitmap? GetPlaceholderAlbumCover()
    {
        var uri = new Uri("avares://CUERipper.Avalonia/Assets/album-placeholder.bmp");
        try
        {
            using var stream = AssetLoader.Open(uri);
            return new Bitmap(stream);
        }
        catch (FileNotFoundException)
        {
            return null;
        }
    }

    private bool _disposed = false;
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _thumbnailJob.Dispose();

        ClearCovers();

        PlaceholderCover?.Dispose();
    }
}