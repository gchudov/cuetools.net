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
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.Utilities;
using CUETools.CTDB;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class CoverViewerViewModel : ViewModelBase, IDisposable
    {
        public ObservableCollection<CoverViewAlbumViewModel> AlbumCovers { get; } = [];

        [ObservableProperty]
        private IBitmap? currentCover;

        [ObservableProperty]
        private bool isReadOnly;

        private readonly IBitmap? _placeholderCover;

        private readonly InterruptibleJob _thumbnailJob = new();

        private readonly IAlbumMetadataService _metaService;
        private readonly IUIDispatcher _dispatcher;
        public CoverViewerViewModel(IAlbumMetadataService metaService
            , IBitmapFactory bitmapFactory
            , IUIDispatcher dispatcher)
        {
            _metaService = metaService;
            _dispatcher = dispatcher;

            _placeholderCover = bitmapFactory.FromAsset("album-placeholder.bmp");
            CurrentCover = _placeholderCover;

            _metaService.OnSelectedMetadataChanged += OnSelectedMetadataChanged;
        }

        private void OnSelectedMetadataChanged(object? sender, SelectedMetadataChangedEventArgs e)
        {
            var releaseCovers = GetReleaseCovers(e.AlbumMetadata);

            var match = AlbumCovers.FirstOrDefault(c => releaseCovers.Any(c.Equals));
            if (match != null) SelectCover(match);
        }

        private static CoverViewAlbumViewModel[] GetReleaseCovers(AlbumMetadata? metadata)
            => (metadata?.Data.AlbumArt ?? [])
                .Where(HasAnyUri)
                .Where(x => x.primary)
                .Select(ToCoverViewModel)
                .ToArray();

        public void Feed()
        {
            var unorderedCovers = _metaService.Search(advancedSearch: false)
                .SelectMany(x => x.Data.AlbumArt)
                .Where(HasAnyUri)
                .Select(ToCoverViewModel)
                .OrderByDescending(x => x.IsPrimary)
                .Distinct()
                .ToArray();

            var releaseCovers = GetReleaseCovers(_metaService.SelectedMetadata);
            var preselectedCover = unorderedCovers.FirstOrDefault(c => releaseCovers.Any(c.Equals));

            var orderedCovers = new[]
            { 
                // Primary artwork (Front cover)
                unorderedCovers.Where(x => x.IsPrimary)
                // Secondary artwork (Photo of CD, back cover, etc.)
                , unorderedCovers.Where(x => !x.IsPrimary)
            };

            _thumbnailJob.Run(async (CancellationToken ct) =>
            {
                using var semaphore = new SemaphoreSlim(Constants.MaxCoverFetchConcurrency);
                foreach (var albumCovers in orderedCovers)
                {
                    await Task.WhenAll(albumCovers.Select(async cover =>
                    {
                        await semaphore.WaitAsync(ct);
                        try
                        {
                            var bitmap = await _metaService.FetchBitmapAsync(cover.Uri150, ct);
                            if (ct.IsCancellationRequested) return;

                            if (bitmap != null)
                            {
                                cover.Bitmap150 = bitmap;
                                _dispatcher.Post(() =>
                                {
                                    AlbumCovers.Add(cover);

                                    if (cover == preselectedCover || AlbumCovers.None(x => x.IsSelected))
                                    {
                                        SelectCover(cover);
                                    }
                                });
                            }
                        }
                        finally
                        {
                            semaphore.Release();
                        }
                    }));
                }
            });
        }

        private static bool HasAnyUri(CTDBResponseMetaImage art)
            => !string.IsNullOrWhiteSpace(art.uri) || !string.IsNullOrWhiteSpace(art.uri150);

        private static CoverViewAlbumViewModel ToCoverViewModel(CTDBResponseMetaImage art)
            => new(!string.IsNullOrWhiteSpace(art.uri) ? art.uri : art.uri150
                , !string.IsNullOrWhiteSpace(art.uri150) ? art.uri150 : art.uri
                , art.primary);

        [RelayCommand]
        private void SelectCover(CoverViewAlbumViewModel cover)
        {
            if (IsReadOnly)
            {
                return;
            }

            foreach (var previous in AlbumCovers.Where(x => x.IsSelected).ToArray())
            {
                previous.IsSelected = false;
            }

            cover.IsSelected = true;
            CurrentCover = cover.Bitmap150;
        }

        public async Task<string> GetCurrentCoverAsync(CancellationToken ct)
        {
            await TryWaitForAtLeastOneThumbnail(ct);

            var cover = AlbumCovers.Where(x => x.IsSelected).FirstOrDefault();
            cover ??= AlbumCovers.Where(x => x.IsPrimary).FirstOrDefault();

            if (cover?.Uri == null) return string.Empty;

            await _metaService.FetchBitmapAsync(cover.Uri, ct);
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
            while (_thumbnailJob.IsExecuting && AlbumCovers.None() && retry < MAX_RETRIES)
            {
                await Task.Delay(MAX_RETRY_MS, ct);
                ++retry;
            }
        }

        public void Clear()
        {
            _thumbnailJob.Interrupt();

            CurrentCover = _placeholderCover;

            foreach (var cover in AlbumCovers)
            {
                cover.Bitmap150?.Dispose();
            }

            AlbumCovers.Clear();
        }

        private bool _disposed = false;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _metaService.OnSelectedMetadataChanged -= OnSelectedMetadataChanged;

            _thumbnailJob.Dispose();

            Clear();

            _placeholderCover?.Dispose();
        }
    }
}
