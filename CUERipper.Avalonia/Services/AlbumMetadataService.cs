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
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.CDImage;
using CUETools.CTDB;
using CUETools.Processor;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public class AlbumMetadataService : IAlbumMetadataService
    {
        private readonly ICUEConfigFacade _cueConfig;
        private readonly ICUEMetadataStore _metadataStore;
        private readonly IRemoteMetadataLookup _remoteLookup;
        private readonly IBitmapFactory _bitmapFactory;
        private readonly HttpClient _httpClient;
        private readonly ILogger _logger;

        private readonly Dictionary<string, IImmutableList<AlbumMetadata>> _cache = [];

        private CDImageLayout _toc = new();

        private AlbumMetadata? _selectedMetadata;
        public AlbumMetadata? SelectedMetadata
        {
            get => _selectedMetadata;
            set
            {
                _selectedMetadata = value;

                var eventArgs = new SelectedMetadataChangedEventArgs(value);
                OnSelectedMetadataChanged?.Invoke(this, eventArgs);
            }
        }
        public event EventHandler<SelectedMetadataChangedEventArgs>? OnSelectedMetadataChanged;

        public AlbumMetadataService(IDiscRippingService rippingService
            , ICUEConfigFacade cueConfig
            , ICUEMetadataStore metadataStore
            , IRemoteMetadataLookup remoteLookup
            , IBitmapFactory bitmapFactory
            , HttpClient httpClient
            , ILogger<AlbumMetadataService> logger)
        {
            _cueConfig = cueConfig;
            _metadataStore = metadataStore;
            _remoteLookup = remoteLookup;
            _bitmapFactory = bitmapFactory;
            _httpClient = httpClient;
            _logger = logger;

            rippingService.OnSelectedDriveChanged += (object? _, DriveChangedEventArgs e) =>
            {
                _toc = rippingService.GetDiscTOC() ?? new();
            };
        }

        private static CUEMetadataEntry CreateDummy(CDImageLayout toc)
        {
            var dummy = new CTDBResponseMeta
            {
                artist = Constants.UnknownArtist
                , album = Constants.UnknownTitle
                , track = new CTDBResponseMetaTrack[toc.AudioTracks]
                , year = string.Empty
                , disccount = "1"
                , discnumber = "1"                
            };

            for (int i = 0; i < dummy.track.Length; ++i)
            {
                dummy.track[i] = new CTDBResponseMetaTrack
                {
                    name = $"{Constants.UnknownTrack} {i + 1}",
                    artist = dummy.album
                };
            }

            var meta = new CUEMetadata(toc.TOCID, (int)toc.AudioTracks);
            meta.FillFromCtdb(dummy, toc.FirstAudio - 1);

            return new CUEMetadataEntry(meta, toc, string.Empty);
        }

        public IImmutableList<AlbumMetadata> Search(bool advancedSearch)
        {
            if (_toc.AudioTracks == 0) return [];

            if (!advancedSearch && _cache.TryGetValue(_toc.TOCID, out var cached))
            {
                _logger.LogInformation("Album is available in cache for {TOCID}", _toc.TOCID);
                return cached;
            }

            _logger.LogInformation("Retrieving album information {TOCID}", _toc.TOCID);

            CUEMetadata? userEntry = null;
            try
            {
                userEntry = _metadataStore.Load(_toc.TOCID);
                _logger.LogInformation("Found user entry for {TOCID}", _toc.TOCID);
            }
            catch (FileNotFoundException)
            {
                _logger.LogInformation("No user entry for {TOCID}", _toc.TOCID);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Non fatal error parsing CUE Metadata cache.");
            }

            List<CUEMetadataEntry> LookupRemote(CTDBMetadataSearch search)
                => _remoteLookup.Lookup(Constants.ApplicationShortName
                    , _toc
                    , _cueConfig.ToCUEConfig()
                    , search
                );

            var metadataSearch = advancedSearch
                ? CTDBMetadataSearch.Extensive
                : _cueConfig.MetadataSearch;

            var remoteResult = LookupRemote(metadataSearch);

            if (remoteResult.Count == 0
                && metadataSearch != CTDBMetadataSearch.Extensive
                && metadataSearch != CTDBMetadataSearch.None)
            {
                _logger.LogInformation("No results for {TOCID}, retrying with an extensive search."
                    , _toc.TOCID);

                remoteResult = LookupRemote(CTDBMetadataSearch.Extensive);
            }

            _logger.LogInformation("{Count} remote results for {TOCID}", remoteResult.Count, _toc.TOCID);

            var result = remoteResult.Concat([CreateDummy(_toc)])
                .Select(entry => new AlbumMetadata(MetaSourceHelper.FromString(entry.ImageKey), entry.metadata))
                .PrependIf(userEntry != null, new AlbumMetadata(MetaSource.Local, userEntry!))
                .ToImmutableList();

            // Only cache if remote call was successful
            if (remoteResult.Count > 0)
            {
                _cache.Remove(_toc.TOCID);
                _cache.Add(_toc.TOCID, result);
            }

            return result;
        }

        public void Reset()
            => _cache.Remove(_toc.TOCID);

        public IEnumerable<string> GetTrackLengths()
        {
            if (_toc.AudioTracks == 0) return [];

            var result = new List<string>();
            for (int i = 1; i <= _toc.TrackCount; ++i)
            {
                var trackLength = _toc[i].LengthMSF;
                var timeParts = trackLength.Split(':').Select(int.Parse).ToArray();
                if (timeParts.Length != 3)
                {
                    _logger.LogWarning("{TrackLength} does not match expected format.", trackLength);
                    return [];
                }

                if (timeParts[2] >= 50) timeParts[1] += 1;

                result.Add($"{timeParts[0]}:{timeParts[1]:00}");
            }

            return result;
        }

        public async Task<IBitmap?> FetchBitmapAsync(string uri, CancellationToken ct)
        {
            _logger.LogInformation("Fetching image from {Uri}.", uri);

            if (string.IsNullOrWhiteSpace(uri)) return null;

            if (!Directory.Exists(Constants.PathImageCache))
            {
                Directory.CreateDirectory(Constants.PathImageCache);
            }

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(uri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");
            if (File.Exists(filePath)) return _bitmapFactory.FromFile(filePath);

            try
            {
                using var response = await _httpClient.GetAsync(uri, ct);
                response.EnsureSuccessStatusCode();

#if NET47
                using var stream = await response.Content.ReadAsStreamAsync();
#else
                using var stream = await response.Content.ReadAsStreamAsync(ct);
#endif

                var bitmapFromStream = _bitmapFactory.FromStream(stream);
                if (bitmapFromStream == null) return null;

                var bitmap = bitmapFromStream.ContainedResize(Constants.HiResImageMaxDimension);
                bitmapFromStream.Dispose();

                bitmap.SaveJpeg(filePath, Constants.JpegQuality);
                return bitmap;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve album cover from {Uri}", uri);
                return null;
            }
        }

        public void Save()
        {
            if (SelectedMetadata == null) return;

            _metadataStore.Save(SelectedMetadata.Data);
        }
    }
}