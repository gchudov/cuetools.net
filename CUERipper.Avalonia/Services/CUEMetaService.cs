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
using Avalonia.Media.Imaging;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
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
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public class CUEMetaService : ICUEMetaService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger _logger;

        private readonly Dictionary<string, IImmutableList<AlbumMetadata>> _cache = [];

        private CDImageLayout _toc = new();
        private string _arName = string.Empty;

        public CUEMetaService(ICUERipperService ripperService
            , HttpClient httpClient
            , ILogger<CUEMetaService> logger)
        {
            _httpClient = httpClient;
            _logger = logger;

            ripperService.OnSelectedDriveChanged += (object? _, DriveChangedEventArgs e) =>
            {
                SetContentInfo(ripperService.GetDiscTOC(), ripperService.GetDriveARName());
            };
        }

        public void SetContentInfo(CDImageLayout? TOC, string ARName)
        {
            _toc = TOC ?? new();
            _arName = ARName;
        }

        private static CTDBResponseMeta CreateDummy(uint audioTracks)
        {
            var dummy = new CTDBResponseMeta
            {
                artist = Constants.UnknownArtist
                , album = Constants.UnknownTitle
                , track = new CTDBResponseMetaTrack[audioTracks]
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

            return dummy;
        }

        public IImmutableList<AlbumMetadata> GetAlbumMetaInformation(bool advancedSearch)
        {
            _logger.LogInformation("Retrieving album information {TOC}", _toc);

            if (_toc.AudioTracks == 0) return [];

            if (!advancedSearch && _cache.TryGetValue(_toc.TOCID, out var cached)) return cached;

            CUEMetadata? userCache = null;
            try
            {
                userCache = CUEMetadata.Load(_toc.TOCID);
            }
            catch (FileNotFoundException)
            {
                _logger.LogInformation("Album not found in user cache.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Non fatal error parsing CUE Metadata cache.");
            }

            var ctdb = new CUEToolsDB(_toc, null);
            ctdb.ContactDB(null, Constants.ApplicationName, _arName, true, false, advancedSearch ? CTDBMetadataSearch.Extensive : CTDBMetadataSearch.Fast);

            switch (ctdb.QueryResponseStatus)
            {
                case HttpStatusCode.OK:
                    {
                        _logger.LogInformation("Retrieved {MetadataCount} releases from the database", ctdb.Metadata.Count());

                        var result = ctdb.Metadata
                            .Concat([CreateDummy(_toc.AudioTracks)])
                            .Select(m =>
                            {
                                var meta = new CUEMetadata(_toc.TOCID, (int)_toc.AudioTracks);
                                meta.FillFromCtdb(m, _toc.FirstAudio - 1);
                                return new AlbumMetadata(MetaSourceHelper.FromString(m.source), meta);
                            }).PrependIf(userCache != null, new AlbumMetadata(MetaSource.Local, userCache!))
                            .ToImmutableList();

                        _cache.Remove(_toc.TOCID);
                        _cache.Add(_toc.TOCID, result);

                        return result;
                    }
                case HttpStatusCode.NotFound:
                    {
                        _logger.LogWarning("Album not found in the database");

                        IEnumerable<CTDBResponseMeta> result = [CreateDummy(_toc.AudioTracks)];
                        return result.Select(m =>
                        {
                            var meta = new CUEMetadata(_toc.TOCID, (int)_toc.AudioTracks);
                            meta.FillFromCtdb(m, _toc.FirstAudio - 1);
                            return new AlbumMetadata(MetaSourceHelper.FromString(m.source), meta);
                        }).PrependIf(userCache != null, new AlbumMetadata(MetaSource.Local, userCache!))
                        .ToImmutableList();
                    }
                default:
                    {
                        _logger.LogError("Unexpected problem contacting the database {StatusCode}", ctdb.QueryResponseStatus);

                        IEnumerable<CTDBResponseMeta> result = [CreateDummy(_toc.AudioTracks)];
                        return result.Select(m =>
                        {
                            var meta = new CUEMetadata(_toc.TOCID, (int)_toc.AudioTracks);
                            meta.FillFromCtdb(m, _toc.FirstAudio - 1);
                            return new AlbumMetadata(MetaSourceHelper.FromString(m.source), meta);
                        }).PrependIf(userCache != null, new AlbumMetadata(MetaSource.Local, userCache!))
                        .ToImmutableList();
                    }
            }
        }

        public void ResetAlbumMetaInformation()
            => _cache.Remove(_toc.TOCID);

        public IEnumerable<string> GetTracksLength()
        {
            _logger.LogInformation("Retrieving album information {TOC}", _toc);
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

        public async Task<Bitmap?> FetchImageAsync(string uri, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(uri)) return null;

            if (!Directory.Exists(Constants.PathImageCache))
            {
                Directory.CreateDirectory(Constants.PathImageCache);
            }

            using var md5 = MD5.Create();
            var fileIdentifier = md5.ComputeHashAsString(uri);
            var filePath = Path.Combine(Constants.PathImageCache, $"{fileIdentifier}{Constants.JpgExtension}");
            if (File.Exists(filePath)) return new Bitmap(filePath);

            try
            {
                using var response = await _httpClient.GetAsync(uri, ct);
                response.EnsureSuccessStatusCode();

#if NET47
                using var stream = await response.Content.ReadAsStreamAsync();
#else
                using var stream = await response.Content.ReadAsStreamAsync(ct);
#endif

                var bitmapFromStream = new Bitmap(stream);
                var bitmap = bitmapFromStream.ContainedResize(Constants.HiResImageMaxDimension);
                bitmapFromStream.Dispose();

                bitmap.Save(filePath);
                return bitmap;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve album cover from {uri}", uri);
                return null;
            }
        }

        public void FinalizeMetadata(AlbumMetadata metadata)
            => metadata.Data.Save();
    }
}