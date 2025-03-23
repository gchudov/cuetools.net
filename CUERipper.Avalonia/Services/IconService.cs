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
using Avalonia.Platform;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;

namespace CUERipper.Avalonia.Services
{
    public sealed class IconService : IIconService, IDisposable
    {
        private readonly Dictionary<AppIcon, string> _appIconPathMapping = new() {
            { AppIcon.Local, $"{Constants.PathNoto}emoji_u1f9e9.png" }
            , { AppIcon.MusicBrainz, "avares://CUERipper.Avalonia/Assets/musicbrainz.ico" }
            , { AppIcon.Freedb, "avares://CUERipper.Avalonia/Assets/freedb16.png" }
            , { AppIcon.Discogs, "avares://CUERipper.Avalonia/Assets/discogs.png" }
            , { AppIcon.Disc, $"{Constants.PathNoto}emoji_u1f4bf.png" }
            , { AppIcon.Search, $"{Constants.PathNoto}emoji_u1f50d.png" }
            , { AppIcon.Eject, $"{Constants.PathNoto}emoji_u23cf.png" }
            , { AppIcon.Cross, $"{Constants.PathNoto}emoji_u274c.png" }
            , { AppIcon.File, $"{Constants.PathNoto}emoji_u1f4c4.png" }
            , { AppIcon.Cog, $"{Constants.PathNoto}emoji_u2699.png"}
            , { AppIcon.Bolt, $"{Constants.PathNoto}emoji_u1f529.png" }
            , { AppIcon.Add, $"{Constants.PathNoto}emoji_u2795.png" }
            , { AppIcon.Subtract, $"{Constants.PathNoto}emoji_u2796.png" }
            , { AppIcon.Multiply, $"{Constants.PathNoto}emoji_u2716.png" }
            , { AppIcon.New, $"{Constants.PathNoto}emoji_u1f195.png" }
        };

        private readonly Dictionary<AppIcon, Bitmap?> _appIconBitmap = [];

        private readonly ILogger _logger;
        public IconService(ILogger<IconService> logger) 
        {
            _logger = logger;

            // Read images
            foreach (var item in _appIconPathMapping)
            {
                try
                {
                    using var stream = AssetLoader.Open(new Uri(item.Value));
                    _appIconBitmap.Add(item.Key, new Bitmap(stream));
                }
                catch(Exception ex)
                {
                    _logger.LogError(ex, "Failed to retrieve icon {path}.", item.Value);
                }
            }
        }

        public Bitmap? GetIcon(AppIcon appIcon)
            => _appIconBitmap.TryGetValue(appIcon, out Bitmap? result) ? result : null;

        public Bitmap? GetIcon(MetaSource metaSource)
            => metaSource switch {
                MetaSource.Local => GetIcon(AppIcon.Local),
                MetaSource.MusicBrainz => GetIcon(AppIcon.MusicBrainz),
                MetaSource.Freedb => GetIcon(AppIcon.Freedb),
                MetaSource.Discogs => GetIcon(AppIcon.Discogs),
                _ => GetIcon(AppIcon.File),
            };

        private bool _disposed;

        /// <summary>
        /// Class is sealed, so no need for inheritance concerns including a complex dispose pattern.
        /// </summary>
        public void Dispose()
        {
            if (_disposed == true) return;
            _disposed = true;

            foreach (var item in _appIconBitmap)
            {
                item.Value?.Dispose();
            }

            _appIconBitmap.Clear();

            GC.SuppressFinalize(this);
        }
    }
}
