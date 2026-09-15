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
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using Microsoft.Extensions.Logging;
using System;
using System.IO;

namespace CUERipper.Avalonia.Services
{
    public sealed class AvaloniaBitmapFactory : IBitmapFactory
    {
        private const string AssetRoot = "avares://CUERipper.Avalonia/Assets/";

        private readonly ILogger _logger;
        public AvaloniaBitmapFactory(ILogger<AvaloniaBitmapFactory> logger)
        {
            _logger = logger;
        }

        public IBitmap? FromFile(string filePath)
        {
            try
            {
                return new AvaloniaBitmap(new Bitmap(filePath));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load bitmap from {Path}.", filePath);
                return null;
            }
        }

        public IBitmap? FromStream(Stream stream)
        {
            try
            {
                return new AvaloniaBitmap(new Bitmap(stream));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load bitmap from stream.");
                return null;
            }
        }

        public IBitmap? FromAsset(string assetName)
        {
            try
            {
                using var stream = AssetLoader.Open(new Uri($"{AssetRoot}{assetName}"));
                return new AvaloniaBitmap(new Bitmap(stream));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load asset {Asset}.", assetName);
                return null;
            }
        }
    }
}
