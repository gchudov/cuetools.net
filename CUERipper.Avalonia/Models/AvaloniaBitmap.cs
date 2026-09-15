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
using Avalonia;
using Avalonia.Media.Imaging;
using CUERipper.Avalonia.Models.Abstractions;
using SkiaSharp;
using System;
using System.IO;

namespace CUERipper.Avalonia.Models
{
    public sealed class AvaloniaBitmap : IBitmap
    {
        public Bitmap Bitmap { get; }

        public int Width => Bitmap.PixelSize.Width;
        public int Height => Bitmap.PixelSize.Height;

        public AvaloniaBitmap(Bitmap bitmap)
        {
            Bitmap = bitmap;
        }

        public IBitmap ContainedResize(int maxDimension)
        {
            var targetSize = Bitmap.PixelSize;

            if (targetSize.Width > maxDimension || targetSize.Height > maxDimension)
            {
                var longestSide = Math.Max(targetSize.Width, targetSize.Height);
                var scaleFactor = (double)maxDimension / longestSide;

                targetSize = new PixelSize((int)(targetSize.Width * scaleFactor)
                    , (int)(targetSize.Height * scaleFactor));
            }

            return new AvaloniaBitmap(Bitmap.CreateScaledBitmap(targetSize, BitmapInterpolationMode.HighQuality));
        }

        public void SaveJpeg(string filePath, int quality)
        {
            using var stream = File.Create(filePath);
            SaveJpeg(stream, quality);
        }

        // Most likely not the most efficient method due to multiple PNG/JPG encodings/decodings, but it's easy to read and does the trick.
        // The current 11.x (old) Avalonia version isn't very supportive of JPEG, raw Skia magic ahead.
        public void SaveJpeg(Stream stream, int quality)
        {
            using var ms = new MemoryStream();
            Bitmap.Save(ms);
            ms.Position = 0;

            using var bitmap = SKBitmap.Decode(ms);
            using var encodedImage = bitmap.Encode(SKEncodedImageFormat.Jpeg, quality);
            encodedImage.SaveTo(stream);
        }

        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            Bitmap.Dispose();
        }
    }
}
