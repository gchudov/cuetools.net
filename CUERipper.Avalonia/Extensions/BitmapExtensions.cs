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
using Avalonia;
using System;

namespace CUERipper.Avalonia.Extensions
{
    public static class BitmapExtensions
    {
        public static Bitmap ContainedResize(this Bitmap bitmap, int maxDimension)
        {
            var targetSize = bitmap.PixelSize;

            if (targetSize.Width > maxDimension || targetSize.Height > maxDimension)
            {
                var longestSide = Math.Max(targetSize.Width, targetSize.Height);
                var scaleFactor = (double)maxDimension / longestSide;

                targetSize = new PixelSize((int)(targetSize.Width * scaleFactor)
                    , (int)(targetSize.Height * scaleFactor));
            }

            return bitmap.CreateScaledBitmap(targetSize, BitmapInterpolationMode.HighQuality);
        }
    }
}
