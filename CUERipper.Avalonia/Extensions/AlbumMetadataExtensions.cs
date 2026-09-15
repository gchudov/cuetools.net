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
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Models;
using CUETools.CDImage;
using CUETools.Processor;

namespace CUERipper.Avalonia.Extensions
{
    internal static class AlbumMetadataExtensions
    {
        // I'll put this here for now. It's a bit of a nasty function, but it does the job.
        /// <summary>
        /// Generate a path based on the current metadata and provided format.
        /// Only call this for UI related functions.
        /// </summary>
        /// <param name="meta"></param>
        /// <param name="format"></param>
        /// <param name="config"></param>
        /// <returns>Formatted path</returns>
        public static string PathStringFromFormat(this AlbumMetadata? meta, string format, ICUEConfigFacade config)
        {
            CUESheet? cueSheet = null;
            if (meta?.Data.Tracks.Count > 0)
            {
                // Dummy layout for the CopyMetadata function
                var cdLayout = new CDImageLayout();
                for (uint i = 0; i < meta.Data.Tracks.Count; ++i)
                {
                    cdLayout.AddTrack(new CDTrack(0, i * 100, 100, true, false));
                }

                cueSheet = new CUESheet(config.ToCUEConfig())
                {
                    Action = CUEAction.Encode
                    , TOC = cdLayout
                };
                cueSheet.CopyMetadata(meta.Data);
            }

            var path = CUESheet.GenerateUniqueOutputPath(
                _config: config.ToCUEConfig()
                , format: format
                , ext: Constants.CueExtension
                , action: CUEAction.Encode
                , vars: []
                , pathIn: null
                , cueSheet: cueSheet);

            cueSheet?.Close();

            return OS.IsWindows() ? path.Replace('/', '\\') : path.Replace('\\', '/');
        }
    }
}
