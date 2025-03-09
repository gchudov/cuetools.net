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
using CUETools.Processor;

namespace CUERipper.Avalonia
{
    public static class Constants
    {
        public const string UnknownArtist = "Unknown artist";
        public const string UnknownTitle = "Unknown title";
        public const string UnknownTrack = "Track";
        public const string TrackNullLength = "-:--";
        public const string NoCDDriveFound = "No CD drive found";

        public static readonly string[] SecureModeValues = ["Burst", "Secure", "Paranoid"];
        public const int SecureModeDefault = 1;

        public const string TempFolderCUERipper = ".cuetmp";
        public const string CueExtension = ".cue";
        public const string JpgExtension = ".jpg";
        public const string HiResCoverName = "cover_hi-res";

        public static readonly string[] DefaultPathFormats = [
            $"%music%/%artist%/[%year% - ]%album%[ '('disc %discnumberandname%')']/%artist% - %album%{CueExtension}",
            $"%music%/%artist%/[%year% - ]%album%[ '('disc %discnumberandname%')'][' ('%releasedateandlabel%')'][' ('%unique%')']/%artist% - %album%{CueExtension}"
        ];
        public const int MaxPathFormats = 10; // Based on the original CUERipper limit

        public const string ApplicationName = $"CUERipper.Avalonia {CUESheet.CUEToolsVersion}";

        public const string PathNoto = "avares://CUERipper.Avalonia/Assets/noto-emoji/32/";
        public const string PathImageCache = "./CUERipper/.AlbumCache/";
        public const string PathUpdate = "./.cueupdate/";

        public const int EmbeddedImageMaxDimension = 500;
        public const int HiResImageMaxDimension = 2048;

        public const string GithubApiUri = "https://api.github.com/repos/gchudov/cuetools.net/releases";
        public const string UserAgent = "Mozilla/5.0";
        public const string UpdaterExecutable = "CUETools.Updater.exe";
    }
}
