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
using CUETools.Processor.Settings;
using System;
using System.IO;

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

        public const int HiResImageMaxDimension = 2048;

        public const string UserAgent = "Mozilla/5.0";
        public const string GithubApiUri = "https://api.github.com/repos/UnknownException/cuetools.net/releases";
        // "https://api.github.com/repos/gchudov/cuetools.net/releases"
        public const string GithubBranch = "cueripper-avalonia"; // "master"

        public const int MaxCoverFetchConcurrency = 4;

        public const char NullDrive = '\0';


        public const string ApplicationUserContentFolder = "CUERipper";
#if NET47
        public static readonly string ApplicationPath = AppDomain.CurrentDomain.BaseDirectory;
#else
        public static readonly string ApplicationPath = Environment.ProcessPath ?? throw new NullReferenceException("Can't determine path.");
#endif
        public static readonly string ProfileDir = SettingsShared.GetProfileDir(ApplicationUserContentFolder, ApplicationPath);

        public static readonly string PathImageCache = Path.Combine(ProfileDir, ".AlbumCache/");
        public static readonly string PathUpdateFolder = Path.Combine(ProfileDir, ".cueupdate/");
        public static readonly string PathUpdateCacheFile = Path.Combine(ProfileDir, "CT_LAST_UPDATE_CHECK");
    }
}
