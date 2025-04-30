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
using CUETools.Codecs;
using CUETools.CTDB;
using CUETools.Processor;
using System.Collections.Generic;

namespace CUERipper.Avalonia.Configuration.Abstractions
{
    public interface ICUEConfigFacade
    {
        /// <summary>
        /// Avoid using this function unless absolutely necessary.
        /// </summary>
        /// <returns></returns>
        CUEConfig ToCUEConfig();

        string Language { get; set; }

        Dictionary<string, CUEToolsFormat> Formats { get; }
        EncoderListViewModel Encoders { get; }
        Dictionary<string, CUEToolsScript> Scripts { get; }

        string DefaultDrive { get; set; }
        SerializableDictionary<string, int> DriveOffsets { get; }
        SerializableDictionary<string, int> DriveC2ErrorModes { get; }
        AudioEncoderType OutputCompression { get; set; }
        string DefaultLosslessFormat { get; set; }
        string DefaultLossyFormat { get; set; }
        string EncodingConfiguration { get; set; }

        string CTDBServer { get; set; }
        CTDBMetadataSearch MetadataSearch { get; set; }
        CUEConfigAdvanced.CTDBCoversSize CoversSize { get; set; }
        CUEConfigAdvanced.CTDBCoversSearch CoversSearch { get; set; }
        bool DetailedCTDBLog { get; set; }

        // Extraction options
        bool PreserveHTOA { get; set; }
        bool DetectGaps { get; set; }
        bool CreateEACLog { get; set; }
        bool CreateM3U { get; set; }
        bool EmbedAlbumArt { get; set; }
        int MaxAlbumArtSize { get; set; }
        bool EjectAfterRip { get; set; }
        bool DisableEjectDisc { get; set; }
        string TrackFilenameFormat { get; set; }
        bool AutomaticRip { get; set; }
        bool SkipRepair { get; set; }

        // Proxy options
        CUEConfigAdvanced.ProxyMode UseProxyMode { get; set; }
        string ProxyServer { get; set; }
        int ProxyPort { get; set; }
        string ProxyUser { get; set; }
        string ProxyPassword { get; set; }

        // Various options
        string FreedbSiteAddress { get; set; }
        bool CheckForUpdates { get; set; }

        // UI
        int CUEStyleIndex { get; set; }
        int SecureModeIndex { get; set; }
        bool TestAndCopyEnabled { get; set; }
        string PathFormat { get; set; }
        List<string> PathFormatTemplates { get; set; }
        bool DetailPaneOpened { get; set; }

        void Save();
    }
}