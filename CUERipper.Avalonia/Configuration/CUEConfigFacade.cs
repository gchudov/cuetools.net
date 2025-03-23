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
using Avalonia.Controls;
using CUETools.Processor;
using System.IO;
using System;
using CUETools.Processor.Settings;
using System.Xml;
using System.Xml.Serialization;
using System.Collections.Generic;
using CUETools.Codecs;
using CUETools.CTDB;
using CUERipper.Avalonia.Configuration.Abstractions;

namespace CUERipper.Avalonia.Configuration
{
    public class CUEConfigFacade : ICUEConfigFacade
    {
        private readonly CUEConfig _cueConfig;
        /// <summary>
        /// Avoid using this function unless absolutely necessary.
        /// </summary>
        /// <returns></returns>
        public CUEConfig ToCUEConfig() => _cueConfig;

        private readonly CUERipperConfig _cueRipperConfig;

        public string Language { get => _cueConfig.language; set => _cueConfig.language = value; }

        public Dictionary<string, CUEToolsFormat> Formats { get => _cueConfig.formats; }
        public EncoderListViewModel Encoders { get => _cueConfig.Encoders; }
        public Dictionary<string, CUEToolsScript> Scripts { get => _cueConfig.scripts; }

        public string DefaultDrive { get => _cueRipperConfig.DefaultDrive; set => _cueRipperConfig.DefaultDrive = value; }
        public SerializableDictionary<string, int> DriveOffsets { get => _cueRipperConfig.DriveOffsets; }
        public SerializableDictionary<string, int> DriveC2ErrorModes { get => _cueRipperConfig.DriveC2ErrorModes; }
        public AudioEncoderType OutputCompression { get; set; } = AudioEncoderType.Lossless;
        public string DefaultLosslessFormat { get => _cueRipperConfig.DefaultLosslessFormat; set => _cueRipperConfig.DefaultLosslessFormat = value; }
        public string DefaultLossyFormat { get => _cueRipperConfig.DefaultLossyFormat; set => _cueRipperConfig.DefaultLossyFormat = value; }
        public string EncodingConfiguration { get => _cueRipperConfig.EncodingConfiguration; set => _cueRipperConfig.EncodingConfiguration = value; }

        // CTDB Options
        public string CTDBServer { get => _cueConfig.advanced.CTDBServer; set => _cueConfig.advanced.CTDBServer = value; }
        public CTDBMetadataSearch MetadataSearch { get => _cueConfig.advanced.metadataSearch; set => _cueConfig.advanced.metadataSearch = value; }
        public CUEConfigAdvanced.CTDBCoversSize CoversSize { get => _cueConfig.advanced.coversSize; set => _cueConfig.advanced.coversSize = value; }
        public CUEConfigAdvanced.CTDBCoversSearch CoversSearch { get => _cueConfig.advanced.coversSearch; set => _cueConfig.advanced.coversSearch = value; }
        public bool DetailedCTDBLog { get => _cueConfig.advanced.DetailedCTDBLog; set => _cueConfig.advanced.DetailedCTDBLog = value; }

        // Extraction options
        public bool PreserveHTOA { get => _cueConfig.preserveHTOA; set => _cueConfig.preserveHTOA = value; }
        public bool DetectGaps { get => _cueConfig.detectGaps; set => _cueConfig.detectGaps = value; }
        public bool CreateEACLog { get => _cueConfig.createEACLOG; set => _cueConfig.createEACLOG = value; }
        public bool CreateM3U { get => _cueConfig.createM3U; set => _cueConfig.createM3U = value; }
        public bool EmbedAlbumArt { get => _cueConfig.embedAlbumArt; set => _cueConfig.embedAlbumArt = value; }
        public int MaxAlbumArtSize { get => _cueConfig.maxAlbumArtSize; set => _cueConfig.maxAlbumArtSize = value; }
        public bool EjectAfterRip { get => _cueConfig.ejectAfterRip; set => _cueConfig.ejectAfterRip = value; }
        public bool DisableEjectDisc { get => _cueConfig.disableEjectDisc; set => _cueConfig.disableEjectDisc = value; }
        public string TrackFilenameFormat { get => _cueConfig.trackFilenameFormat; set => _cueConfig.trackFilenameFormat = value; }
        public bool AutomaticRip { get => _cueRipperConfig.AutomaticRip; set => _cueRipperConfig.AutomaticRip = value; }
        public bool SkipRepair { get => _cueRipperConfig.SkipRepair; set => _cueRipperConfig.SkipRepair = value; }

        // Proxy options
        public CUEConfigAdvanced.ProxyMode UseProxyMode { get => _cueConfig.advanced.UseProxyMode; set => _cueConfig.advanced.UseProxyMode = value; }
        public string ProxyServer { get => _cueConfig.advanced.ProxyServer; set => _cueConfig.advanced.ProxyServer = value; }
        public int ProxyPort { get => _cueConfig.advanced.ProxyPort; set => _cueConfig.advanced.ProxyPort = value; }
        public string ProxyUser { get => _cueConfig.advanced.ProxyUser; set => _cueConfig.advanced.ProxyUser = value; }
        public string ProxyPassword { get => _cueConfig.advanced.ProxyPassword; set => _cueConfig.advanced.ProxyPassword = value; }

        // Various options
        public string FreedbSiteAddress { get => _cueConfig.advanced.FreedbSiteAddress; set => _cueConfig.advanced.FreedbSiteAddress = value; }
        public bool CheckForUpdates { get => _cueConfig.advanced.CheckForUpdates; set => _cueConfig.advanced.CheckForUpdates = value; }

        // UI
        public int CUEStyleIndex { get; set; } = 0;
        public int SecureModeIndex { get; set; } = 0;
        public bool TestAndCopyEnabled { get; set; } = false;
        public string PathFormat { get; set; } = string.Empty;
        public List<string> PathFormatTemplates { get; set; } = [];
        public bool DetailPaneOpened { get => _cueRipperConfig.DetailPaneOpened; set => _cueRipperConfig.DetailPaneOpened = value; }

        private CUEConfigFacade(CUEConfig cueConfig, CUERipperConfig cueRipperConfig)
        {
            _cueConfig = cueConfig;
            _cueRipperConfig = cueRipperConfig;
        }

        public static CUEConfigFacade Create()
        {
            var cueConfig = new CUEConfig();
            var cueRipperConfig = new CUERipperConfig();

            AudioEncoderType outputCompression = AudioEncoderType.Lossless;
            int? cueStyleIndex = null;
            int? secureModeIndex = null;
            bool? testAndCopyEnabled = null;
            string pathFormat = Constants.DefaultPathFormats[0];
            int pathFormatTemplateCount = 0;
            List<string> pathFormatTemplates = [];

            if (!Design.IsDesignMode)
            {
                var settingsReader = new SettingsReader(Constants.ApplicationShortName, "settings.txt", Constants.ApplicationPath);
                cueConfig.Load(settingsReader);

                try
                {
                    outputCompression = (AudioEncoderType?)settingsReader.LoadInt32("OutputAudioType", null, null) ?? AudioEncoderType.Lossless;
                    cueStyleIndex = settingsReader.LoadInt32("ComboImage", int.MinValue, int.MaxValue);
                    secureModeIndex = settingsReader.LoadInt32("SecureMode", int.MinValue, int.MaxValue);
                    testAndCopyEnabled = settingsReader.LoadBoolean("TestAndCopy");

                    pathFormat = settingsReader.Load("PathFormat") ?? Constants.DefaultPathFormats[0];
                    pathFormatTemplateCount = settingsReader.LoadInt32("OutputPathUseTemplates", 0, Constants.MaxPathFormats) ?? 0;
                    for(int i = 0; i < pathFormatTemplateCount; ++i)
                    {
                        var template = settingsReader.Load($"OutputPathUseTemplate{i}") ?? string.Empty;
                        pathFormatTemplates.Add(template);
                    }

                    using TextReader reader = new StringReader(settingsReader.Load("CUERipper"));
                    if (CUERipperConfig.serializer.Deserialize(reader) is CUERipperConfig ripperConfig) cueRipperConfig = ripperConfig;
                }
                catch (Exception)
                {
                    // Do nothing...
                }
            }

            var config = new CUEConfigFacade(cueConfig, cueRipperConfig);
            config.OutputCompression = outputCompression;
            if (cueStyleIndex != null) config.CUEStyleIndex = cueStyleIndex.Value;
            if (secureModeIndex != null) config.SecureModeIndex = secureModeIndex.Value;
            if (testAndCopyEnabled != null) config.TestAndCopyEnabled = testAndCopyEnabled.Value;
            config.PathFormat = pathFormat;
            config.PathFormatTemplates = pathFormatTemplates;

            return config;
        }

        private readonly static XmlSerializerNamespaces xmlEmptyNamespaces = new([XmlQualifiedName.Empty]);
        private readonly static XmlWriterSettings xmlEmptySettings = new() { Indent = true, OmitXmlDeclaration = true };
        public void Save()
        {
            var sw = new SettingsWriter(Constants.ApplicationShortName, "settings.txt", Constants.ApplicationPath);
            _cueConfig.Save(sw);

            sw.Save("OutputAudioType", (int)OutputCompression);
            sw.Save("ComboImage", CUEStyleIndex);
            sw.Save("SecureMode", SecureModeIndex);
            sw.Save("TestAndCopy", TestAndCopyEnabled);
            //           sw.Save("WidthIncrement", SizeIncrement.Width);
            //           sw.Save("HeightIncrement", SizeIncrement.Height);

            sw.Save("PathFormat", PathFormat);
            sw.Save("OutputPathUseTemplates", PathFormatTemplates.Count);
            for (int i = 0; i < PathFormatTemplates.Count; ++i)
            {
                sw.Save($"OutputPathUseTemplate{i}", PathFormatTemplates[i]);
            }

            using TextWriter tw = new StringWriter();
            using XmlWriter xw = XmlTextWriter.Create(tw, xmlEmptySettings);

            CUERipperConfig.serializer.Serialize(xw, _cueRipperConfig, xmlEmptyNamespaces);
            sw.SaveText("CUERipper", tw.ToString());

            sw.Close();
        }
    }
}
