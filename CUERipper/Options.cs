using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB;
using Microsoft.Win32;
using System.Runtime.Serialization.Formatters.Binary;
using CUETools.Processor;

namespace CUERipper
{
    public partial class Options : Form
    {
        private CUEConfig config;

        public Options(CUEConfig config)
        {
            this.config = config;
            this.InitializeComponent();
        }

        private void Options_Load(object sender, EventArgs e)
        {
            this.propertyGrid1.SelectedObject = new CUERipperSettings(this.config);
        }
    }

    public class CUERipperSettings
    {
        private CUEConfig config;

        public CUERipperSettings(CUEConfig config)
        {
            this.config = config;
        }

        [DefaultValue("db.cuetools.net"), Category("CTDB"), DisplayName("CTDB Server")]
        public string CTDBServer { get { return config.advanced.CTDBServer; } set { config.advanced.CTDBServer = value; } }

        [DefaultValue(CUETools.Processor.CUEConfigAdvanced.ProxyMode.System), Category("Proxy"), DisplayName("Proxy mode")]
        public CUETools.Processor.CUEConfigAdvanced.ProxyMode UseProxyMode { get { return config.advanced.UseProxyMode; } set { config.advanced.UseProxyMode = value; } }

        [DefaultValue("127.0.0.1"), Category("Proxy"), DisplayName("Proxy server host")]
        public string ProxyServer { get { return config.advanced.ProxyServer; } set { config.advanced.ProxyServer = value; } }

        [DefaultValue(8080), Category("Proxy"), DisplayName("Proxy server port")]
        public int ProxyPort { get { return config.advanced.ProxyPort; } set { config.advanced.ProxyPort = value; } }

        [DefaultValue(""), Category("Proxy"), DisplayName("Proxy auth user")]
        public string ProxyUser { get { return config.advanced.ProxyUser; } set { config.advanced.ProxyUser = value; } }

        [DefaultValue(""), Category("Proxy"), DisplayName("Proxy auth password")]
        public string ProxyPassword { get { return config.advanced.ProxyPassword; } set { config.advanced.ProxyPassword = value; } }

        [DefaultValue(true), Category("Extraction"), DisplayName("Preserve HTOA")]
        public bool preserveHTOA { get { return config.preserveHTOA; } set { config.preserveHTOA = value; } }

        [DefaultValue(true), Category("Extraction"), DisplayName("Detect Indexes")]
        public bool detectGaps { get { return config.detectGaps; } set { config.detectGaps = value; } }

        [DefaultValue(true), Category("Extraction"), DisplayName("EAC log style")]
        public bool createEACLOG { get { return config.createEACLOG; } set { config.createEACLOG = value; } }

        [DefaultValue(false), Category("Extraction"), DisplayName("Create M3U playlist")]
        public bool createM3U { get { return config.createM3U; } set { config.createM3U = value; } }

        [DefaultValue(true), Category("Extraction"), DisplayName("Embed album art")]
        public bool embedAlbumArt { get { return config.embedAlbumArt; } set { config.embedAlbumArt = value; } }

        [DefaultValue("%tracknumber%. %title%"), Category("Extraction"), DisplayName("Track filename")]
        public string trackFilenameFormat { get { return config.trackFilenameFormat; } set { config.trackFilenameFormat = value; } }

        [DefaultValue(CUETools.CTDB.CTDBMetadataSearch.Default), Category("CTDB"), DisplayName("Metadata search")]
        public CUETools.CTDB.CTDBMetadataSearch metadataSearch { get { return config.advanced.metadataSearch; } set { config.advanced.metadataSearch = value; } }

        [DefaultValue(CUETools.Processor.CUEConfigAdvanced.CTDBCoversSearch.Small), Category("CTDB"), DisplayName("Album art search")]
        public CUETools.Processor.CUEConfigAdvanced.CTDBCoversSearch coversSearch { get { return config.advanced.coversSearch; } set { config.advanced.coversSearch = value; } }

        [DefaultValue(false), Category("CTDB"), DisplayName("Detailed log")]
        public bool DetailedCTDBLog { get { return config.advanced.DetailedCTDBLog; } set { config.advanced.DetailedCTDBLog = value; } }
    }
}
