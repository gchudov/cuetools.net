using CUETools.Codecs;
using System;
using System.ComponentModel;

namespace CUETools.Processor
{
    [Serializable]
    public class CUEConfigAdvanced : CUEToolsCodecsConfig
    {
        public enum ProxyMode
        {
            None,
            System,
            Custom
        }

        public enum CTDBCoversSize
        {
            Small,
            Large
        }

        public enum CTDBCoversSearch
        {
            None,
            Primary,
            Extensive
        }

        public CUEConfigAdvanced()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.ResetValue(this);
            }
        }

        public CUEConfigAdvanced(CUEConfigAdvanced src)
            : base(src)
        {
            // Iterate through each property and call SetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.SetValue(this, property.GetValue(src));
            }
        }

        public void Init()
        {
            Init(CUEProcessorPlugins.encs, CUEProcessorPlugins.decs);
        }

        [DefaultValue("i"), Category("Freedb"), DisplayName("Email user")]
        public string FreedbUser { get; set; }

        [DefaultValue("wont.tell"), Category("Freedb"), DisplayName("Email domain")]
        public string FreedbDomain { get; set; }

        [DefaultValue(ProxyMode.System), Category("Proxy"), DisplayName("Proxy mode")]
        public ProxyMode UseProxyMode { get; set; }

        [DefaultValue("127.0.0.1"), Category("Proxy"), DisplayName("Proxy server host")]
        public string ProxyServer { get; set; }

        [DefaultValue(8080), Category("Proxy"), DisplayName("Proxy server port")]
        public int ProxyPort { get; set; }

        [DefaultValue(""), Category("Proxy"), DisplayName("Proxy auth user")]
        public string ProxyUser { get; set; }

        [DefaultValue(""), Category("Proxy"), DisplayName("Proxy auth password")]
        public string ProxyPassword { get; set; }

        [DefaultValue(true), Category("Tagging"), DisplayName("Cache metadata")]
        public bool CacheMetadata { get; set; }

        [DefaultValue("folder.jpg;cover.jpg;albumart.jpg;thumbnail.jpg;albumartlarge.jpg;front.jpg;%album%.jpg")]
        [Category("Cover Art"), DisplayName("Cover Art Files")]
        public string CoverArtFiles { get; set; }

        [DefaultValue(true)]
        [Category("Cover Art"), DisplayName("Cover Art Extended Search")]
        public bool CoverArtSearchSubdirs { get; set; }

        [DefaultValue(false)]
        [DisplayName("Create TOC files")]
        public bool CreateTOC { get; set; }

        [DefaultValue(true), Category("CTDB"), DisplayName("Submit to CTDB")]
        public bool CTDBSubmit { get; set; }

        [DefaultValue(true), Category("CTDB"), DisplayName("Ask before submitting")]
        public bool CTDBAsk { get; set; }

        [DefaultValue("db.cuetools.net"), Category("CTDB"), DisplayName("CTDB Server")]
        public string CTDBServer { get; set; }

        [DefaultValue(CUETools.CTDB.CTDBMetadataSearch.Default), Category("CTDB"), DisplayName("Metadata search")]
        public CUETools.CTDB.CTDBMetadataSearch metadataSearch { get; set; }

        [DefaultValue(CTDBCoversSize.Large), Category("CTDB"), DisplayName("Album art size")]
        public CTDBCoversSize coversSize { get; set; }

        [DefaultValue(CTDBCoversSearch.Primary), Category("CTDB"), DisplayName("Album art search")]
        public CTDBCoversSearch coversSearch { get; set; }

        [DefaultValue(false), Category("CTDB"), DisplayName("Detailed log")]
        public bool DetailedCTDBLog { get; set; }

        [DefaultValue(true), Category("Tagging"), DisplayName("Write CTDB tags on encode")]
        public bool WriteCTDBTagsOnEncode { get; set; }

        [DefaultValue(false), Category("Tagging"), DisplayName("Write CTDB tags on verify")]
        public bool WriteCTDBTagsOnVerify { get; set; }

        [DefaultValue(false), Category("Tagging"), DisplayName("Use id3v2.4 instead of id3v2.3")]
        public bool UseId3v24 { get; set; }

        [DefaultValue(true), Category("Tagging"), DisplayName("Write CDTOC tag")]
        public bool WriteCDTOCTag { get; set; }
    }
}
