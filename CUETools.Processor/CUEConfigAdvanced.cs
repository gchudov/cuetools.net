using System;
using System.ComponentModel;
using System.Xml.Serialization;

namespace CUETools.Processor
{
    [Serializable]
    public class CUEConfigAdvanced
    {
        public enum ProxyMode
        {
            None,
            System,
            Custom
        }

        public CUEConfigAdvanced()
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(this))
            {
                property.ResetValue(this);
            }
        }

        internal static XmlSerializer serializer = new XmlSerializer(typeof(CUEConfigAdvanced));
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

        [DefaultValue(true), Category("Cache"), DisplayName("Cache metadata")]
        public bool CacheMetadata { get; set; }

        [DefaultValue(new string[] { "folder.jpg", "cover.jpg", "albumart.jpg", "thumbnail.jpg", "albumartlarge.jpg", "front.jpg", "%album%.jpg" })]
        [Category("Cover Art"), DisplayName("Cover Art Files")]
        public string[] CoverArtFiles { get; set; }

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
    }
}
