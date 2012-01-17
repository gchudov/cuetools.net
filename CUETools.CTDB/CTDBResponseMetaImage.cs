using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaImage
    {
        [XmlAttribute]
        public string uri { get; set; }
        [XmlAttribute]
        public string uri150 { get; set; }
        [XmlAttribute]
        public int height { get; set; }
        [XmlAttribute]
        public int width { get; set; }
        [XmlAttribute]
        public bool primary { get; set; }
    }
}
