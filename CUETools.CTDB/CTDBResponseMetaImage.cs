using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaImage
    {
        public CTDBResponseMetaImage()
        {
        }

        public CTDBResponseMetaImage(CTDBResponseMetaImage src)
        {
            this.uri = src.uri;
            this.uri150 = src.uri150;
            this.height = src.height;
            this.width = src.width;
            this.primary = src.primary;
        }

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
