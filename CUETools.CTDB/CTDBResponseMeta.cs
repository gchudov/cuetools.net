using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMeta
    {
        [XmlAttribute]
        public string source { get; set; }
        [XmlAttribute]
        public string id { get; set; }
        [XmlAttribute]
        public string artist { get; set; }
        [XmlAttribute]
        public string album { get; set; }
        [XmlAttribute]
        public string year { get; set; }
        [XmlAttribute]
        public string genre { get; set; }
        [XmlAttribute]
        public string extra { get; set; }
        [XmlAttribute]
        public string country { get; set; }
        [XmlAttribute]
        public string releasedate { get; set; }
        [XmlAttribute]
        public string discnumber { get; set; }
        [XmlAttribute]
        public string disccount { get; set; }
        [XmlAttribute]
        public string discname { get; set; }
        [XmlAttribute]
        public string infourl { get; set; }
        [XmlAttribute]
        public string barcode { get; set; }
        [XmlElement]
        public CTDBResponseMetaImage[] coverart;
        [XmlElement]
        public CTDBResponseMetaTrack[] track;
        [XmlElement]
        public CTDBResponseMetaLabel[] label;
    }
}
