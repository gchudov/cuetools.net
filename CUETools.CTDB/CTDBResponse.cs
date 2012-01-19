using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    [XmlRoot(ElementName = "ctdb", Namespace = "http://db.cuetools.net/ns/mmd-1.0#")]
    public class CTDBResponse
    {
        [XmlIgnore]
        public bool ParityNeeded
        {
            get
            {
                return this.status == "parity needed";
            }
        }

        [XmlAttribute]
        public string status { get; set; }

        [XmlAttribute]
        public string updateurl { get; set; }

        [XmlAttribute]
        public string updatemsg { get; set; }

        [XmlAttribute]
        public string message { get; set; }

        [XmlAttribute]
        public int npar { get; set; }

        [XmlElement]
        public CTDBResponseEntry[] entry;

        [XmlElement]
        public CTDBResponseMeta[] metadata;
    }
}
