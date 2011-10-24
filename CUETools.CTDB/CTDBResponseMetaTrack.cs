using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaTrack
    {
        [XmlAttribute]
        public string name { get; set; }
        [XmlAttribute]
        public string artist { get; set; }
        [XmlAttribute]
        public string extra { get; set; }
    }
}
