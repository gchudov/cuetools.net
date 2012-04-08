using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaTrack
    {
        public CTDBResponseMetaTrack()
        {
        }

        public CTDBResponseMetaTrack(CTDBResponseMetaTrack src)
        {
            this.name = src.name;
            this.artist = src.artist;
            this.extra = src.extra;
        }

        [XmlAttribute]
        public string name { get; set; }
        [XmlAttribute]
        public string artist { get; set; }
        [XmlAttribute]
        public string extra { get; set; }
    }
}
