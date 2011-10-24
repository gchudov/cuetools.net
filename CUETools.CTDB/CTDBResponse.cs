using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    [XmlRoot(ElementName = "ctdb", Namespace = "http://db.cuetools.net/ns/mmd-1.0#")]
    public class CTDBResponse
    {
        [XmlElement]
        public CTDBResponseEntry[] entry;
        [XmlElement]
        public CTDBResponseMeta[] musicbrainz;
    }
}
