using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaLabel
    {
        [XmlAttribute]
        public string name { get; set; }
        [XmlAttribute]
        public string catno { get; set; }
    }
}
