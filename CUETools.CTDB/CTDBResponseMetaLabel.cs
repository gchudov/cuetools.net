using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaLabel
    {
        public CTDBResponseMetaLabel()
        {
        }

        public CTDBResponseMetaLabel(CTDBResponseMetaLabel src)
        {
            this.name = src.name;
            this.catno = src.catno;
        }

        [XmlAttribute]
        public string name { get; set; }
        [XmlAttribute]
        public string catno { get; set; }
    }
}
