using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMetaRelease
    {
        public CTDBResponseMetaRelease()
        {
        }

        public CTDBResponseMetaRelease(CTDBResponseMetaRelease src)
        {
            this.date = src.date;
            this.country = src.country;
        }

        [XmlAttribute]
        public string date { get; set; }
        [XmlAttribute]
        public string country { get; set; }
    }
}
