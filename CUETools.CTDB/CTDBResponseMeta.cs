using System;
using System.Xml.Serialization;

namespace CUETools.CTDB
{
    [Serializable]
    public class CTDBResponseMeta
    {
        public CTDBResponseMeta()
        {
        }

        public CTDBResponseMeta(CTDBResponseMeta src)
        {
            this.source = src.source;
            this.id = src.id;
            this.artist = src.artist;
            this.album = src.album;
            this.year = src.year;
            this.genre = src.genre;
            this.extra = src.extra;
            this.discnumber = src.discnumber;
            this.disccount = src.disccount;
            this.discname = src.discname;
            this.infourl = src.infourl;
            this.barcode = src.barcode;
            if (src.coverart != null)
            {
                this.coverart = new CTDBResponseMetaImage[src.coverart.Length];
                for (int i = 0; i < src.coverart.Length; i++)
                    this.coverart[i] = new CTDBResponseMetaImage(src.coverart[i]);
            }

            if (src.track != null)
            {
                this.track = new CTDBResponseMetaTrack[src.track.Length];
                for (int i = 0; i < src.track.Length; i++)
                    this.track[i] = new CTDBResponseMetaTrack(src.track[i]);
            }

            if (src.label != null)
            {
                this.label = new CTDBResponseMetaLabel[src.label.Length];
                for (int i = 0; i < src.label.Length; i++)
                    this.label[i] = new CTDBResponseMetaLabel(src.label[i]);
            }

            if (src.release != null)
            {
                this.release = new CTDBResponseMetaRelease[src.release.Length];
                for (int i = 0; i < src.release.Length; i++)
                    this.release[i] = new CTDBResponseMetaRelease(src.release[i]);
            }
        }

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
		[XmlElement]
        public string extra { get; set; }
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
        [XmlElement]
        public CTDBResponseMetaRelease[] release;
    }
}
