using CUETools.CDImage;

namespace CUETools.Processor
{
    public class CUEMetadataEntry
    {
        public CUEMetadata metadata { get; set; }
        public CDImageLayout TOC { get; set; }
        public string ImageKey { get; set; }
        public byte[] cover { get; set; }

        public CUEMetadataEntry(CUEMetadata metadata, CDImageLayout TOC, string key)
        {
            this.metadata = new CUEMetadata(metadata);
            this.TOC = TOC;
            this.ImageKey = key;
        }

        public CUEMetadataEntry(CDImageLayout TOC, string key)
            : this(new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks), TOC, key)
        {
        }

        public override string ToString()
        {
            return string.Format("{0}{1} - {2}{3}{4}", metadata.Year != "" ? metadata.Year + ": " : "",
                metadata.Artist == "" ? "Unknown Artist" : metadata.Artist,
                metadata.Title == "" ? "Unknown Title" : metadata.Title,
                metadata.DiscNumberAndName == "" ? "" : " (disc " + metadata.DiscNumberAndName + ")",
                metadata.ReleaseDateAndLabel == "" ? "" : " (" + metadata.ReleaseDateAndLabel + ")");
        }
    }
}
