using System.ComponentModel;

namespace CUETools.Processor
{
    public class CUETrackMetadata
    {
        [DefaultValue("")]
        public string Artist { get; set; }
        [DefaultValue("")]
        public string Title { get; set; }
		[DefaultValue("")]
		public string Composer { get; set; }
		[DefaultValue("")]
		public string Lyricist { get; set; }
		[DefaultValue("")]
		public string Comment { get; set; }
		[DefaultValue("")]
        public string ISRC { get; set; }

        public CUETrackMetadata()
        {
            Artist = "";
            Title = "";
			Composer = "";
			Lyricist = "";
			Comment = "";
			ISRC = "";

		}
    }
}
