using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUETools.CDImage;

namespace CUETools.Processor
{
	public class CueTrackMetaTOCWrapper
	{
		private readonly CUETrackMetadata _trackMetaData;
		private readonly CDTrack _cdTrack;

		public CueTrackMetaTOCWrapper(CUETrackMetadata trackMetaData, CDTrack cdTrack)
		{
			_trackMetaData = trackMetaData;
			_cdTrack = cdTrack;
		}

		public string Artist { get => _trackMetaData.Artist ; set => _trackMetaData.Artist = value;}
		public string Title { get => _trackMetaData.Title; set => _trackMetaData.Title = value; }
		public string Composer { get => _trackMetaData.Composer; set => _trackMetaData.Composer = value; }
		public string Lyricist { get => _trackMetaData.Lyricist; set => _trackMetaData.Lyricist = value; }
		public string Comment { get => _trackMetaData.Comment; set => _trackMetaData.Comment = value; }
		//
		//public uint Start { get => _cdTrack.Start; }
		public string Start { get => _cdTrack.StartMSF; }
		//public uint Length { get => _cdTrack.Length; }
		public string Length { get => _cdTrack.LengthMSF; }
		public string ISRC { get => _cdTrack.ISRC; }
		//public uint End { get => _cdTrack.End; }
		public string End { get => _cdTrack.EndMSF; }
		public uint Number { get => _cdTrack.Number; }
		public uint Pregap { get => _cdTrack.Pregap; }
//		public CDTrackIndex this[int key]
//		public uint LastIndex
		public bool IsAudio { get => _cdTrack.IsAudio; }
		public bool PreEmphasis { get => _cdTrack.PreEmphasis; }
		public bool DCP { get => _cdTrack.DCP; }
	}

}
