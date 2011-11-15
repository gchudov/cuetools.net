using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace CUETools.CDImage
{
	public class CDTrackIndex
	{
		public CDTrackIndex(uint index, uint start)
		{
			_start = start;
			_index = index;
		}

		public CDTrackIndex(CDTrackIndex src)
		{
			_start = src._start;
			_index = src._index;
		}

		public uint Start
		{
			get
			{
				return _start;
			}
			set
			{
				_start = value;
			}
		}

		public uint Index
		{
			get
			{
				return _index;
			}
		}

		public string MSF
		{
			get
			{
				return CDImageLayout.TimeToString(_start);
			}
		}

		uint _start, _index;
	}

	public class CDTrack : ICloneable
	{
		public CDTrack(uint number, uint start, uint length, bool isAudio, bool preEmpasis)
		{
			_number = number;
			_start = start;
			_length = length;
			_isAudio = isAudio;
			_preEmphasis = preEmpasis;
			_indexes = new List<CDTrackIndex>();
			_indexes.Add(new CDTrackIndex(0, start));
			_indexes.Add(new CDTrackIndex(1, start));
		}

		public CDTrack(CDTrack src)
		{
			_number = src._number;
			_start = src._start;
			_length = src._length;
			_isAudio = src._isAudio;
			_preEmphasis = src._preEmphasis;
			_dcp = src._dcp;
			_isrc = src._isrc;
			_indexes = new List<CDTrackIndex>();
			for (int i = 0; i < src._indexes.Count; i++)
				_indexes.Add(new CDTrackIndex(src._indexes[i]));
		}

		public object Clone()
		{
			return new CDTrack(this);
		}

		public uint Start
		{
			get
			{
				return _start;
			}
			set
			{
				_start = value;
			}
		}

		public string StartMSF
		{
			get
			{
				return CDImageLayout.TimeToString(_start);
			}
		}

		public uint Length
		{
			get
			{
				return _length;
			}
			set
			{
				_length = value;
			}
		}

		public string LengthMSF
		{
			get
			{
				return CDImageLayout.TimeToString(_length);
			}
		}

		public string ISRC
		{
			get
			{
				return _isrc;
			}
			set
			{
				_isrc = value;
			}
		}

		public uint End
		{
			get
			{
				return _start + _length - 1;
			}
		}

		public string EndMSF
		{
			get
			{
				return CDImageLayout.TimeToString(End);
			}
		}

		public uint Number
		{
			get
			{
				return _number;
			}
			internal set
			{
				_number = value;
			}
		}

		public uint Pregap
		{
			get
			{
				return _start - _indexes[0].Start;
			}
			set
			{
				_indexes[0].Start = _start - value;
			}
		}

		public CDTrackIndex this[int key]
		{
			get
			{
				return _indexes[key];
			}
		}

		public uint LastIndex
		{
			get
			{
				return (uint) _indexes.Count - 1;
			}
		}

		public bool IsAudio
		{
			get
			{
				return _isAudio;
			}
			set
			{
				_isAudio = value;
			}
		}

		public bool PreEmphasis
		{
			get
			{
				return _preEmphasis;
			}
			set
			{
				_preEmphasis = value;
			}
		}

		public bool DCP
		{
			get
			{
				return _dcp;
			}
			set
			{
				_dcp = value;
			}
		}

		public void AddIndex(CDTrackIndex index)
		{
			if (index.Index < 2)
				_indexes[(int)index.Index] = index;
			else
				_indexes.Add(index);
		}

		IList<CDTrackIndex> _indexes;
		string _isrc;
		bool _isAudio;
		bool _preEmphasis, _dcp;
		uint _start;
		uint _length;
		uint _number;
	}

	public class CDImageLayout : ICloneable
	{
		public CDImageLayout()
		{
			_tracks = new List<CDTrack>();
		}

		public CDImageLayout(CDImageLayout src)
		{
			_barcode = src._barcode;
			_audioTracks = src._audioTracks;
			_firstAudio = src._firstAudio;
			_tracks = new List<CDTrack>();
			for (int i = 0; i < src.TrackCount; i++)
				_tracks.Add(new CDTrack(src._tracks[i]));
		}

		public CDImageLayout(string trackoffsets)
			: this(trackoffsets.Split(' ').Length - 1, trackoffsets.Split(' ').Length - 1, 1, trackoffsets)
		{
		}

		public CDImageLayout(int trackcount, int audiotracks, int firstaudio, string trackoffsets)
		{
			_audioTracks = audiotracks;
			_firstAudio = firstaudio - 1;
			_tracks = new List<CDTrack>();
			string[] n = trackoffsets.Split(' ');
			if (n.Length != trackcount + 1)
				throw new Exception("Invalid trackoffsets.");
			for (int i = 0; i < trackcount; i++)
			{
				uint len = uint.Parse(n[i + 1]) - uint.Parse(n[i]) -
					((i + 1 < _firstAudio + _audioTracks || i + 1 == trackcount) ? 0U : 152U * 75U);
				bool isaudio = i >= _firstAudio && i < _firstAudio + _audioTracks;
				_tracks.Add(new CDTrack((uint)i + 1, uint.Parse(n[i]), len, isaudio, false));
			}
			_tracks[0][0].Start = 0;
			if (TrackOffsets != trackoffsets)
				throw new Exception("TrackOffsets != trackoffsets");
		}

		public object Clone()
		{
			return new CDImageLayout(this);
		}

		public uint Length
		{
			get
			{
				return TrackCount > 0 ? _tracks[TrackCount - 1].End + 1U : 0U;
			}
		}

		public CDTrack this[int key]
		{
			get
			{
				return _tracks[key - 1];
			}
		}

		public int TrackCount
		{
			get
			{
				return _tracks.Count;
			}
		}

		public uint Pregap
		{
			get
			{
				return _tracks[0].Pregap;
			}
		}

		public uint AudioTracks
		{
			get
			{
				return (uint) _audioTracks;
			}

			set
			{
				_audioTracks = (int) value;
			}
		}

		public int FirstAudio
		{
			get
			{
				return _firstAudio + 1;
			}
			set
			{
				_firstAudio = value - 1;
			}
		}

		public int LastAudio
		{
			get
			{
				return _audioTracks + _firstAudio;
			}
		}

		public uint Leadout
		{
			get
			{
				return _tracks[_firstAudio][0].Start + AudioLength;
			}
		}

		public uint AudioLength
		{
			get
			{
				return AudioTracks > 0 ? _tracks[_firstAudio + _audioTracks - 1].End + 1U - _tracks[_firstAudio].Start + _tracks[_firstAudio].Pregap : 0U;
			}
		}

		public string Barcode
		{
			get
			{
				return _barcode;
			}
			set
			{
				_barcode = value;
			}
		}

		public string MusicBrainzTOC
		{
			get
			{
				StringBuilder mbSB = new StringBuilder();
				mbSB.AppendFormat("{0} {1} {2}", 1, LastAudio, _tracks[LastAudio - 1].End + 1 + 150);
				for (int iTrack = 0; iTrack < LastAudio; iTrack++)
					mbSB.AppendFormat(" {0}", _tracks[iTrack].Start + 150);
				return mbSB.ToString();
			}
		}

		public string MusicBrainzId
		{
			get
			{
				StringBuilder mbSB = new StringBuilder();
				mbSB.AppendFormat("{0:X2}{1:X2}", 1, LastAudio);
				mbSB.AppendFormat("{0:X8}", _tracks[LastAudio - 1].End + 1 + 150);
				for (int iTrack = 0; iTrack < LastAudio ; iTrack++)
					mbSB.AppendFormat("{0:X8}", _tracks[iTrack].Start + 150);
				mbSB.Append(new string('0', (99 - LastAudio) * 8));
				byte[] hashBytes = (new SHA1CryptoServiceProvider()).ComputeHash(Encoding.ASCII.GetBytes(mbSB.ToString()));
				return Convert.ToBase64String(hashBytes).Replace('+', '.').Replace('/', '_').Replace('=', '-');
			}
		}

		public string TrackOffsets
		{
			get
			{
				StringBuilder mbSB = new StringBuilder();
				for (int iTrack = 0; iTrack < TrackCount; iTrack++)
					mbSB.AppendFormat("{0} ", _tracks[iTrack].Start);
				mbSB.AppendFormat("{0}", Length);
				return mbSB.ToString();
			}
		}

		public string TOCID
		{
			get
			{
				StringBuilder mbSB = new StringBuilder();
				for (int iTrack = 1; iTrack < AudioTracks; iTrack++)
					mbSB.AppendFormat("{0:X8}", _tracks[_firstAudio + iTrack].Start - _tracks[_firstAudio].Start);
				mbSB.AppendFormat("{0:X8}", _tracks[_firstAudio + (int)AudioTracks - 1].End + 1 - _tracks[_firstAudio].Start);
				mbSB.Append(new string('0', (100 - (int)AudioTracks) * 8));
				byte[] hashBytes = (new SHA1CryptoServiceProvider()).ComputeHash(Encoding.ASCII.GetBytes(mbSB.ToString()));
				return Convert.ToBase64String(hashBytes).Replace('+', '.').Replace('/', '_').Replace('=', '-');
			}
		}

		public static CDImageLayout FromString(string str)
		{
			var ids = str.Split(':');
			int firstaudio = 1;
			int audiotracks = 0;
			int trackcount = ids.Length - 1;
			while (firstaudio < ids.Length && ids[firstaudio - 1][0] == '-')
				firstaudio ++;
            while (firstaudio + audiotracks < ids.Length && ids[firstaudio + audiotracks - 1][0] != '-')
				audiotracks ++;
			for (var i = 0; i < ids.Length; i++)
				if (ids[i][0] == '-')
					ids[i] = ids[i].Substring(1);
			return new CDImageLayout(trackcount, audiotracks, firstaudio, string.Join(" ", ids));
		}

        public override string ToString()
		{
			StringBuilder mbSB = new StringBuilder();
			for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				mbSB.AppendFormat("{0}{1}:", _tracks[iTrack].IsAudio ? "" : "-", _tracks[iTrack].Start);
			mbSB.AppendFormat("{0}", Length);
			return mbSB.ToString();
		}

		public void InsertTrack(CDTrack track)
		{
			_tracks.Insert((int)track.Number - 1, track);
			for (int i = (int)track.Number; i < _tracks.Count; i++)
				_tracks[i].Number++;
			if (track.IsAudio)
				_audioTracks++;
			if (!track.IsAudio && track.Number <= FirstAudio)
				_firstAudio++;
		}

		public void AddTrack(CDTrack track)
		{
			_tracks.Add(track);
			if (track.IsAudio)
			{
				_audioTracks++;
				if (!_tracks[_firstAudio].IsAudio)
					_firstAudio = _tracks.Count - 1;
			}
		}

		public uint IndexLength(int iTrack, int iIndex)
		{
			if (iIndex < _tracks[iTrack - 1].LastIndex)
				return _tracks[iTrack - 1][iIndex + 1].Start - _tracks[iTrack - 1][iIndex].Start;
			if (iTrack < TrackCount && _tracks[iTrack].IsAudio)
				return _tracks[iTrack][0].Start - _tracks[iTrack - 1][iIndex].Start;
			return _tracks[iTrack - 1].End + 1 - _tracks[iTrack - 1][iIndex].Start;
		}

		public static int TimeFromString(string s)
		{
			string[] n = s.Split(':');
			if (n.Length != 3)
			{
				throw new Exception("Invalid timestamp.");
			}
			int min, sec, frame;

			min = Int32.Parse(n[0]);
			sec = Int32.Parse(n[1]);
			frame = Int32.Parse(n[2]);

			return frame + (sec * 75) + (min * 60 * 75);
		}

		public static string TimeToString(string format, uint t)
		{
			uint min, sec, frame;

			frame = t % 75;
			t /= 75;
			sec = t % 60;
			t /= 60;
			min = t;

			return String.Format(format, min, sec, frame);
		}

		public static string TimeToString(uint t)
		{
			return TimeToString("{0:00}:{1:00}:{2:00}", t);
		}

		string _barcode;
		IList<CDTrack> _tracks;
		int _audioTracks;
		int _firstAudio;
	}
}
