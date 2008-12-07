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
		}

		public uint Pregap
		{
			get
			{
				return _start - _indexes[0].Start;
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
		bool _preEmphasis;
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
			_catalog = src._catalog;
			_audioTracks = src._audioTracks;
			_tracks = new List<CDTrack>();
			for (int i = 0; i < src.TrackCount; i++)
				_tracks.Add(new CDTrack(src._tracks[i]));
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
				return _audioTracks;
			}
		}

		public uint AudioLength
		{
			get
			{
				return AudioTracks > 0 ? _tracks[(int)_audioTracks - 1].End + 1U : 0U;
			}
		}

		public string Catalog
		{
			get
			{
				return _catalog;
			}
			set
			{
				_catalog = value;
			}
		}

		public string MusicBrainzId
		{
			get
			{
				StringBuilder mbSB = new StringBuilder();
				mbSB.AppendFormat("{0:X2}{1:X2}{2:X8}", 1, AudioTracks, _tracks[(int)AudioTracks-1].End + 1 + 150);
				for (int iTrack = 0; iTrack < AudioTracks; iTrack++)
					mbSB.AppendFormat("{0:X8}", _tracks[iTrack].Start + 150);
				mbSB.Append(new string('0', (99 - (int)AudioTracks) * 8));
				byte[] hashBytes = (new SHA1CryptoServiceProvider()).ComputeHash(Encoding.ASCII.GetBytes(mbSB.ToString()));
				return Convert.ToBase64String(hashBytes).Replace('+', '.').Replace('/', '_').Replace('=', '-');
			}
		}

		public void AddTrack(CDTrack track)
		{
			_tracks.Add(track);
			if (track.IsAudio)
				_audioTracks++;
		}

		public uint IndexLength(int iTrack, int iIndex)
		{
			if (iIndex < _tracks[iTrack - 1].LastIndex)
				return _tracks[iTrack - 1][iIndex + 1].Start - _tracks[iTrack - 1][iIndex].Start;
			if (iTrack < AudioTracks)
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

		public static string TimeToString(uint t)
		{
			uint min, sec, frame;

			frame = t % 75;
			t /= 75;
			sec = t % 60;
			t /= 60;
			min = t;

			return String.Format("{0:00}:{1:00}:{2:00}", min, sec, frame);
		}

		string _catalog;
		IList<CDTrack> _tracks;
		uint _audioTracks;
	}
}
