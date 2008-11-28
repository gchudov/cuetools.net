using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.CDImage
{
	public class CDTrackIndex
	{
		public CDTrackIndex(uint index, uint start)
		{
			_start = start;
			_index = index;
			_length = 0;
		}

		public CDTrackIndex(uint index, uint start, uint length)
		{
			_length = length;
			_start = start;
			_index = index;
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

		uint _start, _length, _index;
	}

	public class CDTrack
	{
		public CDTrack(uint number, uint start, uint length, bool isAudio)
		{
			_number = number;
			_start = start;
			_length = length;
			_isAudio = isAudio;
			_indexes = new List<CDTrackIndex>();
			_indexes.Add(new CDTrackIndex(0, start, 0));
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
				return _indexes[0].Length;
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

		public void AddIndex(CDTrackIndex index)
		{
			if (index.Index == 0)
				_indexes[0] = index;
			else
				_indexes.Add(index);
		}

		IList<CDTrackIndex> _indexes;
		string _isrc;
		bool _isAudio;
		uint _start;
		uint _length;
		uint _number;
	}

	public class CDImageLayout
	{
		public CDImageLayout(uint length)
		{
			_tracks = new List<CDTrack>();
			_length = length;
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

		public String Catalog
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

		public void AddTrack(CDTrack track)
		{
			_tracks.Add(track);
			if (track.IsAudio)
				_audioTracks++;
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

		public string _cddbId;
		public string _ArId;

		uint _length;
		string _catalog;
		IList<CDTrack> _tracks;
		uint _audioTracks;
	}
}
