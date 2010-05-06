using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using CUETools.CDImage;

namespace CUETools.Processor
{
	[Serializable]
	public class CUEMetadata
	{
		public CUEMetadata()
		{
			TotalDiscs = "";
			DiscNumber = "";
			Year = "";
			Genre = "";
			Artist = "";
			Title = "";
			Catalog = "";
			Tracks = new List<CUETrackMetadata>();
		}

		public CUEMetadata(CUEMetadata src)
			: this(src.Id, src.Tracks.Count)
		{
			CopyMetadata(src);
		}

		public CUEMetadata(string id, int AudioTracks)
			: this()
		{
			Id = id;
			for (int i = 0; i < AudioTracks; i++)
				Tracks.Add(new CUETrackMetadata());
		}

		private static XmlSerializer serializer = new XmlSerializer(typeof(CUEMetadata));

		public static string MetadataPath
		{
			get
			{
				string cache = System.IO.Path.Combine(System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "CUE Tools"), "MetadataCache");
				if (!Directory.Exists(cache))
					Directory.CreateDirectory(cache);
				return cache;
			}
		}

		public string Id { get; set; }
		[DefaultValue("")]
		public string TotalDiscs { get; set; }
		[DefaultValue("")]
		public string DiscNumber { get; set; }
		[DefaultValue("")]
		public string Year { get; set; }
		[DefaultValue("")]
		public string Genre { get; set; }
		[DefaultValue("")]
		public string Artist { get; set; }
		[DefaultValue("")]
		public string Title { get; set; }
		[DefaultValue("")]
		public string Catalog { get; set; }
		public List<CUETrackMetadata> Tracks { get; set; }

		public void Save()
		{
			TextWriter writer = new StreamWriter(Path.Combine(MetadataPath, Id + ".xml"));
			serializer.Serialize(writer, this);
			writer.Close();
		}

		public static CUEMetadata Load(string Id)
		{
			//serializer.UnknownNode += new XmlNodeEventHandler(serializer_UnknownNode);
			//serializer.UnknownAttribute += new XmlAttributeEventHandler(serializer_UnknownAttribute);
			using (FileStream fs = new FileStream(Path.Combine(MetadataPath, Id + ".xml"), FileMode.Open))
				return serializer.Deserialize(fs) as CUEMetadata;
		}

		public void CopyMetadata(CUEMetadata metadata)
		{
			// if (metadata.Tracks.Count != Tracks.Count) throw;
			// Tracks.Count = metadata.Tracks.Count;
			TotalDiscs = metadata.TotalDiscs;
			DiscNumber = metadata.DiscNumber;
			Year = metadata.Year;
			Genre = metadata.Genre;
			Artist = metadata.Artist;
			Title = metadata.Title;
			Catalog = metadata.Catalog;
			for (int i = 0; i < Tracks.Count; i++)
			{
				Tracks[i].Title = metadata.Tracks[i].Title;
				Tracks[i].Artist = metadata.Tracks[i].Artist;
				Tracks[i].ISRC = metadata.Tracks[i].ISRC;
			}
		}

		public void FillFromMusicBrainz(MusicBrainz.Release release, int firstAudio)
		{
			string date = release.GetEvents().Count > 0 ? release.GetEvents()[0].Date : null;
			Year = date == null ? "" : date.Substring(0, 4);
			Artist = release.GetArtist();
			Title = release.GetTitle();
			// How to get Genre: http://mm.musicbrainz.org/ws/1/release/6fe1e218-2aee-49ac-94f0-7910ba2151df.html?type=xml&inc=tags
			//Catalog = release.GetEvents().Count > 0 ? release.GetEvents()[0].Barcode : "";
			for (int i = 0; i < Tracks.Count; i++)
			{
				MusicBrainz.Track track = release.GetTracks()[i + firstAudio]; // !!!!!! - _toc.FirstAudio?
				Tracks[i].Title = track.GetTitle();
				Tracks[i].Artist = track.GetArtist();
			}
		}

		public void FillFromFreedb(Freedb.CDEntry cdEntry, int firstAudio)
		{
			Year = cdEntry.Year;
			Genre = cdEntry.Genre;
			Artist = cdEntry.Artist;
			Title = cdEntry.Title;
			for (int i = 0; i < Tracks.Count; i++)
			{
				Tracks[i].Title = cdEntry.Tracks[i + firstAudio].Title;
				Tracks[i].Artist = cdEntry.Artist;
			}
		}

		private static string FreedbToEncoding(Encoding iso, Encoding def, ref bool changed, ref bool error, string s)
		{
			try
			{
				string res = def.GetString(iso.GetBytes(s));
				changed |= res != s;
				return res;
			}
			catch // EncoderFallbackException, DecoderFallbackException
			{
				error = true;
			}
			return s;
		}

		public bool FreedbToEncoding()
		{
			Encoding iso = Encoding.GetEncoding("iso-8859-1", new EncoderExceptionFallback(), new DecoderExceptionFallback());
			Encoding def = Encoding.GetEncoding(Encoding.Default.CodePage, new EncoderExceptionFallback(), new DecoderExceptionFallback());
			bool different = false;
			bool error = false;
			Artist = FreedbToEncoding(iso, def, ref different, ref error, Artist);
			Title = FreedbToEncoding(iso, def, ref different, ref error, Title);
			for (int i = 0; i < Tracks.Count; i++)
			{
				Tracks[i].Artist = FreedbToEncoding(iso, def, ref different, ref error, Tracks[i].Artist);
				Tracks[i].Title = FreedbToEncoding(iso, def, ref different, ref error, Tracks[i].Title);
			}
			return different && !error;
		}

		public void FreedbToVarious()
		{
			for (int i = 0; i < Tracks.Count; i++)
			{
				string title = Tracks[i].Title;
				int idx = title.IndexOf(" / ");
				if (idx < 0) idx = title.IndexOf(" - ");
				if (idx >= 0)
				{
					Tracks[i].Title = title.Substring(idx + 3);
					Tracks[i].Artist = title.Substring(0, idx);
				}
				else
				{
					Tracks[i].Artist = title;
				}
			}
		}

		public void UpdateArtist(string artist)
		{
			for (int i = 0; i < Tracks.Count; i++)
				if (Tracks[i].Artist == Artist)
					Tracks[i].Artist = artist;
			Artist = artist;
		}

		public bool IsVarious()
		{
			bool isVarious = false;
			for (int i = 0; i < Tracks.Count; i++)
				if (Tracks[i].Artist != Artist)
					isVarious = true;
			return isVarious;
		}
	}

	public class CUETrackMetadata
	{
		public CUETrackMetadata()
		{
			Artist = "";
			Title = "";
			ISRC = "";
		}
		[DefaultValue("")]
		public string Artist { get; set; }
		[DefaultValue("")]
		public string Title { get; set; }
		[DefaultValue("")]
		public string ISRC { get; set; }
	}

	public class CUEMetadataEntry
	{
		public CUEMetadata metadata { get; set; }
		public CDImageLayout TOC { get; set; }
		public string ImageKey { get; set; }
	
		public CUEMetadataEntry(CUEMetadata metadata, CDImageLayout TOC, string key)
		{
			this.metadata = metadata;
			this.TOC = TOC;
			this.ImageKey = key;
		}

		public CUEMetadataEntry(CDImageLayout TOC, string key)
			: this(new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks), TOC, key)
		{
		}

		public override string ToString()
		{
			return string.Format("{0}{1} - {2}", metadata.Year != "" ? metadata.Year + ": " : "", 
				metadata.Artist == "" ? "Unknown Artist" : metadata.Artist,
				metadata.Title == "" ? "Unknown Title" : metadata.Title);
		}
	}
}
