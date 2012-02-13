using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using CUETools.Processor.Settings;

namespace CUETools.Processor
{
	[Serializable]
	public class CUEMetadata
	{
		public CUEMetadata()
		{
			TotalDiscs = "";
			DiscNumber = "";
			DiscName = "";
			Year = "";
			Genre = "";
			Artist = "";
			Title = "";
			Barcode = "";
			ReleaseDate = "";
			Label = "";
			Country = "";
            AlbumArt = new List<CTDB.CTDBResponseMetaImage>();
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
				string cache = Path.Combine(SettingsShared.GetProfileDir("CUE Tools", System.Windows.Forms.Application.ExecutablePath), "MetadataCache");
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
		public string DiscName { get; set; }
		[DefaultValue("")]
		public string Year { get; set; }
		[DefaultValue("")]
		public string Genre { get; set; }
		[DefaultValue("")]
		public string Artist { get; set; }
		[DefaultValue("")]
		public string Title { get; set; }
		[DefaultValue(""), XmlElement(ElementName="Catalog")]
		public string Barcode { get; set; }
		[DefaultValue("")]
		public string ReleaseDate { get; set; }
		[DefaultValue("")]
		public string Label { get; set; }
		[DefaultValue("")]
		public string Country { get; set; }
		public List<CUETrackMetadata> Tracks { get; set; }

        public List<CTDB.CTDBResponseMetaImage> AlbumArt { get; set; }

		[XmlIgnore]
		public string DiscNumber01
		{
			get
			{
				uint td = 0, dn = 0;
				if (uint.TryParse(TotalDiscs, out td) && uint.TryParse(DiscNumber, out dn) && td > 9 && dn > 0)
					return string.Format("{0:00}", dn);
				return DiscNumber;
			}
		}

		[XmlIgnore]
		public string DiscNumberAndTotal
		{
			get
			{
				return (TotalDiscs != "" && TotalDiscs != "1" ? DiscNumber01 + "/" + TotalDiscs : (DiscNumber != "" && DiscNumber != "1" ? DiscNumber01 : ""));
			}
		}

		[XmlIgnore]
		public string ReleaseDateAndLabel
		{
			get
			{
				return Label == "" && ReleaseDate == "" && Country == "" ? ""
					: Country 
					+ (Country != "" && Label != "" ? " - " : "") + Label 
					+ (Label + Country != "" && ReleaseDate != "" ? " - " : "") + ReleaseDate;
			}
		}

		[XmlIgnore]
		public string DiscNumberAndName
		{
			get
			{
				return DiscNumberAndTotal == "" ? ""
					: DiscNumberAndTotal + (DiscName != "" ? " - " + DiscName : "");
			}
		}

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

		public void Merge(CUEMetadata metadata, bool overwrite)
		{
			if ((overwrite || TotalDiscs == "") && metadata.TotalDiscs != "") TotalDiscs = metadata.TotalDiscs;
			if ((overwrite || DiscNumber == "") && metadata.DiscNumber != "") DiscNumber = metadata.DiscNumber;
			if ((overwrite || DiscName == "") && metadata.DiscName != "") DiscName = metadata.DiscName;
			if ((overwrite || Year == "") && metadata.Year != "") Year = metadata.Year;
			if ((overwrite || Genre == "") && metadata.Genre != "") Genre = metadata.Genre;
			if ((overwrite || Artist == "") && metadata.Artist != "") Artist = metadata.Artist;
			if ((overwrite || Title == "") && metadata.Title != "") Title = metadata.Title;
			if ((overwrite || Barcode == "") && metadata.Barcode != "") Barcode = metadata.Barcode;
			if ((overwrite || ReleaseDate == "") && metadata.ReleaseDate != "") ReleaseDate = metadata.ReleaseDate;
			if ((overwrite || Label == "") && metadata.Label != "") Label = metadata.Label;
			if ((overwrite || Country == "") && metadata.Country != "") Country = metadata.Country;
            if ((overwrite || AlbumArt.Count == 0) && metadata.AlbumArt.Count != 0) AlbumArt = metadata.AlbumArt;
			for (int i = 0; i < Tracks.Count; i++)
			{
				if ((overwrite || Tracks[i].Title == "") && metadata.Tracks[i].Title != "") Tracks[i].Title = metadata.Tracks[i].Title;
				if ((overwrite || Tracks[i].Artist == "") && metadata.Tracks[i].Artist != "") Tracks[i].Artist = metadata.Tracks[i].Artist;
				if ((overwrite || Tracks[i].ISRC == "") && metadata.Tracks[i].ISRC != "") Tracks[i].ISRC = metadata.Tracks[i].ISRC;
			}
		}

		public override int GetHashCode()
		{
			return Artist.GetHashCode() ^ Title.GetHashCode() ^ Year.GetHashCode();
		}

		public override bool Equals(object obj)
		{
			CUEMetadata metadata = obj as CUEMetadata;
			if (metadata == null)
				return false;
			if (TotalDiscs != metadata.TotalDiscs ||
				DiscNumber != metadata.DiscNumber ||
				DiscName != metadata.DiscName ||
				Year != metadata.Year ||
				Genre != metadata.Genre ||
				Artist != metadata.Artist ||
				Title != metadata.Title ||
				Barcode != metadata.Barcode ||
				ReleaseDate != metadata.ReleaseDate ||
				Label != metadata.Label ||
				Country != metadata.Country ||
				Tracks.Count != metadata.Tracks.Count
				)
				return false;
			for (int i = 0; i < Tracks.Count; i++)
				if (Tracks[i].Title != metadata.Tracks[i].Title ||
					Tracks[i].Artist != metadata.Tracks[i].Artist ||
					Tracks[i].ISRC != metadata.Tracks[i].ISRC)
					return false;
			return true;
		}

		public bool Contains(CUEMetadata metadata)
		{
			CUEMetadata sum = new CUEMetadata(metadata);
			sum.Merge(this, false);
			return sum.Equals(this);
		}

		public void CopyMetadata(CUEMetadata metadata)
		{
			// if (metadata.Tracks.Count != Tracks.Count) throw;
			// Tracks.Count = metadata.Tracks.Count;
			TotalDiscs = metadata.TotalDiscs;
			DiscNumber = metadata.DiscNumber;
			DiscName = metadata.DiscName;
			Year = metadata.Year;
			Genre = metadata.Genre;
			Artist = metadata.Artist;
			Title = metadata.Title;
			Barcode = metadata.Barcode;
			ReleaseDate = metadata.ReleaseDate;
			Country = metadata.Country;
			Label = metadata.Label;
            AlbumArt = metadata.AlbumArt;
			for (int i = 0; i < Tracks.Count; i++)
			{
				Tracks[i].Title = metadata.Tracks[i].Title;
				Tracks[i].Artist = metadata.Tracks[i].Artist;
				Tracks[i].ISRC = metadata.Tracks[i].ISRC;
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

		public void FillFromCtdb(CUETools.CTDB.CTDBResponseMeta cdEntry, int firstAudio)
		{
			this.Year = cdEntry.year ?? "";
			this.Genre = cdEntry.genre ?? "";
			this.Artist = cdEntry.artist ?? "";
			this.Title = cdEntry.album ?? "";
			this.DiscNumber = cdEntry.discnumber ?? "";
			this.TotalDiscs = cdEntry.disccount ?? "";
			this.DiscName = cdEntry.discname ?? "";
			this.Barcode = cdEntry.barcode ?? "";
			this.ReleaseDate = cdEntry.releasedate ?? "";
			this.Country = cdEntry.country ?? "";
			this.Genre = cdEntry.genre ?? "";
			this.Label = "";
			if (cdEntry.label != null)
				foreach (var l in cdEntry.label)
					this.Label = (this.Label == "" ? "" : this.Label + ": ") + (l.name ?? "") + (l.name != null && l.catno != null ? " " : "") + (l.catno ?? "");
            this.AlbumArt.Clear();
            if (cdEntry.coverart != null)
                this.AlbumArt.AddRange(cdEntry.coverart);
			if (cdEntry.track != null && cdEntry.track.Length >= this.Tracks.Count)
			{
				for (int i = 0; i < this.Tracks.Count; i++)
				{
					this.Tracks[i].Title = cdEntry.track[i].name ?? "";
					this.Tracks[i].Artist = cdEntry.track[i].artist ?? cdEntry.artist ?? "";
				}
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

		public bool FreedbToVarious()
		{
			bool found = false;
			for (int i = 0; i < Tracks.Count; i++)
			{
				string title = Tracks[i].Title;
				int idx = title.IndexOf(" / ");
				if (idx < 0) idx = title.IndexOf(" - ");
				if (idx >= 0)
				{
					Tracks[i].Title = title.Substring(idx + 3);
					Tracks[i].Artist = title.Substring(0, idx);
					found = true;
				}
				else
				{
					Tracks[i].Artist = title;
				}
			}
			return found;
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
}
