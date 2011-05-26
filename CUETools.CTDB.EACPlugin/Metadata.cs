using System;
using System.Collections.Generic;
using System.Text;
using HelperFunctionsLib;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Net;
using System.IO;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.Codecs;
using CUETools.CTDB;
using CUETools.CTDB.EACPlugin.Properties;
using System.Drawing.Imaging;

namespace MetadataPlugIn
{
	[Guid("8271734A-126F-44e9-AC9C-836449B39E51"),
    ClassInterface(ClassInterfaceType.None),
	ComSourceInterfaces(typeof(IMetadataRetriever)),
    ]

	public class MetadataRetriever : IMetadataRetriever
	{
		public bool GetCDInformation(CCDMetadata data, bool cdinfo, bool cover, bool lyrics)
		{
			if (!cdinfo)
				return false;

			var TOC = new CDImageLayout();
			for (int i = 0; i < data.NumberOfTracks; i++)
			{
				uint start = data.GetTrackStartPosition(i);
				uint next = data.GetTrackEndPosition(i);
				TOC.AddTrack(new CDTrack(
					(uint)i + 1,
					start,
					next - start,
					!data.GetTrackDataTrack(i),
					data.GetTrackPreemphasis(i)));
			}
			TOC[1][0].Start = 0U;

			var ctdb = new CUEToolsDB(TOC, null);
			var form = new CUETools.CTDB.EACPlugin.FormMetadata(ctdb, "EAC" + data.HostVersion + " CTDB 2.1.2");
			form.ShowDialog();
			var meta = form.Meta;
			if (meta == null)
				return false;

			int year, disccount, discnumber;
			if (meta.year != null && int.TryParse(meta.year, out year))
				data.Year = year;
			if (meta.disccount != null && int.TryParse(meta.disccount, out disccount))
				data.TotalNumberOfCDs = disccount;
			if (meta.discnumber != null && int.TryParse(meta.discnumber, out discnumber))
				data.CDNumber = discnumber;
			if (meta.album != null)
				data.AlbumTitle = meta.album;
			if (meta.artist != null)
				data.AlbumArtist = meta.artist;
			if (meta.track != null)
				for (int track = 0; track < data.NumberOfTracks; track++)
				{
					if (track < meta.track.Length)
					{
						if (meta.track[track].name != null)
							data.SetTrackTitle(track, meta.track[track].name);
						var trackartist = meta.track[track].artist ?? meta.artist;
						if (trackartist != null)
							data.SetTrackArtist(track, trackartist);
					}
					else if (meta.artist != null)
						data.SetTrackArtist(track, meta.artist);
				}
			return true;
		}

		public string GetPluginGuid()
		{
			return ((GuidAttribute)Attribute.GetCustomAttribute(GetType(), typeof(GuidAttribute))).Value;
		}

		public Array GetPluginLogo()
		{
			MemoryStream ms = new MemoryStream();
			Resources.ctdb64.Save(ms, ImageFormat.Png);
			return ms.ToArray();
		}

		public string GetPluginName()
		{
			return "CUETools DB Metadata Plugin V2.1.2";
		}

		public void ShowOptions()
		{
			AudioDataPlugIn.Options opt = new AudioDataPlugIn.Options();
			opt.ShowDialog();
		}

		public bool SubmitCDInformation(IMetadataLookup data)
		{
			throw new NotSupportedException();
		}

		public bool SupportsCoverRetrieval()
		{
			return false;
		}

		public bool SupportsLyricsRetrieval()
		{
			return false;
		}

		public bool SupportsMetadataRetrieval()
		{
			return true;
		}

		public bool SupportsMetadataSubmission()
		{
			return false;
		}
	}
}
