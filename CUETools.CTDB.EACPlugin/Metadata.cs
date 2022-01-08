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
using AudioDataPlugIn;

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
            if (Options.CoversSearch == CTDBCoversSearch.None)
                cover = false;

            if (!cdinfo && !cover)
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
            var form = new CUETools.CTDB.EACPlugin.FormMetadata(ctdb, "EAC" + data.HostVersion + " CTDB 2.2.0", cdinfo, cover);
            form.ShowDialog();
            var meta = form.Meta;
            if (meta == null)
                return false;

            if (cdinfo)
            {
                int year, disccount, discnumber;
                string extra = meta.extra ?? "";
                if (!string.IsNullOrEmpty(meta.discname))
                    extra += "Disc name: " + meta.discname + "\r\n";
                if (!string.IsNullOrEmpty(meta.infourl))
                    extra += "Info URL: " + meta.infourl + "\r\n";
                if (!string.IsNullOrEmpty(meta.barcode))
                    extra += "Barcode: " + meta.barcode + "\r\n";
                if (meta.release != null)
                    foreach (var release in meta.release)
                    {
                        if (!string.IsNullOrEmpty(release.date))
                            extra += "Release date: " + release.date + "\r\n";
                        if (!string.IsNullOrEmpty(release.country))
                            extra += "Release country: " + release.country + "\r\n";
                    }
                if (meta.label != null)
                    foreach (var label in meta.label)
                    {
                        if (!string.IsNullOrEmpty(label.name))
                            extra += "Release label: " + label.name + "\r\n";
                        if (!string.IsNullOrEmpty(label.catno))
                            extra += "Release catalog#: " + label.catno + "\r\n";
                    }
                data.Year = meta.year != null && int.TryParse(meta.year, out year) ? year : -1;
                data.TotalNumberOfCDs = meta.disccount != null && int.TryParse(meta.disccount, out disccount) ? disccount : 1;
                data.CDNumber = meta.discnumber != null && int.TryParse(meta.discnumber, out discnumber) ? discnumber : 1;
                data.FirstTrackNumber = 1;
                data.AlbumTitle = meta.album ?? "";
                data.AlbumArtist = meta.artist ?? "";
                data.MP3V2Type = meta.genre ?? "";
                data.CDDBMusicType = GetFreeDBMusicType(meta);
                data.MP3Type = GetMP3MusicType(data.CDDBMusicType);
                data.ExtendedDiscInformation = extra;
                data.Revision = -1; // TODO: meta.id? rock/ffffffff/16?
                if (meta.track != null)
                {
                    int firstAudio = meta.track.Length == TOC.AudioTracks ? TOC.FirstAudio - 1 : 0;
                    for (int track = 0; track < data.NumberOfTracks; track++)
                    {
                        if (track - firstAudio >= 0 && track - firstAudio < meta.track.Length)
                        {
                            data.SetTrackTitle(track, meta.track[track - firstAudio].name ?? "");
                            data.SetTrackArtist(track, meta.track[track - firstAudio].artist ?? meta.artist ?? "");
                            data.SetExtendedTrackInformation(track, meta.track[track - firstAudio].extra ?? "");
                        }
                        else if (!TOC[track + 1].IsAudio)
                        {
                            data.SetTrackTitle(track, "[data track]");
                            data.SetTrackArtist(track, meta.artist ?? "");
                            data.SetExtendedTrackInformation(track, "");
                        }
                        else
                        {
                            data.SetTrackTitle(track, "");
                            data.SetTrackArtist(track, meta.artist ?? "");
                            data.SetExtendedTrackInformation(track, "");
                        }
                        data.SetTrackComposer(track, "");
                    }
                }
            }

            if (cover)
            {                
                data.CoverImage = null;
                data.CoverImageURL = "";
                if (form.Image != null)
                {
                    data.CoverImage = form.Image.Data;
                    data.CoverImageURL = form.Image.URL;
                }
            }

            return true;
        }

        public int GetMP3MusicType(int freedbtype)
        {
            int[] list = { 17, 29, 34, 95, 53, 77, 90, 113, 117, 129, 95 };
            return (freedbtype <= 0 || freedbtype >= list.Length) ? -1 : list[freedbtype];
        }

        public int GetFreeDBMusicType(CTDBResponseMeta meta)
        {
            int pos = meta.id.IndexOf('/');
            if (meta.source != "freedb" || pos < 0)
                return -1;
            string freedbtype = meta.id.Substring(0, pos);
            switch (freedbtype.ToUpper())
            {
                case "BLUES":
                    return 0;
                case "CLASSICAL":
                    return 1;
                case "COUNTRY":
                    return 2;
                case "DATA":
                    return 3;
                case "FOLK":
                    return 4;
                case "JAZZ":
                    return 5;
                case "NEWAGE":
                    return 6;
                case "REGGAE":
                    return 7;
                case "ROCK":
                    return 8;
                case "SOUNDTRACK":
                    return 9;
                case "MISC":
                    return 10;
                default:
                    return -1;
            }
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
            return "CUETools DB Metadata Plugin V2.2.0";
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
            return true;
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
