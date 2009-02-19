// ****************************************************************************
// 
// CUE Tools
// Copyright (C) 2006-2007  Moitah (moitah@yahoo.com)
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// ****************************************************************************

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.Globalization;
using System.IO;
using System.Net;
using System.Security.Cryptography;
using System.Threading;
using System.Xml;
using HDCDDotNet;
using CUETools.Codecs;
using CUETools.Codecs.LossyWAV;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.Ripper.SCSI;
using MusicBrainz;
using Freedb;
#if !MONO
using UnRarDotNet;
using CUETools.Codecs.FLAC;
#endif
using ICSharpCode.SharpZipLib.Zip;

namespace CUETools.Processor
{
	public enum OutputAudioFormat
	{
		WAV,
		FLAC,
		WavPack,
		APE,
		TTA,
		NoAudio,
		UDC1
	}

	public enum AccurateRipMode
	{
		None,
		Verify,
		VerifyThenConvert,
		VerifyAndConvert
	}

	public enum CUEStyle
	{
		SingleFileWithCUE,
		SingleFile,
		GapsPrepended,
		GapsAppended,
		GapsLeftOut
	}

	public static class General {
		public static string FormatExtension(OutputAudioFormat value, CUEConfig config)
		{
			switch (value)
			{
				case OutputAudioFormat.FLAC: return ".flac";
				case OutputAudioFormat.WavPack: return ".wv";
				case OutputAudioFormat.APE: return ".ape";
				case OutputAudioFormat.TTA: return ".tta";
				case OutputAudioFormat.WAV: return ".wav";
				case OutputAudioFormat.NoAudio: return ".dummy";
				case OutputAudioFormat.UDC1: return "." + config.udc1Extension;
			}
			return ".wav";
		}

		public static CUELine FindCUELine(List<CUELine> list, string command) {
			command = command.ToUpper();
			foreach (CUELine line in list) {
				if (line.Params[0].ToUpper() == command) {
					return line;
				}
			}
			return null;
		}

		public static CUELine FindCUELine(List<CUELine> list, string command, string command2)
		{
			command = command.ToUpper();
			command2 = command2.ToUpper();
			foreach (CUELine line in list)
			{
				if (line.Params.Count > 1 && line.Params[0].ToUpper() == command && line.Params[1].ToUpper() == command2)
				{
					return line;
				}
			}
			return null;
		}

		//public static CUELine FindCUELine(List<CUELine> list, string [] commands)
		//{
		//    foreach (CUELine line in list)
		//    {
		//        if (line.Params.Count < commands.Length)
		//            continue;
		//        for (int i = 0; i < commands.Length; i++)
		//        {
		//            if (line.Params[i].ToUpper() != commands[i].ToUpper())
		//                break;
		//            if (i == commands.Length - 1)
		//                return line;
		//        }
		//    }
		//    return null;
		//}

		public static void SetCUELine(List<CUELine> list, string command, string value, bool quoted)
		{
			CUELine line = General.FindCUELine(list, command);
			if (line == null)
			{
				line = new CUELine();
				line.Params.Add(command); line.IsQuoted.Add(false);
				line.Params.Add(value); line.IsQuoted.Add(quoted);
				list.Add(line);
			}
			else
			{
				line.Params[1] = value;
				line.IsQuoted[1] = quoted;
			}
		}

		public static void SetCUELine(List<CUELine> list, string command, string command2, string value, bool quoted)
		{
			CUELine line = General.FindCUELine(list, command, command2);
			if (line == null)
			{
				line = new CUELine();
				line.Params.Add(command); line.IsQuoted.Add(false);
				line.Params.Add(command2); line.IsQuoted.Add(false);
				line.Params.Add(value); line.IsQuoted.Add(quoted);
				list.Add(line);
			}
			else
			{
				line.Params[2] = value;
				line.IsQuoted[2] = quoted;
			}
		}

		public static void DelCUELine(List<CUELine> list, string command, string command2)
		{
			CUELine line = General.FindCUELine(list, command, command2);
			if (line == null)
				return;
			list.Remove(line);
		}

		public static void DelCUELine(List<CUELine> list, string command)
		{
			CUELine line = General.FindCUELine(list, command);
			if (line == null)
				return;
			list.Remove(line);
		}

		public static string ReplaceMultiple(string s, List<string> find, List<string> replace)
		{
			if (find.Count != replace.Count)
			{
				throw new ArgumentException();
			}
			StringBuilder sb;
			int iChar, iFind;
			string f;
			bool found;

			sb = new StringBuilder();

			for (iChar = 0; iChar < s.Length; iChar++)
			{
				found = false;
				for (iFind = 0; iFind < find.Count; iFind++)
				{
					f = find[iFind];
					if ((f.Length <= (s.Length - iChar)) && (s.Substring(iChar, f.Length) == f))
					{
						if (replace[iFind] == null)
						{
							return null;
						}
						sb.Append(replace[iFind]);
						iChar += f.Length - 1;
						found = true;
						break;
					}
				}

				if (!found)
				{
					sb.Append(s[iChar]);
				}
			}

			return sb.ToString();
		}

		public static string EmptyStringToNull(string s)
		{
			return ((s != null) && (s.Length == 0)) ? null : s;
		}
	}

	public class CUEConfig {
		public uint fixWhenConfidence;
		public uint fixWhenPercent;
		public uint encodeWhenConfidence;
		public uint encodeWhenPercent;
		public bool encodeWhenZeroOffset;
		public bool writeArTagsOnVerify;
		public bool writeArLogOnVerify;
		public bool writeArTagsOnConvert;
		public bool writeArLogOnConvert;
		public bool fixOffset;
		public bool noUnverifiedOutput;
		public bool autoCorrectFilenames;
		public bool flacVerify;
		public uint flacCompressionLevel;
		public uint apeCompressionLevel;
		public bool preserveHTOA;
		public int wvCompressionMode;
		public int wvExtraMode;
		public bool wvStoreMD5;
		public bool keepOriginalFilenames;
		public string trackFilenameFormat;
		public string singleFilenameFormat;
		public bool removeSpecial;
		public string specialExceptions;
		public bool replaceSpaces;
		public bool embedLog;
		public bool extractLog;
		public bool fillUpCUE;
		public bool overwriteCUEData;
		public bool filenamesANSISafe;
		public bool bruteForceDTL;
		public bool detectHDCD;
		public bool decodeHDCD;
		public bool wait750FramesForHDCD;
		public bool createM3U;
		public bool createTOC;
		public bool createCUEFileWhenEmbedded;
		public bool truncate4608ExtraSamples;
		public int lossyWAVQuality;
		public bool lossyWAVHybrid;
		public bool decodeHDCDtoLW16;
		public bool decodeHDCDto24bit;
		public string udc1Extension, udc1Decoder, udc1Params, udc1Encoder, udc1EncParams;
		public bool udc1APEv2, udc1ID3v2;

		public CUEConfig()
		{
			fixWhenConfidence = 2;
			fixWhenPercent = 51;
			encodeWhenConfidence = 2;
			encodeWhenPercent = 100;
			encodeWhenZeroOffset = false;
			fixOffset = false;
			noUnverifiedOutput = false;
			writeArTagsOnConvert = true;
			writeArLogOnConvert = true;
			writeArTagsOnVerify = false;
			writeArLogOnVerify = true;

			autoCorrectFilenames = true;
			flacVerify = false;
			flacCompressionLevel = 8;
			apeCompressionLevel = 2;
			preserveHTOA = true;
			wvCompressionMode = 1;
			wvExtraMode = 0;
			wvStoreMD5 = false;
			keepOriginalFilenames = true;
			trackFilenameFormat = "%N-%A-%T";
			singleFilenameFormat = "%F";
			removeSpecial = false;
			specialExceptions = "-()";
			replaceSpaces = false;
			embedLog = true;
			extractLog = true;
			fillUpCUE = true;
			overwriteCUEData = false;
			filenamesANSISafe = true;
			bruteForceDTL = false;
			detectHDCD = true;
			wait750FramesForHDCD = true;
			decodeHDCD = false;
			createM3U = false;
			createTOC = false;
			createCUEFileWhenEmbedded = true;
			truncate4608ExtraSamples = true;
			lossyWAVQuality = 5;
			lossyWAVHybrid = true;
			decodeHDCDtoLW16 = false;
			decodeHDCDto24bit = true;

			udc1Extension = udc1Decoder = udc1Params = udc1Encoder = udc1EncParams = "";
			udc1ID3v2 = udc1APEv2 = false;
		}

		public void Save (SettingsWriter sw)
		{
			sw.Save("ArFixWhenConfidence", fixWhenConfidence);
			sw.Save("ArFixWhenPercent", fixWhenPercent);
			sw.Save("ArEncodeWhenConfidence", encodeWhenConfidence);
			sw.Save("ArEncodeWhenPercent", encodeWhenPercent);
			sw.Save("ArEncodeWhenZeroOffset", encodeWhenZeroOffset);
			sw.Save("ArNoUnverifiedOutput", noUnverifiedOutput);
			sw.Save("ArFixOffset", fixOffset);
			sw.Save("ArWriteCRC", writeArTagsOnConvert);
			sw.Save("ArWriteLog", writeArLogOnConvert);
			sw.Save("ArWriteTagsOnVerify", writeArTagsOnVerify);
			sw.Save("ArWriteLogOnVerify", writeArLogOnVerify);

			sw.Save("PreserveHTOA", preserveHTOA);
			sw.Save("AutoCorrectFilenames", autoCorrectFilenames);
			sw.Save("FLACCompressionLevel", flacCompressionLevel);
			sw.Save("APECompressionLevel", apeCompressionLevel);
			sw.Save("FLACVerify", flacVerify);
			sw.Save("WVCompressionMode", wvCompressionMode);
			sw.Save("WVExtraMode", wvExtraMode);
			sw.Save("WVStoreMD5", wvStoreMD5);
			sw.Save("KeepOriginalFilenames", keepOriginalFilenames);
			sw.Save("SingleFilenameFormat", singleFilenameFormat);
			sw.Save("TrackFilenameFormat", trackFilenameFormat);
			sw.Save("RemoveSpecialCharacters", removeSpecial);
			sw.Save("SpecialCharactersExceptions", specialExceptions);
			sw.Save("ReplaceSpaces", replaceSpaces);
			sw.Save("EmbedLog", embedLog);
			sw.Save("ExtractLog", extractLog);
			sw.Save("FillUpCUE", fillUpCUE);
			sw.Save("OverwriteCUEData", overwriteCUEData);			
			sw.Save("FilenamesANSISafe", filenamesANSISafe);
			sw.Save("BruteForceDTL", bruteForceDTL);
			sw.Save("DetectHDCD", detectHDCD);
			sw.Save("Wait750FramesForHDCD", wait750FramesForHDCD);
			sw.Save("DecodeHDCD", decodeHDCD);
			sw.Save("CreateM3U", createM3U);
			sw.Save("CreateTOC", createTOC);
			sw.Save("CreateCUEFileWhenEmbedded", createCUEFileWhenEmbedded);
			sw.Save("Truncate4608ExtraSamples", truncate4608ExtraSamples);
			sw.Save("LossyWAVQuality", lossyWAVQuality);
			sw.Save("LossyWAVHybrid", lossyWAVHybrid);			
			sw.Save("DecodeHDCDToLossyWAV16", decodeHDCDtoLW16);
			sw.Save("DecodeHDCDTo24bit", decodeHDCDto24bit);
			if (udc1Extension != "")
			{
				sw.Save("UDC1Extension", udc1Extension);
				sw.Save("UDC1Decoder", udc1Decoder);
				sw.Save("UDC1Params", udc1Params);
				sw.Save("UDC1Encoder", udc1Encoder);
				sw.Save("UDC1EncParams", udc1EncParams);
				sw.Save("UDC1APEv2", udc1APEv2);
				sw.Save("UDC1ID3v2", udc1ID3v2);
			}
		}

		public void Load(SettingsReader sr)
		{
			fixWhenConfidence = sr.LoadUInt32("ArFixWhenConfidence", 1, 1000) ?? 2;
			fixWhenPercent = sr.LoadUInt32("ArFixWhenPercent", 1, 100) ?? 51;
			encodeWhenConfidence = sr.LoadUInt32("ArEncodeWhenConfidence", 1, 1000) ?? 2;
			encodeWhenPercent = sr.LoadUInt32("ArEncodeWhenPercent", 1, 100) ?? 100;
			encodeWhenZeroOffset = sr.LoadBoolean("ArEncodeWhenZeroOffset") ?? false;
			noUnverifiedOutput = sr.LoadBoolean("ArNoUnverifiedOutput") ?? false;
			fixOffset = sr.LoadBoolean("ArFixOffset") ?? false;
			writeArTagsOnConvert = sr.LoadBoolean("ArWriteCRC") ?? true;
			writeArLogOnConvert = sr.LoadBoolean("ArWriteLog") ?? true;
			writeArTagsOnVerify = sr.LoadBoolean("ArWriteTagsOnVerify") ?? false;
			writeArLogOnVerify = sr.LoadBoolean("ArWriteLogOnVerify") ?? true;

			preserveHTOA = sr.LoadBoolean("PreserveHTOA") ?? true;
			autoCorrectFilenames = sr.LoadBoolean("AutoCorrectFilenames") ?? true;
			flacCompressionLevel = sr.LoadUInt32("FLACCompressionLevel", 0, 8) ?? 8;
			flacVerify = sr.LoadBoolean("FLACVerify") ?? false;
			apeCompressionLevel = sr.LoadUInt32("APECompressionLevel", 1, 5) ?? 2;
			wvCompressionMode = sr.LoadInt32("WVCompressionMode", 0, 3) ?? 1;
			wvExtraMode = sr.LoadInt32("WVExtraMode", 0, 6) ?? 0;
			wvStoreMD5 = sr.LoadBoolean("WVStoreMD5") ?? false;
			keepOriginalFilenames = sr.LoadBoolean("KeepOriginalFilenames") ?? true;
			singleFilenameFormat =  sr.Load("SingleFilenameFormat") ?? "%F";
			trackFilenameFormat = sr.Load("TrackFilenameFormat") ?? "%N-%A-%T";
			removeSpecial = sr.LoadBoolean("RemoveSpecialCharacters") ?? false;
			specialExceptions = sr.Load("SpecialCharactersExceptions") ?? "-()";
			replaceSpaces = sr.LoadBoolean("ReplaceSpaces") ?? false;
			embedLog = sr.LoadBoolean("EmbedLog") ?? true;
			extractLog = sr.LoadBoolean("ExtractLog") ?? true;
			fillUpCUE = sr.LoadBoolean("FillUpCUE") ?? true;
			overwriteCUEData = sr.LoadBoolean("OverwriteCUEData") ?? false;
			filenamesANSISafe = sr.LoadBoolean("FilenamesANSISafe") ?? true;
			bruteForceDTL = sr.LoadBoolean("BruteForceDTL") ?? false;
			detectHDCD = sr.LoadBoolean("DetectHDCD") ?? true;
			wait750FramesForHDCD = sr.LoadBoolean("Wait750FramesForHDCD") ?? true;
			decodeHDCD = sr.LoadBoolean("DecodeHDCD") ?? false;
			createM3U = sr.LoadBoolean("CreateM3U") ?? false;
			createTOC = sr.LoadBoolean("CreateTOC") ?? false;
			createCUEFileWhenEmbedded = sr.LoadBoolean("CreateCUEFileWhenEmbedded") ?? true;
			truncate4608ExtraSamples = sr.LoadBoolean("Truncate4608ExtraSamples") ?? true;
			lossyWAVQuality = sr.LoadInt32("LossyWAVQuality", 0, 10) ?? 5;
			lossyWAVHybrid = sr.LoadBoolean("LossyWAVHybrid") ?? true;
			decodeHDCDtoLW16 = sr.LoadBoolean("DecodeHDCDToLossyWAV16") ?? false;
			decodeHDCDto24bit = sr.LoadBoolean("DecodeHDCDTo24bit") ?? true;

			udc1Extension = sr.Load("UDC1Extension") ?? "";
			udc1Decoder = sr.Load("UDC1Decoder") ?? "";
			udc1Params = sr.Load("UDC1Params") ?? "";
			udc1Encoder = sr.Load("UDC1Encoder") ?? "";
			udc1EncParams = sr.Load("UDC1EncParams") ?? "";
			udc1APEv2 = sr.LoadBoolean("UDC1APEv2") ?? false;
			udc1ID3v2 = sr.LoadBoolean("UDC1ID3v2") ?? false;
		}

		public string CleanseString (string s)
		{
			StringBuilder sb = new StringBuilder();
			char[] invalid = Path.GetInvalidFileNameChars();

			if (filenamesANSISafe)
				s = Encoding.Default.GetString(Encoding.Default.GetBytes(s));

			for (int i = 0; i < s.Length; i++)
			{
				char ch = s[i];
				if (filenamesANSISafe && removeSpecial && specialExceptions.IndexOf(ch) < 0 && !(
					((ch >= 'a') && (ch <= 'z')) ||
					((ch >= 'A') && (ch <= 'Z')) ||
					((ch >= '0') && (ch <= '9')) ||
					(ch == ' ') || (ch == '_')))
					ch = '_';
				if ((Array.IndexOf(invalid, ch) >= 0) || (replaceSpaces && ch == ' '))
					sb.Append("_");
				else
					sb.Append(ch);
			}

			return sb.ToString();
		}
	}

	public class CUEToolsProgressEventArgs
	{
		public string status = string.Empty;
		public double percentTrck = 0;
		public double percentDisk = 0.0;
		public string input = string.Empty;
		public string output = string.Empty;
	}

	public class ArchivePasswordRequiredEventArgs
	{
		public string Password = string.Empty;
		public bool ContinueOperation = true;
	}

	public class CUEToolsSelectionEventArgs
	{
		public string[] choices;
		public int selection = -1;
	}

	public delegate void CUEToolsSelectionHandler(object sender, CUEToolsSelectionEventArgs e);
	public delegate void CUEToolsProgressHandler(object sender, CUEToolsProgressEventArgs e);
	public delegate void ArchivePasswordRequiredHandler(object sender, ArchivePasswordRequiredEventArgs e);

	public class CUESheet {
		private bool _stop, _pause;
		private List<CUELine> _attributes;
		private List<TrackInfo> _tracks;
		private List<SourceInfo> _sources;
		private List<string> _sourcePaths, _trackFilenames;
		private string _htoaFilename, _singleFilename;
		private bool _hasHTOAFilename = false, _hasTrackFilenames = false, _hasSingleFilename = false, _appliedWriteOffset;
		private bool _hasEmbeddedCUESheet;
		private bool _paddedToFrame, _truncated4608, _usePregapForFirstTrackInSingleFile;
		private int _writeOffset;
		private AccurateRipMode _accurateRipMode;
		private uint? _dataTrackLength;
		private uint? _minDataTrackLength;
		private string _accurateRipId;
		private string _accurateRipIdActual;
		private string _mbDiscId;
		private string _mbReleaseId;
		private string _eacLog;
		private string _cuePath;
		private TagLib.File _fileInfo;
		private const int _arOffsetRange = 5 * 588 - 1;
		private HDCDDotNet.HDCDDotNet hdcdDecoder;
		private bool _outputLossyWAV = false;
		private OutputAudioFormat _outputFormat = OutputAudioFormat.WAV;
		private CUEConfig _config;
		private string _cddbDiscIdTag;
		private bool _isCD;
		private CDDriveReader _ripper;
		private bool _isArchive;
		private List<string> _archiveContents;
		private string _archiveCUEpath;
		private string _archivePath;
		private string _archivePassword;
		private CUEToolsProgressEventArgs _progress;
		private AccurateRipVerify _arVerify;
		private CDImageLayout _toc;

		public event ArchivePasswordRequiredHandler PasswordRequired;
		public event CUEToolsProgressHandler CUEToolsProgress;
		public event CUEToolsSelectionHandler CUEToolsSelection;

		public CUESheet(CUEConfig config)
		{
			_config = config;
			_progress = new CUEToolsProgressEventArgs();
			_attributes = new List<CUELine>();
			_tracks = new List<TrackInfo>();
			_trackFilenames = new List<string>();
			_toc = new CDImageLayout();
			_sources = new List<SourceInfo>();
			_sourcePaths = new List<string>();
			_stop = false;
			_pause = false;
			_cuePath = null;
			_paddedToFrame = false;
			_truncated4608 = false;
			_usePregapForFirstTrackInSingleFile = false;
			_accurateRipMode = AccurateRipMode.None;
			_appliedWriteOffset = false;
			_dataTrackLength = null;
			_minDataTrackLength = null;
			hdcdDecoder = null;
			_hasEmbeddedCUESheet = false;
			_isArchive = false;
			_isCD = false;
		}

		public void OpenCD(CDDriveReader ripper)
		{
			_ripper = ripper;
			_toc = (CDImageLayout)_ripper.TOC.Clone();
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				_trackFilenames.Add(string.Format("{0:00}.wav", iTrack + 1));
				_tracks.Add(new TrackInfo());
			}
			_accurateRipId = _accurateRipIdActual = AccurateRipVerify.CalculateAccurateRipId(_toc);
			_arVerify = new AccurateRipVerify(_toc);
			_isCD = true;
			SourceInfo cdInfo;
			cdInfo.Path = _ripper.ARName;
			cdInfo.Offset = 0;
			cdInfo.Length = _toc.AudioLength * 588;
			_sources.Add(cdInfo);
			_ripper.ReadProgress += new EventHandler<ReadProgressArgs>(CDReadProgress);
		}

		public void Close()
		{
			if (_ripper != null)
				_ripper.Close();
			_ripper = null;
		}

		public AccurateRipVerify ArVerify
		{
			get
			{
				return _arVerify;
			}
		}

		public void FillFromMusicBrainz(MusicBrainz.Release release)
		{
			Year = release.GetEvents().Count > 0 ? release.GetEvents()[0].Date.Substring(0, 4) : "";
			Artist = release.GetArtist();
			Title = release.GetTitle();
			//Catalog = release.GetEvents().Count > 0 ? release.GetEvents()[0].Barcode : "";
			for (int i = 1; i <= _toc.AudioTracks; i++)
			{
				MusicBrainz.Track track = release.GetTracks()[(int)_toc[i].Number - 1]; // !!!!!! - _toc.FirstAudio
				Tracks[i - 1].Title = track.GetTitle();
				Tracks[i - 1].Artist = track.GetArtist();
			}
		}

		public void FillFromFreedb(Freedb.CDEntry cdEntry)
		{
			Year = cdEntry.Year;
			Genre = cdEntry.Genre;
			Artist = cdEntry.Artist;
			Title = cdEntry.Title;
			for (int i = 0; i < _toc.AudioTracks; i++)
				Tracks[i].Title = cdEntry.Tracks[i].Title;
		}

		public List<object> LookupAlbumInfo()
		{
			List<object> Releases = new List<object>();

			ReleaseQueryParameters p = new ReleaseQueryParameters();
			p.DiscId = _toc.MusicBrainzId;
			Query<Release> results = Release.Query(p);
			MusicBrainzService.XmlRequest += new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			_progress.percentDisk = 0;
			try
			{
				foreach (MusicBrainz.Release release in results)
				{
					release.GetEvents();
					release.GetTracks();
					try
					{
						foreach (MusicBrainz.Track track in release.GetTracks())
							;
					} catch { }
					try
					{
						foreach (MusicBrainz.Event ev in release.GetEvents())
							;
					} catch { }
					Releases.Add(release);
				}
			} catch { }
			MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			//if (release != null)
			//{
			//    FillFromMusicBrainz(release);
			//    return;
			//}
			//if (cdEntry != null)
			//    FillFromFreedb(cdEntry);

			FreedbHelper m_freedb = new FreedbHelper();

			m_freedb.UserName = "gchudov";
			m_freedb.Hostname = "gmail.com";
			m_freedb.ClientName = "CUETools";
			m_freedb.Version = "1.9.4";
			m_freedb.SetDefaultSiteAddress("freedb.org");

			QueryResult queryResult;
			QueryResultCollection coll;
			string code = string.Empty;
			CDEntry cdEntry = null;
			try
			{
				code = m_freedb.Query(AccurateRipVerify.CalculateCDDBQuery(_toc), out queryResult, out coll);
				if (code == FreedbHelper.ResponseCodes.CODE_200)
				{
					code = m_freedb.Read(queryResult, out cdEntry);
					if (code == FreedbHelper.ResponseCodes.CODE_210)
						Releases.Add(cdEntry);
				}
				else
					if (code == FreedbHelper.ResponseCodes.CODE_210 ||
						code == FreedbHelper.ResponseCodes.CODE_211)
					{
						foreach (QueryResult qr in coll)
						{
							code = m_freedb.Read(qr, out cdEntry);
							if (code == FreedbHelper.ResponseCodes.CODE_210)
								Releases.Add(cdEntry);
						}
					}
			}
			catch (Exception)
			{
			}
			return Releases;
		}

		public void Open(string pathIn)
		{
			string cueDir = Path.GetDirectoryName(pathIn) ?? pathIn;
#if !MONO
			if (cueDir == pathIn)
			{
				CDDriveReader ripper = new CDDriveReader();
				ripper.Open(pathIn[0]);
				if (ripper.TOC.AudioTracks > 0)
				{
					OpenCD(ripper);
					int driveOffset;
					if (!AccurateRipVerify.FindDriveReadOffset(_ripper.ARName, out driveOffset))
						throw new Exception("Failed to find drive read offset for drive" + _ripper.ARName);
					_ripper.DriveOffset = driveOffset;
					LookupAlbumInfo();
					return;
				}
			}
#endif

			SourceInfo sourceInfo;
			string lineStr, command, pathAudio = null, fileType;
			CUELine line;
			TrackInfo trackInfo = null;
			int timeRelativeToFileStart, absoluteFileStartTime = 0;
			int fileTimeLengthSamples = 0, fileTimeLengthFrames = 0, i;
			bool fileIsBinary = false;
			int trackNumber = 0;
			bool isAudioTrack = true;
			bool seenFirstFileIndex = false;
			List<IndexInfo> indexes = new List<IndexInfo>();
			IndexInfo indexInfo;
			TagLib.File _trackFileInfo = null;
			TextReader sr;

			if (Directory.Exists(pathIn))
			{
				if (cueDir + Path.DirectorySeparatorChar != pathIn && cueDir != pathIn)
					throw new Exception("Input directory must end on path separator character.");
				string cueSheet = null;
				string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a", "*.tta" };
				for (i = 0; i < audioExts.Length && cueSheet == null; i++)
					cueSheet = CUESheet.CreateDummyCUESheet(pathIn, audioExts[i]);
				if (_config.udc1Extension != null && cueSheet == null)
					cueSheet = CUESheet.CreateDummyCUESheet(pathIn, "*." + _config.udc1Extension);
				if (cueSheet == null)
					throw new Exception("Input directory doesn't contain supported audio files.");
				sr = new StringReader(cueSheet);				
				if (CUEToolsSelection != null)
				{
					CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
					e.choices = Directory.GetFiles(pathIn, "*.log");
					if (e.choices.Length > 0)
					{
						CUEToolsSelection(this, e);
						if (e.selection != -1)
						{
							StreamReader logReader = new StreamReader(e.choices[e.selection], CUESheet.Encoding);
							_eacLog = logReader.ReadToEnd();
							logReader.Close();
						}
					}
				}
			} 
			else if (Path.GetExtension(pathIn).ToLower() == ".zip" || Path.GetExtension(pathIn).ToLower() == ".rar")
			{				
				_archiveContents = new List<string>();
				_isArchive = true;
				_archivePath = pathIn;

				if (Path.GetExtension(pathIn).ToLower() == ".rar")
				{
#if !MONO
					using (Unrar _unrar = new Unrar())
					{
						_unrar.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
						_unrar.Open(pathIn, Unrar.OpenMode.List);
						while (_unrar.ReadHeader())
						{
							if (!_unrar.CurrentFile.IsDirectory)
								_archiveContents.Add(_unrar.CurrentFile.FileName);
							_unrar.Skip();
						}
						_unrar.Close();
					}
#else
					throw new Exception("rar archives not supported on MONO.");
#endif
				}
				if (Path.GetExtension(pathIn).ToLower() == ".zip")
				{
					using (ZipFile _unzip = new ZipFile(pathIn))
					{
						foreach (ZipEntry e in _unzip)
						{
							if (e.IsFile)
								_archiveContents.Add(e.Name);
						}
						_unzip.Close();
					}
				}

				string cueName = null, cueText = null, logName = null;
				List<string> cueNames = new List<string>();
				List<string> logNames = new List<string>();

				foreach (string s in _archiveContents)
				{
					if (Path.GetExtension(s).ToLower() == ".cue")
						cueNames.Add(s);
					if (Path.GetExtension(s).ToLower() == ".log")
						logNames.Add(s);
				}
				if (cueNames.Count == 0)
					throw new Exception("Input archive doesn't contain a cue sheet.");
				if (cueNames.Count == 1)
					cueName = cueNames[0];
				if (cueName == null && CUEToolsSelection != null)
				{
					CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
					e.choices = cueNames.ToArray();
					CUEToolsSelection(this, e);
					if (e.selection != -1)
						cueName = e.choices[e.selection];
				}
				if (cueName == null)
					throw new Exception("Input archive contains several cue sheets.");

				if (logNames.Contains(Path.ChangeExtension(cueName, ".log")))
					logName = Path.ChangeExtension(cueName, ".log");
				if (logName == null && CUEToolsSelection != null && logNames.Count > 0)
				{
					CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
					e.choices = logNames.ToArray();
					CUEToolsSelection(this, e);
					if (e.selection != -1)
						logName = e.choices[e.selection];
				}

				if (cueName != null)
				{
					Stream archiveStream = OpenArchive(cueName, false);
					StreamReader cueReader = new StreamReader(archiveStream, CUESheet.Encoding);
					cueText = cueReader.ReadToEnd();
					cueReader.Close();
					archiveStream.Close();
					if (cueText == "")
						throw new Exception("Empty cue sheet.");
				}
				if (cueText == null)
					throw new Exception("Input archive doesn't contain a cue sheet.");
				if (logName != null)
				{
					Stream archiveStream = OpenArchive(logName, false);
					StreamReader logReader = new StreamReader(archiveStream, CUESheet.Encoding);
					_eacLog = logReader.ReadToEnd();
					logReader.Close();
					archiveStream.Close();
				}
				_archiveCUEpath = Path.GetDirectoryName(cueName);
				if (_config.autoCorrectFilenames)
					cueText = CorrectAudioFilenames(_archiveCUEpath, cueText, false, _archiveContents);
				sr = new StringReader(cueText);
			}
			else if (Path.GetExtension(pathIn).ToLower() == ".cue")
			{
				if (_config.autoCorrectFilenames)
					sr = new StringReader (CorrectAudioFilenames(pathIn, false));
				else
					sr = new StreamReader (pathIn, CUESheet.Encoding);

				string logPath = Path.ChangeExtension(pathIn, ".log");
				if (System.IO.File.Exists(logPath))
				{
					StreamReader logReader = new StreamReader(logPath, CUESheet.Encoding);
					_eacLog = logReader.ReadToEnd();
					logReader.Close();
				}
				else if (CUEToolsSelection != null)
				{
					CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
					e.choices = Directory.GetFiles(cueDir == "" ? "." : cueDir, "*.log");
					if (e.choices.Length > 0)
					{
						CUEToolsSelection(this, e);
						if (e.selection != -1)
						{
							StreamReader logReader = new StreamReader(e.choices[e.selection], CUESheet.Encoding);
							_eacLog = logReader.ReadToEnd();
							logReader.Close();
						}
					}
				}
			} else
			{
				string cuesheetTag = null;
				TagLib.File fileInfo;
				GetSampleLength(pathIn, out fileInfo);
				NameValueCollection tags = Tagging.Analyze(fileInfo);
				cuesheetTag = tags.Get("CUESHEET");
				_accurateRipId = tags.Get("ACCURATERIPID");
				_eacLog = tags.Get("LOG");
				if (_eacLog == null) _eacLog = tags.Get("LOGFILE");
				if (_eacLog == null) _eacLog = tags.Get("EACLOG");
				if (cuesheetTag == null)
					throw new Exception("Input file does not contain a .cue sheet.");
				sr = new StringReader (cuesheetTag);
				pathAudio = pathIn;
				_hasEmbeddedCUESheet = true;
			}

			using (sr) {
				while ((lineStr = sr.ReadLine()) != null) {
					line = new CUELine(lineStr);
					if (line.Params.Count > 0) {
						command = line.Params[0].ToUpper();

						if (command == "FILE") {
							fileType = line.Params[2].ToUpper();
							if ((fileType == "BINARY") || (fileType == "MOTOROLA"))
								fileIsBinary = true;
							else 
							{
								fileIsBinary = false;
								if (!_hasEmbeddedCUESheet)
								{
									if (_isArchive)
										pathAudio = LocateFile(_archiveCUEpath, line.Params[1], _archiveContents);
                                    else
										pathAudio = LocateFile(cueDir, line.Params[1], null);
									if (pathAudio == null)
										throw new Exception("Unable to locate file \"" + line.Params[1] + "\".");
								} else
								{
									if (_sourcePaths.Count > 0 )
										throw new Exception("Extra file in embedded CUE sheet: \"" + line.Params[1] + "\".");
								}
								_sourcePaths.Add(pathAudio);
								absoluteFileStartTime += fileTimeLengthFrames;
								TagLib.File fileInfo;
								fileTimeLengthSamples = GetSampleLength(pathAudio, out fileInfo);
								if ((fileTimeLengthSamples % 588) == 492 && _config.truncate4608ExtraSamples)
								{
									_truncated4608 = true;
									fileTimeLengthSamples -= 4608;
								}
								fileTimeLengthFrames = (int)((fileTimeLengthSamples + 587) / 588);
								if (_hasEmbeddedCUESheet)
									_fileInfo = fileInfo;
								else
									_trackFileInfo = fileInfo;
								seenFirstFileIndex = false;
							}
						}
						else if (command == "TRACK") 
						{
							isAudioTrack = line.Params[2].ToUpper() == "AUDIO";
							trackNumber = int.Parse(line.Params[1]);
							if (trackNumber != _toc.TrackCount + 1)
								throw new Exception("Invalid track number.");
							_toc.AddTrack(new CDTrack((uint)trackNumber, 0, 0, isAudioTrack, false));
							if (isAudioTrack)
							{
								trackInfo = new TrackInfo();
								_tracks.Add(trackInfo);
							}
						}
						else if (command == "INDEX") 
						{
							timeRelativeToFileStart = CDImageLayout.TimeFromString(line.Params[2]);
							if (!seenFirstFileIndex)
							{
								if (timeRelativeToFileStart != 0)
									throw new Exception("First index must start at file beginning.");
								seenFirstFileIndex = true;
								if (isAudioTrack)
								{
									if (_tracks.Count > 0 && _trackFileInfo != null)
										_tracks[_tracks.Count - 1]._fileInfo = _trackFileInfo;
									_trackFileInfo = null;
									sourceInfo.Path = pathAudio;
									sourceInfo.Offset = 0;
									sourceInfo.Length = (uint)fileTimeLengthSamples;
									_sources.Add(sourceInfo);
									if ((fileTimeLengthSamples % 588) != 0)
									{
										sourceInfo.Path = null;
										sourceInfo.Offset = 0;
										sourceInfo.Length = (uint)((fileTimeLengthFrames * 588) - fileTimeLengthSamples);
										_sources.Add(sourceInfo);
										_paddedToFrame = true;
									}
								}
							}
							else if (fileIsBinary)
							{
								fileTimeLengthFrames = timeRelativeToFileStart + 150;
								sourceInfo.Path = null;
								sourceInfo.Offset = 0;
								sourceInfo.Length = 150 * 588;
								_sources.Add(sourceInfo);
							}
							indexInfo.Track = trackNumber;
							indexInfo.Index = Int32.Parse(line.Params[1]);
							indexInfo.Time = absoluteFileStartTime + timeRelativeToFileStart;
							indexes.Add(indexInfo);
						}
						else if (!isAudioTrack)
						{
							// Ignore lines belonging to data tracks
						}
						else if (command == "PREGAP")
						{
							if (seenFirstFileIndex)
								throw new Exception("Pregap must occur at the beginning of a file.");
							int pregapLength = CDImageLayout.TimeFromString(line.Params[1]);
							indexInfo.Track = trackNumber;
							indexInfo.Index = 0;
							indexInfo.Time = absoluteFileStartTime;
							indexes.Add(indexInfo);
							sourceInfo.Path = null;
							sourceInfo.Offset = 0;
							sourceInfo.Length = (uint)pregapLength * 588;
							_sources.Add(sourceInfo);
							absoluteFileStartTime += pregapLength;
						}
						else if (command == "POSTGAP") {
							throw new Exception("POSTGAP command isn't supported.");
						}
						else if ((command == "REM") &&
							(line.Params.Count >= 3) &&
							(line.Params[1].Length >= 10) &&
							(line.Params[1].Substring(0, 10).ToUpper() == "REPLAYGAIN"))
						{
							// Remove ReplayGain lines
						}
						else if ((command == "REM") &&
							(line.Params.Count == 3) &&
							(line.Params[1].ToUpper() == "DATATRACKLENGTH"))
						{
							_dataTrackLength = (uint)CDImageLayout.TimeFromString(line.Params[2]);
						}
						else if ((command == "REM") &&
						   (line.Params.Count == 3) &&
						   (line.Params[1].ToUpper() == "ACCURATERIPID"))
						{
							_accurateRipId = line.Params[2];
						}
						//else if ((command == "REM") &&
						//   (line.Params.Count == 3) &&
						//   (line.Params[1].ToUpper() == "SHORTEN"))
						//{
						//    fileTimeLengthFrames -= General.TimeFromString(line.Params[2]);
						//}							
						//else if ((command == "REM") &&
						//   (line.Params.Count == 3) &&
						//   (line.Params[1].ToUpper() == "LENGTHEN"))
						//{
						//    fileTimeLengthFrames += General.TimeFromString(line.Params[2]);
						//}							
						else
						{
							if (trackInfo != null)
							{
								trackInfo.Attributes.Add(line);
							}
							else
							{
								if (line.Params.Count > 2 && !line.IsQuoted[1] &&
									(line.Params[0].ToUpper() == "TITLE" || line.Params[0].ToUpper() == "ARTIST" ||
									(line.Params[0].ToUpper() == "REM" && line.Params[1].ToUpper() == "GENRE" && line.Params.Count > 3 && !line.IsQuoted[2])))
								{
									CUELine modline = new CUELine();
									int nParams = line.Params[0].ToUpper() == "REM" ? 2 : 1;
									for (int iParam = 0; iParam < nParams; iParam++)
									{
										modline.Params.Add(line.Params[iParam]);
										modline.IsQuoted.Add(false);
									}
									string s = line.Params[nParams];
									for (int iParam = nParams + 1; iParam < line.Params.Count; iParam++)
										s += " " + line.Params[iParam];
									modline.Params.Add(s); 
									modline.IsQuoted.Add(true);
									line = modline;
								}
								_attributes.Add(line);
							}
						}
					}
				}
				sr.Close();
			}

			if (_tracks.Count == 0)
				throw new Exception("File must contain at least one audio track.");

			// Add dummy index 01 for data track
			if (!_toc[_toc.TrackCount].IsAudio && indexes[indexes.Count - 1].Index == 0)
			{
				indexInfo.Track = trackNumber;
				indexInfo.Index = 1;
				indexInfo.Time = absoluteFileStartTime + fileTimeLengthFrames;
				indexes.Add(indexInfo);
			}

			// Add dummy track for calculation purposes
			indexInfo.Track = trackNumber + 1;
			indexInfo.Index = 1;
			indexInfo.Time = absoluteFileStartTime + fileTimeLengthFrames;
			indexes.Add(indexInfo);

			// Calculate the length of each index
			for (i = 0; i < indexes.Count - 1; i++) 
			{
				if (indexes[i + 1].Time - indexes[i].Time < 0)
					throw new Exception("Indexes must be in chronological order.");
				if ((indexes[i+1].Track != indexes[i].Track || indexes[i+1].Index != indexes[i].Index + 1) &&
					(indexes[i + 1].Track != indexes[i].Track + 1 || indexes[i].Index < 1 || indexes[i + 1].Index > 1))
					throw new Exception("Indexes must be in chronological order.");
				if (indexes[i].Index == 1 && (i == 0 || indexes[i - 1].Index != 0))
					_toc[indexes[i].Track].AddIndex(new CDTrackIndex(0U, (uint)indexes[i].Time));
				_toc[indexes[i].Track].AddIndex(new CDTrackIndex((uint)indexes[i].Index, (uint)indexes[i].Time));
			}

			// Calculate the length of each track
			for (int iTrack = 1; iTrack <= _toc.TrackCount; iTrack++)
			{
				_toc[iTrack].Start = _toc[iTrack][1].Start;
				_toc[iTrack].Length = (iTrack == _toc.TrackCount ? (uint)indexes[indexes.Count - 1].Time - _toc[iTrack].Start : _toc[iTrack + 1][1].Start - _toc[iTrack].Start);
			}

			// Store the audio filenames, generating generic names if necessary
			_hasSingleFilename = (_sourcePaths.Count == 1);
			_singleFilename = _hasSingleFilename ? Path.GetFileName(_sourcePaths[0]) :
				"Range.wav";

			_hasHTOAFilename = (_sourcePaths.Count == (TrackCount + 1));
			_htoaFilename = _hasHTOAFilename ? Path.GetFileName(_sourcePaths[0]) : "01.00.wav";

			_hasTrackFilenames = (_sourcePaths.Count == TrackCount) || _hasHTOAFilename;
			for (i = 0; i < TrackCount; i++) {
				_trackFilenames.Add( _hasTrackFilenames ? Path.GetFileName(
					_sourcePaths[i + (_hasHTOAFilename ? 1 : 0)]) : String.Format("{0:00}.wav", i + 1) );
			}
			if (!_hasEmbeddedCUESheet && _hasSingleFilename)
			{
				_fileInfo = _tracks[0]._fileInfo;
				_tracks[0]._fileInfo = null;
			}
			if (_config.fillUpCUE)
			{
				if (_config.overwriteCUEData || General.FindCUELine(_attributes, "PERFORMER") == null)
				{
					string value = GetCommonTag(delegate(TagLib.File file) { return file.Tag.JoinedAlbumArtists; });
					if (value == null)
						value = GetCommonTag(delegate(TagLib.File file) { return file.Tag.JoinedPerformers; });
					if (value != null)
						General.SetCUELine(_attributes, "PERFORMER", value, true);
				}
				if (_config.overwriteCUEData || General.FindCUELine(_attributes, "TITLE") == null)
				{
					string value = GetCommonTag(delegate(TagLib.File file) { return file.Tag.Album; });
					if (value != null)
						General.SetCUELine(_attributes, "TITLE", value, true);
				}
				if (_config.overwriteCUEData || General.FindCUELine(_attributes, "REM", "DATE") == null)
				{
					string value = GetCommonTag(delegate(TagLib.File file) { return file.Tag.Year != 0 ? file.Tag.Year.ToString() : null; });
					if (value != null)
						General.SetCUELine(_attributes, "REM", "DATE", value, false);
				}
				if (_config.overwriteCUEData || General.FindCUELine(_attributes, "REM", "GENRE") == null)
				{
					string value = GetCommonTag(delegate(TagLib.File file) { return file.Tag.JoinedGenres; });
					if (value != null)
						General.SetCUELine(_attributes, "REM", "GENRE", value, true);
				}
				for (i = 0; i < TrackCount; i++)
				{
					TrackInfo track = _tracks[i];
					string artist = _hasTrackFilenames ? track._fileInfo.Tag.JoinedPerformers :
						_hasEmbeddedCUESheet ? Tagging.TagListToSingleValue(Tagging.GetMiscTag(_fileInfo, String.Format("cue_track{0:00}_ARTIST", i + 1))) :
						null;
					string title = _hasTrackFilenames ? track._fileInfo.Tag.Title :
						_hasEmbeddedCUESheet ? Tagging.TagListToSingleValue(Tagging.GetMiscTag(_fileInfo, String.Format("cue_track{0:00}_TITLE", i + 1))) :
						null;
					if ((_config.overwriteCUEData || track.Artist == "") && artist != null)
						track.Artist = artist;
					if ((_config.overwriteCUEData || track.Title == "") && title != null)
						track.Title = title;
				}
			}

			CUELine cddbDiscIdLine = General.FindCUELine(_attributes, "REM", "DISCID");
			_cddbDiscIdTag = cddbDiscIdLine != null && cddbDiscIdLine.Params.Count == 3 ? cddbDiscIdLine.Params[2] : null;
			if (_cddbDiscIdTag == null)
				_cddbDiscIdTag = GetCommonMiscTag("DISCID");

			if (_accurateRipId == null)
				_accurateRipId = GetCommonMiscTag("ACCURATERIPID");

			if (_accurateRipId == null)
			{
				if (_dataTrackLength != null)
				{
					// TODO: check if we have a data track of unknown length already, and just change it's length!
					CDImageLayout toc2 = new CDImageLayout(_toc);
					toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152U * 75U, _dataTrackLength.Value, false, false));
					_accurateRipId = AccurateRipVerify.CalculateAccurateRipId(toc2);
				}
				else
				{
					bool dtlFound = false;
					if (_eacLog != null)
					{
						sr = new StringReader(_eacLog);
						bool isEACLog = false;
						CDImageLayout tocFromLog = new CDImageLayout();
						while ((lineStr = sr.ReadLine()) != null)
						{
							if (isEACLog)
							{
								string[] n = lineStr.Split('|');
								uint trNo, trStart, trEnd;
								if (n.Length == 5 && uint.TryParse(n[0], out trNo) && uint.TryParse(n[3], out trStart) && uint.TryParse(n[4], out trEnd) && trNo == tocFromLog.TrackCount + 1)
								{
									bool isAudio = true;
									if (tocFromLog.TrackCount >= _toc.TrackCount &&
										trStart == tocFromLog[tocFromLog.TrackCount].End + 1U + 152U * 75U
										)
										isAudio = false;
									if (tocFromLog.TrackCount < _toc.TrackCount &&
										!_toc[tocFromLog.TrackCount + 1].IsAudio
										)
										isAudio = false;
									tocFromLog.AddTrack(new CDTrack(trNo, trStart, trEnd + 1 - trStart, isAudio, false));
								}
							}
							else
								if (lineStr.StartsWith("TOC of the extracted CD")
									|| lineStr.StartsWith("Exact Audio Copy")
									|| lineStr.StartsWith("CUERipper"))
									isEACLog = true;
						}
						if (tocFromLog.TrackCount == _toc.TrackCount + 1 && !tocFromLog[tocFromLog.TrackCount].IsAudio)
						{
							//_accurateRipId = AccurateRipVerify.CalculateAccurateRipId(tocFromLog);
							_toc.AddTrack(new CDTrack((uint)tocFromLog.TrackCount, tocFromLog[tocFromLog.TrackCount].Start, tocFromLog[tocFromLog.TrackCount].Length, false, false));
							dtlFound = true;
						}
						else if (tocFromLog.TrackCount == _toc.TrackCount)
						{
							if (!tocFromLog[1].IsAudio)
							{
								for (i = 2; i <= _toc.TrackCount; i++)
								{
									_toc[i].Start += tocFromLog[1].Length - _toc[1].Length;
									for (int j = 0; j <= _toc[i].LastIndex; j++)
										_toc[i][j].Start += tocFromLog[1].Length - _toc[1].Length;
								}
								_toc[1].Length = tocFromLog[1].Length;
								dtlFound = true;
							}
							else if (!tocFromLog[tocFromLog.TrackCount].IsAudio)
							{
								_toc[_toc.TrackCount].Start = tocFromLog[_toc.TrackCount].Start;
								_toc[_toc.TrackCount].Length = tocFromLog[_toc.TrackCount].Length;
								_toc[_toc.TrackCount][0].Start = tocFromLog[_toc.TrackCount].Start;
								_toc[_toc.TrackCount][1].Start = tocFromLog[_toc.TrackCount].Start;
								dtlFound = true;
							}
						}
					}
					if (!dtlFound && _cddbDiscIdTag != null)
					{
						uint cddbDiscIdNum;
						if (uint.TryParse(_cddbDiscIdTag, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out cddbDiscIdNum) && (cddbDiscIdNum & 0xff) == _toc.TrackCount + 1)
						{
							uint lengthFromTag = ((cddbDiscIdNum >> 8) & 0xffff);
							_minDataTrackLength = ((lengthFromTag + _toc[1].Start / 75) - 152) * 75 - _toc.Length;
						}
					}
				}
			}

			_accurateRipIdActual = AccurateRipVerify.CalculateAccurateRipId(_toc);

			if (_accurateRipId == null)
				_accurateRipId = _accurateRipIdActual;

			_arVerify = new AccurateRipVerify(_toc);

			if (_eacLog != null)
			{
				sr = new StringReader(_eacLog);
				bool isEACLog = false;
				int trNo = 1;
				while ((lineStr = sr.ReadLine()) != null)
				{
					if (isEACLog && trNo <= TrackCount)
					{
						string[] s = { "Copy CRC ", "CRC копии" };
						string[] n = lineStr.Split(s, StringSplitOptions.None);
						uint crc;
						if (n.Length == 2 && uint.TryParse(n[1], NumberStyles.HexNumber, CultureInfo.CurrentCulture, out crc))
							_arVerify.CRCLOG(trNo++, crc);
					}
					else
						if (lineStr.StartsWith("Exact Audio Copy"))
							isEACLog = true;
				}
				if (trNo == 2)
				{
					_arVerify.CRCLOG(0, _arVerify.CRCLOG(1));
					if (TrackCount > 1)
						_arVerify.CRCLOG(1, 0);
				}
			}

			//if (!_dataTrackLength.HasValue && _cddbDiscIdTag != null)
			//{
			//    uint cddbDiscIdNum = UInt32.Parse(_cddbDiscIdTag, NumberStyles.HexNumber);
			//    if ((cddbDiscIdNum & 0xff) == TrackCount)
			//    {
			//        _cutOneFrame = true;
			//        string cddbDiscIdTagCut = CalculateAccurateRipId().Split('-')[2];
			//        if (cddbDiscIdTagCut.ToUpper() != _cddbDiscIdTag.ToUpper())
			//            _cutOneFrame = false;
			//    }
			//}
		}

		public static Encoding Encoding {
			get {
				return Encoding.Default;
			}
		}

		internal Stream OpenArchive(string fileName, bool showProgress)
		{
#if !MONO
			if (Path.GetExtension(_archivePath).ToLower() == ".rar")
			{
				RarStream rarStream = new RarStream(_archivePath, fileName);
				rarStream.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
				if (showProgress)
					rarStream.ExtractionProgress += new ExtractionProgressHandler(unrar_ExtractionProgress);
				return rarStream;
			}
#endif
			if (Path.GetExtension(_archivePath).ToLower() == ".zip")
			{
				SeekableZipStream zipStream = new SeekableZipStream(_archivePath, fileName);
				zipStream.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
				if (showProgress)
					zipStream.ExtractionProgress += new ExtractionProgressHandler(unrar_ExtractionProgress);
				return zipStream;
			}
			throw new Exception("Unknown archive type.");
		}

		private void ShowProgress(string status, double percentTrack, double percentDisk, string input, string output)
		{
			if (this.CUEToolsProgress == null)
				return;
			_progress.status = status;
			_progress.percentTrck = percentTrack;
			_progress.percentDisk = percentDisk;
			_progress.input = input;
			_progress.output = output;
			this.CUEToolsProgress(this, _progress);
		}

#if !MONO
		private void CDReadProgress(object sender, ReadProgressArgs e)
		{
			lock (this)
			{
				if (_stop)
					throw new StopException();
				if (_pause)
				{
					ShowProgress("Paused...", 0, 0, null, null);
					Monitor.Wait(this);
				}
			}
			if (this.CUEToolsProgress == null)
				return;
			CDDriveReader audioSource = (CDDriveReader)sender;
			int processed = e.Position - e.PassStart;
			TimeSpan elapsed = DateTime.Now - e.PassTime;
			double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;
			_progress.percentDisk = (double)(e.PassStart + (processed + e.Pass * (e.PassEnd - e.PassStart)) / (audioSource.CorrectionQuality + 1)) / audioSource.TOC.AudioLength;
			_progress.percentTrck = (double) (e.Position - e.PassStart) / (e.PassEnd - e.PassStart);
			_progress.status = string.Format("Ripping @{0:00.00}x {1}", speed, e.Pass > 0 ? " (Retry " + e.Pass.ToString() + ")" : "");
			this.CUEToolsProgress(this, _progress);
		}

		private void MusicBrainz_LookupProgress(object sender, XmlRequestEventArgs e)
		{
			if (this.CUEToolsProgress == null)
				return;
			_progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
			_progress.percentTrck = 0;
			_progress.input = e.Uri.ToString();
			_progress.output = null;
			_progress.status = "Looking up album via MusicBrainz";
			this.CUEToolsProgress(this, _progress);
		}

		private void unrar_ExtractionProgress(object sender, ExtractionProgressEventArgs e)
		{
			if (this.CUEToolsProgress == null)
				return;
			_progress.percentTrck = e.PercentComplete/100;
			this.CUEToolsProgress(this, _progress);
		}

		private void unrar_PasswordRequired(object sender, PasswordRequiredEventArgs e)
		{
			if (_archivePassword != null)
			{
				e.ContinueOperation = true;
				e.Password = _archivePassword;
				return;
			}
			if (this.PasswordRequired != null)
			{
				ArchivePasswordRequiredEventArgs e1 = new ArchivePasswordRequiredEventArgs();
				this.PasswordRequired(this, e1);
				if (e1.ContinueOperation && e1.Password != "")
				{
					_archivePassword = e1.Password;
					e.ContinueOperation = true;
					e.Password = e1.Password;
					return;
				} 
			}
			throw new IOException("Password is required for extraction.");
		}
#endif

		public delegate string GetStringTagProvider(TagLib.File file);
		
		public string GetCommonTag(GetStringTagProvider provider)
		{
			if (_hasEmbeddedCUESheet || _hasSingleFilename)
				return General.EmptyStringToNull(provider(_fileInfo));
			if (_hasTrackFilenames)
			{
				string tagValue = null;
				bool commonValue = true;
				for (int i = 0; i < TrackCount; i++)
				{
					TrackInfo track = _tracks[i];
					string newValue = General.EmptyStringToNull(provider(track._fileInfo));
					if (tagValue == null)
						tagValue = newValue;
					else
						commonValue = (newValue == null || tagValue == newValue);
				}
				return commonValue ? tagValue : null;
			}
			return null;
		}

		public string GetCommonMiscTag(string tagName)
		{
			return GetCommonTag(delegate(TagLib.File file) { return Tagging.TagListToSingleValue(Tagging.GetMiscTag(file, tagName)); });
		}

		private static string LocateFile(string dir, string file, List<string> contents) {
			List<string> dirList, fileList;
			string altDir;

			dirList = new List<string>();
			fileList = new List<string>();
			altDir = Path.GetDirectoryName(file);
			file = Path.GetFileName(file);

			dirList.Add(dir);
			if (altDir.Length != 0) {
				dirList.Add(Path.IsPathRooted(altDir) ? altDir : Path.Combine(dir, altDir));
			}

			fileList.Add(file);
			fileList.Add(file.Replace(' ', '_'));
			fileList.Add(file.Replace('_', ' '));

			for (int iDir = 0; iDir < dirList.Count; iDir++) {
				for (int iFile = 0; iFile < fileList.Count; iFile++) {
					string path = Path.Combine(dirList[iDir], fileList[iFile]);
					if ((contents == null && System.IO.File.Exists(path))
						|| (contents != null && contents.Contains(path)))
						return path;
					path = dirList[iDir] + '/' + fileList[iFile];
					if (contents != null && contents.Contains(path))
						return path;
				}
			}

			return null;
		}

		public void GenerateFilenames(OutputAudioFormat format, bool outputLossyWAV, string outputPath)
		{
			_outputLossyWAV = outputLossyWAV;
			_outputFormat = format;
			_cuePath = outputPath;

			string extension = General.FormatExtension(format, _config);
			List<string> find, replace;
			string filename;
			int iTrack;

			find = new List<string>();
			replace = new List<string>();

			find.Add("%D"); // 0: Album artist
			find.Add("%C"); // 1: Album title
			find.Add("%N"); // 2: Track number
			find.Add("%A"); // 3: Track artist
			find.Add("%T"); // 4: Track title
			find.Add("%F"); // 5: Input filename
			find.Add("%Y"); // 6: Album date

			replace.Add(General.EmptyStringToNull(_config.CleanseString(Artist)));
			replace.Add(General.EmptyStringToNull(_config.CleanseString(Title)));
			replace.Add(null);
			replace.Add(null);
			replace.Add(null);
			replace.Add(Path.GetFileNameWithoutExtension(outputPath));
			replace.Add(General.EmptyStringToNull(_config.CleanseString(Year)));
			
			if (_outputLossyWAV)
				extension = ".lossy" + extension;
			if (_config.detectHDCD && _config.decodeHDCD && (!_outputLossyWAV || !_config.decodeHDCDtoLW16))
			{
				if (_config.decodeHDCDto24bit )
					extension = ".24bit" + extension;
				else
					extension = ".20bit" + extension;
			}

			if (_config.keepOriginalFilenames && HasSingleFilename)
			{
				SingleFilename = Path.ChangeExtension(SingleFilename, extension);
			}
			else
			{
				filename = General.ReplaceMultiple(_config.singleFilenameFormat, find, replace);
				if (filename == null)
					filename = "Range";
				filename += extension;
				SingleFilename = filename;
			}

			for (iTrack = -1; iTrack < TrackCount; iTrack++)
			{
				bool htoa = (iTrack == -1);

				if (_config.keepOriginalFilenames && htoa && HasHTOAFilename)
				{
					HTOAFilename = Path.ChangeExtension(HTOAFilename, extension);
				}
				else if (_config.keepOriginalFilenames && !htoa && HasTrackFilenames)
				{
					TrackFilenames[iTrack] = Path.ChangeExtension(
						TrackFilenames[iTrack], extension);
				}
				else
				{
					string trackStr = htoa ? "01.00" : String.Format("{0:00}", iTrack + 1);
					string artist = Tracks[htoa ? 0 : iTrack].Artist;
					string title = htoa ? "(HTOA)" : Tracks[iTrack].Title;

					replace[2] = trackStr;
					replace[3] = General.EmptyStringToNull(_config.CleanseString(artist==""?Artist:artist));
					replace[4] = General.EmptyStringToNull(_config.CleanseString(title));

					filename = General.ReplaceMultiple(_config.trackFilenameFormat, find, replace);
					if (filename == null)
						filename = replace[2];
					filename += extension;

					if (htoa)
					{
						HTOAFilename = filename;
					}
					else
					{
						TrackFilenames[iTrack] = filename;
					}
				}
			}
		}

		private int GetSampleLength(string path, out TagLib.File fileInfo)
		{
			ShowProgress("Analyzing input file...", 0.0, 0.0, path, null);
			
			TagLib.UserDefined.AdditionalFileTypes.Config = _config;
			TagLib.File.IFileAbstraction file = _isArchive
				? (TagLib.File.IFileAbstraction) new ArchiveFileAbstraction(this, path)
				: (TagLib.File.IFileAbstraction) new TagLib.File.LocalFileAbstraction(path);
			fileInfo = TagLib.File.Create(file);

			IAudioSource audioSource = AudioReadWrite.GetAudioSource(path, _isArchive ? OpenArchive(path, true) : null, _config);
			if ((audioSource.BitsPerSample != 16) ||
				(audioSource.ChannelCount != 2) ||
				(audioSource.SampleRate != 44100) ||
				(audioSource.Length > Int32.MaxValue))
			{
				audioSource.Close();
				throw new Exception("Audio format is invalid.");
			}
			audioSource.Close();
			return (int)audioSource.Length;
		}

		public static void WriteText(string path, string text)
		{
			bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(text)) != text;
			StreamWriter sw1 = new StreamWriter(path, false, utf8Required ? Encoding.UTF8 : CUESheet.Encoding);
			sw1.Write(text);
			sw1.Close();
		}

		public string LOGContents()
		{
			if (!_isCD || _ripper == null)
				return null;
#if !MONO
			StringWriter logWriter = new StringWriter();
			logWriter.WriteLine("{0}", CDDriveReader.RipperVersion());
			logWriter.WriteLine("Extraction logfile from : {0}", DateTime.Now);
			logWriter.WriteLine("Used drive              : {0}", _ripper.ARName);
			logWriter.WriteLine("Read offset correction  : {0}", _ripper.DriveOffset);
			logWriter.WriteLine("Read command            : {0}", _ripper.CurrentReadCommand);
			logWriter.WriteLine("Secure mode             : {0}", _ripper.CorrectionQuality);
			logWriter.WriteLine("Disk length             : {0}", CDImageLayout.TimeToString(_toc.AudioLength));
			logWriter.WriteLine("AccurateRip             : {0}", _arVerify.ARStatus == null ? "ok" : _arVerify.ARStatus);
			if (hdcdDecoder != null && hdcdDecoder.Detected)
			{
				hdcd_decoder_statistics stats;
				hdcdDecoder.GetStatistics(out stats);
				logWriter.WriteLine("HDCD                    : peak extend: {0}, transient filter: {1}, gain: {2}",
					(stats.enabled_peak_extend ? (stats.disabled_peak_extend ? "some" : "yes") : "none"),
					(stats.enabled_transient_filter ? (stats.disabled_transient_filter ? "some" : "yes") : "none"),
					stats.min_gain_adjustment == stats.max_gain_adjustment ?
					(stats.min_gain_adjustment == 1.0 ? "none" : String.Format("{0:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20))) :
					String.Format("{0:0.0}dB..{1:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20), (Math.Log10(stats.max_gain_adjustment) * 20))
					);
				logWriter.WriteLine();
			}
			logWriter.WriteLine();
			logWriter.WriteLine("TOC of the extracted CD");
			logWriter.WriteLine();
			logWriter.WriteLine("     Track |   Start  |  Length  | Start sector | End sector");
			logWriter.WriteLine("    ---------------------------------------------------------");
			for (int track = 1; track <= _toc.TrackCount; track++)
				logWriter.WriteLine("{0,9}  | {1,8} | {2,8} | {3,9}    | {4,9}",
					_toc[track].Number,
					_toc[track].StartMSF,
					_toc[track].LengthMSF,
					_toc[track].Start,
					_toc[track].End);
			bool wereErrors = false;
			for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
			{
				int cdErrors = 0;
				bool crcMismatch = _accurateRipMode == AccurateRipMode.VerifyThenConvert &&
					_arVerify.BackupCRC(iTrack) != _arVerify.CRC(iTrack);
				for (uint iSector = _toc[iTrack + 1].Start; iSector <= _toc[iTrack + 1].End; iSector++)
					if (_ripper.Errors[(int)iSector])
						cdErrors++;
				if (crcMismatch || cdErrors != 0)
				{
					if (!wereErrors)
					{
						logWriter.WriteLine();
						logWriter.WriteLine("Errors detected");
						logWriter.WriteLine();
					}
					wereErrors = true;
					if (crcMismatch)
						logWriter.WriteLine("Track {0} contains {1} errors, CRC mismatch: test {2:X8} vs copy {3:X8}", iTrack + 1, cdErrors, _arVerify.BackupCRC(iTrack), _arVerify.CRC(iTrack));
					else
						logWriter.WriteLine("Track {0} contains {1} errors", iTrack + 1, cdErrors);
				}
			}
			if (_accurateRipMode != AccurateRipMode.None)
			{
				logWriter.WriteLine();
				logWriter.WriteLine("AccurateRip summary");
				logWriter.WriteLine();
				_arVerify.GenerateFullLog(logWriter, 0);
				logWriter.WriteLine();
			}
			logWriter.WriteLine();
			logWriter.WriteLine("End of status report");
			logWriter.Close();
			return logWriter.ToString();
#else
			return null;
#endif
		}

		public string M3UContents(CUEStyle style)
		{
			StringWriter sw = new StringWriter();
			if (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0)
				WriteLine(sw, 0, _htoaFilename);
			for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				WriteLine(sw, 0, _trackFilenames[iTrack]);
			sw.Close();
			return sw.ToString();
		}

		public string TOCContents()
		{
			StringWriter sw = new StringWriter();
			for (int iTrack = 1; iTrack <= _toc.TrackCount; iTrack++)
				sw.WriteLine("\t{0}", _toc[iTrack].Start + 150);
			sw.Close();
			return sw.ToString();
		}

		public string CUESheetContents(CUEStyle style) 
		{
			StringWriter sw = new StringWriter();
			int i, iTrack, iIndex;
			bool htoaToFile = (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0);

			uint timeRelativeToFileStart = 0;

			using (sw) 
			{
				if (_config.writeArTagsOnConvert)
					WriteLine(sw, 0, "REM ACCURATERIPID " + _accurateRipId);

				for (i = 0; i < _attributes.Count; i++)
					WriteLine(sw, 0, _attributes[i]);

				if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
					WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _singleFilename));

				if (htoaToFile)
					WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _htoaFilename));

				for (iTrack = 0; iTrack < TrackCount; iTrack++) 
				{
					if ((style == CUEStyle.GapsPrepended) ||
						(style == CUEStyle.GapsLeftOut) ||
						((style == CUEStyle.GapsAppended) &&
						((_toc[_toc.FirstAudio + iTrack].Pregap == 0) || ((iTrack == 0) && !htoaToFile))))
					{
						WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
						timeRelativeToFileStart = 0;
					}

					WriteLine(sw, 1, String.Format("TRACK {0:00} AUDIO", iTrack + 1));
					for (i = 0; i < _tracks[iTrack].Attributes.Count; i++)
						WriteLine(sw, 2, _tracks[iTrack].Attributes[i]);

					if (_toc[_toc.FirstAudio + iTrack].Pregap != 0)
					{
						if (((style == CUEStyle.GapsLeftOut) ||
							((style == CUEStyle.GapsAppended) && (iTrack == 0) && !htoaToFile) ||
							((style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) && (iTrack == 0) && _usePregapForFirstTrackInSingleFile)))
							WriteLine(sw, 2, "PREGAP " + CDImageLayout.TimeToString(_toc[_toc.FirstAudio + iTrack].Pregap));
						else
						{
							WriteLine(sw, 2, String.Format("INDEX 00 {0}", CDImageLayout.TimeToString(timeRelativeToFileStart)));
							timeRelativeToFileStart += _toc[_toc.FirstAudio + iTrack].Pregap;
							if (style == CUEStyle.GapsAppended)
							{
								WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
								timeRelativeToFileStart = 0;
							}
						}
					}
					for (iIndex = 1; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
					{
						WriteLine(sw, 2, String.Format( "INDEX {0:00} {1}", iIndex, CDImageLayout.TimeToString(timeRelativeToFileStart)));
						timeRelativeToFileStart += _toc.IndexLength(_toc.FirstAudio + iTrack, iIndex);
					}
				}
			}
			sw.Close();
			return sw.ToString();
		}

		public void GenerateAccurateRipLog(TextWriter sw)
		{
			sw.WriteLine("[Verification date: {0}]", DateTime.Now);
			sw.WriteLine("[Disc ID: {0}]", _accurateRipId);
			if (_dataTrackLength.HasValue)
				sw.WriteLine("Assuming a data track was present, length {0}.", CDImageLayout.TimeToString(_dataTrackLength.Value));
			else
			{
				if (_cddbDiscIdTag != null && _accurateRipId.Split('-')[2].ToUpper() != _cddbDiscIdTag.ToUpper())
					sw.WriteLine("CDDBId mismatch: {0} vs {1}", _cddbDiscIdTag.ToUpper(), _accurateRipId.Split('-')[2].ToUpper());
				if (_minDataTrackLength.HasValue)
					sw.WriteLine("Data track was probably present, length {0}-{1}.", CDImageLayout.TimeToString(_minDataTrackLength.Value), CDImageLayout.TimeToString(_minDataTrackLength.Value + 74));
				if (_accurateRipIdActual != _accurateRipId)
					sw.WriteLine("Using preserved id, actual id is {0}.", _accurateRipIdActual);
				if (_truncated4608)
					sw.WriteLine("Truncated 4608 extra samples in some input files.");
				if (_paddedToFrame)
					sw.WriteLine("Padded some input files to a frame boundary.");
			}

			if (hdcdDecoder != null && hdcdDecoder.Detected)
			{
				hdcd_decoder_statistics stats;
				hdcdDecoder.GetStatistics(out stats);
				sw.WriteLine("HDCD: peak extend: {0}, transient filter: {1}, gain: {2}",
					(stats.enabled_peak_extend ? (stats.disabled_peak_extend ? "some" : "yes") : "none"),
					(stats.enabled_transient_filter ? (stats.disabled_transient_filter ? "some" : "yes") : "none"),
					stats.min_gain_adjustment == stats.max_gain_adjustment ? 
					(stats.min_gain_adjustment == 1.0 ? "none" : String.Format ("{0:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20))) :
					String.Format ("{0:0.0}dB..{1:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20), (Math.Log10(stats.max_gain_adjustment) * 20))
					);
			}

			if (0 != _writeOffset)
				sw.WriteLine("Offset applied: {0}", _writeOffset);
			_arVerify.GenerateFullLog(sw, 0);
		}

		public void GenerateAccurateRipTagsForTrack(NameValueCollection tags, int offset, int bestOffset, int iTrack, string prefix)
		{
			uint total = 0;
			uint matching = 0;
			uint matching2 = 0;
			uint matching3 = 0;
			for (int iDisk = 0; iDisk < _arVerify.AccDisks.Count; iDisk++)
			{
				total += _arVerify.AccDisks[iDisk].tracks[iTrack].count;
				if (_arVerify.CRC(iTrack, offset) ==
					_arVerify.AccDisks[iDisk].tracks[iTrack].CRC)
					matching += _arVerify.AccDisks[iDisk].tracks[iTrack].count;
				if (_arVerify.CRC(iTrack, bestOffset) ==
					_arVerify.AccDisks[iDisk].tracks[iTrack].CRC)
					matching2 += _arVerify.AccDisks[iDisk].tracks[iTrack].count;
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					if (_arVerify.CRC(iTrack, oi) ==
						_arVerify.AccDisks[iDisk].tracks[iTrack].CRC)
						matching3 += _arVerify.AccDisks[iDisk].tracks[iTrack].count;
			}
			tags.Add(String.Format("{0}ACCURATERIPCRC", prefix), String.Format("{0:x8}", _arVerify.CRC(iTrack, offset)));
			tags.Add(String.Format("{0}AccurateRipDiscId", prefix), String.Format("{0:000}-{1}-{2:00}", TrackCount, _accurateRipId, iTrack+1));
			tags.Add(String.Format("{0}ACCURATERIPCOUNT", prefix), String.Format("{0}", matching));
			tags.Add(String.Format("{0}ACCURATERIPCOUNTALLOFFSETS", prefix), String.Format("{0}", matching3));
			tags.Add(String.Format("{0}ACCURATERIPTOTAL", prefix), String.Format("{0}", total));
			if (bestOffset != offset)
				tags.Add(String.Format("{0}ACCURATERIPCOUNTWITHOFFSET", prefix), String.Format("{0}", matching2));
		}

		public void GenerateAccurateRipTags(NameValueCollection tags, int offset, int bestOffset, int iTrack)
		{
			tags.Add("ACCURATERIPID", _accurateRipId);
			if (bestOffset != offset)
				tags.Add("ACCURATERIPOFFSET", String.Format("{1}{0}", bestOffset - offset, bestOffset > offset ? "+" : ""));
			if (iTrack != -1)
				GenerateAccurateRipTagsForTrack(tags, offset, bestOffset, iTrack, "");
			else
			for (iTrack = 0; iTrack < TrackCount; iTrack++)
			{
				GenerateAccurateRipTagsForTrack(tags, offset, bestOffset, iTrack,
					String.Format("cue_track{0:00}_", iTrack + 1));
			}
		}

		public void CleanupTags (NameValueCollection tags, string substring)
		{
			string [] keys = tags.AllKeys;
			for (int i = 0; i < keys.Length; i++)
				if (keys[i].ToUpper().Contains(substring))
					tags.Remove (keys[i]);
		}

		private void FindBestOffset(uint minConfidence, bool optimizeConfidence, out uint outTracksMatch, out int outBestOffset)
		{
			uint bestTracksMatch = 0;
			uint bestConfidence = 0;
			int bestOffset = 0;

			for (int offset = -_arOffsetRange; offset <= _arOffsetRange; offset++)
			{
				uint tracksMatch = 0;
				uint sumConfidence = 0;

				for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				{
					uint confidence = 0;

					for (int di = 0; di < (int)_arVerify.AccDisks.Count; di++)
						if (_arVerify.CRC(iTrack, offset) == _arVerify.AccDisks[di].tracks[iTrack].CRC)
							confidence += _arVerify.AccDisks[di].tracks[iTrack].count;

					if (confidence >= minConfidence)
						tracksMatch++;

					sumConfidence += confidence;
				}

				if (tracksMatch > bestTracksMatch
					|| (tracksMatch == bestTracksMatch && optimizeConfidence && sumConfidence > bestConfidence)
					|| (tracksMatch == bestTracksMatch && optimizeConfidence && sumConfidence == bestConfidence && Math.Abs(offset) < Math.Abs(bestOffset))
					|| (tracksMatch == bestTracksMatch && !optimizeConfidence && Math.Abs(offset) < Math.Abs(bestOffset))
					)
				{
					bestTracksMatch = tracksMatch;
					bestConfidence = sumConfidence;
					bestOffset = offset;
				}
			}
			outBestOffset = bestOffset;
			outTracksMatch = bestTracksMatch;
		}

		public void WriteAudioFiles(string dir, CUEStyle style) {
			string[] destPaths;
			int[] destLengths;
			bool htoaToFile = ((style == CUEStyle.GapsAppended) && _config.preserveHTOA &&
				(_toc.Pregap != 0));

			if (_isCD && (style == CUEStyle.GapsLeftOut || style == CUEStyle.GapsPrepended) && (_accurateRipMode == AccurateRipMode.None || _accurateRipMode == AccurateRipMode.VerifyAndConvert))
				throw new Exception("When ripping a CD, gaps Left Out/Gaps prepended modes can only be used in verify-then-convert mode");

			if (_usePregapForFirstTrackInSingleFile)
				throw new Exception("UsePregapForFirstTrackInSingleFile is not supported for writing audio files.");

			if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) {
				destPaths = new string[1];
				destPaths[0] = Path.Combine(dir, _singleFilename);
			}
			else {
				destPaths = new string[TrackCount + (htoaToFile ? 1 : 0)];
				if (htoaToFile) {
					destPaths[0] = Path.Combine(dir, _htoaFilename);
				}
				for (int i = 0; i < TrackCount; i++) {
					destPaths[i + (htoaToFile ? 1 : 0)] = Path.Combine(dir, _trackFilenames[i]);
				}
			}

			if (_accurateRipMode != AccurateRipMode.Verify)
				for (int i = 0; i < destPaths.Length; i++)
					for (int j = 0; j < _sourcePaths.Count; j++)
						if (destPaths[i].ToLower() == _sourcePaths[j].ToLower())
							throw new Exception("Source and destination audio file paths cannot be the same.");

			destLengths = CalculateAudioFileLengths(style);

			bool SkipOutput = false;

			if (_accurateRipMode != AccurateRipMode.None)
			{
				ShowProgress((string)"Contacting AccurateRip database...", 0, 0, null, null);
				if (!_dataTrackLength.HasValue && _minDataTrackLength.HasValue && _accurateRipId == _accurateRipIdActual && _config.bruteForceDTL)
				{
					uint minDTL = _minDataTrackLength.Value;
					CDImageLayout toc2 = new CDImageLayout(_toc);
					toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152 * 75, minDTL, false, false));
					for (uint dtl = minDTL; dtl < minDTL + 75; dtl++)
					{
						toc2[toc2.TrackCount].Length = dtl;
						_accurateRipId = AccurateRipVerify.CalculateAccurateRipId(toc2);
						_arVerify.ContactAccurateRip(_accurateRipId);
						if (_arVerify.AccResult != HttpStatusCode.NotFound)
						{
							_dataTrackLength = dtl;
							break;
						}
						ShowProgress((string)"Contacting AccurateRip database...", 0, (dtl - minDTL) / 75.0, null, null);
						lock (this)
						{
							if (_stop)
								throw new StopException();
							if (_pause)
							{
								ShowProgress("Paused...", 0, 0, null, null);
								Monitor.Wait(this);
							}
							else
								Monitor.Wait(this, 1000);
						}
					}
					if (_arVerify.AccResult != HttpStatusCode.OK)
					{
						_accurateRipId = _accurateRipIdActual;
					}
				}
				else
					_arVerify.ContactAccurateRip(_accurateRipId);

				if (_accurateRipMode == AccurateRipMode.VerifyThenConvert)
				{
					if (_arVerify.AccResult != HttpStatusCode.OK && !_isCD)
					{
						if (_config.noUnverifiedOutput)
						{
							if (_config.writeArLogOnConvert)
							{
								if (!Directory.Exists(dir))
									Directory.CreateDirectory(dir);
								StreamWriter sw = new StreamWriter(Path.ChangeExtension(_cuePath, ".accurip"),
									false, CUESheet.Encoding);
								GenerateAccurateRipLog(sw);
								sw.Close();
							}
							if (_config.createTOC)
							{
								if (!Directory.Exists(dir))
									Directory.CreateDirectory(dir);
								WriteText(Path.ChangeExtension(_cuePath, ".toc"), TOCContents());
							}
							return;
						}
					}
					else
					{
						_writeOffset = 0;
						WriteAudioFilesPass(dir, style, destPaths, destLengths, htoaToFile, true);
						if (!_isCD)
						{
							uint tracksMatch;
							int bestOffset;

							if (_config.noUnverifiedOutput)
							{
								FindBestOffset(_config.encodeWhenConfidence, false, out tracksMatch, out bestOffset);
								if (tracksMatch * 100 < _config.encodeWhenPercent * TrackCount || (_config.encodeWhenZeroOffset && bestOffset != 0))
									SkipOutput = true;
							}

							if (!SkipOutput && _config.fixOffset)
							{
								FindBestOffset(_config.fixWhenConfidence, false, out tracksMatch, out bestOffset);
								if (tracksMatch * 100 >= _config.fixWhenPercent * TrackCount)
									_writeOffset = bestOffset;
							}
						}
						_arVerify.CreateBackup(_writeOffset);
					}
				}
			}

			if (!SkipOutput)
			{
				if (_accurateRipMode != AccurateRipMode.Verify)
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
				}
				if (_isCD)
					destLengths = CalculateAudioFileLengths(style); // need to recalc, might have changed after scanning the CD
				if (_outputFormat != OutputAudioFormat.NoAudio || _accurateRipMode == AccurateRipMode.Verify)
					WriteAudioFilesPass(dir, style, destPaths, destLengths, htoaToFile, _accurateRipMode == AccurateRipMode.Verify);
				if (_accurateRipMode != AccurateRipMode.Verify)
				{
					string logContents = LOGContents();
					string cueContents = CUESheetContents(style);
					uint tracksMatch = 0;
					int bestOffset = 0;
					
					if (_accurateRipMode != AccurateRipMode.None &&
						_config.writeArTagsOnConvert &&
						_arVerify.AccResult == HttpStatusCode.OK)
						FindBestOffset(1, true, out tracksMatch, out bestOffset);

					if (logContents != null)
						WriteText(Path.ChangeExtension(_cuePath, ".log"), logContents);
					else
					if (_eacLog != null && _config.extractLog)
						WriteText(Path.ChangeExtension(_cuePath, ".log"), _eacLog);

					if (style == CUEStyle.SingleFileWithCUE || style == CUEStyle.SingleFile)
					{
						if (style == CUEStyle.SingleFileWithCUE && _config.createCUEFileWhenEmbedded)
							WriteText(Path.ChangeExtension(_cuePath, ".cue"), cueContents);
						if (style == CUEStyle.SingleFile)
							WriteText(_cuePath, cueContents);
						if (_outputFormat != OutputAudioFormat.NoAudio)
						{
							NameValueCollection tags = GenerateAlbumTags(bestOffset, style == CUEStyle.SingleFileWithCUE);
							TagLib.UserDefined.AdditionalFileTypes.Config = _config;
							TagLib.File fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(destPaths[0]));
							if (Tagging.UpdateTags(fileInfo, tags, _config))
							{
								fileInfo.Tag.DiscCount = (_tracks[0]._fileInfo ?? _fileInfo).Tag.DiscCount; // TODO: GetCommonTag?
								fileInfo.Tag.Disc = (_tracks[0]._fileInfo ?? _fileInfo).Tag.Disc;
								//fileInfo.Tag.Title = null;
								//fileInfo.Tag.Performers = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Performers;
								//if (fileInfo.Tag.Performers.Length == 0) fileInfo.Tag.Performers = new string[] { _tracks[iTrack].Artist != "" ? _tracks[iTrack].Artist : Artist };
								fileInfo.Tag.AlbumArtists = (_tracks[0]._fileInfo ?? _fileInfo).Tag.AlbumArtists;
								if (fileInfo.Tag.AlbumArtists.Length == 0) fileInfo.Tag.AlbumArtists = new string[] { Artist };
								fileInfo.Tag.Album = (_tracks[0]._fileInfo ?? _fileInfo).Tag.Album ?? Title;
								uint year = (_tracks[0]._fileInfo ?? _fileInfo).Tag.Year;
								fileInfo.Tag.Year = year != 0 ? year : ("" != Year && uint.TryParse(Year, out year)) ? year : 0;
								fileInfo.Tag.Genres = (_tracks[0]._fileInfo ?? _fileInfo).Tag.Genres;
								if (fileInfo.Tag.Genres.Length == 0) fileInfo.Tag.Genres = new string[] { Genre };
								fileInfo.Tag.Pictures = (_tracks[0]._fileInfo ?? _fileInfo).Tag.Pictures;
								fileInfo.Save();
							}
						}
					}
					else
					{
						WriteText(_cuePath, cueContents);
						if (_config.createM3U)
							WriteText(Path.ChangeExtension(_cuePath, ".m3u"), M3UContents(style));
						if (_outputFormat != OutputAudioFormat.NoAudio)
							for (int iTrack = 0; iTrack < TrackCount; iTrack++)
							{
								string path = destPaths[iTrack + (htoaToFile ? 1 : 0)];
								NameValueCollection tags = GenerateTrackTags(iTrack, bestOffset);
								TagLib.UserDefined.AdditionalFileTypes.Config = _config;
								TagLib.File fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(path));
								if (Tagging.UpdateTags(fileInfo, tags, _config))
								{
									fileInfo.Tag.TrackCount = (uint) TrackCount;
									fileInfo.Tag.Track = (uint) iTrack + 1;
									fileInfo.Tag.DiscCount = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.DiscCount;
									fileInfo.Tag.Disc = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Disc;
									fileInfo.Tag.Title = _tracks[iTrack]._fileInfo != null ? _tracks[iTrack]._fileInfo.Tag.Title : _tracks[iTrack].Title;
									fileInfo.Tag.Performers = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Performers;
									if (fileInfo.Tag.Performers.Length == 0) fileInfo.Tag.Performers = new string[] { _tracks[iTrack].Artist != "" ? _tracks[iTrack].Artist : Artist };
									fileInfo.Tag.AlbumArtists = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.AlbumArtists;
									if (fileInfo.Tag.AlbumArtists.Length == 0) fileInfo.Tag.AlbumArtists = new string[] { Artist };
									fileInfo.Tag.Album = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Album ?? Title;
									uint year = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Year;
									fileInfo.Tag.Year = year != 0 ? year : ("" != Year && uint.TryParse(Year, out year)) ? year : 0;
									fileInfo.Tag.Genres = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Genres;
									if (fileInfo.Tag.Genres.Length == 0) fileInfo.Tag.Genres = new string[] { Genre };
									fileInfo.Tag.Pictures = (_tracks[iTrack]._fileInfo ?? _fileInfo).Tag.Pictures;
									fileInfo.Save();
								}
							}
					}
				}
			}

			if (_accurateRipMode == AccurateRipMode.Verify || 
				(_accurateRipMode != AccurateRipMode.None && _outputFormat != OutputAudioFormat.NoAudio))
			{
				ShowProgress((string)"Generating AccurateRip report...", 0, 0, null, null);
				if (_accurateRipMode == AccurateRipMode.Verify && _config.writeArTagsOnVerify && _writeOffset == 0 && !_isArchive && !_isCD)
				{
					uint tracksMatch;
					int bestOffset;
					FindBestOffset(1, true, out tracksMatch, out bestOffset);

					if (_hasEmbeddedCUESheet)
					{
						if (_fileInfo is TagLib.Flac.File)
						{
							NameValueCollection tags = Tagging.Analyze(_fileInfo);
							CleanupTags(tags, "ACCURATERIP");
							GenerateAccurateRipTags(tags, 0, bestOffset, -1);
							if (Tagging.UpdateTags(_fileInfo, tags, _config))
								_fileInfo.Save();
						}
					} else if (_hasTrackFilenames)
					{
						for (int iTrack = 0; iTrack < TrackCount; iTrack++)
							if (_tracks[iTrack]._fileInfo is TagLib.Flac.File)
							{
								NameValueCollection tags = Tagging.Analyze(_tracks[iTrack]._fileInfo);
								CleanupTags(tags, "ACCURATERIP");
								GenerateAccurateRipTags(tags, 0, bestOffset, iTrack);
								if (Tagging.UpdateTags(_tracks[iTrack]._fileInfo, tags, _config))
									_tracks[iTrack]._fileInfo.Save();
							}
					}
				}

				if ((_accurateRipMode != AccurateRipMode.Verify && _config.writeArLogOnConvert) ||
					(_accurateRipMode == AccurateRipMode.Verify && _config.writeArLogOnVerify))
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
					StreamWriter sw = new StreamWriter(Path.ChangeExtension(_cuePath, ".accurip"),
						false, CUESheet.Encoding);
					GenerateAccurateRipLog(sw);
					sw.Close();
				}
				if (_config.createTOC)
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
					WriteText(Path.ChangeExtension(_cuePath, ".toc"), TOCContents());
				}
			}
		}

		private NameValueCollection GenerateTrackTags(int iTrack, int bestOffset)
		{
			NameValueCollection destTags = new NameValueCollection();
			
			if (_hasEmbeddedCUESheet)
			{
				string trackPrefix = String.Format("cue_track{0:00}_", iTrack + 1);
				NameValueCollection albumTags = Tagging.Analyze(_fileInfo);
				foreach (string key in albumTags.AllKeys)
				{
					if (key.ToLower().StartsWith(trackPrefix)
						|| !key.ToLower().StartsWith("cue_track"))
					{
						string name = key.ToLower().StartsWith(trackPrefix) ?
							key.Substring(trackPrefix.Length) : key;
						string[] values = albumTags.GetValues(key);
						for (int j = 0; j < values.Length; j++)
							destTags.Add(name, values[j]);
					}
				}
			}
			else if (_hasTrackFilenames)
				destTags.Add(Tagging.Analyze(_tracks[iTrack]._fileInfo));
			else if (_hasSingleFilename)
			{
				// TODO?
			}

			// these will be set explicitely
			destTags.Remove("ARTIST");
			destTags.Remove("TITLE");
			destTags.Remove("ALBUM");
			destTags.Remove("ALBUMARTIST");
			destTags.Remove("DATE");
			destTags.Remove("GENRE");
			destTags.Remove("TRACKNUMBER");
			destTags.Remove("TRACKTOTAL");
			destTags.Remove("TOTALTRACKS");
			destTags.Remove("DISCNUMBER");
			destTags.Remove("DISCTOTAL");
			destTags.Remove("TOTALDISCS");

			destTags.Remove("LOG");
			destTags.Remove("LOGFILE");
			destTags.Remove("EACLOG");

			// these are not valid
			destTags.Remove("CUESHEET");
			CleanupTags(destTags, "ACCURATERIP");
			CleanupTags(destTags, "REPLAYGAIN");

			if (_config.writeArTagsOnConvert)
			{
				if (_accurateRipMode != AccurateRipMode.None && _arVerify.AccResult == HttpStatusCode.OK)
					GenerateAccurateRipTags(destTags, _writeOffset, bestOffset, iTrack);
				else
					destTags.Add("ACCURATERIPID", _accurateRipId);
			}
			return destTags;
		}

		private NameValueCollection GenerateAlbumTags(int bestOffset, bool fWithCUE)
		{
			NameValueCollection destTags = new NameValueCollection();

			if (_hasEmbeddedCUESheet || _hasSingleFilename)
			{
				destTags.Add(Tagging.Analyze(_fileInfo));
				if (!fWithCUE)
					CleanupTags(destTags, "CUE_TRACK");
			}
			else if (_hasTrackFilenames)
			{
				for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				{
					NameValueCollection trackTags = Tagging.Analyze(_tracks[iTrack]._fileInfo);
					foreach (string key in trackTags.AllKeys)
					{
						string singleValue = GetCommonMiscTag(key);
						if (singleValue != null)
						{
							if (destTags.Get(key) == null)
								destTags.Add(key, singleValue);
						}
						else if (fWithCUE && key.ToUpper() != "TRACKNUMBER")
						{
							string[] values = trackTags.GetValues(key);
							for (int j = 0; j < values.Length; j++)
								destTags.Add(String.Format("cue_track{0:00}_{1}", iTrack + 1, key), values[j]);
						}
					}
				}
			}

			// these will be set explicitely
			destTags.Remove("ARTIST");
			destTags.Remove("TITLE");
			destTags.Remove("ALBUM");
			destTags.Remove("ALBUMARTIST");
			destTags.Remove("DATE");
			destTags.Remove("GENRE");
			destTags.Remove("TRACKNUMBER");
			destTags.Remove("TRACKTOTAL");
			destTags.Remove("TOTALTRACKS");
			destTags.Remove("DISCNUMBER");
			destTags.Remove("DISCTOTAL");
			destTags.Remove("TOTALDISCS");

			// these are not valid
			CleanupTags(destTags, "ACCURATERIP");
			CleanupTags(destTags, "REPLAYGAIN");

			destTags.Remove("CUESHEET");
			if (fWithCUE)
				destTags.Add("CUESHEET", CUESheetContents(CUEStyle.SingleFileWithCUE));

			if (_config.embedLog)
			{
				destTags.Remove("LOG");
				destTags.Remove("LOGFILE");
				destTags.Remove("EACLOG");
				string logContents = LOGContents();
				if (logContents != null)
					destTags.Add("LOG", logContents);
				else if (_eacLog != null)
					destTags.Add("LOG", _eacLog);
			}

			if (_config.writeArTagsOnConvert)
			{
				if (fWithCUE && _accurateRipMode != AccurateRipMode.None && _arVerify.AccResult == HttpStatusCode.OK)
					GenerateAccurateRipTags(destTags, _writeOffset, bestOffset, -1);
				else
					destTags.Add("ACCURATERIPID", _accurateRipId);
			}
			return destTags;
		}

		public void WriteAudioFilesPass(string dir, CUEStyle style, string[] destPaths, int[] destLengths, bool htoaToFile, bool noOutput)
		{
			const int buffLen = 16384;
			int iTrack, iIndex;
			int[,] sampleBuffer = new int[buffLen, 2];
			TrackInfo track;
			IAudioSource audioSource = null;
			IAudioDest audioDest = null;
			bool discardOutput;
			int iSource = -1;
			int iDest = -1;
			uint samplesRemSource = 0;
			//CDImageLayout updatedTOC = null;

			if (_writeOffset != 0)
			{
				uint absOffset = (uint)Math.Abs(_writeOffset);
				SourceInfo sourceInfo;

				sourceInfo.Path = null;
				sourceInfo.Offset = 0;
				sourceInfo.Length = absOffset;

				if (_writeOffset < 0)
				{
					_sources.Insert(0, sourceInfo);

					int last = _sources.Count - 1;
					while (absOffset >= _sources[last].Length)
					{
						absOffset -= _sources[last].Length;
						_sources.RemoveAt(last--);
					}
					sourceInfo = _sources[last];
					sourceInfo.Length -= absOffset;
					_sources[last] = sourceInfo;
				}
				else
				{
					_sources.Add(sourceInfo);

					while (absOffset >= _sources[0].Length)
					{
						absOffset -= _sources[0].Length;
						_sources.RemoveAt(0);
					}
					sourceInfo = _sources[0];
					sourceInfo.Offset += absOffset;
					sourceInfo.Length -= absOffset;
					_sources[0] = sourceInfo;
				}

				_appliedWriteOffset = true;
			}

			if (_config.detectHDCD)
			{
				// currently broken verifyThenConvert on HDCD detection!!!! need to check for HDCD results higher
				try { hdcdDecoder = new HDCDDotNet.HDCDDotNet(2, 44100, ((_outputLossyWAV && _config.decodeHDCDtoLW16) || !_config.decodeHDCDto24bit) ? 20 : 24, _config.decodeHDCD); }
				catch { }
			}

			if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
			{
				iDest++;
				audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], hdcdDecoder != null && hdcdDecoder.Decoding ? hdcdDecoder.BitsPerSample : 16, noOutput);
			}

			uint currentOffset = 0, previousOffset = 0;
			uint trackLength = _toc.Pregap * 588;
			uint diskLength = 588 * _toc.AudioLength;
			uint diskOffset = 0;

			if (_accurateRipMode != AccurateRipMode.None)
				_arVerify.Init();

			ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", 0, 0, noOutput ? "Verifying" : "Writing"), 0, 0.0, null, null);

#if !DEBUG
			try
#endif
			{
				for (iTrack = 0; iTrack < TrackCount; iTrack++)
				{
					track = _tracks[iTrack];

					if ((style == CUEStyle.GapsPrepended) || (style == CUEStyle.GapsLeftOut))
					{
						iDest++;
						if (hdcdDecoder != null)
							hdcdDecoder.AudioDest = null;
						if (audioDest != null)
							audioDest.Close();
						audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], hdcdDecoder != null && hdcdDecoder.Decoding ? hdcdDecoder.BitsPerSample : 16, noOutput);
					}

					for (iIndex = 0; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
					{
						uint samplesRemIndex = _toc.IndexLength(_toc.FirstAudio + iTrack, iIndex) * 588;

						if (iIndex == 1)
						{
							previousOffset = currentOffset;
							currentOffset = 0;
							trackLength = _toc[_toc.FirstAudio + iTrack].Length * 588;
						}

						if ((style == CUEStyle.GapsAppended) && (iIndex == 1))
						{
							if (hdcdDecoder != null)
								hdcdDecoder.AudioDest = null;
							if (audioDest != null)
								audioDest.Close();
							iDest++;
							audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], hdcdDecoder != null && hdcdDecoder.Decoding ? hdcdDecoder.BitsPerSample : 16, noOutput);
						}

						if ((style == CUEStyle.GapsAppended) && (iIndex == 0) && (iTrack == 0))
						{
							discardOutput = !htoaToFile;
							if (htoaToFile)
							{
								iDest++;
								audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], hdcdDecoder != null && hdcdDecoder.Decoding ? hdcdDecoder.BitsPerSample : 16, noOutput);
							}
						}
						else if ((style == CUEStyle.GapsLeftOut) && (iIndex == 0))
						{
							discardOutput = true;
						}
						else
						{
							discardOutput = false;
						}

						while (samplesRemIndex != 0)
						{
							if (samplesRemSource == 0)
							{
//#if !MONO
//                                if (_isCD && audioSource != null && audioSource is CDDriveReader)
//                                    updatedTOC = ((CDDriveReader)audioSource).TOC;
//#endif
								if (audioSource != null && !_isCD) audioSource.Close();
								audioSource = GetAudioSource(++iSource);
								samplesRemSource = (uint)_sources[iSource].Length;
							}

							uint copyCount = (uint)Math.Min(Math.Min(samplesRemIndex, samplesRemSource), buffLen);

							if (trackLength > 0 && !_isCD)
							{
								double trackPercent = (double)currentOffset / trackLength;
								double diskPercent = (double)diskOffset / diskLength;
								ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", iIndex > 0 ? iTrack + 1 : iTrack, (uint)(100*trackPercent),
									noOutput ? "Verifying" : "Writing"), trackPercent, diskPercent,
									_isCD ? string.Format("{0}: {1:00} - {2}", audioSource.Path, iTrack + 1, _tracks[iTrack].Title) : audioSource.Path, discardOutput ? null : audioDest.Path);
							}

							if (audioSource.Read(sampleBuffer, copyCount) != copyCount)
								throw new Exception("samples read != samples expected");
							if (!discardOutput)
							{
								if (!_config.detectHDCD || !_config.decodeHDCD)
									audioDest.Write(sampleBuffer, copyCount);
								if (_config.detectHDCD && hdcdDecoder != null)
								{
									if (_config.wait750FramesForHDCD && diskOffset > 750 * 588 && !hdcdDecoder.Detected)
									{
										hdcdDecoder.AudioDest = null;
										hdcdDecoder = null;
										if (_config.decodeHDCD)
										{
											if (!_isCD) audioSource.Close();
											audioDest.Delete();
											throw new Exception("HDCD not detected.");
										}
									}
									else
									{
										if (_config.decodeHDCD)
											hdcdDecoder.AudioDest = (discardOutput || noOutput) ? null : audioDest;
										hdcdDecoder.Process(sampleBuffer, copyCount);
									}
								}
							}
							if (_accurateRipMode != AccurateRipMode.None)
								_arVerify.Write(sampleBuffer, copyCount);

							currentOffset += copyCount;
							diskOffset += copyCount;
							samplesRemIndex -= copyCount;
							samplesRemSource -= copyCount;

							lock (this)
							{
								if (_stop)
									throw new StopException();
								if (_pause)
								{
									ShowProgress("Paused...", 0, 0, null, null);
									Monitor.Wait(this);
								}
							}
						}
					}
				}
			}
#if !DEBUG
			catch (Exception ex)
			{
				if (hdcdDecoder != null)
					hdcdDecoder.AudioDest = null;
				hdcdDecoder = null;
				try { if (audioSource != null && !_isCD) audioSource.Close(); }
				catch { }
				audioSource = null;
				try { if (audioDest != null) audioDest.Close(); } 
				catch { }
				audioDest = null;
				throw ex;
			}
#endif

#if !MONO
			//if (_isCD && audioSource != null && audioSource is CDDriveReader)
			//    updatedTOC = ((CDDriveReader)audioSource).TOC;
			if (_isCD)
			{
				_toc = (CDImageLayout)_ripper.TOC.Clone();
				if (_toc.Catalog != null)
					Catalog = _toc.Catalog;
				for (iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
				{
					if (_toc[_toc.FirstAudio + iTrack].ISRC != null)
						General.SetCUELine(_tracks[iTrack].Attributes, "ISRC", _toc[_toc.FirstAudio + iTrack].ISRC, false);
					//if (_toc[_toc.FirstAudio + iTrack].DCP || _toc[_toc.FirstAudio + iTrack].PreEmphasis)
					//cueWriter.WriteLine("    FLAGS{0}{1}", audioSource.TOC[track].PreEmphasis ? " PRE" : "", audioSource.TOC[track].DCP ? " DCP" : "");
					if (_toc[_toc.FirstAudio + iTrack].DCP)
						General.SetCUELine(_tracks[iTrack].Attributes, "FLAGS", "DCP", false);
					if (_toc[_toc.FirstAudio + iTrack].PreEmphasis)
						General.SetCUELine(_tracks[iTrack].Attributes, "FLAGS", "PRE", false);
				}
			}
#endif

			if (hdcdDecoder != null)
				hdcdDecoder.AudioDest = null;
			if (audioSource != null && !_isCD)
				audioSource.Close();
			if (audioDest != null)
				audioDest.Close();
		}

		public static string CreateDummyCUESheet(string path, string extension)
		{
			string[] audioFiles = Directory.GetFiles(path, extension);
			if (audioFiles.Length < 2)
				return null;
			Array.Sort(audioFiles);
			StringWriter sw = new StringWriter();
			sw.WriteLine(String.Format("REM COMMENT \"CUETools generated dummy CUE sheet\""));
			for (int iFile = 0; iFile < audioFiles.Length; iFile++)
			{
				sw.WriteLine(String.Format("FILE \"{0}\" WAVE", Path.GetFileName(audioFiles[iFile])));
				sw.WriteLine(String.Format("  TRACK {0:00} AUDIO", iFile + 1));
				sw.WriteLine(String.Format("    INDEX 01 00:00:00"));
			}
			sw.Close();
			return sw.ToString();
		}

		public static string CorrectAudioFilenames(string path, bool always)
		{
			StreamReader sr = new StreamReader(path, CUESheet.Encoding);
			string cue = sr.ReadToEnd();
			sr.Close();
			return CorrectAudioFilenames(Path.GetDirectoryName(path), cue, always, null);
		}

		public static string CorrectAudioFilenames(string dir, string cue, bool always, List<string> files) 
		{
			string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a", "*.tta" };
			List<string> lines = new List<string>();
			List<int> filePos = new List<int>();
			List<string> origFiles = new List<string>();
			bool foundAll = true;
			string[] audioFiles = null;
			string lineStr;
			CUELine line;
			int i;

			using (StringReader sr = new StringReader(cue)) {
				while ((lineStr = sr.ReadLine()) != null) {
					lines.Add(lineStr);
					line = new CUELine(lineStr);
					if ((line.Params.Count == 3) && (line.Params[0].ToUpper() == "FILE")) {
						string fileType = line.Params[2].ToUpper();
						if ((fileType != "BINARY") && (fileType != "MOTOROLA")) {
							filePos.Add(lines.Count - 1);
							origFiles.Add(line.Params[1]);
							foundAll &= (LocateFile(dir, line.Params[1], files) != null);
						}
					}
				}
				sr.Close();
			}

			if (!foundAll || always)
			{
				foundAll = false;
				for (i = 0; i < audioExts.Length; i++)
				{
					List<string> newFiles = new List<string>();
					for (int j = 0; j < origFiles.Count; j++)
					{
						string newFilename = Path.ChangeExtension(Path.GetFileName(origFiles[j]), audioExts[i].Substring(1));
						string locatedFilename = LocateFile(dir, newFilename, files);
						if (locatedFilename != null)
							newFiles.Add(locatedFilename);
					}
					if (newFiles.Count == origFiles.Count)
					{
						audioFiles = newFiles.ToArray();
						foundAll = true;
						break;
					}
				}
				if (!foundAll)
				for (i = 0; i < audioExts.Length; i++)
				{
					if (files == null)
						audioFiles = Directory.GetFiles(dir == "" ? "." : dir, audioExts[i]);
					else
					{
						audioFiles = files.FindAll(delegate(string s)
						{
							return Path.GetDirectoryName(s) == dir && Path.GetExtension(s) == audioExts[i].Substring(1);
						}).ToArray();
					}
					if (audioFiles.Length == filePos.Count)
					{
						Array.Sort(audioFiles);
						foundAll = true;
						break;
					}
				}
				if (!foundAll)
					throw new Exception("Unable to locate the audio files.");

				for (i = 0; i < filePos.Count; i++)
					lines[filePos[i]] = "FILE \"" + Path.GetFileName(audioFiles[i]) + "\" WAVE";
			}

			using (StringWriter sw = new StringWriter()) {
				for (i = 0; i < lines.Count; i++) {
					sw.WriteLine(lines[i]);
				}
				return sw.ToString ();
			}
		}

		private int[] CalculateAudioFileLengths(CUEStyle style) 
		{
			int iTrack, iIndex, iFile;
			TrackInfo track;
			int[] fileLengths;
			bool htoaToFile = (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0);
			bool discardOutput;

			if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) {
				fileLengths = new int[1];
				iFile = 0;
			}
			else {
				fileLengths = new int[TrackCount + (htoaToFile ? 1 : 0)];
				iFile = -1;
			}

			for (iTrack = 0; iTrack < TrackCount; iTrack++) {
				track = _tracks[iTrack];

				if (style == CUEStyle.GapsPrepended || style == CUEStyle.GapsLeftOut)
					iFile++;

				for (iIndex = 0; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
				{
					if (style == CUEStyle.GapsAppended && (iIndex == 1 || (iIndex == 0 && iTrack == 0 && htoaToFile)))
						iFile++;

					if (style == CUEStyle.GapsAppended && iIndex == 0 && iTrack == 0) 
						discardOutput = !htoaToFile;
					else 
						discardOutput = (style == CUEStyle.GapsLeftOut && iIndex == 0);

					if (!discardOutput)
						fileLengths[iFile] += (int)_toc.IndexLength(_toc.FirstAudio + iTrack, iIndex) * 588;
				}
			}

			return fileLengths;
		}

		public void Stop() {
			lock (this) {
				if (_pause)
				{
					_pause = false;
					Monitor.Pulse(this);
				}
				_stop = true;
			}
		}

		public void Pause()
		{
			lock (this)
			{
				if (_pause)
				{
					_pause = false;
					Monitor.Pulse(this);
				} else
				{
					_pause = true;
				}
			}
		}

		public int TrackCount {
			get {
				return _tracks.Count;
			}
		}

		private IAudioDest GetAudioDest(string path, int finalSampleCount, int bps, bool noOutput) 
		{
			if (noOutput)
				return new DummyWriter(path, bps, 2, 44100);
			return AudioReadWrite.GetAudioDest(path, finalSampleCount, bps, 44100, _config);
		}

		private IAudioSource GetAudioSource(int sourceIndex) {
			SourceInfo sourceInfo = _sources[sourceIndex];
			IAudioSource audioSource;

			if (sourceInfo.Path == null) {
				audioSource = new SilenceGenerator(sourceInfo.Offset + sourceInfo.Length);
			}
			else {
#if !MONO
				if (_isCD)
				{
					_ripper.Position = 0;
					//audioSource = _ripper;
					audioSource = new AudioPipe(_ripper, 3);
				} else
#endif
				if (_isArchive)
					audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, OpenArchive(sourceInfo.Path, false), _config);
				else
					audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, null, _config);
			}

			if (sourceInfo.Offset != 0)
				audioSource.Position = sourceInfo.Offset;

			return audioSource;
		}

		private void WriteLine(TextWriter sw, int level, CUELine line) {
			WriteLine(sw, level, line.ToString());
		}

		private void WriteLine(TextWriter sw, int level, string line) {
			sw.Write(new string(' ', level * 2));
			sw.WriteLine(line);
		}

		public List<CUELine> Attributes {
			get {
				return _attributes;
			}
		}

		public List<TrackInfo> Tracks {
			get { 
				return _tracks;
			}
		}

		public bool HasHTOAFilename {
			get {
				return _hasHTOAFilename;
			}
		}

		public string HTOAFilename {
			get {
				return _htoaFilename;
			}
			set {
				_htoaFilename = value;
			}
		}

		public bool HasTrackFilenames {
			get {
				return _hasTrackFilenames;
			}
		}

		public List<string> TrackFilenames {
			get {
				return _trackFilenames;
			}
		}

		public bool HasSingleFilename {
			get {
				return _hasSingleFilename;
			}
		}

		public string SingleFilename {
			get {
				return _singleFilename;
			}
			set {
				_singleFilename = value;
			}
		}

		public string Artist {
			get {
				CUELine line = General.FindCUELine(_attributes, "PERFORMER");
				return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
			}
			set {
				General.SetCUELine(_attributes, "PERFORMER", value, true);
			}
		}

		public string Year
		{
			get
			{
				CUELine line = General.FindCUELine(_attributes, "REM", "DATE");
				return ( line == null || line.Params.Count < 3 ) ? String.Empty : line.Params[2];
			}
			set
			{
				if (value != "")
					General.SetCUELine(_attributes, "REM", "DATE", value, false);
				else
					General.DelCUELine(_attributes, "REM", "DATE");
			}
		}

		public string Genre
		{
			get
			{
				CUELine line = General.FindCUELine(_attributes, "REM", "GENRE");
				return (line == null  || line.Params.Count < 3) ? String.Empty : line.Params[2];
			}
			set
			{
				if (value != "")
					General.SetCUELine(_attributes, "REM", "GENRE", value, true);
				else
					General.DelCUELine(_attributes, "REM", "GENRE");
			}
		}

		public string Catalog
		{
			get
			{
				CUELine line = General.FindCUELine(_attributes, "CATALOG");
				return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
			}
			set
			{
				if (value != "")
					General.SetCUELine(_attributes, "CATALOG", value, false);
				else
					General.DelCUELine(_attributes, "CATALOG");
			}
		}

		public string Title {
			get {
				CUELine line = General.FindCUELine(_attributes, "TITLE");
				return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
			}
			set {
				General.SetCUELine(_attributes, "TITLE", value, true);
			}
		}

		public int WriteOffset {
			get {
				return _writeOffset;
			}
			set {
				if (_appliedWriteOffset) {
					throw new Exception("Cannot change write offset after audio files have been written.");
				}
				_writeOffset = value;
			}
		}

		public bool PaddedToFrame {
			get {
				return _paddedToFrame;
			}
		}

		public string DataTrackLength
		{
			get
			{
				return CDImageLayout.TimeToString(_dataTrackLength.HasValue ? _dataTrackLength.Value : 0);
			}
			set
			{
				uint dtl = (uint)CDImageLayout.TimeFromString(value);
				if (dtl != 0)
				{
					_dataTrackLength = dtl;
					CDImageLayout toc2 = new CDImageLayout(_toc);
					toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152 * 75, dtl, false, false));
					_accurateRipIdActual = _accurateRipId = AccurateRipVerify.CalculateAccurateRipId(toc2);
				}
			}
		}

		public bool UsePregapForFirstTrackInSingleFile {
			get {
				return _usePregapForFirstTrackInSingleFile;
			}
			set{
				_usePregapForFirstTrackInSingleFile = value;
			}
		}

		public CUEConfig Config
		{
			get
			{
				return _config;
			}
		}

		public AccurateRipMode AccurateRip
		{
			get
			{
				return _accurateRipMode;
			}
			set
			{
				_accurateRipMode = value;
			}
		}

		public bool IsCD
		{
			get
			{
				return _isCD;
			}
		}
	}

	public class SeekableZipStream : Stream
	{
		ZipFile zipFile;
		ZipEntry zipEntry;
		Stream zipStream;
		long position;
		byte[] temp;

		public SeekableZipStream(string path, string fileName)
		{
			zipFile = new ZipFile(path);
			zipEntry = zipFile.GetEntry(fileName);
			if (zipEntry == null)
				throw new Exception("Archive entry not found.");
			zipStream = zipFile.GetInputStream(zipEntry);
			temp = new byte[65536];
			position = 0;
		}
		public override bool CanRead
		{
			get { return true; }
		}
		public override bool CanSeek
		{
			get { return true; }
		}
		public override bool CanWrite
		{
			get { return false; }
		}
		public override long Length
		{
			get
			{
				return zipEntry.Size;
			}
		}
		public override long Position
		{
			get { return position; }
			set { Seek(value, SeekOrigin.Begin); }
		}
		public override void Close()
		{
			zipStream.Close();
			zipEntry = null;
			zipFile.Close();
		}
		public override void Flush()
		{
			throw new NotSupportedException();
		}
		public override void SetLength(long value)
		{
			throw new NotSupportedException();
		}
		public override int Read(byte[] buffer, int offset, int count)
		{
			if (position == 0 && zipEntry.IsCrypted && ((ZipInputStream)zipStream).Password == null && PasswordRequired != null)
			{
				PasswordRequiredEventArgs e = new PasswordRequiredEventArgs();
				PasswordRequired(this, e);
				if (e.ContinueOperation && e.Password.Length > 0)
					((ZipInputStream)zipStream).Password = e.Password;
			}
			// TODO: always save to a local temp circular buffer for optimization of the backwards seek.
			int total = zipStream.Read(buffer, offset, count);
			position += total;
			if (ExtractionProgress != null)
			{
				ExtractionProgressEventArgs e = new ExtractionProgressEventArgs();
				e.BytesExtracted = position;
				e.FileName = zipEntry.Name;
				e.FileSize = zipEntry.Size;
				e.PercentComplete = 100.0 * position / zipEntry.Size;
				ExtractionProgress(this, e);
			}
			return total;
		}
		public override long Seek(long offset, SeekOrigin origin)
		{
			long seek_to;
			switch (origin)
			{
				case SeekOrigin.Begin:
					seek_to = offset;
					break;
				case SeekOrigin.Current:
					seek_to = Position + offset;
					break;
				case SeekOrigin.End:
					seek_to = Length + offset;
					break;
				default:
					throw new NotSupportedException();
			}
			if (seek_to < 0 || seek_to > Length)
				throw new IOException("Invalid seek");
			if (seek_to < position)
			{
				zipStream.Close();
				zipStream = zipFile.GetInputStream(zipEntry);
				position = 0;
			}
			while (seek_to > position)
				if (Read(temp, 0, (int) Math.Min(seek_to - position, (long) temp.Length)) <= 0)
					throw new IOException("Invalid seek");
			return position;
		}
		public override void Write(byte[] array, int offset, int count)
		{
			throw new NotSupportedException();
		}
		public event PasswordRequiredHandler PasswordRequired;
		public event ExtractionProgressHandler ExtractionProgress;
	}

	public class ArchiveFileAbstraction : TagLib.File.IFileAbstraction
	{
		private string name;
		private CUESheet _cueSheet;

		public ArchiveFileAbstraction(CUESheet cueSheet, string file)
		{
			name = file;
			_cueSheet = cueSheet;
		}

		public string Name
		{
			get { return name; }
		}

		public System.IO.Stream ReadStream
		{
			get { return _cueSheet.OpenArchive(Name, true); }
		}

		public System.IO.Stream WriteStream
		{
			get { return null; }
		}

		public void CloseStream(System.IO.Stream stream)
		{
			stream.Close();
		}
	}

	public class CUELine {
		private List<String> _params;
		private List<bool> _quoted;

		public CUELine() {
			_params = new List<string>();
			_quoted = new List<bool>();
		}

		public CUELine(string line) {
			int start, end, lineLen;
			bool isQuoted;

			_params = new List<string>();
			_quoted = new List<bool>();

			start = 0;
			lineLen = line.Length;

			while (true) {
				while ((start < lineLen) && (line[start] == ' ')) {
					start++;
				}
				if (start >= lineLen) {
					break;
				}

				isQuoted = (line[start] == '"');
				if (isQuoted) {
					start++;
				}

				end = line.IndexOf(isQuoted ? '"' : ' ', start);
				if (end == -1) {
					end = lineLen;
				}

				_params.Add(line.Substring(start, end - start));
				_quoted.Add(isQuoted);

				start = isQuoted ? end + 1 : end;
			}
		}

		public List<string> Params {
			get {
				return _params;
			}
		}

		public List<bool> IsQuoted {
			get {
				return _quoted;
			}
		}

		public override string ToString() {
			if (_params.Count != _quoted.Count) {
				throw new Exception("Parameter and IsQuoted lists must match.");
			}

			StringBuilder sb = new StringBuilder();
			int last = _params.Count - 1;

			for (int i = 0; i <= last; i++) {
				if (_quoted[i] || _params[i].Contains(" ")) sb.Append('"');
				sb.Append(_params[i].Replace('"', '\''));
				if (_quoted[i] || _params[i].Contains(" ")) sb.Append('"');
				if (i < last) sb.Append(' ');
			}

			return sb.ToString();
		}
	}

	public class TrackInfo {
		private List<CUELine> _attributes;
		public TagLib.File _fileInfo;

		public TrackInfo() {
			_attributes = new List<CUELine>();
			_fileInfo = null;
		}

		public List<CUELine> Attributes {
			get {
				return _attributes;
			}
		}

		public string Artist {
			get {
				CUELine line = General.FindCUELine(_attributes, "PERFORMER");
				return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
			}
			set
			{
				General.SetCUELine(_attributes, "PERFORMER", value, true);
			}
		}

		public string Title {
			get {
				CUELine line = General.FindCUELine(_attributes, "TITLE");
				return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
			}
			set
			{
				General.SetCUELine(_attributes, "TITLE", value, true);
			}
		}
	}

	struct IndexInfo {
		public int Track;
		public int Index;
		public int Time;
	}

	struct SourceInfo {
		public string Path;
		public uint Offset;
		public uint Length;
	}

	public class StopException : Exception {
		public StopException() : base() {
		}
	}
}