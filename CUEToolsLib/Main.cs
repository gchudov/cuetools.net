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
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.Globalization;
using System.IO;
using System.Net;
using System.Threading;
using AudioCodecsDotNet;
using HDCDDotNet;

namespace CUEToolsLib
{

	public enum OutputAudioFormat
	{
		WAV,
		FLAC,
		WavPack,
		APE,
		NoAudio
	}

	public static class General {
		public static string FormatExtension(OutputAudioFormat value)
		{
			switch (value)
			{
				case OutputAudioFormat.FLAC: return ".flac";
				case OutputAudioFormat.WavPack: return ".wv";
				case OutputAudioFormat.APE: return ".ape";
				case OutputAudioFormat.WAV: return ".wav";
				case OutputAudioFormat.NoAudio: return ".dummy";
			}
			return ".wav";
		}

		public static int TimeFromString(string s) {
			string[] n = s.Split(':');
			if (n.Length != 3) {
				throw new Exception("Invalid timestamp.");
			}
			int min, sec, frame;

			min = Int32.Parse(n[0]);
			sec = Int32.Parse(n[1]);
			frame = Int32.Parse(n[2]);

			return frame + (sec * 75) + (min * 60 * 75);
		}

		public static string TimeToString(uint t) {
			uint min, sec, frame;

			frame = t % 75;
			t /= 75;
			sec = t % 60;
			t /= 60;
			min = t;

			return String.Format("{0:00}:{1:00}:{2:00}", min, sec, frame);
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

		public static void SetCUELine(List<CUELine> list, string command, string value, bool quoted)
		{
			CUELine line = General.FindCUELine(list, command);
			if (line == null)
			{
				line = new CUELine();
				line.Params.Add(command); line.IsQuoted.Add(false);
				line.Params.Add(value); line.IsQuoted.Add(true);
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
				line.Params.Add(value); line.IsQuoted.Add(true);
				list.Add(line);
			}
			else
			{
				line.Params[2] = value;
				line.IsQuoted[2] = quoted;
			}
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

	public delegate void SetStatus(string status, uint percentTrack, double percentDisk, string input, string output);

	public enum CUEStyle {
		SingleFileWithCUE,
		SingleFile,
		GapsPrepended,
		GapsAppended,
		GapsLeftOut
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
		public bool fillUpCUE;
		public bool filenamesANSISafe;
		public bool bruteForceDTL;
		public bool detectHDCD;
		public bool decodeHDCD;
		public bool wait750FramesForHDCD;
		public bool createM3U;
		public bool createCUEFileWhenEmbedded;
		public bool truncate4608ExtraSamples;

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
			replaceSpaces = true;
			embedLog = true;
			fillUpCUE = true;
			filenamesANSISafe = true;
			bruteForceDTL = false;
			detectHDCD = true;
			wait750FramesForHDCD = true;
			decodeHDCD = false;
			createM3U = false;
			createCUEFileWhenEmbedded = false;
			truncate4608ExtraSamples = true;
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
			sw.Save("FillUpCUE", fillUpCUE);
			sw.Save("FilenamesANSISafe", filenamesANSISafe);
			sw.Save("BruteForceDTL", bruteForceDTL);
			sw.Save("DetectHDCD", detectHDCD);
			sw.Save("Wait750FramesForHDCD", wait750FramesForHDCD);
			sw.Save("DecodeHDCD", decodeHDCD);
			sw.Save("CreateM3U", createM3U);
			sw.Save("CreateCUEFileWhenEmbedded", createCUEFileWhenEmbedded);
			sw.Save("Truncate4608ExtraSamples", truncate4608ExtraSamples);
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
			replaceSpaces = sr.LoadBoolean("ReplaceSpaces") ?? true;
			embedLog = sr.LoadBoolean("EmbedLog") ?? true;
			fillUpCUE = sr.LoadBoolean("FillUpCUE") ?? true;
			filenamesANSISafe = sr.LoadBoolean("FilenamesANSISafe") ?? true;
			bruteForceDTL = sr.LoadBoolean("BruteForceDTL") ?? false;
			detectHDCD = sr.LoadBoolean("DetectHDCD") ?? true;
			wait750FramesForHDCD = sr.LoadBoolean("Wait750FramesForHDCD") ?? true;
			decodeHDCD = sr.LoadBoolean("DecodeHDCD") ?? false;
			createM3U = sr.LoadBoolean("CreateM3U") ?? false;
			createCUEFileWhenEmbedded = sr.LoadBoolean("CreateCUEFileWhenEmbedded") ?? false;
			truncate4608ExtraSamples = sr.LoadBoolean("Truncate4608ExtraSamples") ?? true;
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
				if (removeSpecial && specialExceptions.IndexOf(ch) < 0 && !(
					((ch >= 'a') && (ch <= 'z')) ||
					((ch >= 'A') && (ch <= 'Z')) ||
					((ch >= '0') && (ch <= '9')) ||
					(ch == ' ') || (ch == '_')))
					ch = '_';
				if (Array.IndexOf(invalid, ch) >= 0)
					sb.Append("_");
				else
					sb.Append (ch);
			}

			return sb.ToString();
		}
	}

	public class CUESheet {
		private bool _stop, _pause;
		private List<CUELine> _attributes;
		private List<TrackInfo> _tracks;
		private List<SourceInfo> _sources;
		private List<string> _sourcePaths, _trackFilenames;
		private string _htoaFilename, _singleFilename;
		private bool _hasHTOAFilename, _hasTrackFilenames, _hasSingleFilename, _appliedWriteOffset;
		private bool _hasEmbeddedCUESheet;
		private bool _paddedToFrame, _truncated4608, _usePregapForFirstTrackInSingleFile;
		private int _writeOffset;
		private bool _accurateRip, _accurateOffset;
		private uint? _dataTrackLength;
		private uint? _minDataTrackLength;
		private string _accurateRipId;
		private string _accurateRipIdActual;
		private string _eacLog;
		private string _cuePath;
		private NameValueCollection _albumTags;
		private List<AccDisk> accDisks;
		private HttpStatusCode accResult;
		private const int _arOffsetRange = 5 * 588 - 1;
		private HDCDDotNet.HDCDDotNet hdcdDecoder;
		CUEConfig _config;
		string _cddbDiscIdTag;

		public CUESheet(string pathIn, CUEConfig config)
		{
			_config = config;

			hdcdDecoder = null;
			if (_config.detectHDCD)
			{
				try { hdcdDecoder = new HDCDDotNet.HDCDDotNet(2, 44100, _config.decodeHDCD); }
				catch { }
			}

			string cueDir, lineStr, command, pathAudio, fileType;
			CUELine line;
			TrackInfo trackInfo;
			int tempTimeLength, timeRelativeToFileStart, absoluteFileStartTime;
			int fileTimeLengthSamples, fileTimeLengthFrames, i, trackNumber;
			bool seenFirstFileIndex, seenDataTrack;
			List<IndexInfo> indexes;
			IndexInfo indexInfo;
			SourceInfo sourceInfo;
			NameValueCollection _trackTags = null;

			_stop = false;
			_pause = false;
			_attributes = new List<CUELine>();
			_tracks = new List<TrackInfo>();
			_sources = new List<SourceInfo>();
			_sourcePaths = new List<string>();
			_cuePath = null;
			_paddedToFrame = false;
			_truncated4608 = false;
			_usePregapForFirstTrackInSingleFile = false;
			_accurateRip = false;
			_accurateOffset = false;
			_appliedWriteOffset = false;
			_dataTrackLength = null;
			_minDataTrackLength = null;
			_albumTags = new NameValueCollection();
			cueDir = Path.GetDirectoryName(pathIn);
			pathAudio = null;
			indexes = new List<IndexInfo>();
			trackInfo = null;
			absoluteFileStartTime = 0;
			fileTimeLengthSamples = 0;
			fileTimeLengthFrames = 0;
			trackNumber = 0;
			seenFirstFileIndex = false;
			seenDataTrack = false;
			accDisks = new List<AccDisk>();
			_hasEmbeddedCUESheet = false;

			TextReader sr;

			if (Path.GetExtension(pathIn).ToLower() != ".cue")
			{
				IAudioSource audioSource;
				NameValueCollection tags;
				string cuesheetTag = null;

				audioSource = AudioReadWrite.GetAudioSource(pathIn);
				tags = audioSource.Tags;
				cuesheetTag = tags.Get("CUESHEET");
				_accurateRipId = tags.Get("ACCURATERIPID");
				_eacLog = tags.Get("LOG");
				if (_eacLog == null) _eacLog = tags.Get("LOGFILE");
				if (_eacLog == null) _eacLog = tags.Get("EACLOG");
				audioSource.Close();
				if (cuesheetTag == null)
					throw new Exception("Input file does not contain a .cue sheet.");
				sr = new StringReader (cuesheetTag);
				pathAudio = pathIn;
				_hasEmbeddedCUESheet = true;
			} else
			{
				if (_config.autoCorrectFilenames)
					sr = new StringReader (CorrectAudioFilenames(pathIn, false));
				else
					sr = new StreamReader (pathIn, CUESheet.Encoding);

				try
				{
					StreamReader logReader = new StreamReader(Path.ChangeExtension(pathIn, ".log"), CUESheet.Encoding);
					_eacLog = logReader.ReadToEnd();
					logReader.Close();
				}
				catch { }
			}

			using (sr) {
				while ((lineStr = sr.ReadLine()) != null) {
					line = new CUELine(lineStr);
					if (line.Params.Count > 0) {
						command = line.Params[0].ToUpper();

						if (command == "FILE") {
							fileType = line.Params[2].ToUpper();
							if ((fileType == "BINARY") || (fileType == "MOTOROLA")) {
								seenDataTrack = true;
							}
							else if (seenDataTrack) {
								throw new Exception("Audio tracks cannot appear after data tracks.");
							}
							else {
								if (!_hasEmbeddedCUESheet)
								{
									pathAudio = LocateFile(cueDir, line.Params[1]);
									if (pathAudio == null)
									{
										throw new Exception("Unable to locate file \"" + line.Params[1] + "\".");
									}
								} else
								{
									if (_sourcePaths.Count > 0 )
										throw new Exception("Extra file in embedded CUE sheet: \"" + line.Params[1] + "\".");
								}
								_sourcePaths.Add(pathAudio);
								absoluteFileStartTime += fileTimeLengthFrames;
								NameValueCollection tags;
								fileTimeLengthSamples = GetSampleLength(pathAudio, out tags);
								if ((fileTimeLengthSamples % 588) == 492 && _config.truncate4608ExtraSamples)
								{
									_truncated4608 = true;
									fileTimeLengthSamples -= 4608;
								}
								fileTimeLengthFrames = (int)((fileTimeLengthSamples + 587) / 588);
								if (_hasEmbeddedCUESheet)
									_albumTags = tags;
								else
									_trackTags = tags;
								seenFirstFileIndex = false;
							}
						}
						else if (command == "TRACK") {
							if (line.Params[2].ToUpper() != "AUDIO") {
								seenDataTrack = true;
							}
							else if (seenDataTrack) {
								throw new Exception("Audio tracks cannot appear after data tracks.");
							}
							else {
								trackNumber = Int32.Parse(line.Params[1]);
								if (trackNumber != _tracks.Count + 1) {
									throw new Exception("Invalid track number.");
								}
								trackInfo = new TrackInfo();
								_tracks.Add(trackInfo);
							}
						}
						else if (seenDataTrack) {
							// Ignore lines belonging to data tracks
						}
						else if (command == "INDEX") {
							timeRelativeToFileStart = General.TimeFromString(line.Params[2]);
							if (!seenFirstFileIndex)
							{
								if (timeRelativeToFileStart != 0)
								{
									throw new Exception("First index must start at file beginning.");
								}
								if (trackNumber > 0 && _trackTags != null && _trackTags.Count != 0)
									_tracks[trackNumber-1]._trackTags = _trackTags;
								seenFirstFileIndex = true;
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
							indexInfo.Track = trackNumber;
							indexInfo.Index = Int32.Parse(line.Params[1]);
							indexInfo.Time = absoluteFileStartTime + timeRelativeToFileStart;
							indexes.Add(indexInfo);
						}
						else if (command == "PREGAP") {
							if (seenFirstFileIndex) {
								throw new Exception("Pregap must occur at the beginning of a file.");
							}
							tempTimeLength = General.TimeFromString(line.Params[1]);
							indexInfo.Track = trackNumber;
							indexInfo.Index = 0;
							indexInfo.Time = absoluteFileStartTime;
							indexes.Add(indexInfo);
							sourceInfo.Path = null;
							sourceInfo.Offset = 0;
							sourceInfo.Length = (uint) tempTimeLength * 588;
							_sources.Add(sourceInfo);
							absoluteFileStartTime += tempTimeLength;
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
							_dataTrackLength = (uint)General.TimeFromString(line.Params[2]);
						}
						else if ((command == "REM") &&
						   (line.Params.Count == 3) &&
						   (line.Params[1].ToUpper() == "ACCURATERIPID"))
						{
							_accurateRipId = line.Params[2];
						}
						else
						{
							if (trackInfo != null)
							{
								trackInfo.Attributes.Add(line);
							}
							else
							{
								_attributes.Add(line);
							}
						}
					}
				}
				sr.Close();
			}

			if (trackNumber == 0) {
				throw new Exception("File must contain at least one audio track.");
			}

			// Add dummy track for calculation purposes
			indexInfo.Track = trackNumber + 1;
			indexInfo.Index = 1;
			indexInfo.Time = absoluteFileStartTime + fileTimeLengthFrames;
			indexes.Add(indexInfo);

			// Calculate the length of each index
			for (i = 0; i < indexes.Count - 1; i++) {
				indexInfo = indexes[i];

				tempTimeLength = indexes[i + 1].Time - indexInfo.Time;
				if (tempTimeLength > 0) {
					_tracks[indexInfo.Track - 1].AddIndex((indexInfo.Index == 0), (uint) tempTimeLength);
				}
				else if (tempTimeLength < 0) {
					throw new Exception("Indexes must be in chronological order.");
				}
			}

			for (i = 0; i < TrackCount; i++) {
				if (_tracks[i].LastIndex < 1) {
					throw new Exception("Track must have an INDEX 01.");
				}
			}

			// Store the audio filenames, generating generic names if necessary
			_hasSingleFilename = (_sourcePaths.Count == 1);
			_singleFilename = _hasSingleFilename ? Path.GetFileName(_sourcePaths[0]) :
				"Range.wav";

			_hasHTOAFilename = (_sourcePaths.Count == (TrackCount + 1));
			_htoaFilename = _hasHTOAFilename ? Path.GetFileName(_sourcePaths[0]) : "01.00.wav";

			_hasTrackFilenames = (_sourcePaths.Count == TrackCount) || _hasHTOAFilename;
			_trackFilenames = new List<string>();
			for (i = 0; i < TrackCount; i++) {
				_trackFilenames.Add( _hasTrackFilenames ? Path.GetFileName(
					_sourcePaths[i + (_hasHTOAFilename ? 1 : 0)]) : String.Format("{0:00}.wav", i + 1) );
			}

			if (_hasTrackFilenames)
				for (i = 0; i < TrackCount; i++)
				{
					TrackInfo track = _tracks[i];
					string artist = track._trackTags.Get("ARTIST");
					string title = track._trackTags.Get("TITLE");
					if (track.Artist == "" && artist != null)
						track.Artist = artist;
					if (track.Title == "" && title != null)
						track.Title = title;
				}
			if (!_hasEmbeddedCUESheet && _hasSingleFilename)
			{
				_albumTags = _tracks[0]._trackTags;
				_tracks[0]._trackTags = new NameValueCollection();
			}
			if (_config.fillUpCUE)
			{
				if (General.FindCUELine(_attributes, "PERFORMER") == null && GetCommonTag("ALBUM ARTIST") != null)
					General.SetCUELine(_attributes, "PERFORMER", GetCommonTag("ALBUM ARTIST"), true);
				if (General.FindCUELine(_attributes, "PERFORMER") == null && GetCommonTag("ARTIST") != null)
					General.SetCUELine(_attributes, "PERFORMER", GetCommonTag("ARTIST"), true);
				if (General.FindCUELine(_attributes, "TITLE") == null && GetCommonTag("ALBUM") != null)
					General.SetCUELine(_attributes, "TITLE", GetCommonTag("ALBUM"), true);
				if (General.FindCUELine(_attributes, "REM", "DATE") == null && GetCommonTag("DATE") != null)
					General.SetCUELine(_attributes, "REM", "DATE", GetCommonTag("DATE"), false);
				if (General.FindCUELine(_attributes, "REM", "DATE") == null && GetCommonTag("YEAR") != null)
					General.SetCUELine(_attributes, "REM", "DATE", GetCommonTag("YEAR"), false);
				if (General.FindCUELine(_attributes, "REM", "GENRE") == null && GetCommonTag("GENRE") != null)
					General.SetCUELine(_attributes, "REM", "GENRE", GetCommonTag("GENRE"), true);
			}
			if (_accurateRipId == null)
				_accurateRipId = GetCommonTag("ACCURATERIPID");

			if (_accurateRipId == null && _dataTrackLength == null && _eacLog != null)
			{
				sr = new StringReader(_eacLog);
				int lastAudioSector = -1;
				bool isEACLog = false;
				while ((lineStr = sr.ReadLine()) != null)
				{
					if (!isEACLog)
					{
						if (!lineStr.StartsWith("Exact Audio Copy"))
							break;
						isEACLog = true;
					}
					string[] n = lineStr.Split('|');
					if (n.Length == 5)
						try
						{
							int trNo = Int32.Parse(n[0]);
							int trStart = Int32.Parse(n[3]);
							int trEnd = Int32.Parse(n[4]);
							if (trNo == TrackCount && trEnd > 0)
								lastAudioSector = trEnd;
							if (trNo == TrackCount + 1 && lastAudioSector != -1 && trEnd > lastAudioSector + (90 + 60) * 75 + 150)
							{
								_dataTrackLength = (uint)(trEnd - lastAudioSector - (90 + 60) * 75 - 150);
								break;
							}
						}
						catch { }
				}
			}

			CUELine cddbDiscIdLine = General.FindCUELine(_attributes, "REM", "DISCID");
			_cddbDiscIdTag = cddbDiscIdLine != null && cddbDiscIdLine.Params.Count == 3 ? cddbDiscIdLine.Params[2] : null;
			if (_cddbDiscIdTag == null) _cddbDiscIdTag = GetCommonTag("DISCID");

			if (_dataTrackLength != null)
				_accurateRipIdActual = _accurateRipId = CalculateAccurateRipId();
			else
			{
				_accurateRipIdActual = CalculateAccurateRipId();
				if (_accurateRipId == null)
					_accurateRipId = _accurateRipIdActual;
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

		public string GetCommonTag(string tagName)
		{
			if (_hasEmbeddedCUESheet || _hasSingleFilename)
				return _albumTags.Get(tagName);
			if (_hasTrackFilenames)
			{
				string tagValue = null;
				bool commonValue = true;
				for (int i = 0; i < TrackCount; i++)
				{
					TrackInfo track = _tracks[i];
					string newValue = track._trackTags.Get (tagName);
					if (tagValue == null)
						tagValue = newValue;
					else
						commonValue = (newValue == null || tagValue == newValue);
				}
				return commonValue ? tagValue : null;
			}
			return null;
		}

		private static string LocateFile(string dir, string file) {
			List<string> dirList, fileList;
			string altDir, path;

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
					path = Path.Combine(dirList[iDir], fileList[iFile]);

					if (File.Exists(path)) {
						return path;
					}
				}
			}

			return null;
		}

		public void GenerateFilenames (OutputAudioFormat format, string outputPath)
		{
			_cuePath = outputPath;

			string extension = General.FormatExtension(format);
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

			replace.Add(General.EmptyStringToNull(_config.CleanseString(Artist)));
			replace.Add(General.EmptyStringToNull(_config.CleanseString(Title)));
			replace.Add(null);
			replace.Add(null);
			replace.Add(null);
			replace.Add(Path.GetFileNameWithoutExtension(outputPath));

			if (_config.keepOriginalFilenames && HasSingleFilename)
			{
				SingleFilename = Path.ChangeExtension(SingleFilename, extension);
			}
			else
			{
				filename = General.ReplaceMultiple(_config.singleFilenameFormat, find, replace);
				if (filename == null)
				{
					filename = "Range";
				}
				if (_config.replaceSpaces)
				{
					filename = filename.Replace(' ', '_');
				}
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
					{
						filename = replace[2];
					}
					if (_config.replaceSpaces)
					{
						filename = filename.Replace(' ', '_');
					}
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

		private int GetSampleLength(string path, out NameValueCollection tags)
		{
			IAudioSource audioSource;

			audioSource = AudioReadWrite.GetAudioSource(path);

			if ((audioSource.BitsPerSample != 16) ||
				(audioSource.ChannelCount != 2) ||
				(audioSource.SampleRate != 44100) ||
				(audioSource.Length > Int32.MaxValue))
			{
				audioSource.Close();
				throw new Exception("Audio format is invalid.");
			}

			tags = audioSource.Tags;
			audioSource.Close();
			return (int)audioSource.Length;
		}

		public void WriteM3U(string path, CUEStyle style)
		{
			StringWriter sw = new StringWriter();
			WriteM3U(sw, style);
			sw.Close();
			bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(sw.ToString())) != sw.ToString();
			StreamWriter sw1 = new StreamWriter(path, false, utf8Required ? Encoding.UTF8 : CUESheet.Encoding);
			sw1.Write(sw.ToString());
			sw1.Close();
		}

		public void WriteM3U(TextWriter sw, CUEStyle style)
		{
			int iTrack;
			bool htoaToFile = ((style == CUEStyle.GapsAppended) && _config.preserveHTOA &&
				(_tracks[0].IndexLengths[0] != 0));

			if (htoaToFile) {
				WriteLine(sw, 0, _htoaFilename);
			}
			for (iTrack = 0; iTrack < TrackCount; iTrack++) {
				WriteLine(sw, 0, _trackFilenames[iTrack]);
			}
		}

		public void Write(string path, CUEStyle style) {
			StringWriter sw = new StringWriter();
			Write(sw, style);
			sw.Close();
			bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(sw.ToString())) != sw.ToString();
			StreamWriter sw1 = new StreamWriter(path, false, utf8Required?Encoding.UTF8:CUESheet.Encoding);
			sw1.Write(sw.ToString());
			sw1.Close();
		}

		public void Write(TextWriter sw, CUEStyle style) {
			int i, iTrack, iIndex;
			TrackInfo track;
			bool htoaToFile = ((style == CUEStyle.GapsAppended) && _config.preserveHTOA &&
				(_tracks[0].IndexLengths[0] != 0));

			uint timeRelativeToFileStart = 0;

			using (sw) {
				if (_accurateRipId != null && _config.writeArTagsOnConvert)
					WriteLine(sw, 0, "REM ACCURATERIPID " +
						_accurateRipId);

				for (i = 0; i < _attributes.Count; i++) {
					WriteLine(sw, 0, _attributes[i]);
				}

				if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) {
					WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _singleFilename));
				}
				if (htoaToFile) {
					WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _htoaFilename));
				}

				for (iTrack = 0; iTrack < TrackCount; iTrack++) {
					track = _tracks[iTrack];

					if ((style == CUEStyle.GapsPrepended) ||
						(style == CUEStyle.GapsLeftOut) ||
						((style == CUEStyle.GapsAppended) &&
						((track.IndexLengths[0] == 0) || ((iTrack == 0) && !htoaToFile))) )
					{
						WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
						timeRelativeToFileStart = 0;
					}

					WriteLine(sw, 1, String.Format("TRACK {0:00} AUDIO", iTrack + 1));
					for (i = 0; i < track.Attributes.Count; i++) {
						WriteLine(sw, 2, track.Attributes[i]);
					}

					for (iIndex = 0; iIndex <= track.LastIndex; iIndex++) {
						if (track.IndexLengths[iIndex] != 0) {
							if ((iIndex == 0) &&
								((style == CUEStyle.GapsLeftOut) ||
								((style == CUEStyle.GapsAppended) && (iTrack == 0) && !htoaToFile) ||
								((style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) && (iTrack == 0) && _usePregapForFirstTrackInSingleFile)))
							{
								WriteLine(sw, 2, "PREGAP " + General.TimeToString(track.IndexLengths[iIndex]));
							}
							else {
								WriteLine(sw, 2, String.Format( "INDEX {0:00} {1}", iIndex,
									General.TimeToString(timeRelativeToFileStart) ));
								timeRelativeToFileStart += track.IndexLengths[iIndex];

								if ((style == CUEStyle.GapsAppended) && (iIndex == 0)) {
									WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
									timeRelativeToFileStart = 0;
								}
							}
						}
					}
				}
			}
		}

		private uint sumDigits(uint n) 
		{
		    uint r = 0;
		    while (n > 0) 
		    {
				r = r + (n % 10);
				n = n / 10;
		    }
		    return r;
		}
		private uint readIntLE(byte[] data, int pos)
		{
			return (uint) (data[pos] + ( data[pos+1] << 8 ) + ( data[pos+2] << 16 ) + ( data[pos+3] << 24) );
		}

		private string CalculateAccurateRipId ()
		{
			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;
			uint trackOffset = 0;

			for (int iTrack = 0; iTrack < TrackCount; iTrack++)
			{
				TrackInfo track = _tracks[iTrack];

				trackOffset += track.IndexLengths[0];
				discId1 += trackOffset;
				discId2 += (trackOffset == 0 ? 1 : trackOffset) * ((uint)iTrack + 1);
				cddbDiscId += sumDigits((uint)(trackOffset / 75) + 2);

				for (int iIndex = 1; iIndex <= track.LastIndex; iIndex++)
					trackOffset += track.IndexLengths[iIndex];
			}
			if (_dataTrackLength.HasValue)
			{
				trackOffset += ((90 + 60) * 75) + 150; // 90 second lead-out, 60 second lead-in, 150 sector gap
				cddbDiscId += sumDigits((uint)(trackOffset / 75) + 2);
				trackOffset += _dataTrackLength.Value;
			}
			discId1 += trackOffset;
			discId2 += (trackOffset == 0 ? 1 : trackOffset) * ((uint)TrackCount + 1);

			if (!_dataTrackLength.HasValue && _cddbDiscIdTag != null)
			{
				uint cddbDiscIdNum = UInt32.Parse(_cddbDiscIdTag, NumberStyles.HexNumber);
				if ((cddbDiscIdNum & 0xff) == TrackCount + 1)
				{
					uint lengthFromTag = ((cddbDiscIdNum >> 8) & 0xffff);
					_minDataTrackLength = ((lengthFromTag + (uint)(_tracks[0].IndexLengths[0] / 75)) - 152) * 75 - trackOffset;
				}
			}

			cddbDiscId = ((cddbDiscId % 255) << 24) + 
				(((uint)(trackOffset / 75) - (uint)(_tracks[0].IndexLengths[0] / 75)) << 8) + 
				(uint)(TrackCount + (_dataTrackLength.HasValue  ? 1 : 0));

			discId1 &= 0xFFFFFFFF;
			discId2 &= 0xFFFFFFFF;
			cddbDiscId &= 0xFFFFFFFF;

			return String.Format("{0:x8}-{1:x8}-{2:x8}", discId1, discId2, cddbDiscId);
		}

		public void ContactAccurateRip()
		{
		    // Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;
		
			string[] n = _accurateRipId.Split('-');
			if (n.Length != 3) {
				throw new Exception("Invalid accurateRipId.");
			}
			discId1 = UInt32.Parse(n[0], NumberStyles.HexNumber);
			discId2 = UInt32.Parse(n[1], NumberStyles.HexNumber);
			cddbDiscId = UInt32.Parse(n[2], NumberStyles.HexNumber);			
	
			string url = String.Format("http://www.accuraterip.com/accuraterip/{0:x}/{1:x}/{2:x}/dBAR-{3:d3}-{4:x8}-{5:x8}-{6:x8}.bin", 
				discId1 & 0xF, discId1>>4 & 0xF, discId1>>8 & 0xF, TrackCount, discId1, discId2, cddbDiscId);

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.Method = "GET";

			try
			{
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				accResult = resp.StatusCode;

				if (accResult == HttpStatusCode.OK)
				{
					// Retrieve response stream and wrap in StreamReader
					Stream respStream = resp.GetResponseStream();

					// Allocate byte buffer to hold stream contents
					byte [] urlData = new byte[13];
					int urlDataLen, bytesRead;
		
					accDisks.Clear();
					while ( true )
					{
						for (urlDataLen = 0; urlDataLen < 13; urlDataLen += bytesRead)
						{
							bytesRead = respStream.Read(urlData, urlDataLen, 13 - urlDataLen);
							if (0 == bytesRead)
								break;
						}
						if (urlDataLen == 0)
							break;
						if (urlDataLen < 13)
						{
							accResult = HttpStatusCode.PartialContent;
							return;
						}
						AccDisk dsk = new AccDisk();
						dsk.count = urlData[0];
						dsk.discId1 = readIntLE(urlData, 1);
						dsk.discId2 = readIntLE(urlData, 5);
						dsk.cddbDiscId = readIntLE(urlData, 9);

						for (int i = 0; i < dsk.count; i++)
						{
							for (urlDataLen = 0; urlDataLen < 9; urlDataLen += bytesRead)
							{
								bytesRead = respStream.Read(urlData, urlDataLen, 9 - urlDataLen);
								if (0 == bytesRead)
								{
									accResult = HttpStatusCode.PartialContent;
									return;
								}
							}
							AccTrack trk = new AccTrack();
							trk.count = urlData[0];
							trk.CRC = readIntLE(urlData, 1);
							trk.Frame450CRC = readIntLE(urlData, 5);
							dsk.tracks.Add(trk);
						}
						accDisks.Add(dsk);
					}
					respStream.Close();
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}		 
		}

		unsafe private void CalculateAccurateRipCRCsSemifast(int* samples, uint count, int iTrack, uint currentOffset, uint previousOffset, uint trackLength)
		{
			fixed (uint* CRCsA = iTrack != 0 ? _tracks[iTrack - 1].OffsetedCRC : null,
				CRCsB = _tracks[iTrack].OffsetedCRC,
				CRCsC = iTrack != TrackCount - 1 ? _tracks[iTrack + 1].OffsetedCRC : null)
			{
				for (uint si = 0; si < count; si++)
				{
					uint sampleValue = (uint) ((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
					int i;
					int iB = Math.Max(0, _arOffsetRange - (int)(currentOffset + si));
					int iC = Math.Min(2 * _arOffsetRange + 1, _arOffsetRange + (int)trackLength - (int)(currentOffset + si));

					uint baseSumA = sampleValue * (uint)(previousOffset + 1 - iB);
					for (i = 0; i < iB; i++)
					{
						CRCsA[i] += baseSumA;
						baseSumA += sampleValue;
					}
					uint baseSumB = sampleValue * (uint)Math.Max(1, (int)(currentOffset + si) - _arOffsetRange + 1);
					for (i = iB; i < iC; i++)
					{
						CRCsB[i] += baseSumB;
						baseSumB += sampleValue;
					}
					uint baseSumC = sampleValue;
					for (i = iC; i <= 2 * _arOffsetRange; i++)
					{
						CRCsC[i] += baseSumC;
						baseSumC += sampleValue;
					}
				}
				return;
			}
		}

		unsafe private void CalculateAccurateRipCRCs(int* samples, uint count, int iTrack, uint currentOffset, uint previousOffset, uint trackLength)
		{
			for (int si = 0; si < count; si++)
			{
				uint sampleValue = (uint)((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
				{
					int iTrack2 = iTrack;
					int currentOffset2 = (int)currentOffset + si - oi;

					if (currentOffset2 < 5 * 588 - 1 && iTrack == 0)
					// we are in the skipped area at the start of the disk
					{
						continue;
					}
					else if (currentOffset2 < 0)
					// offset takes us to previous track
					{
						iTrack2--;
						currentOffset2 += (int)previousOffset;
					}
					else if (currentOffset2 >= trackLength - 5 * 588 && iTrack == TrackCount - 1)
					// we are in the skipped area at the end of the disc
					{
						continue;
					}
					else if (currentOffset2 >= trackLength)
					// offset takes us to the next track
					{
						iTrack2++;
						currentOffset2 -= (int)trackLength;
					}
					_tracks[iTrack2].OffsetedCRC[_arOffsetRange - oi] += sampleValue * (uint) (currentOffset2 + 1);
				}
			}
		}

		unsafe private void CalculateAccurateRipCRCsFast(int* samples, uint count, int iTrack, uint currentOffset)
		{
			int s1 = (int) Math.Min(count, Math.Max(0, 450 * 588 - _arOffsetRange - (int)currentOffset));
			int s2 = (int) Math.Min(count, Math.Max(0, 451 * 588 + _arOffsetRange - (int)currentOffset));
			if ( s1 < s2 )
				fixed (uint* FrameCRCs = _tracks[iTrack].OffsetedFrame450CRC)
					for (int sj = s1; sj < s2; sj++)
					{
						int magicFrameOffset = (int)currentOffset + sj - 450 * 588 + 1;
						int firstOffset = Math.Max(-_arOffsetRange, magicFrameOffset - 588);
						int lastOffset = Math.Min(magicFrameOffset - 1, _arOffsetRange);
						uint sampleValue = (uint)((samples[2 * sj] & 0xffff) + (samples[2 * sj + 1] << 16));
						for (int oi = firstOffset; oi <= lastOffset; oi++)
							FrameCRCs[_arOffsetRange - oi] += sampleValue * (uint)(magicFrameOffset - oi);
					}
			fixed (uint* CRCs = _tracks[iTrack].OffsetedCRC)
			{
				uint baseSum = 0, stepSum = 0;
				currentOffset += (uint) _arOffsetRange + 1;
				for (uint si = 0; si < count; si++)
				{
					uint sampleValue = (uint)((samples[2 * si] & 0xffff) + (samples[2 * si + 1] << 16));
					stepSum += sampleValue;
					baseSum += sampleValue * (uint)(currentOffset + si);
				}
				for (int i = 2 * _arOffsetRange; i >= 0; i--)
				{
					CRCs[i] += baseSum;
					baseSum -= stepSum;
				}
			}
		}

		public void GenerateAccurateRipLog(TextWriter sw, int oi)
		{
			for (int iTrack = 0; iTrack < TrackCount; iTrack++)
			{
				uint count = 0;
				uint partials = 0;
				uint conf = 0;
				string pressings = "";
				string partpressings = "";
				for (int di = 0; di < (int)accDisks.Count; di++)
				{
					count += accDisks[di].tracks[iTrack].count;
					if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - oi] == accDisks[di].tracks[iTrack].CRC)
					{
						conf += accDisks[di].tracks[iTrack].count;
						if (pressings != "")
							pressings = pressings + ",";
						pressings = pressings + (di + 1).ToString();
					}
					if (_tracks[iTrack].OffsetedFrame450CRC[_arOffsetRange - oi] == accDisks[di].tracks[iTrack].Frame450CRC)
					{
						partials += accDisks[di].tracks[iTrack].count;
						if (partpressings != "")
							partpressings = partpressings + ",";
						partpressings = partpressings + (di + 1).ToString();
					}
				}
				if (conf > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) Accurately ripped as in pressing(s) #{4}", iTrack + 1, _tracks[iTrack].OffsetedCRC[_arOffsetRange - oi], count, conf, pressings));
				else if (partials > 0)
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] ({3:00}/{2:00}) Partial match to pressing(s) #{4} ", iTrack + 1, _tracks[iTrack].OffsetedCRC[_arOffsetRange - oi], count, partials, partpressings));
				else
					sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] (00/{2:00}) No matches", iTrack + 1, _tracks[iTrack].OffsetedCRC[_arOffsetRange - oi], count));
			}
		}

		public void GenerateAccurateRipLog(TextWriter sw)
		{
			int iTrack;
			sw.WriteLine (String.Format("[Disc ID: {0}]", _accurateRipId));
			if (_dataTrackLength.HasValue)
				sw.WriteLine("Assuming a data track was present, length {0}.", General.TimeToString(_dataTrackLength.Value));
			else
			{
				if (_minDataTrackLength.HasValue)
					sw.WriteLine("Data track was probably present, length {0}-{1}.", General.TimeToString(_minDataTrackLength.Value), General.TimeToString(_minDataTrackLength.Value + 74));
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

			if (accResult == HttpStatusCode.NotFound)
			{
				sw.WriteLine("Disk not present in database.");
				//for (iTrack = 0; iTrack < TrackCount; iTrack++)
				//    sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] Disk not present in database", iTrack + 1, _tracks[iTrack].CRC));
			}
			else if (accResult != HttpStatusCode.OK)
			{
				sw.WriteLine("Database access error: " + accResult.ToString());
				//for (iTrack = 0; iTrack < TrackCount; iTrack++)
				//    sw.WriteLine(String.Format(" {0:00}\t[{1:x8}] Database access error {2}", iTrack + 1, _tracks[iTrack].CRC, accResult.ToString()));
			}
			else
			{
				if (0 != _writeOffset)
					sw.WriteLine(String.Format("Offset applied: {0}", _writeOffset));
				int offsetApplied = _accurateOffset ? _writeOffset : 0;
				sw.WriteLine(String.Format("Track\t[ CRC    ] Status"));
				GenerateAccurateRipLog(sw, offsetApplied);
				uint offsets_match = 0;
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
				{
					uint matches = 0;
					for (iTrack = 0; iTrack < TrackCount; iTrack++)
						for (int di = 0; di < (int)accDisks.Count; di++)
							if ( (_tracks[iTrack].OffsetedCRC[_arOffsetRange - oi] == accDisks[di].tracks[iTrack].CRC && accDisks[di].tracks[iTrack].CRC != 0) ||
								 (_tracks[iTrack].OffsetedFrame450CRC[_arOffsetRange - oi] == accDisks[di].tracks[iTrack].Frame450CRC && accDisks[di].tracks[iTrack].Frame450CRC != 0) )
								matches++;
					if (matches != 0 && oi != offsetApplied)
					{
						if (offsets_match++ > 10)
						{
							sw.WriteLine("More than 10 offsets match!");
							break;
						}
						sw.WriteLine(String.Format("Offsetted by {0}:", oi));
						GenerateAccurateRipLog(sw, oi);
					}
				}
			}
		}

		public void GenerateAccurateRipTagsForTrack(NameValueCollection tags, int offset, int bestOffset, int iTrack, string prefix)
		{
			uint total = 0;
			uint matching = 0;
			uint matching2 = 0;
			uint matching3 = 0;
			for (int iDisk = 0; iDisk < accDisks.Count; iDisk++)
			{
				total += accDisks[iDisk].tracks[iTrack].count;
				if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - offset] ==
					accDisks[iDisk].tracks[iTrack].CRC)
					matching += accDisks[iDisk].tracks[iTrack].count;
				if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - bestOffset] ==
					accDisks[iDisk].tracks[iTrack].CRC)
					matching2 += accDisks[iDisk].tracks[iTrack].count;
				for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
					if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - oi] ==
						accDisks[iDisk].tracks[iTrack].CRC)
						matching3 += accDisks[iDisk].tracks[iTrack].count;
			}
			tags.Add(String.Format("{0}ACCURATERIPCRC", prefix), String.Format("{0:x8}", _tracks[iTrack].OffsetedCRC[_arOffsetRange - offset]));
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

					for (int di = 0; di < (int)accDisks.Count; di++)
						if (_tracks[iTrack].OffsetedCRC[_arOffsetRange - offset] == accDisks[di].tracks[iTrack].CRC)
							confidence += accDisks[di].tracks[iTrack].count;

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

		public void WriteAudioFiles(string dir, CUEStyle style, SetStatus statusDel) {
			string[] destPaths;
			int[] destLengths;
			bool htoaToFile = ((style == CUEStyle.GapsAppended) && _config.preserveHTOA &&
				(_tracks[0].IndexLengths[0] != 0));

			if (_usePregapForFirstTrackInSingleFile) {
				throw new Exception("UsePregapForFirstTrackInSingleFile is not supported for writing audio files.");
			}

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

			if ( !_accurateRip || _accurateOffset )
			for (int i = 0; i < destPaths.Length; i++) {
				for (int j = 0; j < _sourcePaths.Count; j++) {
					if (destPaths[i].ToLower() == _sourcePaths[j].ToLower()) {
						throw new Exception("Source and destination audio file paths cannot be the same.");
					}
				}
			}

			destLengths = CalculateAudioFileLengths(style);

			bool SkipOutput = false;

			if (_accurateRip) {
				statusDel((string)"Contacting AccurateRip database...", 0, 0, null, null);
				if (!_dataTrackLength.HasValue && _minDataTrackLength.HasValue && _accurateRipId == _accurateRipIdActual && _config.bruteForceDTL)
				{
					uint minDTL = _minDataTrackLength.Value;
					for (uint dtl = minDTL; dtl < minDTL + 75; dtl++)
					{
						_dataTrackLength = dtl;
						_accurateRipId = CalculateAccurateRipId();
						ContactAccurateRip();
						if (accResult != HttpStatusCode.NotFound)
							break;
						statusDel((string)"Contacting AccurateRip database...", 0, (dtl-minDTL)/75.0, null, null);
						lock (this) {
							if (_stop)
								throw new StopException();
							if (_pause)
							{
								statusDel("Paused...", 0, 0, null, null);
								Monitor.Wait(this);
							}
							else
								Monitor.Wait(this, 1000);
						}
					}
					if (accResult != HttpStatusCode.OK)
					{
						_dataTrackLength = null;
						_accurateRipId = _accurateRipIdActual;
					}
				} else
					ContactAccurateRip();

				if (accResult != HttpStatusCode.OK)
				{
					if (!_accurateOffset || _config.noUnverifiedOutput)
					{
						if ((_accurateOffset && _config.writeArLogOnConvert)  || 
							(!_accurateOffset && _config.writeArLogOnVerify))
						{
							if (!Directory.Exists(dir))
								Directory.CreateDirectory(dir);
							StreamWriter sw = new StreamWriter(Path.ChangeExtension(_cuePath, ".accurip"),
								false, CUESheet.Encoding);
							GenerateAccurateRipLog(sw);
							sw.Close();
						}
						return;
					}
				}
				else if (_accurateOffset)
				{
					_writeOffset = 0;
					WriteAudioFilesPass(dir, style, statusDel, destPaths, destLengths, htoaToFile, true);

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
			}

			if (!SkipOutput)
			{
				bool verifyOnly = _accurateRip && !_accurateOffset;
				if (!verifyOnly)
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
					if (style != CUEStyle.SingleFileWithCUE)
						Write(_cuePath, style);
					else if (_config.createCUEFileWhenEmbedded)
						Write(Path.ChangeExtension(_cuePath, ".cue"), style);
					if (style != CUEStyle.SingleFileWithCUE && style != CUEStyle.SingleFile && _config.createM3U)
						WriteM3U(Path.ChangeExtension(_cuePath, ".m3u"), style);
				}
				WriteAudioFilesPass(dir, style, statusDel, destPaths, destLengths, htoaToFile, verifyOnly);
			}

			if (_accurateRip)
			{
				statusDel((string)"Generating AccurateRip report...", 0, 0, null, null);
				if (!_accurateOffset && _config.writeArTagsOnVerify && _writeOffset == 0)
				{
					uint tracksMatch;
					int bestOffset;
					FindBestOffset(1, true, out tracksMatch, out bestOffset);

					if (_hasEmbeddedCUESheet)
					{
						IAudioSource audioSource = AudioReadWrite.GetAudioSource(_sourcePaths[0]);
						NameValueCollection tags = audioSource.Tags;
						CleanupTags(tags, "ACCURATERIP");
						GenerateAccurateRipTags (tags, 0, bestOffset, -1);
#if !MONO
						if (audioSource is FLACReader)
							((FLACReader)audioSource).UpdateTags (true);
#endif
						audioSource.Close();
						audioSource = null;
					} else if (_hasTrackFilenames)
					{
						for (int iTrack = 0; iTrack < TrackCount; iTrack++)
						{
							string src = _sourcePaths[iTrack + (_hasHTOAFilename ? 1 : 0)];
							IAudioSource audioSource = AudioReadWrite.GetAudioSource(src);
#if !MONO
							if (audioSource is FLACReader)
							{
								NameValueCollection tags = audioSource.Tags;
								CleanupTags(tags, "ACCURATERIP");
								GenerateAccurateRipTags (tags, 0, bestOffset, iTrack);
								((FLACReader)audioSource).UpdateTags(true);
							}
#endif
							audioSource.Close();
							audioSource = null;
						}
					}
				}

				if ((_accurateOffset && _config.writeArLogOnConvert) ||
					(!_accurateOffset && _config.writeArLogOnVerify))
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
					StreamWriter sw = new StreamWriter(Path.ChangeExtension(_cuePath, ".accurip"),
						false, CUESheet.Encoding);
					GenerateAccurateRipLog(sw);
					sw.Close();
				}
			}
		}

		private void SetTrackTags(IAudioDest audioDest, int iTrack, int bestOffset)
		{
			NameValueCollection destTags = new NameValueCollection();

			if (_hasEmbeddedCUESheet)
			{
				string trackPrefix = String.Format ("cue_track{0:00}_", iTrack + 1);
				string[] keys = _albumTags.AllKeys;
				for (int i = 0; i < keys.Length; i++)
				{
					if (keys[i].ToLower().StartsWith(trackPrefix)
						|| !keys[i].ToLower().StartsWith("cue_track"))
					{
						string name = keys[i].ToLower().StartsWith(trackPrefix) ? 
							keys[i].Substring(trackPrefix.Length) : keys[i];
						string[] values = _albumTags.GetValues(keys[i]);
						for (int j = 0; j < values.Length; j++)
							destTags.Add(name, values[j]);
					}
				}
			}
			else if (_hasTrackFilenames)
				destTags.Add(_tracks[iTrack]._trackTags);
			else if (_hasSingleFilename)
			{
				// TODO?
			}

			destTags.Remove("CUESHEET");
			destTags.Remove("TRACKNUMBER");
			destTags.Remove("LOG");
			destTags.Remove("LOGFILE");
			destTags.Remove("EACLOG");
			CleanupTags(destTags, "ACCURATERIP");
			CleanupTags(destTags, "REPLAYGAIN");

			if (destTags.Get("TITLE") == null && "" != _tracks[iTrack].Title)
				destTags.Add("TITLE", _tracks[iTrack].Title);
			if (destTags.Get("ARTIST") == null && "" != _tracks[iTrack].Artist)
				destTags.Add("ARTIST", _tracks[iTrack].Artist);
			destTags.Add("TRACKNUMBER", (iTrack + 1).ToString());
			if (_accurateRipId != null && _config.writeArTagsOnConvert)
			{
				if (_accurateOffset && accResult == HttpStatusCode.OK)
					GenerateAccurateRipTags(destTags, _writeOffset, bestOffset, iTrack);
				else
					destTags.Add("ACCURATERIPID", _accurateRipId);
			}
			audioDest.SetTags(destTags);
		}

		private void SetAlbumTags(IAudioDest audioDest, int bestOffset, bool fWithCUE)
		{
			NameValueCollection destTags = new NameValueCollection();

			if (_hasEmbeddedCUESheet || _hasSingleFilename)
			{
				destTags.Add(_albumTags);
				if (!fWithCUE)
					CleanupTags(destTags, "CUE_TRACK");
			}
			else if (_hasTrackFilenames)
			{
				for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				{
					string[] keys = _tracks[iTrack]._trackTags.AllKeys;
					for (int i = 0; i < keys.Length; i++)
					{
						string singleValue = GetCommonTag (keys[i]);
						if (singleValue != null)
						{
							if (destTags.Get(keys[i]) == null)
								destTags.Add(keys[i], singleValue);
						}
						else if (fWithCUE && keys[i].ToUpper() != "TRACKNUMBER")
						{
							string[] values = _tracks[iTrack]._trackTags.GetValues(keys[i]);
							for (int j = 0; j < values.Length; j++)
								destTags.Add(String.Format("cue_track{0:00}_{1}", iTrack + 1, keys[i]), values[j]);
						}
					}
				}
			}

			destTags.Remove("CUESHEET");
			destTags.Remove("TITLE");
			destTags.Remove("TRACKNUMBER");
			CleanupTags(destTags, "ACCURATERIP");
			CleanupTags(destTags, "REPLAYGAIN");

			if (fWithCUE)
			{
				StringWriter sw = new StringWriter();
				Write(sw, CUEStyle.SingleFileWithCUE);
				destTags.Add("CUESHEET", sw.ToString());
				sw.Close();
			}

			if (_config.embedLog)
			{
				destTags.Remove("LOG");
				destTags.Remove("LOGFILE");
				destTags.Remove("EACLOG");
				if (_eacLog != null)
					destTags.Add("LOG", _eacLog);
			}

			if (_accurateRipId != null && _config.writeArTagsOnConvert)
			{
				if (fWithCUE && _accurateOffset && accResult == HttpStatusCode.OK)
					GenerateAccurateRipTags(destTags, _writeOffset, bestOffset, -1);
				else
					destTags.Add("ACCURATERIPID", _accurateRipId);
			}
			audioDest.SetTags(destTags);
		}

		public void WriteAudioFilesPass(string dir, CUEStyle style, SetStatus statusDel, string[] destPaths, int[] destLengths, bool htoaToFile, bool noOutput)
		{
			const int buffLen = 16384;
			int iTrack, iIndex;
			int[,] sampleBuffer = new int[buffLen, 2];
			TrackInfo track;
			IAudioSource audioSource = null;
			IAudioDest audioDest = null;
			IAudioDest decodedAudioDest = null;
			bool discardOutput;
			int iSource = -1;
			int iDest = -1;
			uint samplesRemSource = 0;

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

			uint tracksMatch;
			int bestOffset = _writeOffset;
			if (!noOutput && _accurateRipId != null && _config.writeArTagsOnConvert && _accurateOffset && accResult == HttpStatusCode.OK)
				FindBestOffset(1, true, out tracksMatch, out bestOffset);

			if (hdcdDecoder != null)
				hdcdDecoder.Reset();

			if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
			{
				iDest++;
				audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput, 16);
				if (!noOutput)
					SetAlbumTags(audioDest, bestOffset, style == CUEStyle.SingleFileWithCUE);
				if (_config.detectHDCD && hdcdDecoder != null && _config.decodeHDCD && !noOutput)
				{
					decodedAudioDest = GetAudioDest(Path.ChangeExtension(destPaths[iDest], ".24bit" + Path.GetExtension(destPaths[iDest])),
						destLengths[iDest], noOutput, 24);
					SetAlbumTags(decodedAudioDest, bestOffset, style == CUEStyle.SingleFileWithCUE);
				}
			}

			if (_accurateRip && noOutput)
				for (iTrack = 0; iTrack < TrackCount; iTrack++)
					for (int iCRC = 0; iCRC < 10 * 588; iCRC++)
					{
						_tracks[iTrack].OffsetedCRC[iCRC] = 0;
						_tracks[iTrack].OffsetedFrame450CRC[iCRC] = 0;
					}

			uint currentOffset = 0, previousOffset = 0;
			uint trackLength = _tracks[0].IndexLengths[0] * 588;
			uint diskLength = 0, diskOffset = 0;

			for (iTrack = 0; iTrack < TrackCount; iTrack++)
				for (iIndex = 0; iIndex <= _tracks[iTrack].LastIndex; iIndex++)
					diskLength += _tracks[iTrack].IndexLengths[iIndex] * 588;


			statusDel(String.Format("{2} track {0:00} ({1:00}%)...", 0, 0, noOutput ? "Verifying" : "Writing"), 0, 0.0, null, null);

			for (iTrack = 0; iTrack < TrackCount; iTrack++) {
				track = _tracks[iTrack];

				if ((style == CUEStyle.GapsPrepended) || (style == CUEStyle.GapsLeftOut)) {
					if (audioDest != null) audioDest.Close();
					iDest++;
					audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput, 16);
					if (!noOutput)
						SetTrackTags(audioDest, iTrack, bestOffset);
					if (_config.detectHDCD && hdcdDecoder != null && _config.decodeHDCD && !noOutput)
					{
						hdcdDecoder.AudioDest = null;
						if (decodedAudioDest != null)
							decodedAudioDest.Close();
						decodedAudioDest = GetAudioDest(Path.ChangeExtension(destPaths[iDest], ".24bit" + Path.GetExtension(destPaths[iDest])),
							destLengths[iDest], noOutput, 24);
						SetTrackTags(decodedAudioDest, iTrack, bestOffset);
					}
				}		

				for (iIndex = 0; iIndex <= track.LastIndex; iIndex++) {
					uint trackPercent= 0, lastTrackPercent= 101;
					uint samplesRemIndex = track.IndexLengths[iIndex] * 588;

					if (iIndex == 1)
					{
						previousOffset = currentOffset;
						currentOffset = 0;
						trackLength  = 0;
						for (int iIndex2 = 1; iIndex2 <= track.LastIndex; iIndex2++)
							trackLength += _tracks[iTrack].IndexLengths[iIndex2] * 588;
						if (iTrack != TrackCount -1)
							trackLength += _tracks[iTrack + 1].IndexLengths[0] * 588;
					}

					if ((style == CUEStyle.GapsAppended) && (iIndex == 1)) {
						if (audioDest != null) audioDest.Close();
						iDest++;
						audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput, 16);
						if (!noOutput)
							SetTrackTags(audioDest, iTrack, bestOffset);
						if (_config.detectHDCD && hdcdDecoder != null && _config.decodeHDCD && !noOutput)
						{
							hdcdDecoder.AudioDest = null;
							if (decodedAudioDest != null)
								decodedAudioDest.Close();
							decodedAudioDest = GetAudioDest(Path.ChangeExtension(destPaths[iDest], ".24bit" + Path.GetExtension(destPaths[iDest])),
								destLengths[iDest], noOutput, 24);
							SetTrackTags(decodedAudioDest, iTrack, bestOffset);
						}
					}

					if ((style == CUEStyle.GapsAppended) && (iIndex == 0) && (iTrack == 0)) {
						discardOutput = !htoaToFile;
						if (htoaToFile) {
							iDest++;
							audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput, 16);
							if (_config.detectHDCD && hdcdDecoder != null && _config.decodeHDCD && !noOutput)
							{
								decodedAudioDest = GetAudioDest(Path.ChangeExtension(destPaths[iDest], ".24bit" + Path.GetExtension(destPaths[iDest])),
									destLengths[iDest], noOutput, 24);
							}
						}
					}
					else if ((style == CUEStyle.GapsLeftOut) && (iIndex == 0)) {
						discardOutput = true;
					}
					else {
						discardOutput = false;
					}

					while (samplesRemIndex != 0) {
						if (samplesRemSource == 0) {
							if (audioSource != null) audioSource.Close();
							audioSource = GetAudioSource(++iSource);
							samplesRemSource = (uint) _sources[iSource].Length;
						}

						uint copyCount = (uint) Math.Min(Math.Min(samplesRemIndex, samplesRemSource), buffLen);

						if ( trackLength > 0 )
						{
							trackPercent = (uint)(currentOffset / 0.01 / trackLength);
							double diskPercent = ((float)diskOffset) / diskLength;
							if (trackPercent != lastTrackPercent)
								statusDel(String.Format("{2} track {0:00} ({1:00}%)...", iIndex > 0 ? iTrack + 1 : iTrack, trackPercent,
									noOutput ? "Verifying" : "Writing"), trackPercent, diskPercent, 
									audioSource.Path, discardOutput ? null : audioDest.Path);
							lastTrackPercent = trackPercent;
						}

						audioSource.Read(sampleBuffer, copyCount);					
						if (!discardOutput) audioDest.Write(sampleBuffer, copyCount);
						if (_config.detectHDCD && hdcdDecoder != null)
						{
							if (_config.wait750FramesForHDCD && diskOffset > 750 * 588 && !hdcdDecoder.Detected)
							{
								hdcdDecoder.AudioDest = null;
								hdcdDecoder = null;
							}
							else
							{
								if (_config.decodeHDCD)
									hdcdDecoder.AudioDest = (discardOutput || noOutput) ? null : decodedAudioDest;
								hdcdDecoder.Process(sampleBuffer, copyCount);
							}
						}
						if (_accurateRip && noOutput && (iTrack != 0 || iIndex != 0))
						unsafe {
							fixed (int *pSampleBuff = &sampleBuffer[0,0])
							{
								int iTrack2 = iTrack - (iIndex == 0 ? 1 : 0);

								uint si1 = (uint) Math.Min(copyCount, Math.Max(0, 588*(iTrack2 == 0?10:5) - (int) currentOffset));
								uint si2 = (uint) Math.Min(copyCount, Math.Max(si1, trackLength - (int) currentOffset - 588 * (iTrack2 == TrackCount - 1?10:5)));
								if (iTrack2 == 0)
									CalculateAccurateRipCRCs (pSampleBuff, si1, iTrack2, currentOffset, previousOffset, trackLength);
								else
									CalculateAccurateRipCRCsSemifast(pSampleBuff, si1, iTrack2, currentOffset, previousOffset, trackLength);
								if (si2 > si1)
									CalculateAccurateRipCRCsFast(pSampleBuff + si1 * 2, (uint)(si2 - si1), iTrack2, currentOffset + si1);
								if (iTrack2 == TrackCount - 1)
									CalculateAccurateRipCRCs(pSampleBuff + si2*2, copyCount - si2, iTrack2, currentOffset + si2, previousOffset, trackLength);
								else
									CalculateAccurateRipCRCsSemifast(pSampleBuff + si2*2, copyCount - si2, iTrack2, currentOffset + si2, previousOffset, trackLength);
							}
						}
						currentOffset += copyCount;
						diskOffset += copyCount;
						samplesRemIndex -= copyCount;
						samplesRemSource -= copyCount;

						lock (this) {
							if (_stop) {
								if (hdcdDecoder != null)
									hdcdDecoder.AudioDest = null;
								audioSource.Close();
								try {
									if (audioDest != null) audioDest.Close();									
								} catch { }
								// need two separate try/catches, 
								// because Close always throws an exception, and 
								// we want both streams closed.
								try {
									if (decodedAudioDest != null) decodedAudioDest.Close();
								} catch { }
								throw new StopException();
							}
							if (_pause)
							{
								statusDel ("Paused...", 0, 0, null, null);
								Monitor.Wait(this);
							}
						}
					}
				}
			}

			if (hdcdDecoder != null)
				hdcdDecoder.AudioDest = null;
			if (audioSource != null) audioSource.Close();
			audioDest.Close();
			if (decodedAudioDest != null)
				decodedAudioDest.Close();
		}

		public static string CorrectAudioFilenames(string path, bool always) {
			string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a" };
			List<string> lines = new List<string>();
			List<int> filePos = new List<int>();
			List<string> origFiles = new List<string>();
			bool foundAll = true;
			string[] audioFiles = null;
			string lineStr;
			CUELine line;
			string dir;
			int i;

			dir = Path.GetDirectoryName(path);

			using (StreamReader sr = new StreamReader(path, CUESheet.Encoding)) {
				while ((lineStr = sr.ReadLine()) != null) {
					lines.Add(lineStr);
					line = new CUELine(lineStr);
					if ((line.Params.Count == 3) && (line.Params[0].ToUpper() == "FILE")) {
						string fileType = line.Params[2].ToUpper();
						if ((fileType != "BINARY") && (fileType != "MOTOROLA")) {
							filePos.Add(lines.Count - 1);
							origFiles.Add(line.Params[1]);
							foundAll &= (LocateFile(dir, line.Params[1]) != null);
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
					foundAll = true;
					List<string> newFiles = new List<string>();
					for (int j = 0; j < origFiles.Count; j++)
					{
						string newFilename = Path.ChangeExtension(Path.GetFileName(origFiles[j]), audioExts[i].Substring(1));
						foundAll &= LocateFile(dir, newFilename) != null;
						newFiles.Add (newFilename);
					}
					if (foundAll)
					{
						audioFiles = newFiles.ToArray();
						break;
					}
				}
				if (!foundAll)
				for (i = 0; i < audioExts.Length; i++)
				{
					audioFiles = Directory.GetFiles(dir == "" ? "." : dir, audioExts[i]);
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

		private int[] CalculateAudioFileLengths(CUEStyle style) {
			int iTrack, iIndex, iFile;
			TrackInfo track;
			int[] fileLengths;
			bool htoaToFile = ((style == CUEStyle.GapsAppended) && _config.preserveHTOA &&
				(_tracks[0].IndexLengths[0] != 0));
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

				if ((style == CUEStyle.GapsPrepended) || (style == CUEStyle.GapsLeftOut)) {
					iFile++;
				}

				for (iIndex = 0; iIndex <= track.LastIndex; iIndex++) {
					if ((style == CUEStyle.GapsAppended) && (iIndex == 1)) {
						iFile++;
					}

					if ((style == CUEStyle.GapsAppended) && (iIndex == 0) && (iTrack == 0)) {
						discardOutput = !htoaToFile;
						if (htoaToFile) {
							iFile++;
						}
					}
					else if ((style == CUEStyle.GapsLeftOut) && (iIndex == 0)) {
						discardOutput = true;
					}
					else {
						discardOutput = false;
					}

					if (!discardOutput) {
						fileLengths[iFile] += (int) track.IndexLengths[iIndex] * 588;
					}
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

		private IAudioDest GetAudioDest(string path, int finalSampleCount, bool noOutput, int bitsPerSample) {
			if (noOutput)
				return new DummyWriter(path, bitsPerSample, 2, 44100);

			IAudioDest dest = AudioReadWrite.GetAudioDest(path, bitsPerSample, 2, 44100, finalSampleCount);

#if !MONO
			if (dest is FLACWriter) {
				FLACWriter w = (FLACWriter)dest;
				w.CompressionLevel = (int) _config.flacCompressionLevel;
				w.Verify = _config.flacVerify;
			}
			if (dest is WavPackWriter) {
				WavPackWriter w = (WavPackWriter)dest;
				w.CompressionMode = _config.wvCompressionMode;
				w.ExtraMode = _config.wvExtraMode;
				w.MD5Sum = _config.wvStoreMD5;
			}
			if (dest is APEWriter)
			{
				APEWriter w = (APEWriter)dest;
				w.CompressionLevel = (int) _config.apeCompressionLevel;
			}
#endif

			return dest;
		}

		private IAudioSource GetAudioSource(int sourceIndex) {
			SourceInfo sourceInfo = _sources[sourceIndex];
			IAudioSource audioSource;

			if (sourceInfo.Path == null) {
				audioSource = new SilenceGenerator(sourceInfo.Offset + sourceInfo.Length);
			}
			else {
				audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path);
			}

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
				return (line == null) ? String.Empty : line.Params[1];
			}
		}

		public string Title {
			get {
				CUELine line = General.FindCUELine(_attributes, "TITLE");
				return (line == null) ? String.Empty : line.Params[1];
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
				return General.TimeToString(_dataTrackLength.HasValue ? _dataTrackLength.Value : 0);
			}
			set
			{
				uint dtl = (uint) General.TimeFromString(value);
				if (dtl != 0)
				{
					_dataTrackLength = dtl;
					_accurateRipId = _accurateRipIdActual = CalculateAccurateRipId();
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

		public CUEConfig Config {
			get {
				return _config;
			}
		}

		public bool AccurateRip {
			get {
				return _accurateRip;
			}
			set {
				_accurateRip = value;
			}
		}

		public bool AccurateOffset {
			get {
				return _accurateOffset;
			}
			set {
				_accurateOffset = value;
			}
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
				if (_quoted[i]) sb.Append('"');
				sb.Append(_params[i]);
				if (_quoted[i]) sb.Append('"');
				if (i < last) sb.Append(' ');
			}

			return sb.ToString();
		}
	}

	public class TrackInfo {
		private List<uint> _indexLengths;
		private List<CUELine> _attributes;
		public uint[] OffsetedCRC;
		public uint[] OffsetedFrame450CRC;
		public NameValueCollection _trackTags;

		public TrackInfo() {
			_indexLengths = new List<uint>();
			_attributes = new List<CUELine>();
			_trackTags = new NameValueCollection();
			OffsetedCRC = new uint[10 * 588];
			OffsetedFrame450CRC = new uint[10 * 588];

			_indexLengths.Add(0);
		}

		public uint CRC
		{
			get
			{
				return OffsetedCRC[5 * 588 - 1];
			}
		}

		public int LastIndex {
			get {
				return _indexLengths.Count - 1;
			}
		}

		public List<uint> IndexLengths {
			get {
				return _indexLengths;
			}
		}

		public void AddIndex(bool isGap, uint length) {
			if (isGap) {
				_indexLengths[0] = length;
			}
			else {
				_indexLengths.Add(length);
			}
		}

		public List<CUELine> Attributes {
			get {
				return _attributes;
			}
		}

		public string Artist {
			get {
				CUELine line = General.FindCUELine(_attributes, "PERFORMER");
				return (line == null) ? String.Empty : line.Params[1];
			}
			set
			{
				General.SetCUELine(_attributes, "PERFORMER", value, true);
			}
		}

		public string Title {
			get {
				CUELine line = General.FindCUELine(_attributes, "TITLE");
				return (line == null) ? String.Empty : line.Params[1];
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

	struct AccTrack
	{
		public uint count;
		public uint CRC;
		public uint Frame450CRC;
	}

	class	AccDisk
	{
		public uint		count;
		public uint		discId1;
		public uint		discId2;
		public uint		cddbDiscId;
		public List<AccTrack>	tracks;

		public AccDisk() {
			tracks = new List<AccTrack>();
		}
	}


	public class StopException : Exception {
		public StopException() : base() {
		}
	}
}