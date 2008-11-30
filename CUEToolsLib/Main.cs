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
using System.Security.Cryptography;
using System.Threading;
using System.Xml;
using HDCDDotNet;
using CUETools.Codecs;
using CUETools.Codecs.LossyWAV;
using CUETools.CDImage;
using CUETools.AccurateRip;
#if !MONO
using UnRarDotNet;
using FLACDotNet;
#endif

namespace CUETools.Processor
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
		public bool createTOC;
		public bool createCUEFileWhenEmbedded;
		public bool truncate4608ExtraSamples;
		public int lossyWAVQuality;
		public bool decodeHDCDtoLW16;
		public bool decodeHDCDto24bit;

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
			createTOC = false;
			createCUEFileWhenEmbedded = false;
			truncate4608ExtraSamples = true;
			lossyWAVQuality = 5;
			decodeHDCDtoLW16 = false;
			decodeHDCDto24bit = true;
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
			sw.Save("CreateTOC", createTOC);
			sw.Save("CreateCUEFileWhenEmbedded", createCUEFileWhenEmbedded);
			sw.Save("Truncate4608ExtraSamples", truncate4608ExtraSamples);
			sw.Save("LossyWAVQuality", lossyWAVQuality);
			sw.Save("DecodeHDCDToLossyWAV16", decodeHDCDtoLW16);
			sw.Save("DecodeHDCDTo24bit", decodeHDCDto24bit);
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
			createTOC = sr.LoadBoolean("CreateTOC") ?? false;
			createCUEFileWhenEmbedded = sr.LoadBoolean("CreateCUEFileWhenEmbedded") ?? false;
			truncate4608ExtraSamples = sr.LoadBoolean("Truncate4608ExtraSamples") ?? true;
			lossyWAVQuality = sr.LoadInt32("LossyWAVQuality", 0, 10) ?? 5;
			decodeHDCDtoLW16 = sr.LoadBoolean("DecodeHDCDToLossyWAV16") ?? false;
			decodeHDCDto24bit = sr.LoadBoolean("DecodeHDCDTo24bit") ?? true;
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
		public uint percentTrack = 0;
		public double percentDisk = 0.0;
		public string input = string.Empty;
		public string output = string.Empty;
	}

	public class ArchivePasswordRequiredEventArgs
	{
		public string Password = string.Empty;
		public bool ContinueOperation = true;
	}

	public delegate void CUEToolsProgressHandler(object sender, CUEToolsProgressEventArgs e);
	public delegate void ArchivePasswordRequiredHandler(object sender, ArchivePasswordRequiredEventArgs e);

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
		private string _mbDiscId;
		private string _mbReleaseId;
		private string _eacLog;
		private string _cuePath;
		private NameValueCollection _albumTags;
		private const int _arOffsetRange = 5 * 588 - 1;
		private HDCDDotNet.HDCDDotNet hdcdDecoder;
		private bool _outputLossyWAV = false;
		CUEConfig _config;
		string _cddbDiscIdTag;
		private bool _isArchive;
		private List<string> _archiveContents;
		private string _archiveCUEpath;
		private string _archivePath;
		private string _archivePassword;
		private CUEToolsProgressEventArgs _progress;
		private AccurateRipVerify _arVerify;

		public event ArchivePasswordRequiredHandler PasswordRequired;
		public event CUEToolsProgressHandler CUEToolsProgress;

		public CUESheet(CUEConfig config)
		{
			_config = config;
			_progress = new CUEToolsProgressEventArgs();
			_attributes = new List<CUELine>();
			_tracks = new List<TrackInfo>();
			_toc = new CDImageLayout();
			_sources = new List<SourceInfo>();
			_sourcePaths = new List<string>();
			_albumTags = new NameValueCollection();
			_stop = false;
			_pause = false;
			_cuePath = null;
			_paddedToFrame = false;
			_truncated4608 = false;
			_usePregapForFirstTrackInSingleFile = false;
			_accurateRip = false;
			_accurateOffset = false;
			_appliedWriteOffset = false;
			_dataTrackLength = null;
			_minDataTrackLength = null;
			hdcdDecoder = null;
			_hasEmbeddedCUESheet = false;
			_isArchive = false;
		}

		public void Open(string pathIn, bool outputLossyWAV)
		{
			_outputLossyWAV = outputLossyWAV;
			if (_config.detectHDCD)
			{
				try { hdcdDecoder = new HDCDDotNet.HDCDDotNet(2, 44100, ((_outputLossyWAV && _config.decodeHDCDtoLW16) || !_config.decodeHDCDto24bit) ? 20 : 24, _config.decodeHDCD); }
				catch { }
			}

			string cueDir, lineStr, command, pathAudio = null, fileType;
			CUELine line;
			TrackInfo trackInfo;
			int timeRelativeToFileStart, absoluteFileStartTime;
			int fileTimeLengthSamples, fileTimeLengthFrames, i;
			int trackNumber = 0;
			bool seenFirstFileIndex = false, seenDataTrack = false;
			List<IndexInfo> indexes = new List<IndexInfo>();
			IndexInfo indexInfo;
			SourceInfo sourceInfo;
			NameValueCollection _trackTags = null;

			cueDir = Path.GetDirectoryName(pathIn);
			trackInfo = null;
			absoluteFileStartTime = 0;
			fileTimeLengthSamples = 0;
			fileTimeLengthFrames = 0;
			TextReader sr;

			if (Directory.Exists(pathIn))
			{
				if (cueDir + Path.DirectorySeparatorChar != pathIn)
					throw new Exception("Input directory must end on path separator character.");
				string cueSheet = null;
				string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a" };
				for (i = 0; i < audioExts.Length && cueSheet == null; i++)
					cueSheet = CUESheet.CreateDummyCUESheet(pathIn, audioExts[i]);
				if (cueSheet == null)
					throw new Exception("Input directory doesn't contain supported audio files.");
				sr = new StringReader(cueSheet);				
			} 
#if !MONO
			else if (Path.GetExtension(pathIn).ToLower() == ".rar")
			{
				Unrar _unrar = new Unrar();
				_unrar.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
				string cueName = null, cueText = null;
				_unrar.Open(pathIn, Unrar.OpenMode.List);
				_archiveContents = new List<string>();
				while (_unrar.ReadHeader())
				{
					if (!_unrar.CurrentFile.IsDirectory)
					{
						_archiveContents.Add(_unrar.CurrentFile.FileName);
						if (Path.GetExtension(_unrar.CurrentFile.FileName).ToLower() == ".cue")
							cueName = _unrar.CurrentFile.FileName;
					}
					_unrar.Skip();
				}
				_unrar.Close();
				if (cueName != null)
				{
					RarStream rarStream = new RarStream(pathIn, cueName);
					rarStream.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
					StreamReader cueReader = new StreamReader(rarStream, CUESheet.Encoding);
					cueText = cueReader.ReadToEnd();
					cueReader.Close();
					rarStream.Close();
					if (cueText == "")
						throw new Exception("Empty cue sheet.");
				}
				if (cueText == null)
					throw new Exception("Input archive doesn't contain a cue sheet.");
				_archiveCUEpath = Path.GetDirectoryName(cueName);
				sr = new StringReader(cueText);
				_isArchive = true;
				_archivePath = pathIn;
			}
#endif
			else if (Path.GetExtension(pathIn).ToLower() == ".cue")
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
			} else
			{
				IAudioSource audioSource;
				NameValueCollection tags;
				string cuesheetTag = null;

				audioSource = AudioReadWrite.GetAudioSource(pathIn,null);
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
								trackNumber = int.Parse(line.Params[1]);
								if (trackNumber != _tracks.Count + 1) {
									throw new Exception("Invalid track number.");
								}
								trackInfo = new TrackInfo();
								_tracks.Add(trackInfo);
								_toc.AddTrack(new CDTrack((uint)trackNumber, 0, 0, true));
							}
						}
						else if (seenDataTrack) {
							// Ignore lines belonging to data tracks
						}
						else if (command == "INDEX") {
							timeRelativeToFileStart = CDImageLayout.TimeFromString(line.Params[2]);
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
			for (i = 0; i < indexes.Count - 1; i++) 
			{
				int length = indexes[i + 1].Time - indexes[i].Time;
				if (length < 0)
					throw new Exception("Indexes must be in chronological order.");
				_toc[indexes[i].Track].AddIndex(new CDTrackIndex((uint)indexes[i].Index, (uint)indexes[i].Time, (uint)length));
			}
			// Calculate the length of each track
			for (i = 1; i <= TrackCount; i++)
			{
				if (_toc[i].LastIndex < 1)
					throw new Exception("Track must have an INDEX 01.");
				_toc[i].Start = _toc[i][1].Start;
				_toc[i].Length = (i == TrackCount ? (uint)indexes[indexes.Count - 1].Time - _toc[i].Start : _toc[i + 1][1].Start - _toc[i].Start);
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

			CUELine cddbDiscIdLine = General.FindCUELine(_attributes, "REM", "DISCID");
			_cddbDiscIdTag = cddbDiscIdLine != null && cddbDiscIdLine.Params.Count == 3 ? cddbDiscIdLine.Params[2] : null;
			if (_cddbDiscIdTag == null) _cddbDiscIdTag = GetCommonTag("DISCID");

			if (_accurateRipId == null)
				_accurateRipId = GetCommonTag("ACCURATERIPID");

			if (_accurateRipId == null && _dataTrackLength == null && _eacLog != null)
			{
				sr = new StringReader(_eacLog);
				uint lastAudioSector = 0;
				bool isEACLog = false;
				CDImageLayout tocFromLog = new CDImageLayout();
				while ((lineStr = sr.ReadLine()) != null)
				{
					if (isEACLog)
					{
						string[] n = lineStr.Split('|');
						uint trNo, trStart, trEnd;
						if (n.Length == 5 && uint.TryParse(n[0], out trNo) && uint.TryParse(n[3], out trStart) && uint.TryParse(n[4], out trEnd))
							tocFromLog.AddTrack(new CDTrack(trNo, trStart, trEnd + 1 - trStart,
								tocFromLog.TrackCount < _toc.TrackCount || trStart != tocFromLog[tocFromLog.TrackCount].End + 1U + 152U * 75U));
					} else
						if (lineStr.StartsWith("TOC of the extracted CD") 
							|| lineStr.StartsWith("Exact Audio Copy")
							|| lineStr.StartsWith("CUERipper"))
							isEACLog = true;
				}
				if (tocFromLog.TrackCount == _toc.TrackCount + 1 && !tocFromLog[tocFromLog.TrackCount].IsAudio)
					_accurateRipId = AccurateRipVerify.CalculateAccurateRipId(tocFromLog);
			}

			if (_accurateRipId == null && _dataTrackLength != null)
			{
				CDImageLayout toc2 = new CDImageLayout(_toc);
				toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152U * 75U, _dataTrackLength.Value, false));
				_accurateRipId = AccurateRipVerify.CalculateAccurateRipId(toc2);
			}

			if (_dataTrackLength == null && _cddbDiscIdTag != null)
			{
				uint cddbDiscIdNum;
				if (uint.TryParse(_cddbDiscIdTag, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out cddbDiscIdNum) && (cddbDiscIdNum & 0xff) == TrackCount + 1)
				{
					uint lengthFromTag = ((cddbDiscIdNum >> 8) & 0xffff);
					_minDataTrackLength = ((lengthFromTag + _toc[1].Start / 75) - 152) * 75 - _toc.Length;
				}
			}

			_accurateRipIdActual = AccurateRipVerify.CalculateAccurateRipId(_toc);
			if (_accurateRipId == null)
				_accurateRipId = _accurateRipIdActual;

			_arVerify = new AccurateRipVerify(_toc);

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

		private void ShowProgress(string status, uint percentTrack, double percentDisk, string input, string output)
		{
			if (this.CUEToolsProgress == null)
				return;
			_progress.status = status;
			_progress.percentTrack = percentTrack;
			_progress.percentDisk = percentDisk;
			_progress.input = input;
			_progress.output = output;
			this.CUEToolsProgress(this, _progress);
		}

#if !MONO
		private void unrar_ExtractionProgress(object sender, ExtractionProgressEventArgs e)
		{
			if (this.CUEToolsProgress == null)
				return;
			_progress.percentTrack = (uint)Math.Round(e.PercentComplete);
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

		private static string LocateFile(string dir, string file, List<string> contents) {
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
					if ( (contents == null && File.Exists(path))
						|| (contents != null && contents.Contains (path)))
						return path;
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

		private int GetSampleLength(string path, out NameValueCollection tags)
		{
			IAudioSource audioSource;

			ShowProgress("Analyzing input file...", 0, 0.0, path, null);
#if !MONO
			if (_isArchive)
			{
				RarStream IO = new RarStream(_archivePath, path);
				IO.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
				IO.ExtractionProgress += new ExtractionProgressHandler(unrar_ExtractionProgress);
				audioSource = AudioReadWrite.GetAudioSource(path, IO);
			} else
#endif
				audioSource = AudioReadWrite.GetAudioSource(path, null);

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
				(_toc.Pregap != 0));

			if (htoaToFile) {
				WriteLine(sw, 0, _htoaFilename);
			}
			for (iTrack = 0; iTrack < TrackCount; iTrack++) {
				WriteLine(sw, 0, _trackFilenames[iTrack]);
			}
		}

		public void WriteTOC(string path)
		{
			StreamWriter sw = new StreamWriter(path, false, CUESheet.Encoding);
			WriteTOC(sw);
			sw.Close();
		}

		public void WriteTOC(TextWriter sw)
		{
			for (int iTrack = 0; iTrack < TrackCount; iTrack++)
				WriteLine(sw, 0, "\t" + _toc[iTrack+1].Start + 150);
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
				(_toc.Pregap != 0));

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
						((_toc[iTrack+1].Pregap == 0) || ((iTrack == 0) && !htoaToFile))))
					{
						WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
						timeRelativeToFileStart = 0;
					}

					WriteLine(sw, 1, String.Format("TRACK {0:00} AUDIO", iTrack + 1));
					for (i = 0; i < track.Attributes.Count; i++) {
						WriteLine(sw, 2, track.Attributes[i]);
					}

					for (iIndex = 0; iIndex <= _toc[iTrack+1].LastIndex; iIndex++) {
						if (_toc[iTrack+1][iIndex].Length != 0) {
							if ((iIndex == 0) &&
								((style == CUEStyle.GapsLeftOut) ||
								((style == CUEStyle.GapsAppended) && (iTrack == 0) && !htoaToFile) ||
								((style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) && (iTrack == 0) && _usePregapForFirstTrackInSingleFile)))
							{
								WriteLine(sw, 2, "PREGAP " + CDImageLayout.TimeToString(_toc[iTrack + 1][iIndex].Length));
							}
							else {
								WriteLine(sw, 2, String.Format( "INDEX {0:00} {1}", iIndex,
									CDImageLayout.TimeToString(timeRelativeToFileStart)));
								timeRelativeToFileStart += _toc[iTrack + 1][iIndex].Length;

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

		private void CalculateMusicBrainzDiscID() {
			StringBuilder mbSB = new StringBuilder();
			mbSB.AppendFormat("{0:X2}{1:X2}{2:X8}", 1, TrackCount, _toc.Length + 150);
			for (int iTrack = 1; iTrack <= _toc.TrackCount; iTrack++)
				mbSB.AppendFormat("{0:X8}", _toc[iTrack].Start + 150);
			mbSB.Append(new string('0', (99 - TrackCount) * 8));

			byte[] hashBytes = (new SHA1CryptoServiceProvider()).ComputeHash(Encoding.ASCII.GetBytes(mbSB.ToString()));
			_mbDiscId = Convert.ToBase64String(hashBytes).Replace('+', '.').Replace('/', '_').Replace('=', '-');
			System.Diagnostics.Debug.WriteLine(_mbDiscId);
		}

		private void GetMetadataFromMusicBrainz() {
			if (_mbDiscId == null) return;

			using (Stream respStream = HttpGetToStream(
				"http://musicbrainz.org/ws/1/release/?type=xml&limit=1&discid=" + _mbDiscId))
			{
				XmlDocument xd = GetXmlDocument(respStream);
				XmlNode xn;
				
				xn = xd.SelectSingleNode("/metadata/release-list/release");
				if (xn != null)
					_mbReleaseId = xn.Attributes["id"].InnerText;
			}

			if (_mbReleaseId == null) return;

			using (Stream respStream = HttpGetToStream(String.Format(
				"http://musicbrainz.org/ws/1/release/{0}?type=xml&inc=artist+tracks", _mbReleaseId)))
			{
				string discArtist = null;
				string discTitle = null;
				XmlDocument xd = GetXmlDocument(respStream);
				XmlNode xn;

				XmlNode xnRelease = xd.DocumentElement.SelectSingleNode("/metadata/release");
				if (xnRelease == null) return;

				XmlNodeList xnlTracks = xnRelease.SelectNodes("track-list/track");
				if (xnlTracks.Count != TrackCount) return;

				xn = xnRelease.SelectSingleNode("title");
				if (xn != null)
					discTitle = xn.InnerText;
				
				xn = xnRelease.SelectSingleNode("artist/name");
				if (xn != null)
					discArtist = xn.InnerText;

				Artist = discArtist;
				Title = discTitle;

				for (int iTrack = 0; iTrack < TrackCount; iTrack++) {
					string trackArtist = null;
					string trackTitle = null;
					XmlNode xnTrack = xnlTracks[iTrack];
					TrackInfo trackInfo = Tracks[iTrack];

					xn = xnTrack.SelectSingleNode("title");
					if (xn != null)
						trackTitle = xn.InnerText;

					xn = xnTrack.SelectSingleNode("artist/name");
					if (xn != null)
						trackArtist = xn.InnerText;

					trackInfo.Artist = trackArtist ?? discArtist;
					trackInfo.Title = trackTitle;
				}
			}
		}

		private XmlDocument GetXmlDocument(Stream stream) {
			XmlDocument xd = new XmlDocument();
			
			xd.Load(stream);

			if (xd.DocumentElement.NamespaceURI.Length > 0) {
				// Strip namespace to simplify xpath expressions
				XmlDocument xdNew = new XmlDocument();
				xd.DocumentElement.SetAttribute("xmlns", String.Empty);
				xdNew.LoadXml(xd.OuterXml);
				xd = xdNew;
			}

			return xd;
		}

		private Stream HttpGetToStream(string url) {
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.UserAgent = "CUE Tools";
			try {
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				return resp.GetResponseStream();
			}
			catch (WebException ex) {
				if (ex.Status == WebExceptionStatus.ProtocolError) {
					HttpStatusCode code = ((HttpWebResponse)ex.Response).StatusCode;
					if (code == HttpStatusCode.NotFound) {
						throw new HttpNotFoundException();
					}
				}
				throw;
			}
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
			_arVerify.GenerateFullLog(sw, _accurateOffset ? _writeOffset : 0);
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
				ShowProgress((string)"Contacting AccurateRip database...", 0, 0, null, null);
				if (!_dataTrackLength.HasValue && _minDataTrackLength.HasValue && _accurateRipId == _accurateRipIdActual && _config.bruteForceDTL)
				{
					uint minDTL = _minDataTrackLength.Value;
					CDImageLayout toc2 = new CDImageLayout(_toc);
					toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152 * 75, minDTL, false));
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
						lock (this) {
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
				} else
					_arVerify.ContactAccurateRip(_accurateRipId);

				if (_arVerify.AccResult != HttpStatusCode.OK)
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
						if (_config.createTOC)
						{
							if (!Directory.Exists(dir))
								Directory.CreateDirectory(dir);
							WriteTOC(Path.ChangeExtension(_cuePath, ".toc"));
						}
						return;
					}
				}
				else if (_accurateOffset)
				{
					_writeOffset = 0;
					WriteAudioFilesPass(dir, style, destPaths, destLengths, htoaToFile, true);

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
				WriteAudioFilesPass(dir, style, destPaths, destLengths, htoaToFile, verifyOnly);
			}

			if (_accurateRip)
			{
				ShowProgress((string)"Generating AccurateRip report...", 0, 0, null, null);
				if (!_accurateOffset && _config.writeArTagsOnVerify && _writeOffset == 0 && !_isArchive)
				{
					uint tracksMatch;
					int bestOffset;
					FindBestOffset(1, true, out tracksMatch, out bestOffset);

					if (_hasEmbeddedCUESheet)
					{
						IAudioSource audioSource = AudioReadWrite.GetAudioSource(_sourcePaths[0], null);
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
							IAudioSource audioSource = AudioReadWrite.GetAudioSource(src, null);
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
				if (_config.createTOC)
				{
					if (!Directory.Exists(dir))
						Directory.CreateDirectory(dir);
					WriteTOC(Path.ChangeExtension(_cuePath, ".toc"));
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
				if (_accurateOffset && _arVerify.AccResult == HttpStatusCode.OK)
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
				if (fWithCUE && _accurateOffset && _arVerify.AccResult == HttpStatusCode.OK)
					GenerateAccurateRipTags(destTags, _writeOffset, bestOffset, -1);
				else
					destTags.Add("ACCURATERIPID", _accurateRipId);
			}
			audioDest.SetTags(destTags);
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
			if (!noOutput && _accurateRipId != null && _config.writeArTagsOnConvert && _accurateOffset && _arVerify.AccResult == HttpStatusCode.OK)
				FindBestOffset(1, true, out tracksMatch, out bestOffset);

			if (hdcdDecoder != null)
				hdcdDecoder.Reset();

			if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
			{
				iDest++;
				audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput);
				if (!noOutput)
					SetAlbumTags(audioDest, bestOffset, style == CUEStyle.SingleFileWithCUE);
			}

			uint currentOffset = 0, previousOffset = 0;
			uint trackLength = _toc.Pregap * 588;
			uint diskLength = _toc.Length * 588, diskOffset = 0;

			if (_accurateRip && noOutput)
				_arVerify.Init();

			ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", 0, 0, noOutput ? "Verifying" : "Writing"), 0, 0.0, null, null);

			for (iTrack = 0; iTrack < TrackCount; iTrack++) {
				track = _tracks[iTrack];

				if ((style == CUEStyle.GapsPrepended) || (style == CUEStyle.GapsLeftOut)) {
					iDest++;
					if (hdcdDecoder != null)
						hdcdDecoder.AudioDest = null;
					if (audioDest != null)
						audioDest.Close();
					audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput);
					if (!noOutput)
						SetTrackTags(audioDest, iTrack, bestOffset);
				}		

				for (iIndex = 0; iIndex <= _toc[iTrack+1].LastIndex; iIndex++) {
					uint trackPercent= 0, lastTrackPercent= 101;
					uint samplesRemIndex = _toc[iTrack + 1][iIndex].Length * 588;

					if (iIndex == 1)
					{
						previousOffset = currentOffset;
						currentOffset = 0;
						trackLength = _toc[iTrack + 1].Length * 588;
					}

					if ((style == CUEStyle.GapsAppended) && (iIndex == 1)) 
					{
						if (hdcdDecoder != null)
							hdcdDecoder.AudioDest = null;
						if (audioDest != null)
							audioDest.Close();
						iDest++;
						audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput);
						if (!noOutput)
							SetTrackTags(audioDest, iTrack, bestOffset);
					}

					if ((style == CUEStyle.GapsAppended) && (iIndex == 0) && (iTrack == 0)) {
						discardOutput = !htoaToFile;
						if (htoaToFile) {
							iDest++;
							audioDest = GetAudioDest(destPaths[iDest], destLengths[iDest], noOutput);
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
								ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", iIndex > 0 ? iTrack + 1 : iTrack, trackPercent,
									noOutput ? "Verifying" : "Writing"), trackPercent, diskPercent, 
									audioSource.Path, discardOutput ? null : audioDest.Path);
							lastTrackPercent = trackPercent;
						}

						audioSource.Read(sampleBuffer, copyCount);
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
										audioSource.Close();
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
						if (_accurateRip && noOutput)
							_arVerify.Write(sampleBuffer, copyCount);

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
								throw new StopException();
							}
							if (_pause)
							{
								ShowProgress("Paused...", 0, 0, null, null);
								Monitor.Wait(this);
							}
						}
					}
				}
			}

			if (hdcdDecoder != null)
				hdcdDecoder.AudioDest = null;
			if (audioSource != null) 
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
			return CorrectAudioFilenames(Path.GetDirectoryName(path), cue, always);
		}

		public static string CorrectAudioFilenames(string dir, string cue, bool always) {
			string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a" };
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
							foundAll &= (LocateFile(dir, line.Params[1], null) != null);
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
						foundAll &= LocateFile(dir, newFilename, null) != null;
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

				for (iIndex = 0; iIndex <= _toc[iTrack+1].LastIndex; iIndex++) {
					if (style == CUEStyle.GapsAppended && (iIndex == 1 || (iIndex == 0 && iTrack == 0 && htoaToFile)))
						iFile++;

					if (style == CUEStyle.GapsAppended && iIndex == 0 && iTrack == 0) 
						discardOutput = !htoaToFile;
					else 
						discardOutput = (style == CUEStyle.GapsLeftOut && iIndex == 0);

					if (!discardOutput)
						fileLengths[iFile] += (int) _toc[iTrack+1][iIndex].Length * 588;
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

		private IAudioDest GetAudioDest(string path, int finalSampleCount, bool noOutput) 
		{
			if (noOutput)
				return new DummyWriter(path, (_config.detectHDCD && _config.decodeHDCD) ? 24 : 16, 2, 44100);
			return AudioReadWrite.GetAudioDest(path, finalSampleCount, _config);
		}

		private IAudioSource GetAudioSource(int sourceIndex) {
			SourceInfo sourceInfo = _sources[sourceIndex];
			IAudioSource audioSource;

			if (sourceInfo.Path == null) {
				audioSource = new SilenceGenerator(sourceInfo.Offset + sourceInfo.Length);
			}
			else {
#if !MONO
				if (_isArchive)
				{
					RarStream IO = new RarStream(_archivePath, sourceInfo.Path);
					IO.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
					audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, IO);
				}
				else
#endif
					audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, null);
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
				return (line == null) ? String.Empty : line.Params[1];
			}
			set {
				General.SetCUELine(_attributes, "PERFORMER", value, true);
			}
		}

		public string Title {
			get {
				CUELine line = General.FindCUELine(_attributes, "TITLE");
				return (line == null) ? String.Empty : line.Params[1];
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
					toc2.AddTrack(new CDTrack((uint)_toc.TrackCount, _toc.Length + 152 * 75, dtl, false));
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

		CDImageLayout _toc;
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
		private List<CUELine> _attributes;
		public NameValueCollection _trackTags;

		public TrackInfo() {
			_attributes = new List<CUELine>();
			_trackTags = new NameValueCollection();
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

	public class StopException : Exception {
		public StopException() : base() {
		}
	}

	class HttpNotFoundException : Exception {
	}
}