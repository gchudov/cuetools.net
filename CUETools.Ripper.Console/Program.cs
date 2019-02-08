// ****************************************************************************
// 
// CUERipper
// Copyright (C) 2008-13 Grigory Chudov (gchudov@gmail.com)
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
using System.IO;
using System.Net;
using CUETools.Ripper;
using CUETools.Ripper.SCSI;
using CUETools.Codecs;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.CTDB;

namespace CUETools.ConsoleRipper
{
    using System.Globalization;
    using System.Text.RegularExpressions;

    using TagLib;
    using TagLib.Ogg;

    class ProgressMeter
	{
		public DateTime realStart;

		public ProgressMeter()
		{
			realStart = DateTime.Now;
			//TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
		}

		public void ReadProgress(object sender, ReadProgressArgs e)
		{
			CDDriveReader audioSource = (CDDriveReader)sender;
			int processed = e.Position - e.PassStart;
			TimeSpan elapsed = DateTime.Now - e.PassTime;
			double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;
			TimeSpan totalElapsed = DateTime.Now - realStart;
			TimeSpan totalEstimated = TimeSpan.FromMilliseconds(totalElapsed.TotalMilliseconds / Math.Max(1, (e.PassStart + (processed + e.Pass * (e.PassEnd - e.PassStart)) / (audioSource.CorrectionQuality + 1))) * audioSource.TOC.AudioLength);

			// if ((elapsed - lastPrint).TotalMilliseconds > 60) ;
			Console.Write("\r{9} : {0:00}%; {1:00.00}x; {2} ({10:0.00}%) errors; {3:d2}:{4:d2}:{5:d2}/{6:d2}:{7:d2}:{8:d2}        ",
				100.0 * e.Position / audioSource.TOC.AudioLength,
				speed,
				e.ErrorsCount,
				totalElapsed.Hours, totalElapsed.Minutes, totalElapsed.Seconds,
				totalEstimated.Hours, totalEstimated.Minutes, totalEstimated.Seconds,
				e.Pass < 1 ? "Progress   " : string.Format("Retry {0:00}   ", e.Pass),
				processed > 0 ? 100.0 * e.ErrorsCount / processed / (4 * 588) : 0
			);
			//lastPrint = elapsed;
		}
	}

	class Program
	{
		static void Usage()
		{
			string drives = "";
			char[] drivesAvailable = CDDrivesList.DrivesAvailable();
			for (int i = 0; i < drivesAvailable.Length; i++)
				drives += string.Format("{0}: ", drivesAvailable[i]);
			Console.WriteLine("Usage    : CUERipper.exe <options>");
			Console.WriteLine();
			Console.WriteLine("-S, --secure             secure mode, read each block twice (default);");
			Console.WriteLine("-B, --burst              burst (1 pass) mode;");
			Console.WriteLine("-P, --paranoid           maximum level of error correction;");
			Console.WriteLine("-D, --drive <letter>     use a specific CD drive, e.g. {0};", drives);
			Console.WriteLine("-O, --offset <samples>   use specific drive read offset;");
			Console.WriteLine("-T, --test               detect read command;");
			Console.WriteLine("--d8                     force D8h read command;");
			Console.WriteLine("--be                     force BEh read command;");
		    Console.WriteLine("-F, --flac               rip to FLAC instead of WAV (uses Flake with compression level 8)");
		    Console.WriteLine("-I, --info               only output information on the CD; do not rip it");
		    Console.WriteLine("-N, --filename           specify the output file name (do not specify file extension)");
		}

		static void Main(string[] args)
		{
			Console.SetOut(Console.Error);
			Console.WriteLine("CUERipper v2.1.7 Copyright (C) 2008-10 Grigory Chudov");
			Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
			Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");

			int correctionQuality = 1;
			string driveLetter = null;
			int driveOffset = 0;
			bool test = false;
			bool forceD8 = false, forceBE = false, quiet = false;
            bool flac = false;
            bool infoOnly = false;
            string destFileNameWithoutExtension = null;

			for (int arg = 0; arg < args.Length; arg++)
			{
				bool ok = true;
                if (args[arg] == "-P" || args[arg] == "--paranoid")
                    correctionQuality = 2;
                else if (args[arg] == "-S" || args[arg] == "--secure")
                    correctionQuality = 1;
                else if (args[arg] == "-B" || args[arg] == "--burst")
                    correctionQuality = 0;
                else if (args[arg] == "-T" || args[arg] == "--test")
                    test = true;
                else if (args[arg] == "--d8")
                    forceD8 = true;
                else if (args[arg] == "--be")
                    forceBE = true;
                else if (args[arg] == "-Q" || args[arg] == "--quiet")
                    quiet = true;
                else if ((args[arg] == "-D" || args[arg] == "--drive") && ++arg < args.Length)
                    driveLetter = args[arg];
                else if ((args[arg] == "-O" || args[arg] == "--offset") && ++arg < args.Length)
                    ok = int.TryParse(args[arg], out driveOffset);
                else if (args[arg] == "--flac" || args[arg] == "-F")
                    flac = true;
                else if (args[arg] == "-I" || args[arg] == "--info")
                    infoOnly = true;
                else if ((args[arg] == "-N" || args[arg] == "--filename") && ++arg < args.Length)
                    destFileNameWithoutExtension = args[arg];
                else
                    ok = false;
				if (!ok)
				{
					Usage();
					return;
				}
			}
			
			char[] drives;
			if (driveLetter == null || driveLetter.Length < 1)
			{
				drives = CDDrivesList.DrivesAvailable();
				if (drives.Length < 1)
				{
					Console.WriteLine("No CD drives found.");
					return;
				}
			}
			else
			{
				drives = new char[1];
				drives[0] = driveLetter[0];
			}

#if !DEBUG
			try
#endif
			{
				CDDriveReader audioSource = new CDDriveReader();
				audioSource.Open(drives[0]);
				
				if (audioSource.TOC.AudioTracks < 1)
				{
					Console.WriteLine("{0}: CD does not contain any audio tracks.", audioSource.Path);
					audioSource.Close();
					return;
				}
				if (driveOffset == 0)
					if (!AccurateRipVerify.FindDriveReadOffset(audioSource.ARName, out driveOffset))
						Console.WriteLine("Unknown read offset for drive {0}!!!", audioSource.Path);
						//throw new Exception("Failed to find drive read offset for drive" + audioSource.ARName);

				audioSource.DriveOffset = driveOffset;
				audioSource.CorrectionQuality = correctionQuality;
				audioSource.DebugMessages = !quiet;
				if (forceD8) audioSource.ForceD8 = true;
				if (forceBE) audioSource.ForceBE = true;
				string readCmd = audioSource.AutoDetectReadCommand;
				if (test)
				{
					Console.Write(readCmd);
					return;
				}

				AccurateRipVerify arVerify = new AccurateRipVerify(audioSource.TOC, WebRequest.GetSystemWebProxy());
				AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);
				string CDDBId = AccurateRipVerify.CalculateCDDBId(audioSource.TOC);
				string ArId = AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC);
				var ctdb = new CUEToolsDB(audioSource.TOC, null);
				ctdb.Init(arVerify);
				ctdb.ContactDB(null, "CUETools.ConsoleRipper 2.1.7", audioSource.ARName, true, false, CTDBMetadataSearch.Fast);
				arVerify.ContactAccurateRip(ArId);
				CTDBResponseMeta meta = null;
				foreach (var imeta in ctdb.Metadata)
				{
					meta = imeta;
					break;
				}

			    string destFile;

			    if (flac)
			    {
			        if (!string.IsNullOrEmpty(destFileNameWithoutExtension))
			        {
			            destFile = destFileNameWithoutExtension + ".flac";
			        }
                    else if (meta != null)
			        {
			            destFile = Program.CoerceValidFileName(meta.artist + " - " + meta.album + ".flac");
			        }
			        else
			        {
			            destFile = "cdimage.flac";
			        }
			    }
			    else
			    {
			        if (!string.IsNullOrEmpty(destFileNameWithoutExtension))
			        {
			            destFile = destFileNameWithoutExtension + ".wav";
			        }
			        else if (meta != null)
			        {
			            destFile = Program.CoerceValidFileName(meta.artist + " - " + meta.album + ".wav");
			        }
			        else
			        {
			            destFile = "cdimage.wav";
			        }
                }

				Console.WriteLine("Drive       : {0}", audioSource.Path);
				Console.WriteLine("Read offset : {0}", audioSource.DriveOffset);
				Console.WriteLine("Read cmd    : {0}", audioSource.CurrentReadCommand);
				Console.WriteLine("Secure mode : {0}", audioSource.CorrectionQuality);
				Console.WriteLine("Filename    : {0}", destFile);
				Console.WriteLine("Disk length : {0}", CDImageLayout.TimeToString(audioSource.TOC.AudioLength));
				Console.WriteLine("AccurateRip : {0}", arVerify.ARStatus == null ? "ok" : arVerify.ARStatus);
				Console.WriteLine("MusicBrainz : {0}", meta == null ? "not found" : meta.artist + " - " + meta.album);
			    Console.WriteLine("Artist      : {0}", meta == null ? string.Empty : meta.artist);
			    Console.WriteLine("Album       : {0}", meta == null ? string.Empty : meta.album);
			    Console.WriteLine("Year        : {0}", meta == null ? string.Empty : meta.year);
			    Console.WriteLine("Genre       : {0}", meta == null ? string.Empty : meta.genre);
			    Console.WriteLine("Disc Number : {0}", meta == null ? string.Empty : meta.discnumber);
			    Console.WriteLine("Disc Name   : {0}", meta == null ? string.Empty : meta.discname);
			    Console.WriteLine("Disc Count  : {0}", meta == null ? string.Empty : meta.disccount);

			    if (infoOnly)
			    {
                    return;
			    }

				ProgressMeter meter = new ProgressMeter();
				audioSource.ReadProgress += new EventHandler<ReadProgressArgs>(meter.ReadProgress);

				audioSource.DetectGaps();

				StringWriter cueWriter = new StringWriter();
				cueWriter.WriteLine("REM DISCID {0}", CDDBId);
				cueWriter.WriteLine("REM ACCURATERIPID {0}", ArId);
				cueWriter.WriteLine("REM COMMENT \"{0}\"", audioSource.RipperVersion);
				if (meta != null && meta.year != "")
					cueWriter.WriteLine("REM DATE {0}", meta.year);
				if (audioSource.TOC.Barcode != null)
					cueWriter.WriteLine("CATALOG {0}", audioSource.TOC.Barcode);
				if (meta != null)
				{
					cueWriter.WriteLine("PERFORMER \"{0}\"", meta.artist);
					cueWriter.WriteLine("TITLE \"{0}\"", meta.album);
				}
				cueWriter.WriteLine("FILE \"{0}\" WAVE", destFile);
				for (int track = 1; track <= audioSource.TOC.TrackCount; track++)
					if (audioSource.TOC[track].IsAudio)
					{
						cueWriter.WriteLine("  TRACK {0:00} AUDIO", audioSource.TOC[track].Number);
						if (meta != null && meta.track.Length >= audioSource.TOC[track].Number)
						{
							cueWriter.WriteLine("    TITLE \"{0}\"", meta.track[(int)audioSource.TOC[track].Number - 1].name);
							cueWriter.WriteLine("    PERFORMER \"{0}\"", meta.track[(int)audioSource.TOC[track].Number - 1].artist);
						}
						if (audioSource.TOC[track].ISRC != null)
							cueWriter.WriteLine("    ISRC {0}", audioSource.TOC[track].ISRC);
						if (audioSource.TOC[track].DCP || audioSource.TOC[track].PreEmphasis)
							cueWriter.WriteLine("    FLAGS{0}{1}", audioSource.TOC[track].PreEmphasis ? " PRE" : "", audioSource.TOC[track].DCP ? " DCP" : "");
						for (int index = audioSource.TOC[track].Pregap > 0 ? 0 : 1; index <= audioSource.TOC[track].LastIndex; index++)
							cueWriter.WriteLine("    INDEX {0:00} {1}", index, audioSource.TOC[track][index].MSF);
					}
				cueWriter.Close();
				StreamWriter cueFile = new StreamWriter(Path.ChangeExtension(destFile, ".cue"));
				cueFile.Write(cueWriter.ToString());
				cueFile.Close();

                IAudioDest audioDest;

			    if (flac)
			    {
			        var settings = new Codecs.Flake.EncoderSettings();
                    settings.PCM = audioSource.PCM;
			        settings.SetEncoderModeIndex(8);
                    audioDest = new Codecs.Flake.AudioEncoder(settings, destFile);
			    }
			    else
			    {
                    audioDest = new Codecs.WAV.AudioEncoder(new Codecs.WAV.EncoderSettings(audioSource.PCM), destFile);
                }

				audioDest.FinalSampleCount = audioSource.Length;
				while (audioSource.Read(buff, -1) != 0)
				{
					arVerify.Write(buff);
					audioDest.Write(buff);
				}

				TimeSpan totalElapsed = DateTime.Now - meter.realStart;
				Console.Write("\r                                                                             \r");
				Console.WriteLine("Results     : {0:0.00}x; {1:d5} errors; {2:d2}:{3:d2}:{4:d2}",
					audioSource.Length / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
                    audioSource.FailedSectors.PopulationCount(),
					totalElapsed.Hours, totalElapsed.Minutes, totalElapsed.Seconds
					);
				audioDest.Close();

				StringWriter logWriter = new StringWriter();
				logWriter.WriteLine("{0}", audioSource.RipperVersion);
				logWriter.WriteLine("Extraction logfile from {0}", DateTime.Now);
				logWriter.WriteLine("Used drive  : {0}", audioSource.Path);
				logWriter.WriteLine("Read offset correction                      : {0}", audioSource.DriveOffset);
				bool wereErrors = false;
				for (int iTrack = 1; iTrack <= audioSource.TOC.AudioTracks; iTrack++)
					for (uint iSector = audioSource.TOC[iTrack].Start; iSector <= audioSource.TOC[iTrack].End; iSector ++)
						if (audioSource.FailedSectors[(int)iSector])
						{
							if (!wereErrors)
							{
								logWriter.WriteLine();
								logWriter.WriteLine("Errors detected");
								logWriter.WriteLine();
							}
							wereErrors = true;
							logWriter.WriteLine("Track {0} contains errors", iTrack);
							break;
						}
				logWriter.WriteLine();
				logWriter.WriteLine("TOC of the extracted CD");
				logWriter.WriteLine();
				logWriter.WriteLine("     Track |   Start  |  Length  | Start sector | End sector");
				logWriter.WriteLine("    ---------------------------------------------------------");
				for (int track = 1; track <= audioSource.TOC.TrackCount; track++)
					logWriter.WriteLine("{0,9}  | {1,8} | {2,8} | {3,9}    | {4,9}",
						audioSource.TOC[track].Number,
						audioSource.TOC[track].StartMSF,
						audioSource.TOC[track].LengthMSF,
						audioSource.TOC[track].Start,
						audioSource.TOC[track].End);
				logWriter.WriteLine();
				logWriter.WriteLine("AccurateRip summary");
				logWriter.WriteLine();
				arVerify.GenerateFullLog(logWriter, true, ArId);
				logWriter.WriteLine();
				logWriter.WriteLine("End of status report");
				logWriter.Close();				
				StreamWriter logFile = new StreamWriter(Path.ChangeExtension(destFile, ".log"));
				logFile.Write(logWriter.ToString());
				logFile.Close();

				audioSource.Close();

			    if (flac)
			    {
			        var fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(destFile));

			        var xiph = (XiphComment)fileInfo.GetTag(TagTypes.Xiph);
			        xiph.SetField("CUESHEET", cueWriter.ToString());
			        xiph.SetField("LOG", logWriter.ToString());

			        if (meta != null)
			        {
			            fileInfo.Tag.Album = meta.album;
			            fileInfo.Tag.AlbumArtists = new[] { meta.artist };
			            fileInfo.Tag.Performers = new[] { meta.artist };
			            fileInfo.Tag.Genres = new[] { meta.genre };
			            fileInfo.Tag.DiscSubtitle = meta.discname;

			            if (uint.TryParse(meta.disccount, out uint discCount))
			            {
			                fileInfo.Tag.DiscCount = discCount;
			            }

			            if (uint.TryParse(meta.discnumber, out uint discNumber))
			            {
			                fileInfo.Tag.Disc = discNumber;
			            }

			            if (uint.TryParse(meta.year, out uint year))
			            {
			                fileInfo.Tag.Year = year;
			            }
			        }

			        fileInfo.Save();
			    }
			}
#if !DEBUG
			catch (Exception ex)
			{
				Console.WriteLine();
				Console.WriteLine("Error: {0}", ex.Message);
				Console.WriteLine("{0}", ex.StackTrace);
			}
#endif
		}

	    private static readonly string[] reservedWords = new[] { "CON", "PRN", "AUX", "CLOCK$", "NUL", "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT0", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9" };
	    private static readonly string invalidChars = Regex.Escape(new string(Path.GetInvalidFileNameChars()) + "'");

        private static string CoerceValidFileName(string filename)
	    {
	        string invalidReStr = string.Format(CultureInfo.InvariantCulture, @"[{0}]+", Program.invalidChars);
	        string sanitisedNamePart = Regex.Replace(filename, invalidReStr, "_");

	        var pieces = sanitisedNamePart.Split('.');
	        bool needsRegEx = false;

	        foreach (string piece in pieces)
	        {
	            foreach (string word in Program.reservedWords)
	            {
	                if (word.Equals(piece, StringComparison.OrdinalIgnoreCase))
	                {
	                    needsRegEx = true;
	                    break;
	                }
	            }

	            if (needsRegEx)
	            {
	                break;
	            }
	        }

	        if (needsRegEx)
	        {
	            foreach (string reservedWord in Program.reservedWords)
	            {
	                string reservedWordPattern = string.Format(CultureInfo.InvariantCulture, "^{0}\\.", reservedWord);
	                sanitisedNamePart = Regex.Replace(sanitisedNamePart, reservedWordPattern, "_reservedWord_.", RegexOptions.IgnoreCase);
	            }
	        }

	        return sanitisedNamePart;
	    }

        //private void MusicBrainz_LookupProgress(object sender, XmlRequestEventArgs e)
        //{
        //    if (this.CUEToolsProgress == null)
        //        return;
        //    _progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
        //    _progress.percentTrack = 0;
        //    _progress.input = e.Uri.ToString();
        //    _progress.output = null;
        //    _progress.status = "Looking up album via MusicBrainz";
        //    this.CUEToolsProgress(this, _progress);
        //}
    }
}
