// ****************************************************************************
// 
// CUERipper
// Copyright (C) 2008 Gregory S. Chudov (gchudov@gmail.com)
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
using System.Text;
using System.Collections.Generic;
using CUETools.Ripper.SCSI;
using CUETools.Codecs;
using CUETools.CDImage;
using CUETools.AccurateRip;
using FLACDotNet;
using MusicBrainz;

namespace CUERipper
{
	class Program
	{
		static void Main(string[] args)
		{
			string programVersion = "CUERipper v1.9.3 Copyright (C) 2008 Gregory S. Chudov";
			Console.SetOut(Console.Error);
			Console.WriteLine("{0}", programVersion);
			Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
			Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");

			char[] drives = CDDriveReader.DrivesAvailable();
			if (drives.Length < 1)
			{
				Console.WriteLine("No CD drives found.");
				return;
			}
			char driveLetter = drives[0];
#if !DEBUG
			try
#endif
			{
				CDDriveReader audioSource = new CDDriveReader();
				audioSource.Open(driveLetter);
				int driveOffset;
				if (!AccurateRipVerify.FindDriveReadOffset(audioSource.ARName, out driveOffset))
					throw new Exception("Failed to find drive read offset for drive" + audioSource.ARName);
				audioSource.DriveOffset = driveOffset;
			
				//bool toStdout = false;
				AccurateRipVerify arVerify = new AccurateRipVerify(audioSource.TOC);
				int[,] buff = new int[audioSource.BestBlockSize, audioSource.ChannelCount];
				string CDDBId = AccurateRipVerify.CalculateCDDBId(audioSource.TOC);
				string ArId = AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC);
				Release release;
				ReleaseQueryParameters p = new ReleaseQueryParameters();
				p.DiscId = audioSource.TOC.MusicBrainzId;
				Query<Release> results = Release.Query(p);

				arVerify.ContactAccurateRip(ArId);

				try
				{
					release = results.First();
				}
				catch
				{
					release = null;
				}

				string destFile = (release == null) ? "cdimage.flac" : release.GetArtist() + " - " + release.GetTitle() + ".flac";

				Console.WriteLine("Drive       : {0}", audioSource.Path);
				Console.WriteLine("Read offset : {0}", audioSource.DriveOffset);
				Console.WriteLine("Filename    : {0}", destFile);
				Console.WriteLine("Disk length : {0}", CDImageLayout.TimeToString(audioSource.TOC.AudioLength));
				Console.WriteLine("AccurateRip : {0}", arVerify.ARStatus == null ? "ok" : arVerify.ARStatus);
				Console.WriteLine("MusicBrainz : {0}", release == null ? "not found" : release.GetArtist() + " - " + release.GetTitle());

				IAudioDest audioDest = new FLACWriter(destFile, audioSource.BitsPerSample, audioSource.ChannelCount, audioSource.SampleRate);
				audioDest.FinalSampleCount = (long)audioSource.Length;


				DateTime start = DateTime.Now;
				TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);

				do
				{
					uint samplesRead = audioSource.Read(buff, Math.Min((uint)buff.GetLength(0), (uint)audioSource.Remaining));
					if (samplesRead == 0) break;
					arVerify.Write(buff, samplesRead);
					audioDest.Write(buff, samplesRead);
					TimeSpan elapsed = DateTime.Now - start;
					if ((elapsed - lastPrint).TotalMilliseconds > 60)
					{
						Console.Write("\rProgress    : {0:00}%; {1:0.00}x; {2}/{3}",
							100.0 * audioSource.Position / audioSource.Length,
							audioSource.Position / elapsed.TotalSeconds / audioSource.SampleRate,
							elapsed,
							TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * audioSource.Length)
							);
						lastPrint = elapsed;
					}
				} while (true);

				TimeSpan totalElapsed = DateTime.Now - start;
				Console.Write("\r                                                                           \r");
				Console.WriteLine("Results     : {0:0.00}x; {1}",
					audioSource.Length / totalElapsed.TotalSeconds / audioSource.SampleRate,
					totalElapsed
					);
				audioDest.Close();

				StringWriter logWriter = new StringWriter();
				logWriter.WriteLine("{0}", programVersion);
				logWriter.WriteLine();
				logWriter.WriteLine("Extraction logfile from {0}", DateTime.Now);
				logWriter.WriteLine();
				logWriter.WriteLine("Used drive  : {0}", audioSource.Path);
				logWriter.WriteLine();
				logWriter.WriteLine("Read offset correction                      : {0}", audioSource.DriveOffset);
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
				arVerify.GenerateFullLog(logWriter, 0);
				logWriter.WriteLine();
				logWriter.WriteLine("End of status report");
				logWriter.Close();				
				StreamWriter logFile = new StreamWriter(Path.ChangeExtension(destFile, ".log"));
				logFile.Write(logWriter.ToString());
				logFile.Close();

				StringWriter cueWriter = new StringWriter();
				cueWriter.WriteLine("REM DISCID {0}", CDDBId);
				cueWriter.WriteLine("REM ACCURATERIPID {0}", ArId);
				cueWriter.WriteLine("REM COMMENT \"{0}\"", programVersion);
				if (release != null && release.GetEvents().Count > 0)
					cueWriter.WriteLine("REM DATE {0}", release.GetEvents()[0].Date.Substring(0,4));
				if (audioSource.TOC.Catalog != null)
					cueWriter.WriteLine("CATALOG {0}", audioSource.TOC.Catalog);
				if (release != null)
				{
					cueWriter.WriteLine("PERFORMER \"{0}\"", release.GetArtist());
					cueWriter.WriteLine("TITLE \"{0}\"", release.GetTitle());
				}
				cueWriter.WriteLine("FILE \"{0}\" WAVE", destFile);
				for (int track = 1; track <= audioSource.TOC.TrackCount; track++)
				if (audioSource.TOC[track].IsAudio)
				{
					cueWriter.WriteLine("  TRACK {0:00} AUDIO", audioSource.TOC[track].Number);
					if (release != null && release.GetTracks().Count >= audioSource.TOC[track].Number)
					{
						cueWriter.WriteLine("    TITLE \"{0}\"", release.GetTracks()[(int)audioSource.TOC[track].Number - 1].GetTitle());
						cueWriter.WriteLine("    PERFORMER \"{0}\"", release.GetTracks()[(int)audioSource.TOC[track].Number - 1].GetArtist());
					}
					if (audioSource.TOC[track].ISRC != null)
						cueWriter.WriteLine("    ISRC {0}", audioSource.TOC[track].ISRC);
					for (int index = audioSource.TOC[track].Pregap > 0 ? 0 : 1; index <= audioSource.TOC[track].LastIndex; index++)
						cueWriter.WriteLine("    INDEX {0:00} {1}", index, audioSource.TOC[track][index].MSF);
				}
				cueWriter.Close();
				StreamWriter cueFile = new StreamWriter(Path.ChangeExtension(destFile, ".cue"));
				cueFile.Write(cueWriter.ToString());
				cueFile.Close();

				audioSource.Close();

				FLACReader tagger = new FLACReader(destFile, null);
				tagger.Tags.Add("CUESHEET", cueWriter.ToString());
				tagger.Tags.Add("LOG", logWriter.ToString());
				tagger.UpdateTags(false);
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
	}
}
