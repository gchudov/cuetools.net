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
using CUETools.AccurateRip;

namespace CUERipper
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUERipper.exe <file.wav>");
			Console.WriteLine();
		}

		static void Main(string[] args)
		{
			string programVersion = "CUERipper v1.9.3";
			Console.SetOut(Console.Error);
			Console.WriteLine("{0}", programVersion);
			Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
			Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			if (args.Length < 1)
			{
				Usage();
				return;
			}
			string destFile = args[0];
#if !DEBUG
			try
#endif
			{
				CDDriveReader audioSource = new CDDriveReader();
				audioSource.Open('D');
				audioSource.DriveOffset = 48;
			
				bool toStdout = false;
				AccurateRipVerify arVerify = new AccurateRipVerify(audioSource.TOC);
				WAVWriter audioDest = new WAVWriter(destFile, audioSource.BitsPerSample, audioSource.ChannelCount, audioSource.SampleRate, toStdout ? Console.OpenStandardOutput() : null);
				int[,] buff = new int[audioSource.BestBlockSize, audioSource.ChannelCount];
				string CDDBId = AccurateRipVerify.CalculateCDDBId(audioSource.TOC);
				string ArId = AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC);

				arVerify.ContactAccurateRip(ArId);

				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.SampleRate, audioSource.ChannelCount, audioSource.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.SampleRate));
				Console.WriteLine("Filename  : {0}", destFile);
				Console.WriteLine("AR status : {0}", arVerify.ARStatus == null ? "ok" : arVerify.ARStatus);

				audioDest.FinalSampleCount = (long) audioSource.Length;

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
						Console.Error.Write("\rProgress  : {0:00}%; {1:0.00}x; {2}/{3}",
							100.0 * audioSource.Position / audioSource.Length,
							audioSource.Position / elapsed.TotalSeconds / audioSource.SampleRate,
							elapsed,
							TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * audioSource.Length)
							);
						lastPrint = elapsed;
					}
				} while (true);

				TimeSpan totalElapsed = DateTime.Now - start;
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.00}x; {1}",
					audioSource.Length / totalElapsed.TotalSeconds / audioSource.SampleRate,
					totalElapsed
					);
				audioDest.Close();

				StreamWriter logWriter = new StreamWriter(Path.ChangeExtension(destFile, ".log"));
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

				StreamWriter cueWriter = new StreamWriter(Path.ChangeExtension(destFile, ".cue"));
				cueWriter.WriteLine("REM DISCID {0}", CDDBId);
				cueWriter.WriteLine("REM ACCURATERIPID {0}", ArId);
				cueWriter.WriteLine("REM COMMENT \"{0}\"", programVersion);
				if (audioSource.TOC.Catalog != null)
					cueWriter.WriteLine("CATALOG {0}", audioSource.TOC.Catalog);
				cueWriter.WriteLine("FILE \"{0}\" WAVE", destFile);
				for (int track = 1; track <= audioSource.TOC.TrackCount; track++)
				if (audioSource.TOC[track].IsAudio)
				{
					cueWriter.WriteLine("  TRACK {0:00} AUDIO", audioSource.TOC[track].Number);
					if (audioSource.TOC[track].ISRC != null)
						cueWriter.WriteLine("    ISRC {0}", audioSource.TOC[track].ISRC);
					for (int index = audioSource.TOC[track].Pregap > 0 ? 0 : 1; index <= audioSource.TOC[track].LastIndex; index++)
						cueWriter.WriteLine("    INDEX {0:00} {1}", index, audioSource.TOC[track][index].MSF);
				}
				cueWriter.Close();

				audioSource.Close();
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
