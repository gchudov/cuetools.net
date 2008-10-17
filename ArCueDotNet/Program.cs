using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUEToolsLib;

namespace ArCueDotNet
{
	class Program
	{
		static void Main(string[] args)
		{
			if (args.Length != 1)
			{
				Console.WriteLine("Usage: ArCueDotNet <filename>");
				return;
			}
			string pathIn = args[0];
			if (!File.Exists(pathIn))
			{
				Console.WriteLine("Input CUE Sheet not found.");
				return;
			}
			CUEConfig config = new CUEConfig();
			config.writeArLog = false;
			config.writeArTags = false;
			config.autoCorrectFilenames = true;
			StringWriter sw = new StringWriter();
			try
			{
				CUESheet cueSheet = new CUESheet(pathIn, config);
				cueSheet.GenerateFilenames(OutputAudioFormat.NoAudio, pathIn);
				cueSheet.AccurateRip = true;
				cueSheet.WriteAudioFiles(Path.GetDirectoryName(pathIn), CUEStyle.SingleFile, new SetStatus(ArCueSetStatus));
				cueSheet.GenerateAccurateRipLog(sw);
			}
			catch (Exception ex)
			{
				Console.WriteLine("Error: " + ex.Message);
			}
			sw.Close();
			Console.Write(sw.ToString());
		}
		public static void ArCueSetStatus(string status, uint percentTrack, double percentDisk)
		{
		}
	}
}
