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
			config.writeArLogOnVerify = false;
			config.writeArTagsOnVerify = false;
			config.autoCorrectFilenames = true;
			StringWriter sw = new StringWriter();
			try
			{
				CUESheet cueSheet = new CUESheet(config);
				cueSheet.Open(pathIn);
				cueSheet.GenerateFilenames(OutputAudioFormat.NoAudio, pathIn);
				cueSheet.AccurateRip = true;
				cueSheet.WriteAudioFiles(Path.GetDirectoryName(pathIn), CUEStyle.SingleFile);
				cueSheet.GenerateAccurateRipLog(sw);
			}
			catch (Exception ex)
			{
				Console.WriteLine("Error: " + ex.Message);
			}
			sw.Close();
			Console.Write(sw.ToString());
		}
	}
}
