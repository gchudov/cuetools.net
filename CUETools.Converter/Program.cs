using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using CUETools.Codecs;
using CUETools.Processor;

namespace CUETools.Converter
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.Converter.exe <infile> <outfile>");
			Console.WriteLine();
			//Console.WriteLine("-N  --stdinname <t>  pseudo filename to use when input from STDIN.");
			//Console.WriteLine("    --stdout         write processed WAV output to STDOUT.");
		}

		static void Main(string[] args)
		{
			Console.SetOut(Console.Error);
			Console.WriteLine("CUETools.Converter, Copyright (C) 2009 Gregory S. Chudov.");
			Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
			Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			if (args.Length < 2)
			{
				Usage();
				return;
			}
			string sourceFile = args[0];
			string destFile = args[1];

			DateTime start = DateTime.Now;
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
			CUEConfig config = new CUEConfig();

			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt");
			config.Load(sr);
			config.lossyWAVHybrid = false;
#if !DEBUG
			try
#endif
			{
				IAudioSource audioSource = AudioReadWrite.GetAudioSource(sourceFile, null, config);
				IAudioDest audioDest = AudioReadWrite.GetAudioDest(destFile, (long)audioSource.Length, audioSource.BitsPerSample, audioSource.SampleRate, config);
				int[,] buff = new int[0x4000, audioSource.ChannelCount];

				Console.WriteLine("Filename  : {0}", sourceFile);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.SampleRate, audioSource.ChannelCount, audioSource.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.SampleRate));

				do
				{
					uint samplesRead = audioSource.Read(buff, Math.Min((uint)buff.GetLength(0), (uint)audioSource.Remaining));
					if (samplesRead == 0) break;
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
					audioSource.Position / totalElapsed.TotalSeconds / audioSource.SampleRate,
					totalElapsed
					);
				audioSource.Close();
				audioDest.Close();

				TagLib.UserDefined.AdditionalFileTypes.Config = config;
				TagLib.File sourceInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(sourceFile));
				TagLib.File destInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(destFile));
				if (Tagging.UpdateTags(destInfo, Tagging.Analyze(sourceInfo), config))
				{
					sourceInfo.Tag.CopyTo(destInfo.Tag, true);
					destInfo.Tag.Pictures = sourceInfo.Tag.Pictures;
					destInfo.Save();
				}
			}
#if !DEBUG
			catch (Exception ex)
			{
				Console.WriteLine();
				Console.WriteLine("Error: {0}", ex.Message);
				//Console.WriteLine("{0}", ex.StackTrace);
			}
#endif
		}
	}
}
