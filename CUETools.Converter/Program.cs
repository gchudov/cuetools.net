using System;
using System.IO;
using CUETools.Codecs;
using CUETools.Processor;
using CUETools.Processor.Settings;

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
			TextWriter stdout = Console.Out;
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

			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt", null);
			config.Load(sr);
#if !DEBUG
			try
#endif
			{
				IAudioSource audioSource = AudioReadWrite.GetAudioSource(sourceFile, null, config);
				IAudioDest audioDest = AudioReadWrite.GetAudioDest(AudioEncoderType.Lossless, destFile, (long)audioSource.Length, audioSource.PCM.BitsPerSample, audioSource.PCM.SampleRate, 8192, config);
				AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);

				Console.WriteLine("Filename  : {0}", sourceFile);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.PCM.SampleRate));

				while (audioSource.Read(buff, -1) != 0)
				{
					audioDest.Write(buff);
					TimeSpan elapsed = DateTime.Now - start;
					if ((elapsed - lastPrint).TotalMilliseconds > 60)
					{
						Console.Error.Write("\rProgress  : {0:00}%; {1:0.00}x; {2}/{3}",
							100.0 * audioSource.Position / audioSource.Length,
							audioSource.Position / elapsed.TotalSeconds / audioSource.PCM.SampleRate,
							elapsed,
							TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * audioSource.Length)
							);
						lastPrint = elapsed;
					}
				}

				TimeSpan totalElapsed = DateTime.Now - start;
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.00}x; {1}",
					audioSource.Position / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
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
