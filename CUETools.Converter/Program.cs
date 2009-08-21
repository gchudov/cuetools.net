using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;
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
				IAudioDest audioDest;
				FlakeWriter flake = null;
				if (destFile == "$flaketest$")
				{					
					flake = new FlakeWriter("", audioSource.BitsPerSample, audioSource.ChannelCount, audioSource.SampleRate, new NullStream());
					//((FlakeWriter)audioDest).CompressionLevel = 6;
					flake.PredictionType = Flake.LookupPredictionType(args[2]);
					flake.StereoMethod = Flake.LookupStereoMethod(args[3]);
					flake.OrderMethod = Flake.LookupOrderMethod(args[4]);
					flake.WindowFunction = Flake.LookupWindowFunction(args[5]);
					flake.MinPartitionOrder = Int32.Parse(args[6]);
					flake.MaxPartitionOrder = Int32.Parse(args[7]);
					flake.MinLPCOrder = Int32.Parse(args[8]);
					flake.MaxLPCOrder = Int32.Parse(args[9]);
					flake.MinFixedOrder = Int32.Parse(args[10]);
					flake.MaxFixedOrder = Int32.Parse(args[11]);
					flake.MaxPrecisionSearch = Int32.Parse(args[12]);
					flake.BlockSize = Int32.Parse(args[13]);
					audioDest = new BufferedWriter(flake, 512 * 1024);
				}
				else
					audioDest = AudioReadWrite.GetAudioDest(AudioEncoderType.Lossless, destFile, (long)audioSource.Length, audioSource.BitsPerSample, audioSource.SampleRate, 8192, config);
				int[,] buff = new int[0x4000, audioSource.ChannelCount];

				Console.WriteLine("Filename  : {0}", sourceFile);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.SampleRate, audioSource.ChannelCount, audioSource.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.SampleRate));

				do
				{
					uint samplesRead = audioSource.Read(buff, Math.Min((uint)buff.GetLength(0), (uint)audioSource.Remaining));
					if (samplesRead == 0) break;
					audioDest.Write(buff, 0, (int)samplesRead);
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

				if (destFile != "$flaketest$")
				{
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
				else
				{
					Console.SetOut(stdout);
					//Console.Out.WriteLine("{0}\t{6}\t{1}\t{2}\t{3}\t{4}\t{5}", 
					//    "Size    ", "MaxPart", "MaxPred", "Pred      ", "Stereo", "Order", "Time  ");
					Console.Out.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}..{7}\t{8}..{9}\t{10}..{11}\t{12}\t{13}",
						flake.TotalSize,
						flake.UserProcessorTime.TotalSeconds,
						flake.PredictionType.ToString().PadRight(15),
						flake.StereoMethod.ToString().PadRight(15),
						flake.OrderMethod.ToString().PadRight(15),
						flake.WindowFunction,
						flake.MinPartitionOrder,
						flake.MaxPartitionOrder,
						flake.MinLPCOrder,
						flake.MaxLPCOrder,
						flake.MinFixedOrder,
						flake.MaxFixedOrder,
						flake.MaxPrecisionSearch,
						flake.BlockSize
						);
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
