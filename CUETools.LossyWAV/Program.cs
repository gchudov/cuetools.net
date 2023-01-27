using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUETools;
using CUETools.Codecs;
using CUETools.Codecs.LossyWAV;

namespace CUETools.LossyWAVSharp
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.LossyWAV.exe <file.wav> <options>");
			Console.WriteLine();
			Console.WriteLine("Quality Options:");
			Console.WriteLine();
			Console.WriteLine("-I, --insane        highest quality output, suitable for transcoding;");
			Console.WriteLine("-E, --extreme       high quality output, also suitable for transcoding;");
			Console.WriteLine("-S, --standard      default quality output, considered to be transparent;");
			Console.WriteLine("-P, --portable      good quality output for DAP use. Not considered to be fully");
			Console.WriteLine("                    transparent, but considered fit for its intended purpose.");
			Console.WriteLine();
			Console.WriteLine("Standard Options:");
			Console.WriteLine();
			Console.WriteLine("-C, --correction    write correction file for processed WAV file; default=off.");
			Console.WriteLine("-h, --help          display help.");
			Console.WriteLine();
			Console.WriteLine("Advanced Options:");
			Console.WriteLine();
			Console.WriteLine("-                    if filename=\"-\" then WAV input is taken from STDIN.");
			Console.WriteLine("-q, --quality <n>    quality preset (10=highest quality, 0=lowest bitrate;");
 			Console.WriteLine("                     default = --standard = 5; --insane = 10; --extreme = 7.5;");
 			Console.WriteLine("                     --portable = 2.5)");
			Console.WriteLine("-N  --stdinname <t>  pseudo filename to use when input from STDIN.");
			Console.WriteLine("    --stdout         write processed WAV output to STDOUT.");
		}

		static void Main(string[] args)
		{
			Console.SetOut(Console.Error);
			Console.WriteLine("LossyWAV {0}, Copyright (C) 2007,2008 Nick Currie, Copyleft.", LossyWAVWriter.version_string);
			Console.WriteLine("C# port Copyright (C) 2008-2023 Grigory Chudov.");
			Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
			Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			if (args.Length < 1 || (args[0].StartsWith("-") && args[0] != "-"))
			{
				Usage();
				return;
			}
			string sourceFile = args[0];
			string stdinName = null;
			double quality = 5.0;
			bool createCorrection = false;
			bool toStdout = false;
			int outputBPS = 0;

			for (int arg = 1; arg < args.Length; arg++)
			{
				bool ok = true;
				if (args[arg] == "-I" || args[arg] == "--insane")
					quality = 10;
				else if (args[arg] == "-E" || args[arg] == "--extreme")
					quality = 7.5;
				else if (args[arg] == "-S" || args[arg] == "--standard")
					quality = 5.0;
				else if (args[arg] == "-P" || args[arg] == "--portable")
					quality = 2.5;
				else if (args[arg] == "-C" || args[arg] == "--correction")
					createCorrection = true;
				else if (args[arg] == "--16")
					outputBPS = 16;
				else if ((args[arg] == "-N" || args[arg] == "--stdinname") && ++arg < args.Length)
					stdinName = args[arg];
				else if ((args[arg] == "-q" || args[arg] == "--quality") && ++arg < args.Length)
					ok = double.TryParse(args[arg], out quality);
				else if (args[arg] == "--stdout")
					toStdout = true;
				else
					ok = false;
				if (!ok)
				{
					Usage();
					return;
				}
			}
			DateTime start = DateTime.Now;
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
#if !DEBUG
			try
#endif
			{
                Codecs.WAV.AudioDecoder audioSource = new Codecs.WAV.AudioDecoder(new Codecs.WAV.DecoderSettings() { IgnoreChunkSizes = (sourceFile == "-") }, sourceFile, (sourceFile == "-" ? Console.OpenStandardInput() : null));
				if (sourceFile == "-" && stdinName != null) sourceFile = stdinName;
				AudioPCMConfig pcm = outputBPS == 0 ? audioSource.PCM : new AudioPCMConfig(outputBPS, audioSource.PCM.ChannelCount, audioSource.PCM.SampleRate, audioSource.PCM.ChannelMask);
                Codecs.WAV.AudioEncoder audioDest = new Codecs.WAV.AudioEncoder(new Codecs.WAV.EncoderSettings(pcm), Path.ChangeExtension(sourceFile, ".lossy.wav"), toStdout ? Console.OpenStandardOutput() : null);
                Codecs.WAV.AudioEncoder lwcdfDest = createCorrection ? new Codecs.WAV.AudioEncoder(new Codecs.WAV.EncoderSettings(audioSource.PCM), Path.ChangeExtension(sourceFile, ".lwcdf.wav")) : null;
				LossyWAVWriter lossyWAV = new LossyWAVWriter(audioDest, lwcdfDest, quality, new Codecs.WAV.EncoderSettings(audioSource.PCM));
				AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);

				Console.WriteLine("Filename  : {0}", sourceFile);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.PCM.SampleRate));
				lossyWAV.FinalSampleCount = audioSource.Length;

				while (audioSource.Read(buff, -1) != 0)
				{
					lossyWAV.Write(buff);
					TimeSpan elapsed = DateTime.Now - start;
					if ((elapsed - lastPrint).TotalMilliseconds > 60)
					{
						Console.Error.Write("\rProgress  : {0:00}%; {1:0.0000} bits; {2:0.00}x; {3}/{4}",
							100.0 * audioSource.Position / audioSource.Length,
							1.0 * lossyWAV.OverallBitsRemoved / audioSource.PCM.ChannelCount / lossyWAV.BlocksProcessed,
							lossyWAV.SamplesProcessed / elapsed.TotalSeconds / audioSource.PCM.SampleRate,
							elapsed,
							TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / lossyWAV.SamplesProcessed * audioSource.Length)
							);
						lastPrint = elapsed;
					}
				}

				TimeSpan totalElapsed = DateTime.Now - start;
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.0000} bits; {1:0.00}x; {2}",
					(1.0 * lossyWAV.OverallBitsRemoved) / audioSource.PCM.ChannelCount / lossyWAV.BlocksProcessed,
					lossyWAV.SamplesProcessed / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
					totalElapsed
					);
				audioSource.Close();
				lossyWAV.Close();
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
