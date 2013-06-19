using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.ALAC;

namespace CUETools.ALACEnc
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.ALACEnc.exe [options] <input.wav>");
			Console.WriteLine();
			Console.WriteLine("Options:");
			Console.WriteLine();
			Console.WriteLine(" -0 .. -10            Compression level, default 5.");
			Console.WriteLine(" -o <file>            Output filename, or \"-\" for stdout, or nul.");
			//Console.WriteLine(" -p #                 Padding bytes.");
			Console.WriteLine(" -q --quiet           Quiet mode.");
			Console.WriteLine(" --verify             Verify during encoding.");
			//Console.WriteLine(" --no-seektable       Don't generate a seektable.");
			Console.WriteLine();
			Console.WriteLine("Advanced Options:");
			Console.WriteLine();
			Console.WriteLine(" -b #                 Block size (samples).");
			Console.WriteLine(" -p #                 Padding (bytes).");
			Console.WriteLine(" -s <method>          Stereo decorrelation (independent,estimate,evaluate,search).");
			Console.WriteLine(" --history-modifier # Rice history modifier {max} or {min},{max}, default 4,4.");
			Console.WriteLine();
			Console.WriteLine("LPC options:");
			Console.WriteLine();
			Console.WriteLine(" -m <method>          Prediction order search (estimate,estsearch,logfast,search).");
			Console.WriteLine(" -e #                 Estimation depth (1..30).");
			Console.WriteLine(" -w <func>[,<func>]   One or more window functions (welch,hann,flattop,tukey).");
			Console.WriteLine("    --window-method   Window selection method (estimate,evaluate,search).");
			Console.WriteLine(" -l #[,#]             Prediction order {max} or {min},{max} (1..30).");
			Console.WriteLine();
		}

		static int Main(string[] args)
		{
			TextWriter stdout = Console.Out;
			Console.SetOut(Console.Error);

			DateTime start = DateTime.Now;
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
			bool debug = false, quiet = false;
			string stereo_method = null;
			string window_method = null;
			string order_method = null;
			string window_function = null;
			string input_file = null;
			string output_file = null;
			int min_lpc_order = -1, max_lpc_order = -1,
				estimation_depth = -1,
				min_modifier = -1, max_modifier = -1;
			int intarg = -1;
			int initial_history = -1, history_mult = -1;
			int adaptive_passes = -1;
			bool do_seektable = true;
			bool buffered = false;
            var settings = new ALACWriterSettings();

			for (int arg = 0; arg < args.Length; arg++)
			{
				bool ok = true;
				if (args[arg].Length == 0)
					ok = false;
				else if (args[arg] == "--debug")
					debug = true;
				else if ((args[arg] == "-q" || args[arg] == "--quiet"))
					quiet = true;
				else if (args[arg] == "--verify")
					settings.DoVerify = true;
				else if (args[arg] == "--no-seektable")
					do_seektable = false;
				else if (args[arg] == "--buffered")
					buffered = true;
				else if ((args[arg] == "-o" || args[arg] == "--output") && ++arg < args.Length)
					output_file = args[arg];
				else if ((args[arg] == "-s" || args[arg] == "--stereo") && ++arg < args.Length)
					stereo_method = args[arg];
				else if ((args[arg] == "-m" || args[arg] == "--order-method") && ++arg < args.Length)
					order_method = args[arg];
				else if ((args[arg] == "-w" || args[arg] == "--window") && ++arg < args.Length)
					window_function = args[arg];
				else if (args[arg] == "--window-method" && ++arg < args.Length)
					window_method = args[arg];
				else if ((args[arg] == "-l" || args[arg] == "--lpc-order") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_lpc_order) &&
						int.TryParse(args[arg].Split(',')[1], out max_lpc_order)) ||
						int.TryParse(args[arg], out max_lpc_order);
				}
				else if ((args[arg] == "--history-modifier") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_modifier) &&
						int.TryParse(args[arg].Split(',')[1], out max_modifier)) ||
						int.TryParse(args[arg], out max_modifier);
				}
				else if ((args[arg] == "--initial-history") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out initial_history);
				else if ((args[arg] == "--adaptive-passes") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out adaptive_passes);
				else if ((args[arg] == "--history-mult") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out history_mult);
				else if ((args[arg] == "-e" || args[arg] == "--estimation-depth") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out estimation_depth);
                else if ((args[arg] == "-b" || args[arg] == "--blocksize") && ++arg < args.Length)
                {
                    ok = int.TryParse(args[arg], out intarg);
                    settings.BlockSize = intarg;
                }
                else if ((args[arg] == "-p" || args[arg] == "--padding") && ++arg < args.Length)
                {
                    ok = int.TryParse(args[arg], out intarg);
                    settings.Padding = intarg;
                }
                else if (args[arg] != "-" && args[arg][0] == '-' && int.TryParse(args[arg].Substring(1), out intarg))
                {
                    ok = intarg >= 0 && intarg <= 11;
                    settings.EncoderModeIndex = intarg;
                }
                else if ((args[arg][0] != '-' || args[arg] == "-") && input_file == null)
                    input_file = args[arg];
                else
                    ok = false;
				if (!ok)
				{
					Usage();
					return 1;
				}
			}
			if (input_file == null || ((input_file == "-" || Path.GetExtension(input_file) == ".m4a") && output_file == null))
			{
				Usage();
				return 2;
			}

			if (!quiet)
			{
				Console.WriteLine("CUETools.ALACEnc, Copyright (C) 2009 Grigory Chudov.");
				Console.WriteLine("Based on ffdshow ALAC audio encoder");
				Console.WriteLine("Copyright (c) 2008  Jaikrishnan Menon, <realityman@gmx.net>");
				Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
				Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			}

			//byte [] b = new byte[0x10000];
			//int len = 0;
			//Stream si = Console.OpenStandardInput();
			//Stream so = new FileStream(output_file, FileMode.Create);
			//do
			//{
			//    len = si.Read(b, 0, 0x10000);
			//    so.Write(b, 0, len);
			//} while (len > 0);
			//return 0;
			IAudioSource audioSource;
			if (input_file == "-")
				audioSource = new WAVReader("", Console.OpenStandardInput());
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".wav")
				audioSource = new WAVReader(input_file, null);
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".m4a")
				audioSource = new ALACReader(input_file, null);
			else
			{
				Usage();
				return 2;
			}
			if (buffered)
				audioSource = new AudioPipe(audioSource, 0x10000);
			if (output_file == null)
				output_file = Path.ChangeExtension(input_file, "m4a");
            settings.PCM = audioSource.PCM;
			ALACWriter alac = new ALACWriter((output_file == "-" || output_file == "nul") ? "" : output_file,
				output_file == "-" ? Console.OpenStandardOutput() :
				output_file == "nul" ? new NullStream() : null,
				settings);
			alac.FinalSampleCount = audioSource.Length;
			IAudioDest audioDest = alac;
			AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);

			try
			{
				if (stereo_method != null)
					alac.StereoMethod = Alac.LookupStereoMethod(stereo_method);
				if (order_method != null)
					alac.OrderMethod = Alac.LookupOrderMethod(order_method);
				if (window_function != null)
					alac.WindowFunction = Alac.LookupWindowFunction(window_function);
				if (window_method != null)
					alac.WindowMethod = Alac.LookupWindowMethod(window_method);
				if (max_lpc_order >= 0)
					alac.MaxLPCOrder = max_lpc_order;
				if (min_lpc_order >= 0)
					alac.MinLPCOrder = min_lpc_order;
				if (max_modifier >= 0)
					alac.MaxHistoryModifier = max_modifier;
				if (min_modifier >= 0)
					alac.MinHistoryModifier = min_modifier;
				if (history_mult >= 0)
					alac.HistoryMult = history_mult;
				if (initial_history >= 0)
					alac.InitialHistory = initial_history;
				if (estimation_depth >= 0)
					alac.EstimationDepth = estimation_depth;
				if (adaptive_passes >= 0)
					alac.AdaptivePasses = adaptive_passes;
				alac.DoSeekTable = do_seektable;
			}
			catch (Exception ex)
			{
				Usage();
				Console.WriteLine("");
				Console.WriteLine("Error: {0}.", ex.Message);
				return 3;
			}

			if (!quiet)
			{
				Console.WriteLine("Filename  : {0}", input_file);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.PCM.SampleRate));
			}

#if !DEBUG
			try
#endif
			{
				while (audioSource.Read(buff, -1) != 0)
				{
					audioDest.Write(buff);
					TimeSpan elapsed = DateTime.Now - start;
					if (!quiet)
					{
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
				}
				audioDest.Close();
			}
#if !DEBUG
			catch (Exception ex)
			{
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Error     : {0}", ex.Message);
				audioDest.Delete();
				audioSource.Close();
				return 4;
			}
#endif

			if (!quiet)
			{
				TimeSpan totalElapsed = DateTime.Now - start;
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.00}x; {1}",
					audioSource.Position / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
					totalElapsed
					);
			}
			audioSource.Close();

			if (debug)
			{
				Console.SetOut(stdout);
				Console.Out.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}..{6}\t{7}..{8}\t{9}",
					alac.TotalSize,
					alac.UserProcessorTime.TotalSeconds,
					alac.StereoMethod.ToString().PadRight(15),
					(alac.OrderMethod.ToString() + (alac.OrderMethod == OrderMethod.Estimate ? "(" + alac.EstimationDepth.ToString() + ")" : "")).PadRight(15),
					alac.WindowFunction.ToString() + "(" + alac.AdaptivePasses.ToString() + ")",
					alac.MinLPCOrder,
					alac.MaxLPCOrder,
					alac.MinHistoryModifier,
					alac.MaxHistoryModifier,
					alac.Settings.BlockSize
					);
			}
			//File.SetAttributes(output_file, FileAttributes.ReadOnly);
			return 0;
		}
	}
}
