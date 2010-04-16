using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;

namespace CUETools.FlakeExe
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.Flake.exe [options] <input.wav>");
			Console.WriteLine();
			Console.WriteLine("Options:");
			Console.WriteLine();
			Console.WriteLine(" -0 .. -11            Compression level, default 7.");
			Console.WriteLine(" -o <file>            Output filename, or \"-\" for stdout, or nul.");
			Console.WriteLine(" -p #                 Padding bytes.");
			Console.WriteLine(" -q --quiet           Quiet mode.");
			Console.WriteLine(" --verify             Verify during encoding.");
			Console.WriteLine(" --no-md5             Don't compute MD5 hash.");
			Console.WriteLine(" --no-seektable       Don't generate a seektable.");
			Console.WriteLine();
			Console.WriteLine("Advanced Options:");
			Console.WriteLine();
			Console.WriteLine(" -b #                 Block size.");
			Console.WriteLine(" -v #                 Variable block size mode (0,4).");
			Console.WriteLine(" -t <type>            Prediction type (fixed,levinson,search).");
			Console.WriteLine(" -s <method>          Stereo decorrelation (independent,estimate,evaluate,search).");
			Console.WriteLine(" -r #[,#]             Rice partition order {max} or {min},{max} (0..8).");
			Console.WriteLine();
			Console.WriteLine("LPC options:");
			Console.WriteLine();
			Console.WriteLine(" -m <method>          Prediction order search (akaike).");
			Console.WriteLine(" -e #                 Prediction order search depth (1..32).");
			Console.WriteLine(" -w <func>[,<func>]   One or more window functions (bartlett,welch,hann,flattop,tukey).");
			Console.WriteLine(" -l #[,#]             Prediction order {max} or {min},{max} (1..32).");
			Console.WriteLine("    --window-method   Window selection method (estimate,evaluate,search).");
			Console.WriteLine("    --max-precision   Coefficients precision search (0..1).");
			Console.WriteLine();
			Console.WriteLine("Fixed prediction options:");
			Console.WriteLine();
			Console.WriteLine(" -f #[,#]             Prediction order {max} or {min},{max} (0..4).");
		}

		static int Main(string[] args)
		{
			TextWriter stdout = Console.Out;
			Console.SetOut(Console.Error);

			DateTime start = DateTime.Now;
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
			bool debug = false, quiet = false;
			string prediction_type = null;
			string stereo_method = null;
			string window_method = null;
			string order_method = null;
			string window_function = null;
			string input_file = null;
			string output_file = null;
			int min_partition_order = -1, max_partition_order = -1,
				min_lpc_order = -1, max_lpc_order = -1,
				min_fixed_order = -1, max_fixed_order = -1,
				min_precision = -1, max_precision = -1,
				blocksize = -1, estimation_depth = -1;
			int level = -1, padding = -1, vbr_mode = -1;
			bool do_md5 = true, do_seektable = true, do_verify = false;
			bool buffered = false;

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
					do_verify = true;
				else if (args[arg] == "--no-seektable")
					do_seektable = false;
				else if (args[arg] == "--no-md5")
					do_md5 = false;
				else if (args[arg] == "--buffered")
					buffered = true;
				else if ((args[arg] == "-o" || args[arg] == "--output") && ++arg < args.Length)
					output_file = args[arg];
				else if ((args[arg] == "-t" || args[arg] == "--prediction-type") && ++arg < args.Length)
					prediction_type = args[arg];
				else if ((args[arg] == "-s" || args[arg] == "--stereo") && ++arg < args.Length)
					stereo_method = args[arg];
				else if ((args[arg] == "-m" || args[arg] == "--order-method") && ++arg < args.Length)
					order_method = args[arg];
				else if ((args[arg] == "-w" || args[arg] == "--window") && ++arg < args.Length)
					window_function = args[arg];
				else if (args[arg] == "--window-method" && ++arg < args.Length)
					window_method = args[arg];
				else if ((args[arg] == "-r" || args[arg] == "--partition-order") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_partition_order) &&
						int.TryParse(args[arg].Split(',')[1], out max_partition_order)) ||
						int.TryParse(args[arg], out max_partition_order);
				}
				else if ((args[arg] == "-l" || args[arg] == "--lpc-order") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_lpc_order) &&
						int.TryParse(args[arg].Split(',')[1], out max_lpc_order)) ||
						int.TryParse(args[arg], out max_lpc_order);
				}
				else if ((args[arg] == "-f" || args[arg] == "--fixed-order") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_fixed_order) &&
						int.TryParse(args[arg].Split(',')[1], out max_fixed_order)) ||
						int.TryParse(args[arg], out max_fixed_order);
				}
				else if ((args[arg] == "-e" || args[arg] == "--estimation-depth") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out estimation_depth);
				else if ((args[arg] == "-c" || args[arg] == "--max-precision") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_precision) &&
						int.TryParse(args[arg].Split(',')[1], out max_precision)) ||
						int.TryParse(args[arg], out max_precision);
				}
				else if ((args[arg] == "-v" || args[arg] == "--vbr"))
					ok = (++arg < args.Length) && int.TryParse(args[arg], out vbr_mode);
				else if ((args[arg] == "-b" || args[arg] == "--blocksize") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out blocksize);
				else if ((args[arg] == "-p" || args[arg] == "--padding") && ++arg < args.Length)
					ok = int.TryParse(args[arg], out padding);
				else if (args[arg] != "-" && args[arg][0] == '-' && int.TryParse(args[arg].Substring(1), out level))
					ok = level >= 0 && level <= 11;
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
			if (input_file == null || ((input_file == "-" || Path.GetExtension(input_file) == ".flac") && output_file == null))
			{
				Usage();
				return 2;
			}

			if (!quiet)
			{
				Console.WriteLine("CUETools.Flake, Copyright (C) 2009 Gregory S. Chudov.");
				Console.WriteLine("Based on Flake encoder by Justin Ruggles, <http://flake-enc.sourceforge.net/>.");
				Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
				Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			}

			IAudioSource audioSource;
			if (input_file == "-")
				audioSource = new WAVReader("", Console.OpenStandardInput());
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".wav")
				audioSource = new WAVReader(input_file, null);
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".flac")
				audioSource = new FlakeReader(input_file, null);
			else
			{
				Usage();
				return 3;
			}
			if (buffered)
				audioSource = new AudioPipe(audioSource, 0x10000);
			if (output_file == null)
				output_file = Path.ChangeExtension(input_file, "flac");
			FlakeWriter flake = new FlakeWriter((output_file == "-" || output_file == "nul") ? "" : output_file,				
				output_file == "-" ? Console.OpenStandardOutput() :
				output_file == "nul" ? new NullStream() : null,
				audioSource.PCM);
			flake.FinalSampleCount = audioSource.Length;
			IAudioDest audioDest = flake;
			AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);

			try
			{
				if (level >= 0)
					flake.CompressionLevel = level;
				if (prediction_type != null)
					flake.PredictionType = Flake.LookupPredictionType(prediction_type);
				if (stereo_method != null)
					flake.StereoMethod = Flake.LookupStereoMethod(stereo_method);
				if (window_method != null)
					flake.WindowMethod = Flake.LookupWindowMethod(window_method);
				if (order_method != null)
					flake.OrderMethod = Flake.LookupOrderMethod(order_method);
				if (window_function != null)
					flake.WindowFunction = Flake.LookupWindowFunction(window_function);
				if (min_partition_order >= 0)
					flake.MinPartitionOrder = min_partition_order;
				if (max_partition_order >= 0)
					flake.MaxPartitionOrder = max_partition_order;
				if (min_lpc_order >= 0)
					flake.MinLPCOrder = min_lpc_order;
				if (max_lpc_order >= 0)
					flake.MaxLPCOrder = max_lpc_order;
				if (min_fixed_order >= 0)
					flake.MinFixedOrder = min_fixed_order;
				if (max_fixed_order >= 0)
					flake.MaxFixedOrder = max_fixed_order;
				if (max_precision >= 0)
					flake.MaxPrecisionSearch = max_precision;
				if (min_precision >= 0)
					flake.MinPrecisionSearch = min_precision;
				if (blocksize >= 0)
					flake.BlockSize = blocksize;
				if (estimation_depth >= 0)
					flake.EstimationDepth = estimation_depth;
				if (padding >= 0)
					flake.PaddingLength = padding;
				if (vbr_mode >= 0)
					flake.VBRMode = vbr_mode;
				flake.DoMD5 = do_md5;
				flake.DoSeekTable = do_seektable;
				flake.DoVerify = do_verify;
			}
			catch (Exception ex)
			{
				Usage();
				Console.WriteLine("");
				Console.WriteLine("Error: {0}.", ex.Message);
				return 4;
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
				return 5;
			}
#endif

			TimeSpan totalElapsed = DateTime.Now - start;
			if (!quiet)
			{
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
				Console.Out.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}..{7}\t{8}..{9}\t{10}..{11}\t{12}..{13}\t{14}\t{15}",
					flake.TotalSize,
					flake.UserProcessorTime.TotalSeconds > 0 ? flake.UserProcessorTime.TotalSeconds : totalElapsed.TotalSeconds,
					flake.PredictionType.ToString().PadRight(15),
					flake.StereoMethod.ToString().PadRight(15),
					(flake.OrderMethod.ToString() + "(" + flake.EstimationDepth.ToString() + ")").PadRight(15),
					flake.WindowFunction,
					flake.MinPartitionOrder,
					flake.MaxPartitionOrder,
					flake.MinLPCOrder,
					flake.MaxLPCOrder,
					flake.MinFixedOrder,
					flake.MaxFixedOrder,
					flake.MinPrecisionSearch,
					flake.MaxPrecisionSearch,
					flake.BlockSize,
					flake.VBRMode
					);
			}
			return 0;
		}
	}
}
