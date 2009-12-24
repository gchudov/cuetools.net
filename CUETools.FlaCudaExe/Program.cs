/**
 * CUETools.FlaCuda: FLAC audio encoder using CUDA
 * Copyright (c) 2009 Gregory S. Chudov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;
using CUETools.Codecs.FlaCuda;

namespace CUETools.FlaCudaExe
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.FlaCuda.exe [options] <input.wav>");
			Console.WriteLine();
			Console.WriteLine("Options:");
			Console.WriteLine();
			Console.WriteLine(" -0 .. -11            Compression level, default 5.");
			Console.WriteLine(" -o <file>            Output filename, or \"-\" for stdout, or nul.");
			Console.WriteLine(" -p #                 Padding bytes.");
			Console.WriteLine(" -q --quiet           Quiet mode.");
			Console.WriteLine(" --verify             Verify during encoding.");
			Console.WriteLine(" --no-md5             Don't compute MD5 hash.");
			Console.WriteLine(" --no-seektable       Don't generate a seektable.");
			Console.WriteLine(" --gpu-only           Do everything on GPU.");
			Console.WriteLine(" --cpu-threads        Use additional CPU threads.");
			Console.WriteLine();
			Console.WriteLine("Advanced Options:");
			Console.WriteLine();
			Console.WriteLine(" -b #                 Block size.");
			Console.WriteLine(" -v #                 Variable block size mode (0,4).");
			Console.WriteLine(" -s <method>          Stereo decorrelation (independent,search).");
			Console.WriteLine(" -r #[,#]             Rice partition order {max} or {min},{max} (0..8).");
			Console.WriteLine();
			Console.WriteLine("LPC options:");
			Console.WriteLine();
			Console.WriteLine(" -w <func>[,<func>]   One or more window functions (bartlett,welch,hann,flattop,tukey).");
			Console.WriteLine(" -l #[,#]             Prediction order {max} or {min},{max} (1..32).");
			Console.WriteLine("    --max-precision   Coefficients precision search (0..1).");
			Console.WriteLine();
		}

		static void Main(string[] args)
		{
			TextWriter stdout = Console.Out;
			Console.SetOut(Console.Error);

			DateTime start = DateTime.Now;
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
			bool debug = false, quiet = false;
			string stereo_method = null;
			string window_function = null;
			string input_file = null;
			string output_file = null;
			int min_partition_order = -1, max_partition_order = -1,
				min_lpc_order = -1, max_lpc_order = -1,
				min_fixed_order = -1, max_fixed_order = -1,
				min_precision = -1, max_precision = -1,
				orders_per_window = -1,
				blocksize = -1;
			int level = -1, padding = -1, vbr_mode = -1;
			bool do_md5 = true, do_seektable = true, do_verify = false, gpu_only = false, use_lattice = false;
			int cpu_threads = 0;
			bool ok = true;

			for (int arg = 0; arg < args.Length; arg++)
			{
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
				else if (args[arg] == "--gpu-only")
					gpu_only = true;
				else if (args[arg] == "--use-lattice")
					use_lattice = true;
				else if (args[arg] == "--no-md5")
					do_md5 = false;
				else if (args[arg] == "--cpu-threads")
					ok = (++arg < args.Length) && int.TryParse(args[arg], out cpu_threads);
				else if ((args[arg] == "-o" || args[arg] == "--output") && ++arg < args.Length)
					output_file = args[arg];
				else if ((args[arg] == "-s" || args[arg] == "--stereo") && ++arg < args.Length)
					stereo_method = args[arg];
				else if ((args[arg] == "-w" || args[arg] == "--window") && ++arg < args.Length)
					window_function = args[arg];
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
				else if (args[arg] == "--fixed-order" && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_fixed_order) &&
						int.TryParse(args[arg].Split(',')[1], out max_fixed_order)) ||
						int.TryParse(args[arg], out max_fixed_order);
				}
				else if ((args[arg] == "-c" || args[arg] == "--max-precision") && ++arg < args.Length)
				{
					ok = (args[arg].Split(',').Length == 2 &&
						int.TryParse(args[arg].Split(',')[0], out min_precision) &&
						int.TryParse(args[arg].Split(',')[1], out max_precision)) ||
						int.TryParse(args[arg], out max_precision);
				}
				else if ((args[arg] == "-v" || args[arg] == "--vbr"))
					ok = (++arg < args.Length) && int.TryParse(args[arg], out vbr_mode);
				else if (args[arg] == "--orders-per-window")
					ok = (++arg < args.Length) && int.TryParse(args[arg], out orders_per_window);					
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
					break;
			}
			if (!quiet)
			{
				Console.WriteLine("CUETools.FlaCuda, Copyright (C) 2009 Gregory S. Chudov.");
				Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
				Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			}
			if (!ok || input_file == null)
			{
				Usage();
				return;
			}

			if (((input_file == "-" || Path.GetExtension(input_file) == ".flac") && output_file == null))
			{
				Console.WriteLine();
				Console.WriteLine("Output file not specified.");
				Console.WriteLine();
				Usage();
				return;
			}

			IAudioSource audioSource;
			if (input_file == "-")
				audioSource = new WAVReader(Console.OpenStandardInput());
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".wav")
				audioSource = new WAVReader(input_file, null);
			else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".flac")
				audioSource = new FlakeReader(input_file, null);
			else
			{
				Usage();
				return;
			}
			if (output_file == null)
				output_file = Path.ChangeExtension(input_file, "flac");
			FlaCudaWriter encoder = new FlaCudaWriter((output_file == "-" || output_file == "nul") ? "" : output_file,
				audioSource.BitsPerSample, audioSource.ChannelCount, audioSource.SampleRate,
				output_file == "-" ? Console.OpenStandardOutput() :
				output_file == "nul" ? new NullStream() : null);
			if (audioSource.Length != 0)
				encoder.FinalSampleCount = (long)audioSource.Length;
			IAudioDest audioDest = encoder;
			int[,] buff = new int[FlaCudaWriter.MAX_BLOCKSIZE, audioSource.ChannelCount];

			try
			{
				encoder.GPUOnly = gpu_only;
				encoder.CPUThreads = cpu_threads;
				if (level >= 0)
					encoder.CompressionLevel = level;
				if (stereo_method != null)
					encoder.StereoMethod = Flake.LookupStereoMethod(stereo_method);
				if (window_function != null)
					encoder.WindowFunction = Flake.LookupWindowFunction(window_function);
				if (min_partition_order >= 0)
					encoder.MinPartitionOrder = min_partition_order;
				if (max_partition_order >= 0)
					encoder.MaxPartitionOrder = max_partition_order;
				if (min_lpc_order >= 0)
					encoder.MinLPCOrder = min_lpc_order;
				if (max_lpc_order >= 0)
					encoder.MaxLPCOrder = max_lpc_order;
				if (min_fixed_order >= 0)
					encoder.MinFixedOrder = min_fixed_order;
				if (max_fixed_order >= 0)
					encoder.MaxFixedOrder = max_fixed_order;
				if (max_precision >= 0)
					encoder.MaxPrecisionSearch = max_precision;
				if (min_precision >= 0)
					encoder.MinPrecisionSearch = min_precision;
				if (blocksize >= 0)
					encoder.BlockSize = blocksize;
				if (padding >= 0)
					encoder.PaddingLength = padding;
				if (vbr_mode >= 0)
					encoder.VBRMode = vbr_mode;
				if (orders_per_window >= 0)
					encoder.OrdersPerWindow = orders_per_window;
				encoder.UseLattice = use_lattice;
				encoder.DoMD5 = do_md5;
				encoder.DoSeekTable = do_seektable;
				encoder.DoVerify = do_verify;
			}
			catch (Exception ex)
			{
				Usage();
				Console.WriteLine("");
				Console.WriteLine("Error: {0}.", ex.Message);
				return;
			}

			if (!quiet)
			{
				Console.WriteLine("Filename  : {0}", input_file);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.SampleRate, audioSource.ChannelCount, audioSource.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.SampleRate));
			}

#if !DEBUG
			try
#endif
			{
				do
				{
					uint toRead = audioSource.Length == 0 ? (uint)buff.GetLength(0) : Math.Min((uint)buff.GetLength(0), (uint)audioSource.Remaining);
					uint samplesRead = audioSource.Read(buff, toRead);
					if (samplesRead == 0) break;
					audioDest.Write(buff, 0, (int)samplesRead);
					TimeSpan elapsed = DateTime.Now - start;
					if (!quiet)
					{
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
					}
				} while (true);
				audioDest.Close();
			}
#if !DEBUG
			catch (Exception ex)
			{
			    Console.Error.Write("\r                                                                         \r");
			    Console.WriteLine("Error     : {0}", ex.Message);
			    audioDest.Delete();
			    audioSource.Close();
			    return;
			}
#endif

			TimeSpan totalElapsed = DateTime.Now - start;
			if (!quiet)
			{
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.00}x; {2} bytes in {1} seconds;",
					audioSource.Position / totalElapsed.TotalSeconds / audioSource.SampleRate,
					totalElapsed,
					encoder.TotalSize
					);
			}
			audioSource.Close();

			if (debug)
			{
				Console.SetOut(stdout);
				Console.Out.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}..{5}\t{6}..{7}\t{8}..{9}\t{10}\t{11}",
					encoder.TotalSize,
					encoder.UserProcessorTime.TotalSeconds > 0 ? encoder.UserProcessorTime.TotalSeconds : totalElapsed.TotalSeconds,
					encoder.StereoMethod.ToString().PadRight(15),
					encoder.WindowFunction.ToString().PadRight(15),
					encoder.MinPartitionOrder,
					encoder.MaxPartitionOrder,
					encoder.MinLPCOrder,
					encoder.MaxLPCOrder,
					encoder.MinPrecisionSearch,
					encoder.MaxPrecisionSearch,
					encoder.BlockSize,
					encoder.VBRMode
					);
			}
		}
	}
}
