/**
 * CUETools.FLACCL: FLAC audio encoder using CUDA
 * Copyright (c) 2009 Grigory Chudov
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
using CUETools.Codecs.Flake;
using CUETools.Codecs.FLACCL;

namespace CUETools.FLACCL.cmd
{
	class Program
	{
		static void Usage()
		{
			Console.WriteLine("Usage    : CUETools.FLACCL.exe [options] <input.wav>");
			Console.WriteLine();
			Console.WriteLine("Options:");
			Console.WriteLine();
			Console.WriteLine(" -0 .. -11             Compression level, default 8; 9..11 require --lax");
			Console.WriteLine(" -o <file>             Output filename, or \"-\" for stdout, or nul");
			Console.WriteLine(" -p #                  Padding bytes");
			Console.WriteLine(" -q --quiet            Quiet mode");
            Console.WriteLine(" --lax                 Allow non-subset modes");
            Console.WriteLine(" --verify              Verify during encoding");
			Console.WriteLine(" --no-md5              Don't compute MD5 hash");
			Console.WriteLine(" --no-seektable        Don't generate a seektable");
            Console.WriteLine(" --ignore-chunk-sizes  Ignore WAV length (for pipe input)");
            Console.WriteLine(" --cpu-threads         Use additional CPU threads");
			Console.WriteLine();
			Console.WriteLine("OpenCL Options:");
			Console.WriteLine();
			Console.WriteLine(" --opencl-type <X>     CPU or GPU, default GPU");
            var platforms = new List<string>();
            foreach (var value in (new EncoderSettingsPlatformConverter()).GetStandardValues(null))
                platforms.Add(value as string);
            Console.WriteLine(" --opencl-platform <X> {0}", platforms.Count == 0 ? "No OpenCL platforms detected" : string.Join(", ", platforms.ConvertAll(s => "\"" + s + "\"").ToArray()));
			Console.WriteLine(" --group-size #        Set GPU workgroup size (64,128,256)");
			Console.WriteLine(" --task-size #         Set number of frames per multiprocessor, default 8");
			Console.WriteLine(" --slow-gpu            Some encoding stages are done on CPU");
			Console.WriteLine(" --fast-gpu            Experimental mode, not recommended");
			Console.WriteLine(" --define <X> <Y>      OpenCL preprocessor definition");
			Console.WriteLine();
			Console.WriteLine("Advanced Options:");
			Console.WriteLine();
			Console.WriteLine(" -b #                  Block size");
			Console.WriteLine(" -s <method>           Stereo decorrelation (independent,search)");
			Console.WriteLine(" -r #[,#]              Rice partition order {max} or {min},{max} (0..8)");
			Console.WriteLine();
			Console.WriteLine("LPC options:");
			Console.WriteLine();
			Console.WriteLine(" -w <func>[,<func>]    Window functions (bartlett,welch,hann,flattop,tukey)");
			Console.WriteLine(" -l #[,#]              Prediction order {max} or {min},{max} (1..32)");
			Console.WriteLine("    --max-precision    Coefficients precision search (0..1)");
			Console.WriteLine();
		}

		static int Main(string[] args)
		{
			TextWriter stdout = Console.Out;
			Console.SetOut(Console.Error);

            var settings = new Codecs.FLACCL.EncoderSettings() { AllowNonSubset = true };
			TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
			bool debug = false, quiet = false;
			string stereo_method = null;
			string window_function = null;
			string input_file = null;
			string output_file = null;
			string device_type = null;
            int min_precision = -1, max_precision = -1,
                orders_per_window = -1, orders_per_channel = -1;
			int input_len = 4096, input_val = 0, input_bps = 16, input_ch = 2, input_rate = 44100;
			int level = -1, vbr_mode = -1;
			bool do_seektable = true;
			bool estimate_window = false;
			bool buffered = false;
			bool ok = true;
            bool allowNonSubset = false;
            bool ignore_chunk_sizes = false;
			int intarg;

			for (int arg = 0; arg < args.Length; arg++)
			{
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
                else if (args[arg] == "--ignore-chunk-sizes")
                    ignore_chunk_sizes = true;
                else if (args[arg] == "--slow-gpu")
					settings.GPUOnly = false;
				else if (args[arg] == "--fast-gpu")
					settings.DoRice = true;
				else if (args[arg] == "--no-md5")
					settings.DoMD5 = false;
                else if (args[arg] == "--lax")
                    allowNonSubset = true;
                else if (args[arg] == "--buffered")
					buffered = true;
				else if (args[arg] == "--cpu-threads")
				{
					int val = settings.CPUThreads;
					ok = (++arg < args.Length) && int.TryParse(args[arg], out val);
					settings.CPUThreads = val;
				}
				else if (args[arg] == "--group-size" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					settings.GroupSize = intarg;
				else if (args[arg] == "--task-size" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					settings.TaskSize = intarg;
				else if (args[arg] == "--define" && arg + 2 < args.Length)
					settings.Defines += "#define " + args[++arg] + " " + args[++arg] + "\n";
				else if (args[arg] == "--opencl-platform" && ++arg < args.Length)
					settings.Platform = args[arg];
				else if (args[arg] == "--mapped-memory")
					settings.MappedMemory = true;
				else if (args[arg] == "--opencl-type" && ++arg < args.Length)
					device_type = args[arg];
				else if (args[arg] == "--input-length" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					input_len = intarg;
				else if (args[arg] == "--input-value" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					input_val = intarg;
				else if (args[arg] == "--input-bps" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					input_bps = intarg;
				else if (args[arg] == "--input-channels" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
					input_ch = intarg;
				else if ((args[arg] == "-o" || args[arg] == "--output") && ++arg < args.Length)
					output_file = args[arg];
				else if ((args[arg] == "-s" || args[arg] == "--stereo") && ++arg < args.Length)
					stereo_method = args[arg];
				else if ((args[arg] == "-w" || args[arg] == "--window") && ++arg < args.Length)
					window_function = args[arg];
				else if ((args[arg] == "-r" || args[arg] == "--partition-order") && ++arg < args.Length)
				{
                    int min_partition_order, max_partition_order;
                    ok = (args[arg].Split(',').Length == 2
                        && int.TryParse(args[arg].Split(',')[0], out min_partition_order)
                        && (settings.MinPartitionOrder = min_partition_order) != -1
                        && int.TryParse(args[arg].Split(',')[1], out max_partition_order)
                        && (settings.MaxPartitionOrder = max_partition_order) != -1)
                        || (int.TryParse(args[arg], out max_partition_order)
                        && (settings.MaxPartitionOrder = max_partition_order) != -1);
				}
				else if ((args[arg] == "-l" || args[arg] == "--lpc-order") && ++arg < args.Length)
				{
                    int min_lpc_order, max_lpc_order;
					ok = (args[arg].Split(',').Length == 2
					    && int.TryParse(args[arg].Split(',')[0], out min_lpc_order)
                        && (settings.MinLPCOrder = min_lpc_order) != -1
						&& int.TryParse(args[arg].Split(',')[1], out max_lpc_order)
                        && (settings.MaxLPCOrder = max_lpc_order) != -1)
						|| (int.TryParse(args[arg], out max_lpc_order)
                        && (settings.MaxLPCOrder = max_lpc_order) != -1);
				}
                else if (args[arg] == "--fixed-order" && ++arg < args.Length)
                {
                    int min_fixed_order, max_fixed_order;
                    ok = (args[arg].Split(',').Length == 2
                        && int.TryParse(args[arg].Split(',')[0], out min_fixed_order)
                        && (settings.MinFixedOrder = min_fixed_order) != -1
                        && int.TryParse(args[arg].Split(',')[1], out max_fixed_order)
                        && (settings.MaxFixedOrder = max_fixed_order) != -1)
                        || (int.TryParse(args[arg], out max_fixed_order)
                        && (settings.MaxFixedOrder = max_fixed_order) != -1);
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
                else if (args[arg] == "--orders-per-window" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    orders_per_window = intarg;
                else if (args[arg] == "--orders-per-channel" && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    orders_per_channel = intarg;
                else if (args[arg] == "--estimate-window")
                    estimate_window = true;
                else if ((args[arg] == "-b" || args[arg] == "--blocksize") && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    settings.BlockSize = intarg;
                else if ((args[arg] == "-p" || args[arg] == "--padding") && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    settings.Padding = intarg;
                else if (args[arg] != "-" && args[arg][0] == '-' && int.TryParse(args[arg].Substring(1), out level))
                {
                    ok = level >= 0 && level <= 11;
                    settings.SetEncoderModeIndex(level);
                }
                else if ((args[arg][0] != '-' || args[arg] == "-") && input_file == null)
                    input_file = args[arg];
                else
                    ok = false;
				if (!ok)
					break;
			}
			if (!quiet)
			{
                Console.WriteLine("{0}, Copyright (C) 2010-2013 Grigory Chudov.", Codecs.FLACCL.AudioEncoder.Vendor);
				Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
				Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
			}
			if (!ok || input_file == null)
			{
				Usage();
				return 1;
			}

			if (((input_file == "-" || Path.GetExtension(input_file) == ".flac") && output_file == null))
			{
                Usage();
                Console.WriteLine();
				Console.WriteLine("Output file not specified.");
				return 2;
			}

			IAudioSource audioSource;
			try
			{
                if (input_file == "-")
                    audioSource = new Codecs.WAV.AudioDecoder(new Codecs.WAV.DecoderSettings() { IgnoreChunkSizes = true }, "", Console.OpenStandardInput());
				else if (input_file == "nul")
					audioSource = new Codecs.NULL.AudioDecoder(new AudioPCMConfig(input_bps, input_ch, input_rate), input_len, input_val);
				else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".wav")
					audioSource = new Codecs.WAV.AudioDecoder(new Codecs.WAV.DecoderSettings(), input_file);
				else if (File.Exists(input_file) && Path.GetExtension(input_file) == ".flac")
					audioSource = new AudioDecoder(new DecoderSettings(), input_file);
				else
				{
                    Usage();
                    Console.WriteLine();
                    Console.WriteLine("Input file \"{0}\" does not exist.", input_file);
					return 2;
				}
			}
			catch (Exception ex)
			{
				Usage();
				Console.WriteLine("");
				Console.WriteLine("Error: {0}.", ex.Message);
				return 3;
			}
			if (buffered)
				audioSource = new AudioPipe(audioSource, Codecs.FLACCL.AudioEncoder.MAX_BLOCKSIZE);
			if (output_file == null)
				output_file = Path.ChangeExtension(input_file, "flac");
            settings.PCM = audioSource.PCM;
            settings.AllowNonSubset = allowNonSubset;
            Codecs.FLACCL.AudioEncoder encoder;

			try
			{
                if (device_type != null)
                    settings.DeviceType = (OpenCLDeviceType)(Enum.Parse(typeof(OpenCLDeviceType), device_type, true));
                encoder = new Codecs.FLACCL.AudioEncoder((output_file == "-" || output_file == "nul") ? "" : output_file,
                    output_file == "-" ? Console.OpenStandardOutput() :
                    output_file == "nul" ? new NullStream() : null,
                    settings);
                settings = encoder.Settings as Codecs.FLACCL.EncoderSettings;
                encoder.FinalSampleCount = audioSource.Length;
				if (stereo_method != null)
					encoder.StereoMethod = FlakeConstants.LookupStereoMethod(stereo_method);
				if (window_function != null)
					encoder.WindowFunction = FlakeConstants.LookupWindowFunction(window_function);
				if (max_precision >= 0)
					encoder.MaxPrecisionSearch = max_precision;
				if (min_precision >= 0)
					encoder.MinPrecisionSearch = min_precision;
				if (vbr_mode >= 0)
					encoder.VBRMode = vbr_mode;
				if (orders_per_window >= 0)
					encoder.OrdersPerWindow = orders_per_window;
				if (orders_per_channel >= 0)
					encoder.OrdersPerChannel = orders_per_channel;
				if (estimate_window)
					encoder.EstimateWindow = estimate_window;
				encoder.DoSeekTable = do_seektable;
			}
			catch (Exception ex)
			{
				Usage();
				Console.WriteLine("");
				Console.WriteLine("Error: {0}.", ex.Message);
				return 3;
			}

            IAudioDest audioDest = encoder;
            AudioBuffer buff = new AudioBuffer(audioSource, Codecs.FLACCL.AudioEncoder.MAX_BLOCKSIZE);

			if (!quiet)
			{
				Console.WriteLine("Filename  : {0}", input_file);
				Console.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.PCM.SampleRate));
                Console.WriteLine("Device    : {0}, Platform: \"{1}\", Version: {2}, Driver: {3}", settings.Device.Trim(), settings.Platform, settings.PlatformVersion.Trim(), settings.DriverVersion.Trim());
            }

            bool keepRunning = true;
            Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
            {
                keepRunning = false;
                if (e.SpecialKey == ConsoleSpecialKey.ControlC)
                    e.Cancel = true;
                else
                    audioDest.Delete();
            };

			DateTime start = DateTime.Now;
			try
			{
				audioDest.Write(buff);
				start = DateTime.Now;
				while (audioSource.Read(buff, -1) != 0)
				{
					audioDest.Write(buff);
					TimeSpan elapsed = DateTime.Now - start;
					if (!quiet)
					{
						if ((elapsed - lastPrint).TotalMilliseconds > 60)
						{
                            long length = Math.Max(audioSource.Position, audioSource.Length);
                            Console.Error.Write("\rProgress  : {0:00}%; {1:0.00}x; {2}/{3}",
                                100.0 * audioSource.Position / length,
								audioSource.Position / elapsed.TotalSeconds / audioSource.PCM.SampleRate,
								elapsed,
                                TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * length)
								);
							lastPrint = elapsed;
						}
					}
                    if (!keepRunning)
                        throw new Exception("Aborted");
                }
				audioDest.Close();
			}
			catch (OpenCLNet.OpenCLBuildException ex)
			{
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Error     : {0}", ex.Message);
				Console.WriteLine("{0}", ex.BuildLogs[0]);
				if (debug)
					using (StreamWriter sw = new StreamWriter("debug.txt", true))
						sw.WriteLine("{0}\n{1}\n{2}", ex.Message, ex.StackTrace, ex.BuildLogs[0]);
				audioDest.Delete();
				audioSource.Close();
				return 4;
			}
#if !DEBUG
			catch (Exception ex)
			{
			    Console.Error.Write("\r                                                                         \r");
			    Console.WriteLine("Error     : {0}", ex.Message);
				if (debug)
					using (StreamWriter sw = new StreamWriter("debug.txt", true))
						sw.WriteLine("{0}\n{1}", ex.Message, ex.StackTrace);
			    audioDest.Delete();
			    audioSource.Close();
			    return 4;
			}
#endif

			TimeSpan totalElapsed = DateTime.Now - start;
			if (!quiet)
			{
				Console.Error.Write("\r                                                                         \r");
				Console.WriteLine("Results   : {0:0.00}x; {2} bytes in {1} seconds;",
					audioSource.Position / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
					totalElapsed,
					encoder.TotalSize
					);
			}
			audioSource.Close();

			if (debug)
			{
				Console.SetOut(stdout);
				Console.Out.WriteLine("{0}\t{1}\t{2}\t{3}\t{4} ({5})\t{6}/{7}+{12}{13}\t{8}..{9}\t{10}\t{11}",
					encoder.TotalSize,
					encoder.UserProcessorTime.TotalSeconds > 0 ? encoder.UserProcessorTime.TotalSeconds : totalElapsed.TotalSeconds,
					(encoder.StereoMethod.ToString() + (encoder.OrdersPerChannel == 32 ? "" : "(" + encoder.OrdersPerChannel.ToString() + ")")).PadRight(15),
					encoder.WindowFunction.ToString().PadRight(15),
					settings.MaxPartitionOrder,
					settings.GPUOnly ? "GPU" : "CPU",
					encoder.OrdersPerWindow,
                    settings.MaxLPCOrder,
					encoder.MinPrecisionSearch,
					encoder.MaxPrecisionSearch,
					encoder.Settings.BlockSize,
					encoder.VBRMode,
                    settings.MaxFixedOrder - settings.MinFixedOrder + 1,
					encoder.DoConstant ? "c" : ""
					);
			}
			return 0;
		}
	}
}
