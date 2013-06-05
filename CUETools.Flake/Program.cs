//#define FINETUNE
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;

namespace CUETools.FlakeExe
{
#if FINETUNE
    class FineTuneTask
    {
        public int code;
        public int size;
        public int no;
        public int gen;
        public int min_depth;
        public List<FineTuneTask> parents;

        public bool IsImprovement
        {
            get
            {
                return this.parents.TrueForAll(p => this.size < p.size);
            }
        }

        public bool done
        {
            get
            {
                return this.size != int.MaxValue;
            }
        }

        public FineTuneTask(int code, int no)
        {
            this.code = code;
            this.gen = 0;
            this.no = no;
            this.size = int.MaxValue;
            this.min_depth = 0;
            this.parents = new List<FineTuneTask>();
        }

        public FineTuneTask(int code, int no, FineTuneTask parent)
            : this(code, no)
        {
            this.gen = parent.gen + 1;
            this.parents.Add(parent);
        }

        public List<int> GetDerivativeCodes(int max_lpc_order)
        {
            var codes = new List<int>();
            for (int ft = 0; ft < max_lpc_order; ft++)
            {
                int c1 = this.code ^ (1 << ft);
                codes.Add(c1);
            }

            for (int ft = 0; ft < max_lpc_order; ft++)
            {
                for (int ft1 = ft + 1; ft1 < max_lpc_order; ft1++)
                    if (((this.code >> ft1) & 1) != ((this.code >> ft) & 1))
                    {
                        int c1 = this.code ^ (1 << ft) ^ (1 << ft1);
                        codes.Add(c1);
                        break;
                    }
                
                for (int ft1 = ft - 1; ft1 >= 0; ft1--)
                    if (((this.code >> ft1) & 1) != ((this.code >> ft) & 1))
                    {
                        int c1 = this.code ^ (1 << ft) ^ (1 << ft1);
                        codes.Add(c1);
                        break;
                    }
            }

            return codes;
        }
    }
#endif

    class Program
    {
        static void Usage()
        {
            Console.WriteLine("Usage    : CUETools.Flake.exe [options] <input.wav>");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine();
            Console.WriteLine(" -0 .. -11            Compression level, default 7; 9..11 require --lax");
            Console.WriteLine(" -o <file>            Output filename, or \"-\" for stdout, or nul.");
            Console.WriteLine(" -p #                 Padding bytes.");
            Console.WriteLine(" -q --quiet           Quiet mode.");
            Console.WriteLine(" --lax                Allow non-subset modes");
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
                estimation_depth = -1;
            int skip_a = 0, skip_b = 0;
            int intarg = -1, vbr_mode = -1, magic = -1;
            bool do_seektable = true;
            bool buffered = false;
            string coeffs = null;
            var settings = new FlakeWriterSettings() { AllowNonSubset = true };
            bool allowNonSubset = false;
#if FINETUNE
            int finetune_depth = -1;
#endif
            
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
                else if (args[arg] == "--no-md5")
                    settings.DoMD5 = false;
                else if (args[arg] == "--lax")
                    allowNonSubset = true;
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
                else if (args[arg] == "--skip" && ++arg < args.Length)
                {
                    ok = (args[arg].Split(',').Length == 2 &&
                        int.TryParse(args[arg].Split(',')[0], out skip_a) &&
                        int.TryParse(args[arg].Split(',')[1], out skip_b)) ||
                        int.TryParse(args[arg], out skip_a);
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
                else if ((args[arg] == "-b" || args[arg] == "--blocksize") && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    settings.BlockSize = intarg;
                else if ((args[arg] == "-p" || args[arg] == "--padding") && ++arg < args.Length && int.TryParse(args[arg], out intarg))
                    settings.Padding = intarg;
                else if (args[arg] == "--magic" && ++arg < args.Length)
                    ok = int.TryParse(args[arg], out magic);
#if FINETUNE
                else if (args[arg] == "--depth" && ++arg < args.Length)
                    ok = int.TryParse(args[arg], out finetune_depth);
#endif
                else if (args[arg] == "--coefs" && ++arg < args.Length)
                    coeffs = args[arg];
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
            if (input_file == null || ((input_file == "-" || Path.GetExtension(input_file) == ".flac") && output_file == null))
            {
                Usage();
                return 2;
            }

            if (!quiet)
            {
                Console.WriteLine("CUETools.Flake, Copyright (C) 2009 Grigory Chudov.");
                Console.WriteLine("Based on Flake encoder by Justin Ruggles, <http://flake-enc.sourceforge.net/>.");
                Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
                Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
            }

#if FINETUNE
            var prefix = new List<int>();
            if (coeffs != null)
            {
                var ps = coeffs.Split(',');
                for (int i = 0; i < ps.Length; i++)
                    prefix.Add(ps[i].StartsWith("0x") ? int.Parse(ps[i].Substring(2), System.Globalization.NumberStyles.HexNumber) : int.Parse(ps[i]));
            }
            string codefmt = "0x{0:x" + (max_lpc_order / 4 + 1).ToString() + "}";

            int max_prefix = magic >= 0 ? magic : 8;
            do
            {
                FineTuneTask best_task = null;
                var tasks = new Dictionary<int, FineTuneTask>();
                if (prefix.Count == 0)
                {
                    tasks.Add((1 << max_lpc_order) - 1, new FineTuneTask((1 << max_lpc_order) - 1, 1));
                    tasks.Add((2 << max_lpc_order) - 1, new FineTuneTask((2 << max_lpc_order) - 1, 2));
                    //tasks.Add((3 << max_lpc_order) - 1, new FineTuneTask((3 << max_lpc_order) - 1, 3));
                    //tasks.Add((4 << max_lpc_order) - 1, new FineTuneTask((4 << max_lpc_order) - 1, 4));
                }
                else
                {
                    foreach (var c in prefix)
                    {
                        tasks.Add(c, new FineTuneTask(c, 1));
                        tasks.Add((1 << max_lpc_order) ^ c, new FineTuneTask((1 << max_lpc_order) ^ c, 2));
                    }
                }
                int total_added_tasks = tasks.Values.Count;

                do
                {
                    foreach (var task in tasks.Values)
                        if (!task.done)
                        {
                            var prefixc = new List<int>(prefix);
                            prefixc.Add(task.code);
                            for (int i = 1; i < max_lpc_order; i++)
                                FlakeWriter.SetCoefs(i, new int[0]);
                            FlakeWriter.SetCoefs(max_lpc_order, prefixc.ToArray());
                            magic = prefixc.Count;
#endif

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
                            settings.PCM = audioSource.PCM;
                            settings.AllowNonSubset = allowNonSubset;
                            FlakeWriter flake = new FlakeWriter((output_file == "-" || output_file == "nul") ? "" : output_file,
                                output_file == "-" ? Console.OpenStandardOutput() :
                                output_file == "nul" ? new NullStream() : null,
                                settings);
                            flake.FinalSampleCount = audioSource.Length - skip_a - skip_b;
                            IAudioDest audioDest = flake;
                            AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);

                            try
                            {
                                settings.Validate();
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
                                if (min_precision >= 0)
                                    flake.MinPrecisionSearch = min_precision;
                                if (max_precision >= 0)
                                    flake.MaxPrecisionSearch = max_precision;
                                if (estimation_depth >= 0)
                                    flake.EstimationDepth = estimation_depth;
                                if (vbr_mode >= 0)
                                    flake.VBRMode = vbr_mode;
                                if (magic >= 0)
                                    flake.DevelopmentMode = magic;
                                flake.DoSeekTable = do_seektable;
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

                            bool keepRunning = true;
                            Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
                            {
                                keepRunning = false;
                                if (e.SpecialKey == ConsoleSpecialKey.ControlC)
                                    e.Cancel = true;
                                else
                                    audioDest.Delete();
                            };

                            start = DateTime.Now;

#if !DEBUG
                            try
#endif
                            {
                                audioSource.Position = skip_a;
                                while (audioSource.Read(buff, skip_b == 0 ? -1 : (int)audioSource.Remaining - skip_b) != 0)
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
                                                elapsed,//.ToString(@"[-][d’.’]hh’:’mm’:’ss[‘.’ff]"),
                                                TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * audioSource.Length)
                                                );
                                            lastPrint = elapsed;
                                        }
                                    }
                                    if (!keepRunning)
                                        throw new Exception("Aborted");
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
                                Console.WriteLine("Results   : {0:0.00}x; {2} bytes in {1} seconds;",
                                    audioSource.Position / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
                                    totalElapsed,
                                    flake.TotalSize
                                    );
                            }
                            audioSource.Close();

#if FINETUNE
                            task.size = flake.TotalSize;
                            Console.SetOut(stdout);
                            Console.Out.WriteLine("{0}\t{1:0.00}\t{2}..{3}\t{4}\t{5}\t{6}/{7}\t{8}\t{9}\t{10}\t{11}",
                                flake.TotalSize,
                                flake.UserProcessorTime.TotalSeconds > 0 ? flake.UserProcessorTime.TotalSeconds : totalElapsed.TotalSeconds,
                                flake.MinLPCOrder,
                                flake.MaxLPCOrder,
                                flake.BlockSize,
                                task.gen,
                                task.no.ToString().PadLeft(total_added_tasks.ToString().Length, '0'),
                                total_added_tasks,
                                task.min_depth,
                                best_task != null && task.size < best_task.size ? "+" + (best_task.size - task.size).ToString() : task.IsImprovement ? "*" : "",
                                string.Join(",", prefixc.ConvertAll(x => string.Format(codefmt, x)).ToArray()),
                                string.Join(",", task.parents.ConvertAll(p => string.Format(codefmt, p.code)).ToArray())
                            );
#else

                            if (debug)
                            {
                                Console.SetOut(stdout);
                                Console.Out.WriteLine("{0}\t{1:0.00}\t{2}\t{3}\t{4}\t{5}\t{6}..{7}\t{8}..{9}\t{10}..{11}\t{12}..{13}\t{14}\t{15}\t{16}",
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
                                    flake.Settings.BlockSize,
                                    flake.VBRMode,
                                    coeffs ?? ""
                                );
                            }
#endif

#if FINETUNE
                            if (best_task == null || task.size < best_task.size)
                                best_task = task;
                        }

                    var promising_tasks = new List<FineTuneTask>();
                    promising_tasks.AddRange(tasks.Values);
                    promising_tasks.RemoveAll(task => !task.IsImprovement);
                    promising_tasks.Sort((task1, task2) => task1.size.CompareTo(task2.size));
                    if (finetune_depth >= 0 && promising_tasks.Count > finetune_depth)
                        promising_tasks.RemoveRange(finetune_depth, promising_tasks.Count - finetune_depth);
                    total_added_tasks = 0;
                    foreach (var c in promising_tasks)
                    {
                        foreach(var c1 in c.GetDerivativeCodes(max_lpc_order))
                            if (!tasks.ContainsKey(c1))
                            {
                                var new_task = new FineTuneTask(c1, ++total_added_tasks, c);
                                foreach(var c2 in new_task.GetDerivativeCodes(max_lpc_order))
                                    if (tasks.ContainsKey(c2) && tasks[c2].done && !new_task.parents.Contains(tasks[c2]))
                                        new_task.parents.Add(tasks[c2]);
                                new_task.min_depth = int.MaxValue;
                                foreach(var p in new_task.parents)
                                    if (promising_tasks.IndexOf(p) >= 0)
                                        new_task.min_depth = Math.Min(new_task.min_depth, Math.Max(p.min_depth, promising_tasks.IndexOf(p) + 1));
                                tasks.Add(c1, new_task);
                            }
                    }
                    if (total_added_tasks == 0)
                        break;
                } while (true);

                prefix.Add(best_task.code);
            } while (prefix.Count < max_prefix);

            Console.Out.WriteLine("\t{0}", string.Join(",", prefix.ConvertAll(x => string.Format(codefmt, x)).ToArray()));
#endif
            return 0;
        }
    }
}
