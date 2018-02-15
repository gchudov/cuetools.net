using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using CUETools.Codecs;
using CUETools.Processor;

namespace CUETools.Converter
{
    class Program
    {
        static void Usage()
        {
            Console.Error.WriteLine("Usage     : CUETools.Converter.exe [options] <infile> <outfile>");
            Console.Error.WriteLine();
            Console.Error.WriteLine("Options:");
            Console.Error.WriteLine();
            Console.Error.WriteLine(" --decoder <name>       Use non-default decoder.");
            Console.Error.WriteLine(" --encoder <name>       Use non-default encoder.");
            Console.Error.WriteLine(" --encoder-format <ext> Use encoder format different from file extension.");
            Console.Error.WriteLine(" --lossy                Use lossy encoder/mode.");
            Console.Error.WriteLine(" --lossless             Use lossless encoder/mode (default).");
            Console.Error.WriteLine(" --ignore-chunk-sizes   Ignore WAV length (for pipe input)");
            Console.Error.WriteLine(" -p #                   Padding bytes.");
            Console.Error.WriteLine(" -m <mode>              Encoder mode (0..8 for flac, V0..V9 for mp3, etc)");
            Console.Error.WriteLine();
        }

        public static CUEToolsUDC GetEncoder(CUEToolsCodecsConfig config, CUEToolsFormat fmt, bool lossless, string chosenEncoder)
        {
            CUEToolsUDC tmpEncoder;
            return chosenEncoder != null ?
                (config.encoders.TryGetValue(fmt.extension, lossless, chosenEncoder, out tmpEncoder) ? tmpEncoder : null) :
                (lossless ? fmt.encoderLossless : fmt.encoderLossy);
        }

        public static IAudioSource GetAudioSource(CUEToolsCodecsConfig config, string path, string chosenDecoder, bool ignore_chunk_sizes)
        {
            if (path == "-")
                return new WAVReader("", Console.OpenStandardInput(), ignore_chunk_sizes);
            string extension = Path.GetExtension(path).ToLower();
            Stream IO = null;
            if (extension == ".bin")
                return new WAVReader(path, IO, AudioPCMConfig.RedBook);
            CUEToolsFormat fmt;
            if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
                throw new Exception("Unsupported audio type: " + path);

            var decoder = fmt.decoder;
            if (chosenDecoder != null && !config.decoders.TryGetValue(fmt.extension, true, chosenDecoder, out decoder))
                throw new Exception("Unknown audio decoder " + chosenDecoder + " or unsupported audio type " + fmt.extension);
            if (decoder == null)
                throw new Exception("Unsupported audio type: " + path);
            if (decoder.path != null)
                return new UserDefinedReader(path, IO, decoder.path, decoder.parameters);
            if (decoder.type == null)
                throw new Exception("Unsupported audio type: " + path);

            try
            {
                object src = Activator.CreateInstance(decoder.type, path, IO);
                if (src == null || !(src is IAudioSource))
                    throw new Exception("Unsupported audio type: " + path + ": " + decoder.type.FullName);
                return src as IAudioSource;
            }
            catch (System.Reflection.TargetInvocationException ex)
            {
                if (ex.InnerException == null)
                    throw ex;
                throw ex.InnerException;
            }
        }

        static int Main(string[] args)
        {
            bool ok = true;
            string sourceFile = null, destFile = null;
            int padding = 8192;
            string encoderMode = null;
            string decoderName = null;
            string encoderName = null;
            string encoderFormat = null;
            bool ignore_chunk_sizes = false;
            AudioEncoderType audioEncoderType = AudioEncoderType.NoAudio;
            var decoderOptions = new Dictionary<string, string>();

            for (int arg = 0; arg < args.Length; arg++)
            {
                if (args[arg].Length == 0)
                    ok = false;
                else if (args[arg] == "--ignore-chunk-sizes")
                    ignore_chunk_sizes = true;
                else if (args[arg] == "--decoder" && ++arg < args.Length)
                    decoderName = args[arg];
                else if (args[arg] == "--decoder-option" && arg + 2 < args.Length)
                {
                    var optionName = args[++arg];
                    var optionValue = args[++arg];
                    decoderOptions.Add(optionName, optionValue);
                }
                else if (args[arg] == "--encoder" && ++arg < args.Length)
                    encoderName = args[arg];
                else if (args[arg] == "--encoder-format" && ++arg < args.Length)
                    encoderFormat = args[arg];
                else if ((args[arg] == "-p" || args[arg] == "--padding") && ++arg < args.Length)
                    ok = int.TryParse(args[arg], out padding);
                else if ((args[arg] == "-m" || args[arg] == "--mode") && ++arg < args.Length)
                    encoderMode = args[arg];
                else if (args[arg] == "--lossy")
                    audioEncoderType = AudioEncoderType.Lossy;
                else if (args[arg] == "--lossless")
                    audioEncoderType = AudioEncoderType.Lossless;
                else if ((args[arg][0] != '-' || args[arg] == "-") && sourceFile == null)
                    sourceFile = args[arg];
                else if ((args[arg][0] != '-' || args[arg] == "-") && sourceFile != null && destFile == null)
                    destFile = args[arg];
                else
                    ok = false;
                if (!ok)
                    break;
            }

            Console.Error.WriteLine("CUETools.Converter, Copyright (C) 2009-2018 Grigory Chudov.");
            Console.Error.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
            Console.Error.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
            if (!ok || sourceFile == null || destFile == null)
            {
                Usage();
                return 22;
            }

            if (destFile != "-" && destFile != "nul" && File.Exists(destFile))
            {
                Console.Error.WriteLine("Error: file already exists.");
                return 17;
            }

            DateTime start = DateTime.Now;
            TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
            CUEToolsCodecsConfig config = new CUEConfig();

#if !DEBUG
            try
#endif
            {
                IAudioSource audioSource = null;
                IAudioDest audioDest = null;
                TagLib.UserDefined.AdditionalFileTypes.Config = config;
                TagLib.File sourceInfo = sourceFile == "-" ? null : TagLib.File.Create(new TagLib.File.LocalFileAbstraction(sourceFile));

                try
                {
                    audioSource = Program.GetAudioSource(config, sourceFile, decoderName, ignore_chunk_sizes);
                    foreach (var decOpt in decoderOptions)
                    {
                        var decoderSettings = audioSource.Settings;
                        if (decoderSettings == null)
                            throw new Exception(String.Format("{0} doesn't have any properties.", audioSource.GetType().Name));
                        var property = TypeDescriptor.GetProperties(decoderSettings).Find(decOpt.Key, true);
                        if (property == null)
                            throw new Exception(String.Format("{0} doesn't have a property named {1}.", audioSource.GetType().Name, decOpt.Key));
                        property.SetValue(decoderSettings, 
                            TypeDescriptor.GetConverter(property.PropertyType).ConvertFromString(decOpt.Value));
                    }

                    AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);
                    Console.Error.WriteLine("Filename  : {0}", sourceFile);
                    Console.Error.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample, TimeSpan.FromSeconds(audioSource.Length * 1.0 / audioSource.PCM.SampleRate));

                    CUEToolsFormat fmt;
                    if (encoderFormat == null)
                    {
                        if (destFile == "-" || destFile == "nul")
                            encoderFormat = "wav";
                        else
                        {
                            string extension = Path.GetExtension(destFile).ToLower();
                            if (!extension.StartsWith("."))
                                throw new Exception("Unknown encoder format: " + destFile);
                            encoderFormat = extension.Substring(1);
                        }
                    }
                    if (!config.formats.TryGetValue(encoderFormat, out fmt))
                        throw new Exception("Unsupported encoder format: " + encoderFormat);
                    CUEToolsUDC encoder =
                        audioEncoderType == AudioEncoderType.Lossless ? Program.GetEncoder(config, fmt, true, encoderName) :
                        audioEncoderType == AudioEncoderType.Lossy ? Program.GetEncoder(config, fmt, false, encoderName) :
                        Program.GetEncoder(config, fmt, true, encoderName) ?? Program.GetEncoder(config, fmt, false, encoderName);
                    if (encoder == null)
                    {
                        var lst = new List<CUEToolsUDC>(config.encoders).FindAll(
                            e => e.extension == fmt.extension && (audioEncoderType == AudioEncoderType.NoAudio || audioEncoderType == (e.Lossless ? AudioEncoderType.Lossless : AudioEncoderType.Lossy))).
                                ConvertAll(e => e.Name + (e.Lossless ? " (lossless)" : " (lossy)"));
                        throw new Exception("Encoders available for format " + fmt.extension + ": " + (lst.Count == 0 ? "none" : string.Join(", ", lst.ToArray())));
                    }
                    var settings = encoder.settings.Clone();
                    settings.PCM = audioSource.PCM;
                    settings.Padding = padding;
                    settings.EncoderMode = encoderMode ?? settings.EncoderMode;
                    object o = null;
                    try
                    {                        
                        o = destFile == "-" ? Activator.CreateInstance(encoder.type, "", Console.OpenStandardOutput(), settings) :
                            destFile == "nul" ? Activator.CreateInstance(encoder.type, "", new NullStream(), settings) :
                            Activator.CreateInstance(encoder.type, destFile, settings);
                    }
                    catch (System.Reflection.TargetInvocationException ex)
                    {
                        throw ex.InnerException;
                    }
                    if (o == null || !(o is IAudioDest))
                        throw new Exception("Unsupported audio type: " + destFile + ": " + encoder.type.FullName);
                    audioDest = o as IAudioDest;
                    audioDest.FinalSampleCount = audioSource.Length;

                    bool keepRunning = true;
                    Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
                    {
                        keepRunning = false;
                        if (e.SpecialKey == ConsoleSpecialKey.ControlC)
                            e.Cancel = true;
                        else
                            audioDest.Delete();
                    };

                    while (audioSource.Read(buff, -1) != 0)
                    {
                        audioDest.Write(buff);
                        TimeSpan elapsed = DateTime.Now - start;
                        if ((elapsed - lastPrint).TotalMilliseconds > 60)
                        {
                            long length = audioSource.Length;
                            if (length < 0 && sourceInfo != null) length = (long)(sourceInfo.Properties.Duration.TotalMilliseconds * audioSource.PCM.SampleRate / 1000);
                            if (length < audioSource.Position) length = audioSource.Position;
                            if (length < 1) length = 1;
                            Console.Error.Write("\rProgress  : {0:00}%; {1:0.00}x; {2}/{3}",
                                100.0 * audioSource.Position / length,
                                audioSource.Position / elapsed.TotalSeconds / audioSource.PCM.SampleRate,
                                elapsed,
                                TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds / audioSource.Position * length)
                                );
                            lastPrint = elapsed;
                        }
                        if (!keepRunning)
                            throw new Exception("Aborted");
                    }

                    TimeSpan totalElapsed = DateTime.Now - start;
                    Console.Error.Write("\r                                                                         \r");
                    Console.Error.WriteLine("Results   : {0:0.00}x; {1}",
                        audioSource.Position / totalElapsed.TotalSeconds / audioSource.PCM.SampleRate,
                        totalElapsed
                        );
                }
                catch (Exception ex)
                {
                    if (audioSource != null) audioSource.Close();
                    if (audioDest != null) audioDest.Delete();
                    throw ex;
                }
                audioSource.Close();
                audioDest.Close();

                if (sourceFile != "-" && destFile != "-" && destFile != "nul")
                {
                    TagLib.File destInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(destFile));
                    if (Tagging.UpdateTags(destInfo, Tagging.Analyze(sourceInfo), config, false))
                    {
                        sourceInfo.Tag.CopyTo(destInfo.Tag, true);
                        destInfo.Tag.Pictures = sourceInfo.Tag.Pictures;
                        destInfo.Save();
                    }
                }
            }
#if !DEBUG
            catch (Exception ex)
            {
                Console.Error.Write("\r                                                                         \r");
                Console.Error.WriteLine("Error     : {0}", ex.Message);
                return 1;
                //Console.WriteLine("{0}", ex.StackTrace);
            }
#endif
            return 0;
        }
    }
}
