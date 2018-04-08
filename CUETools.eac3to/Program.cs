using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.CTDB;
using CUETools.Processor;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CUETools.eac3to
{
    class Program
    {
        static void Usage()
        {
            Console.Error.WriteLine("CUETools.eac3to, Copyright (C) 2018 Grigory Chudov.");
            Console.Error.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
            Console.Error.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");
            Console.Error.WriteLine();
            Console.Error.WriteLine("Usage     : CUETools.eac3to.exe [options] <sourcefile> [trackno:] [destfile]");
            Console.Error.WriteLine();
            Console.Error.WriteLine("Options:");
            Console.Error.WriteLine();
            Console.Error.WriteLine(" --ctdb                 Query CTDB for metadata.");
            Console.Error.WriteLine(" --encoder <name>       Use non-default encoder.");
            Console.Error.WriteLine(" --encoder-format <ext> Use encoder format different from file extension.");
            Console.Error.WriteLine(" --lossy                Use lossy encoder/mode.");
            Console.Error.WriteLine(" --lossless             Use lossless encoder/mode (default).");
            Console.Error.WriteLine(" -p #                   Padding bytes.");
            Console.Error.WriteLine(" -m <mode>              Encoder mode (0..8 for flac, V0..V9 for mp3, etc)");
            Console.Error.WriteLine();
        }

        public static AudioEncoderSettingsViewModel GetEncoder(CUEToolsCodecsConfig config, CUEToolsFormat fmt, bool lossless, string chosenEncoder)
        {
            AudioEncoderSettingsViewModel tmpEncoder;
            return chosenEncoder != null ?
                (config.encodersViewModel.TryGetValue(fmt.extension, lossless, chosenEncoder, out tmpEncoder) ? tmpEncoder : null) :
                (lossless ? fmt.encoderLossless : fmt.encoderLossy);
        }

        static int Main(string[] args)
        {
            bool ok = true;
            string sourceFile = null, destFile = null;
            int padding = 8192;
            int stream = 0;
            string encoderMode = null;
            string encoderName = null;
            string encoderFormat = null;
            AudioEncoderType audioEncoderType = AudioEncoderType.NoAudio;
            bool queryMeta = false;

            for (int arg = 0; arg < args.Length; arg++)
            {
                if (args[arg].Length == 0)
                    ok = false;
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
                else if (args[arg] == "--ctdb")
                    queryMeta = true;
                else if ((args[arg][0] != '-' || args[arg] == "-") && sourceFile == null)
                    sourceFile = args[arg];
                else if ((args[arg][0] != '-' || args[arg] == "-") && sourceFile != null && destFile == null)
                {
                    destFile = args[arg];
                    var x = destFile.Split(':');
                    if (x.Length > 1)
                    {
                        stream = int.Parse(x[0]);
                        if (x[1] != "")
                        {
                            destFile = x[1];
                        }
                        else
                        {
                            arg++;
                            if (arg >= args.Length)
                            {
                                ok = false;
                                break;
                            }
                            destFile = args[arg];
                        }
                    }
                }
                else
                    ok = false;
                if (!ok)
                    break;
            }

            if (!ok || sourceFile == null)
            {
                Usage();
                return 22;
            }

            if (destFile != null && destFile != "-" && destFile != "nul" && File.Exists(destFile))
            {
                Console.Error.WriteLine("Error: file {0} already exists.", destFile);
                return 17;
            }

            DateTime start = DateTime.Now;
            TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
            var config = new CUEConfigAdvanced();
            config.Init();

#if !DEBUG
            try
#endif
            {
                IAudioSource audioSource = null;
                IAudioDest audioDest = null;
                var videos = new List<Codecs.MPEG.MPLS.MPLSStream>();
                var audios = new List<Codecs.MPEG.MPLS.MPLSStream>();
                List<uint> chapters;
                TimeSpan duration;
                TagLib.UserDefined.AdditionalFileTypes.Config = config;

#if !DEBUG
                try
#endif
                {
                    if (true)
                    {
                        var mpls = new Codecs.MPEG.MPLS.AudioDecoder(new Codecs.MPEG.MPLS.DecoderSettings(), sourceFile, null);
                        audioSource = mpls;
                        Console.ForegroundColor = ConsoleColor.White;
                        int frameRate = 0;
                        bool interlaced = false;
                        chapters = mpls.Chapters;
                        mpls.MPLSHeader.play_item.ForEach(i => i.video.ForEach(v => { if (!videos.Exists(v1 => v1.pid == v.pid)) videos.Add(v); }));
                        mpls.MPLSHeader.play_item.ForEach(i => i.audio.ForEach(v => { if (!audios.Exists(v1 => v1.pid == v.pid)) audios.Add(v); }));
                        videos.ForEach(v => { frameRate = v.FrameRate; interlaced = v.Interlaced; });
                        Console.Error.WriteLine("M2TS, {0} video track{1}, {2} audio track{3}, {4}, {5}{6}",
                            videos.Count, videos.Count > 1 ? "s" : "",
                            audios.Count, audios.Count > 1 ? "s" : "",
                            CDImageLayout.TimeToString(mpls.Duration, "{0:0}:{1:00}:{2:00}"), frameRate * (interlaced ? 2 : 1), interlaced ? "i" : "p");
                        //foreach (var item in mpls.MPLSHeader.play_item)
                        //Console.Error.WriteLine("{0}.m2ts", item.clip_id);
                        {
                            Console.ForegroundColor = ConsoleColor.Gray;
                            int id = 1;
                            if (chapters.Count > 1)
                            {
                                Console.ForegroundColor = ConsoleColor.White;
                                Console.Error.Write(id++);
                                Console.Error.Write(": ");
                                Console.ForegroundColor = ConsoleColor.Gray;
                                Console.Error.WriteLine("Chapters, {0} chapters", chapters.Count - 1);
                            }
                            foreach (var video in videos)
                            {
                                Console.ForegroundColor = ConsoleColor.White;
                                Console.Error.Write(id++);
                                Console.Error.Write(": ");
                                Console.ForegroundColor = ConsoleColor.Gray;
                                Console.Error.WriteLine("{0}, {1}{2}", video.CodecString, video.FormatString, video.FrameRate * (video.Interlaced ? 2 : 1));
                            }
                            foreach (var audio in audios)
                            {
                                Console.ForegroundColor = ConsoleColor.White;
                                Console.Error.Write(id++);
                                Console.Error.Write(": ");
                                Console.ForegroundColor = ConsoleColor.Gray;
                                Console.Error.WriteLine("{0}, {1}, {2}, {3}", audio.CodecString, audio.LanguageString, audio.FormatString, audio.RateString);
                            }
                        }

                        duration = mpls.Duration;
                    }

                    if (destFile == null)
                        return 0;

                    string strtoc = "";
                    for (int i = 0; i < chapters.Count; i++)
                        strtoc += string.Format(" {0}", chapters[i] / 600);
                    strtoc = strtoc.Substring(1);
                    CDImageLayout toc = new CDImageLayout(strtoc);
                    CTDBResponseMeta meta = null;
                    if (queryMeta)
                    {
                        var ctdb = new CUEToolsDB(toc, null);
                        Console.Error.WriteLine("Contacting CTDB...");
                        ctdb.ContactDB(null, "CUETools.eac3to 2.1.7", "", false, true, CTDBMetadataSearch.Extensive);
                        foreach (var imeta in ctdb.Metadata)
                        {
                            meta = imeta;
                            break;
                        }
                    }

                    if (stream > 0)
                    {
                        int chapterStreams = chapters.Count > 1 ? 1 : 0;
                        if (stream <= chapterStreams)
                        {
                            if (destFile == "-" || destFile == "nul")
                            {
                                encoderFormat = "txt";
                            }
                            else
                            {
                                string extension = Path.GetExtension(destFile).ToLower();
                                if (!extension.StartsWith("."))
                                {
                                    encoderFormat = destFile;
                                    if (meta == null || meta.artist == null || meta.album == null)
                                        destFile = string.Format("{0}.{1}", Path.GetFileNameWithoutExtension(sourceFile), destFile);
                                    else
                                        destFile = string.Format("{0} - {1} - {2}.{3}", meta.artist, meta.year, meta.album, destFile);
                                }
                                else
                                    encoderFormat = extension.Substring(1);
                                if (encoderFormat != "txt" && encoderFormat != "cue")
                                    throw new Exception(string.Format("Unsupported chapters file format \"{0}\"", encoderFormat));
                            }

                            Console.Error.WriteLine("Creating file \"{0}\"...", destFile);

                            if (encoderFormat == "txt")
                            {
                                using (TextWriter sw = destFile == "nul" ? (TextWriter)new StringWriter() : destFile == "-" ? Console.Out : (TextWriter)new StreamWriter(destFile))
                                {
                                    for (int i = 0; i < chapters.Count - 1; i++)
                                    {
                                        sw.WriteLine("CHAPTER{0:00}={1}", i + 1,
                                            CDImageLayout.TimeToString(TimeSpan.FromSeconds(chapters[i] / 45000.0)));
                                        if (meta != null && meta.track.Length >= toc[i+1].Number)
                                            sw.WriteLine("CHAPTER{0:00}NAME={1}", i + 1, meta.track[(int)toc[i+1].Number - 1].name);
                                        else
                                            sw.WriteLine("CHAPTER{0:00}NAME=", i + 1);
                                    }
                                }
                                Console.BackgroundColor = ConsoleColor.DarkGreen;
                                Console.Error.Write("Done.");
                                Console.BackgroundColor = ConsoleColor.Black;
                                Console.Error.WriteLine();
                                return 0;
                            }

                            if (encoderFormat == "cue")
                            {
                                using (StreamWriter cueWriter = new StreamWriter(destFile, false, Encoding.UTF8))
                                {
                                    cueWriter.WriteLine("REM COMMENT \"{0}\"", "Created by CUETools.eac3to");
                                    if (meta != null && meta.year != null)
                                        cueWriter.WriteLine("REM DATE {0}", meta.year);
                                    else
                                        cueWriter.WriteLine("REM DATE XXXX");
                                    if (meta != null)
                                    {
                                        cueWriter.WriteLine("PERFORMER \"{0}\"", meta.artist);
                                        cueWriter.WriteLine("TITLE \"{0}\"", meta.album);
                                    }
                                    else
                                    {
                                        cueWriter.WriteLine("PERFORMER \"\"");
                                        cueWriter.WriteLine("TITLE \"\"");
                                    }
                                    if (meta != null)
                                    {
                                        //cueWriter.WriteLine("FILE \"{0}\" WAVE", Path.GetFileNameWithoutExtension(destFile) + (extension ?? ".wav"));
                                        cueWriter.WriteLine("FILE \"{0}\" WAVE", Path.GetFileNameWithoutExtension(destFile) + (".wav"));
                                    }
                                    else
                                    {
                                        cueWriter.WriteLine("FILE \"{0}\" WAVE", "");
                                    }
                                    for (int track = 1; track <= toc.TrackCount; track++)
                                        if (toc[track].IsAudio)
                                        {
                                            cueWriter.WriteLine("  TRACK {0:00} AUDIO", toc[track].Number);
                                            if (meta != null && meta.track.Length >= toc[track].Number)
                                            {
                                                cueWriter.WriteLine("    TITLE \"{0}\"", meta.track[(int)toc[track].Number - 1].name);
                                                if (meta.track[(int)toc[track].Number - 1].artist != null)
                                                    cueWriter.WriteLine("    PERFORMER \"{0}\"", meta.track[(int)toc[track].Number - 1].artist);
                                            }
                                            else
                                            {
                                                cueWriter.WriteLine("    TITLE \"\"");
                                            }
                                            if (toc[track].ISRC != null)
                                                cueWriter.WriteLine("    ISRC {0}", toc[track].ISRC);
                                            for (int index = toc[track].Pregap > 0 ? 0 : 1; index <= toc[track].LastIndex; index++)
                                                cueWriter.WriteLine("    INDEX {0:00} {1}", index, toc[track][index].MSF);
                                        }
                                }
                                Console.BackgroundColor = ConsoleColor.DarkGreen;
                                Console.Error.Write("Done.");
                                Console.BackgroundColor = ConsoleColor.Black;
                                Console.Error.WriteLine();
                                return 0;
                            }

                            throw new Exception("Unknown encoder format: " + destFile);
                        }
                        if (audioSource is Codecs.MPEG.MPLS.AudioDecoder)
                        {
                            if (stream - chapterStreams <= videos.Count)
                                throw new Exception("Video extraction not supported.");
                            if (stream - chapterStreams - videos.Count > audios.Count)
                                throw new Exception(string.Format("The source file doesn't contain a track with the number {0}.", stream));
                            int pid = audios[stream - chapterStreams - videos.Count - 1].pid;
                            (audioSource.Settings as Codecs.MPEG.MPLS.DecoderSettings).StreamId = pid;
                        }
                    }

                    AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);
                    Console.Error.WriteLine("Filename  : {0}", sourceFile);
                    Console.Error.WriteLine("File Info : {0}kHz; {1} channel; {2} bit; {3}", audioSource.PCM.SampleRate, audioSource.PCM.ChannelCount, audioSource.PCM.BitsPerSample,
                        duration);

                    CUEToolsFormat fmt;
                    if (encoderFormat == null)
                    {
                        if (destFile == "-" || destFile == "nul")
                            encoderFormat = "wav";
                        else
                        {
                            string extension = Path.GetExtension(destFile).ToLower();
                            if (!extension.StartsWith("."))
                            {
                                encoderFormat = destFile;
                                if (meta == null || meta.artist == null || meta.album == null)
                                    destFile = string.Format("{0} - {1}.{2}", Path.GetFileNameWithoutExtension(sourceFile), stream, destFile);
                                else
                                    destFile = string.Format("{0} - {1} - {2}.{3}", meta.artist, meta.year, meta.album, destFile);
                            }
                            else
                                encoderFormat = extension.Substring(1);
                            if (File.Exists(destFile))
                                throw new Exception(string.Format("Error: file {0} already exists.", destFile));
                        }
                    }
                    if (!config.formats.TryGetValue(encoderFormat, out fmt))
                        throw new Exception("Unsupported encoder format: " + encoderFormat);
                    AudioEncoderSettingsViewModel encoder =
                        audioEncoderType == AudioEncoderType.Lossless ? Program.GetEncoder(config, fmt, true, encoderName) :
                        audioEncoderType == AudioEncoderType.Lossy ? Program.GetEncoder(config, fmt, false, encoderName) :
                        Program.GetEncoder(config, fmt, true, encoderName) ?? Program.GetEncoder(config, fmt, false, encoderName);
                    if (encoder == null)
                    {
                        var lst = new List<AudioEncoderSettingsViewModel>(config.encodersViewModel).FindAll(
                            e => e.Extension == fmt.extension && (audioEncoderType == AudioEncoderType.NoAudio || audioEncoderType == (e.Lossless ? AudioEncoderType.Lossless : AudioEncoderType.Lossy))).
                                ConvertAll(e => e.Name + (e.Lossless ? " (lossless)" : " (lossy)"));
                        throw new Exception("Encoders available for format " + fmt.extension + ": " + (lst.Count == 0 ? "none" : string.Join(", ", lst.ToArray())));
                    }
                    Console.Error.WriteLine("Output {0}  : {1}", stream, destFile);
                    var settings = encoder.Settings.Clone();
                    settings.PCM = audioSource.PCM;
                    settings.Padding = padding;
                    settings.EncoderMode = encoderMode ?? settings.EncoderMode;
                    object o = null;
                    try
                    {
                        o = destFile == "-" ? Activator.CreateInstance(settings.EncoderType, settings, "", Console.OpenStandardOutput()) :
                            destFile == "nul" ? Activator.CreateInstance(settings.EncoderType, settings, "", new NullStream()) :
                            Activator.CreateInstance(settings.EncoderType, settings, destFile, null);
                    }
                    catch (System.Reflection.TargetInvocationException ex)
                    {
                        throw ex.InnerException;
                    }
                    if (o == null || !(o is IAudioDest))
                        throw new Exception("Unsupported audio type: " + destFile + ": " + settings.EncoderType.FullName);
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
                            long length = (long)(duration.TotalSeconds * audioSource.PCM.SampleRate);
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
#if !DEBUG
                catch (Exception ex)
                {
                    if (audioSource != null) audioSource.Close();
                    if (audioDest != null) audioDest.Delete();
                    throw ex;
                }
#endif
                audioSource.Close();
                audioDest.Close();

                if (sourceFile != "-" && destFile != "-" && destFile != "nul")
                {
                    //TagLib.File destInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(destFile));
                    //NameValueCollection tags;
                    //if (Tagging.UpdateTags(destInfo, tags, config, false))
                    //{
                    //    destInfo.Save();
                    //}
                }
            }
#if !DEBUG
            catch (Exception ex)
            {
                Console.Error.Write("\r                                                                         \r");
                Console.BackgroundColor = ConsoleColor.DarkRed;
                Console.Error.Write("Error     : {0}", ex.Message);
                Console.BackgroundColor = ConsoleColor.Black;
                Console.Error.WriteLine();
                return 1;
                //Console.WriteLine("{0}", ex.StackTrace);
            }
#endif
            return 0;
        }
    }
}
