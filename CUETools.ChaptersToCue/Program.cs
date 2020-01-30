// ****************************************************************************
// 
// ChaptersToCue
// Copyright (C) 2018-2020 Grigory Chudov (gchudov@gmail.com)
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// ****************************************************************************

using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using CUETools.CDImage;
using CUETools.CTDB;
using CUETools.Codecs;

namespace CUETools.ChaptersToCue
{
	class Program
	{
		static void Usage()
		{
            Console.WriteLine("Usage    : CUETools.ChaptersToCue.exe <options>");
			Console.WriteLine();
            Console.WriteLine("-i, --input <file>           File containing chapter times");
            Console.WriteLine("-o, --output <file.cue>      Output filename; Default: Artist - year - album.cue");
            Console.WriteLine("-t, --tracks <file>          File containing track filenames");
            Console.WriteLine("-m, --meta                   query CTDB for metadata;");
            Console.WriteLine("--celltimes <fps>            Input is in Celltimes format;");
            Console.WriteLine("--image <file>               Use image file to calculate duration");
		}

        static void Main(string[] args)
        {
            TextWriter stdout = Console.Out;
            Console.SetOut(Console.Error);
            Console.WriteLine("CUETools.ChaptersToCue v2.1.7 Copyright (C) 2017-2020 Grigory Chudov");
            Console.WriteLine("This is free software under the GNU GPLv3+ license; There is NO WARRANTY, to");
            Console.WriteLine("the extent permitted by law. <http://www.gnu.org/licenses/> for details.");

            bool queryMeta = false;
            bool celltimes = false;
            int fps_mul = 0;
            int fps_div = 1;
            string inputPath = "-";
            string outputPath = null;
            string tracksPath = null;
            string imagePath = null;
            for (int arg = 0; arg < args.Length; arg++)
            {
                bool ok = true;
                if ((args[arg] == "-i" || args[arg] == "--input") && ++arg < args.Length)
                    inputPath = args[arg];
                else if ((args[arg] == "-o" || args[arg] == "--output") && ++arg < args.Length)
                    outputPath = args[arg];
                else if ((args[arg] == "-t" || args[arg] == "--tracks") && ++arg < args.Length)
                    tracksPath = args[arg];
                else if (args[arg] == "--image" && ++arg < args.Length)
                    imagePath = args[arg];
                else if (args[arg] == "-m" || args[arg] == "--meta")
                    queryMeta = true;
                else if (args[arg] == "--celltimes" && ++arg < args.Length)
                {
                    celltimes = true;
                    ok = int.TryParse(args[arg], out fps_mul);
                    if (ok && fps_mul == 30)
                    {
                        fps_mul = 30000;
                        fps_div = 1001;
                    }
                }
                else
                    ok = false;
                if (!ok)
                {
                    Usage();
                    return;
                }
            }

            string strtoc = "";
            string extension = null;
            if (tracksPath != null)
            {
                //CUEToolsCodecsConfig config = new CUEConfig();
                //TagLib.UserDefined.AdditionalFileTypes.Config = config;
                TimeSpan pos = new TimeSpan(0);
                using (TextReader sr = tracksPath == "-" ? Console.In : new StreamReader(tracksPath))
                {
                    while (sr.Peek() >= 0)
                    {
                        string line = sr.ReadLine();
                        extension = Path.GetExtension(line);
                        TagLib.File sourceInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(line));
                        strtoc += string.Format(" {0}", (int)(pos.TotalSeconds * 75));
                        pos += sourceInfo.Properties.Duration;
                    }
                }
                strtoc += string.Format(" {0}", (int)(pos.TotalSeconds * 75));
            }
            else
            {
                using (TextReader sr = inputPath == "-" ? Console.In : new StreamReader(inputPath))
                {
                    if (celltimes)
                    {
                        strtoc += string.Format(" {0}", 0);
                        while (sr.Peek() >= 0)
                        {
                            string line = sr.ReadLine();
                            strtoc += string.Format(" {0}", long.Parse(line) * 75 * fps_div / fps_mul);
                        }
                    }
                    else
                    {
                        while (sr.Peek() >= 0)
                        {
                            string line = sr.ReadLine();
                            Regex r = new Regex(@"^CHAPTER(?<number>\d\d)(?<option>[^=]*)=((?<hour>\d+):(?<minute>\d+):(?<second>\d+)\.(?<millisecond>\d+)|(?<text>))");
                            Match m = r.Match(line);
                            if (!m.Success)
                            {
                                Console.Error.WriteLine("Invalid input format: {0}", line);
                                return;
                            }
                            var option = m.Result("${option}");
                            if (option != "") continue;
                            var chapter = int.Parse(m.Result("${number}"));
                            var hour = int.Parse(m.Result("${hour}"));
                            var minute = int.Parse(m.Result("${minute}"));
                            var second = int.Parse(m.Result("${second}"));
                            var millisecond = int.Parse(m.Result("${millisecond}"));
                            strtoc += string.Format(" {0}", ((hour * 60 + minute) * 60 + second) * 75 + millisecond * 75 / 1000);
                        }
                        if (imagePath != null)
                        {
                            TagLib.File sourceInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(imagePath));
                            strtoc += string.Format(" {0}", (int)(sourceInfo.Properties.Duration.TotalSeconds * 75));
                        }
                        else
                        {
                            strtoc += string.Format(" {0}", (int)(75 * 60 * 60 * 2));
                            //strtoc += string.Format(" {0}", (int)(75 * 259570688.0 / 96000));
                        }
                    }
                }
            }
            strtoc = strtoc.Substring(1);
            CDImageLayout toc = new CDImageLayout(strtoc);
            CTDBResponseMeta meta = null;
            if (queryMeta)
            {
                var ctdb = new CUEToolsDB(toc, null);
                ctdb.ContactDB(null, "CUETools.ChaptersToCue 2.1.7", "", false, true, CTDBMetadataSearch.Extensive);
                foreach (var imeta in ctdb.Metadata)
                {
                    meta = imeta;
                    break;
                }
            }

            if (outputPath == null)
            {
                if (meta != null)
                    outputPath = string.Format("{0} - {1} - {2}.cue", meta.artist ?? "Unknown Artist", meta.year ?? "XXXX", meta.album ?? "Unknown Album");
                else
                    outputPath = "unknown.cue";
            }

            StringWriter cueWriter = new StringWriter();
            cueWriter.WriteLine("REM COMMENT \"{0}\"", "Created by ChaptersToCue");
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
                cueWriter.WriteLine("FILE \"{0}\" WAVE", Path.GetFileNameWithoutExtension(outputPath) + (extension ?? ".wav"));
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
            cueWriter.Close();
            if (outputPath == "-")
            {
                stdout.Write(cueWriter.ToString());
            }
            else
            {
                try
                {
                    using (var ofs = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write))
                    using (var cueFile = new StreamWriter(ofs))
                    {
                        cueFile.Write(cueWriter.ToString());
                        cueFile.Close();
                    }
                }
                catch (System.IO.IOException ex)
                {
                    Console.Error.WriteLine("{0}", ex.Message);
                }
            }
        }
	}
}
