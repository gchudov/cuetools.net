using System;
using System.IO;
using CUETools.CTDB;
using CUETools.Processor;

namespace ArCueDotNet
{
    class Program
    {
        static int Main(string[] args)
        {
            bool ok = true;
            int offset = 0;
            bool verbose = false;
            string pathIn = null;
            for (int arg = 0; arg < args.Length; arg++)
            {
                if (args[arg].Length == 0)
                    ok = false;
                else if ((args[arg] == "-O" || args[arg] == "--offset") && ++arg < args.Length)
                    ok = int.TryParse(args[arg], out offset);
                else if ((args[arg] == "-v" || args[arg] == "--verbose"))
                    verbose = true;
                else if (args[arg][0] != '-' && pathIn == null)
                    pathIn = args[arg];
                else
                    ok = false;
                if (!ok)
                    break;
            }

            if (!ok || pathIn == null)
            {
                Console.SetOut(Console.Error);
                Console.WriteLine("Usage    : CUETools.ARCUE.exe [options] <filename>");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine();
                Console.WriteLine(" -O, --offset <samples>   Use specific offset;");
                Console.WriteLine(" -v, --verbose            Verbose mode");
                return 1;
            }
            if (!File.Exists(pathIn))
            {
                Console.SetOut(Console.Error);
                Console.WriteLine("Input CUE Sheet not found.");
                return 2;
            }

            CUEConfig config = new CUEConfig();
            config.writeArLogOnVerify = false;
            config.writeArTagsOnVerify = false;
            config.autoCorrectFilenames = true;
            config.extractAlbumArt = false;
            config.embedAlbumArt = false;
            config.advanced.DetailedCTDBLog = verbose;

            string accurateRipLog;
            try
            {
                CUESheet cueSheet = new CUESheet(config);
                cueSheet.Action = CUEAction.Verify;
                //cueSheet.OutputStyle = CUEStyle.SingleFile;
                cueSheet.WriteOffset = offset;
                cueSheet.Open(pathIn);
                cueSheet.UseAccurateRip();
                cueSheet.UseCUEToolsDB("ARCUE " + CUESheet.CUEToolsVersion, null, true, CTDBMetadataSearch.None);
                cueSheet.GenerateFilenames(AudioEncoderType.NoAudio, "dummy", pathIn);
                cueSheet.Go();

                accurateRipLog = CUESheetLogWriter.GetAccurateRipLog(cueSheet);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
                return 3;
            }

            Console.Write(accurateRipLog);
            return 0;
        }
    }
}
