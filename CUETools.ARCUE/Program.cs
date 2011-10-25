using System;
using System.IO;
using CUETools.Processor;

namespace ArCueDotNet
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("Usage: ArCueDotNet <filename>");
                return;
            }
            string pathIn = args[0];
            if (!File.Exists(pathIn))
            {
                Console.WriteLine("Input CUE Sheet not found.");
                return;
            }
            CUEConfig config = new CUEConfig();
            config.writeArLogOnVerify = false;
            config.writeArTagsOnVerify = false;
            config.autoCorrectFilenames = true;
            config.extractAlbumArt = false;
            config.embedAlbumArt = false;

            string accurateRipLog;
            try
            {
                CUESheet cueSheet = new CUESheet(config);
                cueSheet.Action = CUEAction.Verify;
                //cueSheet.OutputStyle = CUEStyle.SingleFile;
                cueSheet.Open(pathIn);
                cueSheet.UseAccurateRip();
                cueSheet.GenerateFilenames(AudioEncoderType.NoAudio, "dummy", pathIn);
                cueSheet.Go();

                accurateRipLog = CUESheetLogWriter.GetAccurateRipLog(cueSheet);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
                return;
            }

            Console.Write(accurateRipLog);
        }
    }
}
