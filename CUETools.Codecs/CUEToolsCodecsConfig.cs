using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Serialization;

namespace CUETools.Codecs
{
    public class CUEToolsCodecsConfig
    {
        [JsonIgnore]
        public Dictionary<string, CUEToolsFormat> formats;
        public List<IAudioEncoderSettings> encoders;
        public List<IAudioDecoderSettings> decoders;
        [JsonIgnore]
        public EncoderListViewModel encodersViewModel;
        [JsonIgnore]
        public DecoderListViewModel decodersViewModel;

        public CUEToolsCodecsConfig()
        {
            encoders = new List<IAudioEncoderSettings>();
            decoders = new List<IAudioDecoderSettings>();
            encodersViewModel = new EncoderListViewModel(encoders);
            decodersViewModel = new DecoderListViewModel(decoders);
            formats = new Dictionary<string, CUEToolsFormat>();
        }

        public CUEToolsCodecsConfig(CUEToolsCodecsConfig src)
        {
            encoders = new List<IAudioEncoderSettings>();
            decoders = new List<IAudioDecoderSettings>();
            src.encoders.ForEach(item => encoders.Add(item.Clone()));
            src.decoders.ForEach(item => decoders.Add(item.Clone()));
            encodersViewModel = new EncoderListViewModel(encoders);
            decodersViewModel = new DecoderListViewModel(decoders);
            formats = new Dictionary<string, CUEToolsFormat>();
            foreach (var fmt in src.formats)
                formats.Add(fmt.Key, fmt.Value.Clone(this));
        }

        public void Init(List<IAudioEncoderSettings> src_encoders, List<IAudioDecoderSettings> src_decoders)
        {
            encoders = new List<IAudioEncoderSettings>();
            decoders = new List<IAudioDecoderSettings>();
            src_encoders.ForEach(item => encoders.Add(item.Clone()));
            src_decoders.ForEach(item => decoders.Add(item.Clone()));

            if (Type.GetType("Mono.Runtime", false) == null)
            {
                encoders.Add(new CommandLine.EncoderSettings("flac.exe", "flac", true, "0 1 2 3 4 5 6 7 8", "5", "flac.exe", "-%M -P %P - -o %O"));
                encoders.Add(new CommandLine.EncoderSettings("flake.exe", "flac", true, "0 1 2 3 4 5 6 7 8 9 10 11 12", "8", "flake.exe", "-%M - -o %O -p %P"));
                encoders.Add(new CommandLine.EncoderSettings("takc.exe", "tak", true, "0 1 2 2e 2m 3 3e 3m 4 4e 4m", "2", "takc.exe", "-e -p%M -overwrite - %O"));
                encoders.Add(new CommandLine.EncoderSettings("ffmpeg.exe", "m4a", true, "", "", "ffmpeg.exe", "-i - -f ipod -acodec alac -y %O"));
                encoders.Add(new CommandLine.EncoderSettings("lame.exe (VBR)", "mp3", false, "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2", "lame.exe", "--vbr-new -%M - %O"));
                encoders.Add(new CommandLine.EncoderSettings("lame.exe (CBR)", "mp3", false, "96 128 192 256 320", "256", "lame.exe", "-m s -q 0 -b %M --noreplaygain - %O"));
                encoders.Add(new CommandLine.EncoderSettings("oggenc.exe", "ogg", false, "-1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8", "3", "oggenc.exe", "-q %M - -o %O"));
                encoders.Add(new CommandLine.EncoderSettings("opusenc.exe", "opus", false, "6 16 32 48 64 96 128 192 256", "128", "opusenc.exe", "--bitrate %M - %O"));
                encoders.Add(new CommandLine.EncoderSettings("neroAacEnc.exe", "m4a", false, "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9", "0.4", "neroAacEnc.exe", "-q %M -if - -of %O"));
                encoders.Add(new CommandLine.EncoderSettings("qaac.exe (tvbr)", "m4a", false, "10 20 30 40 50 60 70 80 90 100 110 127", "80", "qaac.exe", "-s -V %M -q 2 - -o %O"));

                decoders.Add(new CommandLine.DecoderSettings("takc.exe", "tak", "takc.exe", "-d %I -"));
                decoders.Add(new CommandLine.DecoderSettings("ffmpeg.exe", "m4a", "ffmpeg.exe", "-v 0 -i %I -f wav -"));
            }
            else
            {
                // !!!
            }

            encodersViewModel = new EncoderListViewModel(encoders);
            decodersViewModel = new DecoderListViewModel(decoders);

            formats = new Dictionary<string, CUEToolsFormat>();
            formats.Add("flac", new CUEToolsFormat("flac", CUEToolsTagger.TagLibSharp, true, false, true, true, encodersViewModel.GetDefault("flac", true), null, decodersViewModel.GetDefault("flac")));
            formats.Add("wv", new CUEToolsFormat("wv", CUEToolsTagger.TagLibSharp, true, false, true, true, encodersViewModel.GetDefault("wv", true), null, decodersViewModel.GetDefault("wv")));
            formats.Add("ape", new CUEToolsFormat("ape", CUEToolsTagger.TagLibSharp, true, false, true, true, encodersViewModel.GetDefault("ape", true), null, decodersViewModel.GetDefault("ape")));
            formats.Add("tta", new CUEToolsFormat("tta", CUEToolsTagger.APEv2, true, false, false, true, encodersViewModel.GetDefault("tta", true), null, decodersViewModel.GetDefault("tta")));
            formats.Add("m2ts", new CUEToolsFormat("m2ts", CUEToolsTagger.APEv2, true, false, false, true, null, null, decodersViewModel.GetDefault("m2ts")));
            formats.Add("mpls", new CUEToolsFormat("mpls", CUEToolsTagger.APEv2, true, false, false, true, null, null, decodersViewModel.GetDefault("mpls")));
            formats.Add("wav", new CUEToolsFormat("wav", CUEToolsTagger.TagLibSharp, true, false, false, true, encodersViewModel.GetDefault("wav", true), null, decodersViewModel.GetDefault("wav")));
            formats.Add("m4a", new CUEToolsFormat("m4a", CUEToolsTagger.TagLibSharp, true, true, false, true, encodersViewModel.GetDefault("m4a", true), encodersViewModel.GetDefault("m4a", false), decodersViewModel.GetDefault("m4a")));
            formats.Add("tak", new CUEToolsFormat("tak", CUEToolsTagger.APEv2, true, false, true, true, encodersViewModel.GetDefault("tak", true), null, decodersViewModel.GetDefault("tak")));
            formats.Add("wma", new CUEToolsFormat("wma", CUEToolsTagger.TagLibSharp, true, true, false, true, encodersViewModel.GetDefault("wma", true), encodersViewModel.GetDefault("wma", false), decodersViewModel.GetDefault("wma")));
            formats.Add("mp3", new CUEToolsFormat("mp3", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encodersViewModel.GetDefault("mp3", false), null));
            formats.Add("ogg", new CUEToolsFormat("ogg", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encodersViewModel.GetDefault("ogg", false), null));
            formats.Add("opus", new CUEToolsFormat("opus", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encodersViewModel.GetDefault("opus", false), null));
            formats.Add("mlp", new CUEToolsFormat("mlp", CUEToolsTagger.APEv2, true, false, false, false, null, null, decodersViewModel.GetDefault("mlp")));
            formats.Add("aob", new CUEToolsFormat("aob", CUEToolsTagger.APEv2, true, false, false, false, null, null, decodersViewModel.GetDefault("aob")));
        }
    }
}
