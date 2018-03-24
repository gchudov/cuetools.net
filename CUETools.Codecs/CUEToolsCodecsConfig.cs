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
        public Dictionary<string, CUEToolsFormat> formats;
        public EncoderListViewModel encoders;
        public DecoderListViewModel decoders;

        public CUEToolsCodecsConfig(CUEToolsCodecsConfig src)
        {
            encoders = new EncoderListViewModel();
            foreach (var enc in src.encoders)
                encoders.Add(enc.Clone());
            decoders = new DecoderListViewModel();
            foreach (var dec in src.decoders)
                decoders.Add(dec.Clone());
            formats = new Dictionary<string, CUEToolsFormat>();
            foreach (var fmt in src.formats)
                formats.Add(fmt.Key, fmt.Value.Clone(this));
        }

        public CUEToolsCodecsConfig(List<Type> encs, List<Type> decs)
        {
            encoders = new EncoderListViewModel();
            foreach (Type type in encs)
                foreach (AudioEncoderClassAttribute enc in Attribute.GetCustomAttributes(type, typeof(AudioEncoderClassAttribute)))
                    try
                    {
                        encoders.Add(new AudioEncoderSettingsViewModel(enc));
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.WriteLine(ex.Message);
                    }

            decoders = new DecoderListViewModel();
            foreach (Type type in decs)
                foreach (AudioDecoderClassAttribute dec in Attribute.GetCustomAttributes(type, typeof(AudioDecoderClassAttribute)))
                    try
                    {
                        decoders.Add(new AudioDecoderSettingsViewModel(Activator.CreateInstance(dec.Settings) as AudioDecoderSettings));
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.WriteLine(ex.Message);
                    }

            if (Type.GetType("Mono.Runtime", false) == null)
            {
                encoders.Add(new AudioEncoderSettingsViewModel("flake", "flac", true, "0 1 2 3 4 5 6 7 8 9 10 11 12", "8", "flake.exe", "-%M - -o %O -p %P"));
                encoders.Add(new AudioEncoderSettingsViewModel("takc", "tak", true, "0 1 2 2e 2m 3 3e 3m 4 4e 4m", "2", "takc.exe", "-e -p%M -overwrite - %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("ffmpeg alac", "m4a", true, "", "", "ffmpeg.exe", "-i - -f ipod -acodec alac -y %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("VBR (lame.exe)", "mp3", false, "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2", "lame.exe", "--vbr-new -%M - %O"));
				encoders.Add(new AudioEncoderSettingsViewModel("CBR (lame.exe)", "mp3", false, "96 128 192 256 320", "256", "lame.exe", "-m s -q 0 -b %M --noreplaygain - %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("oggenc", "ogg", false, "-1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8", "3", "oggenc.exe", "-q %M - -o %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("opusenc", "opus", false, "6 16 32 48 64 96 128 192 256", "128", "opusenc.exe", "--bitrate %M - %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("nero aac", "m4a", false, "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9", "0.4", "neroAacEnc.exe", "-q %M -if - -of %O"));
                encoders.Add(new AudioEncoderSettingsViewModel("qaac tvbr", "m4a", false, "10 20 30 40 50 60 70 80 90 100 110 127", "80", "qaac.exe", "-s -V %M -q 2 - -o %O"));

                decoders.Add(new AudioDecoderSettingsViewModel(new CommandLine.DecoderSettings("takc", "tak", "takc.exe", "-d %I -")));
                decoders.Add(new AudioDecoderSettingsViewModel(new CommandLine.DecoderSettings("ffmpeg alac", "m4a", "ffmpeg.exe", "-v 0 -i %I -f wav -")));
            }
            else
            {
                // !!!
            }

            formats = new Dictionary<string, CUEToolsFormat>();
            formats.Add("flac", new CUEToolsFormat("flac", CUEToolsTagger.TagLibSharp, true, false, true, true, encoders.GetDefault("flac", true), null, decoders.GetDefault("flac", true)));
            formats.Add("wv", new CUEToolsFormat("wv", CUEToolsTagger.TagLibSharp, true, false, true, true, encoders.GetDefault("wv", true), null, decoders.GetDefault("wv", true)));
            formats.Add("ape", new CUEToolsFormat("ape", CUEToolsTagger.TagLibSharp, true, false, true, true, encoders.GetDefault("ape", true), null, decoders.GetDefault("ape", true)));
            formats.Add("tta", new CUEToolsFormat("tta", CUEToolsTagger.APEv2, true, false, false, true, encoders.GetDefault("tta", true), null, decoders.GetDefault("tta", true)));
            formats.Add("m2ts", new CUEToolsFormat("m2ts", CUEToolsTagger.APEv2, true, false, false, true, null, null, decoders.GetDefault("m2ts", true)));
            formats.Add("mpls", new CUEToolsFormat("mpls", CUEToolsTagger.APEv2, true, false, false, true, null, null, decoders.GetDefault("mpls", true)));
            formats.Add("wav", new CUEToolsFormat("wav", CUEToolsTagger.TagLibSharp, true, false, false, true, encoders.GetDefault("wav", true), null, decoders.GetDefault("wav", true)));
            formats.Add("m4a", new CUEToolsFormat("m4a", CUEToolsTagger.TagLibSharp, true, true, false, true, encoders.GetDefault("m4a", true), encoders.GetDefault("m4a", false), decoders.GetDefault("m4a", true)));
            formats.Add("tak", new CUEToolsFormat("tak", CUEToolsTagger.APEv2, true, false, true, true, encoders.GetDefault("tak", true), null, decoders.GetDefault("tak", true)));
            formats.Add("wma", new CUEToolsFormat("wma", CUEToolsTagger.TagLibSharp, true, true, false, true, encoders.GetDefault("wma", true), encoders.GetDefault("wma", false), decoders.GetDefault("wma", true)));
            formats.Add("mp3", new CUEToolsFormat("mp3", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encoders.GetDefault("mp3", false), null));
            formats.Add("ogg", new CUEToolsFormat("ogg", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encoders.GetDefault("ogg", false), null));
            formats.Add("opus", new CUEToolsFormat("opus", CUEToolsTagger.TagLibSharp, false, true, false, true, null, encoders.GetDefault("opus", false), null));
        }
    }
}
