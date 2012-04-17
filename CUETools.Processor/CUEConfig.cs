using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Serialization;
using CUETools.Codecs;
using CUETools.Processor.Settings;

namespace CUETools.Processor
{
    public class CUEConfig
    {
        public readonly static XmlSerializerNamespaces xmlEmptyNamespaces = new XmlSerializerNamespaces(new XmlQualifiedName[] { XmlQualifiedName.Empty });
        public readonly static XmlWriterSettings xmlEmptySettings = new XmlWriterSettings { Indent = true, OmitXmlDeclaration = true };

        public uint fixOffsetMinimumConfidence;
        public uint fixOffsetMinimumTracksPercent;
        public uint encodeWhenConfidence;
        public uint encodeWhenPercent;
        public bool encodeWhenZeroOffset;
        public bool writeArTagsOnVerify;
        public bool writeArLogOnVerify;
        public bool writeArTagsOnEncode;
        public bool writeArLogOnConvert;
        public bool fixOffset;
        public bool noUnverifiedOutput;
        public bool autoCorrectFilenames;
        public bool detectGaps;
        public bool preserveHTOA;
        public bool keepOriginalFilenames;
        public string trackFilenameFormat;
        public string singleFilenameFormat;
        public bool removeSpecial;
        public string specialExceptions;
        public bool replaceSpaces;
        public bool embedLog;
        public bool extractLog;
        public bool fillUpCUE;
        public bool overwriteCUEData;
        public bool filenamesANSISafe;
        public bool bruteForceDTL;
        public bool createEACLOG;
        public bool detectHDCD;
        public bool decodeHDCD;
        public bool wait750FramesForHDCD;
        public bool createM3U;
        public bool createCUEFileWhenEmbedded;
        public bool truncate4608ExtraSamples;
        public int lossyWAVQuality;
        public bool decodeHDCDtoLW16;
        public bool decodeHDCDto24bit;
        public bool oneInstance;
        public bool checkForUpdates;
        public string language;
        public Dictionary<string, CUEToolsFormat> formats;
        public CUEToolsUDCList encoders;
        public Dictionary<string, CUEToolsUDC> decoders;
        public Dictionary<string, CUEToolsScript> scripts;
        public string defaultVerifyScript;
        public string defaultEncodeScript;
        public bool writeBasicTagsFromCUEData;
        public bool copyBasicTags;
        public bool copyUnknownTags;
        public bool embedAlbumArt;
        public bool extractAlbumArt;
        public bool arLogToSourceFolder;
        public bool arLogVerbose;
        public bool fixOffsetToNearest;
        public int maxAlbumArtSize;
        public CUEStyle gapsHandling;
        public bool separateDecodingThread;

        public CUEConfigAdvanced advanced { get; private set; }
        public bool CopyAlbumArt { get; set; }
        public string ArLogFilenameFormat { get; set; }
        public string AlArtFilenameFormat { get; set; }
        public CUEToolsUDCList Encoders
        {
            get { return encoders; }
        }

        public CUEConfig()
        {
            fixOffsetMinimumConfidence = 2;
            fixOffsetMinimumTracksPercent = 51;
            encodeWhenConfidence = 2;
            encodeWhenPercent = 100;
            encodeWhenZeroOffset = false;
            fixOffset = false;
            noUnverifiedOutput = false;
            writeArTagsOnEncode = false;
            writeArLogOnConvert = true;
            writeArTagsOnVerify = false;
            writeArLogOnVerify = false;

            autoCorrectFilenames = true;
            preserveHTOA = true;
            detectGaps = true;
            keepOriginalFilenames = false;
            trackFilenameFormat = "%tracknumber%. %title%";
            singleFilenameFormat = "%filename%";
            removeSpecial = false;
            specialExceptions = "-()";
            replaceSpaces = false;
            embedLog = true;
            extractLog = true;
            fillUpCUE = true;
            overwriteCUEData = false;
            filenamesANSISafe = true;
            bruteForceDTL = false;
            createEACLOG = true;
            detectHDCD = true;
            wait750FramesForHDCD = true;
            decodeHDCD = false;
            createM3U = false;
            createCUEFileWhenEmbedded = true;
            truncate4608ExtraSamples = true;
            lossyWAVQuality = 5;
            decodeHDCDtoLW16 = false;
            decodeHDCDto24bit = true;

            oneInstance = true;
            checkForUpdates = true;

            writeBasicTagsFromCUEData = true;
            copyBasicTags = true;
            copyUnknownTags = true;
            CopyAlbumArt = true;
            embedAlbumArt = true;
            extractAlbumArt = true;
            maxAlbumArtSize = 300;

            arLogToSourceFolder = false;
            arLogVerbose = true;
            fixOffsetToNearest = true;
            ArLogFilenameFormat = "%filename%.accurip";
            AlArtFilenameFormat = "folder.jpg";

            separateDecodingThread = true;

            gapsHandling = CUEStyle.GapsAppended;

            advanced = new CUEConfigAdvanced();

            language = Thread.CurrentThread.CurrentUICulture.Name;

            encoders = new CUEToolsUDCList();
            foreach (Type type in CUEProcessorPlugins.encs)
                foreach (AudioEncoderClass enc in Attribute.GetCustomAttributes(type, typeof(AudioEncoderClass)))
                    encoders.Add(new CUEToolsUDC(enc, type));
            decoders = new Dictionary<string, CUEToolsUDC>();
            foreach (Type type in CUEProcessorPlugins.decs)
            {
                AudioDecoderClass dec = Attribute.GetCustomAttribute(type, typeof(AudioDecoderClass)) as AudioDecoderClass;
                decoders.Add(dec.DecoderName, new CUEToolsUDC(dec, type));
            }
            if (Type.GetType("Mono.Runtime", false) == null)
            {
                encoders.Add(new CUEToolsUDC("flake", "flac", true, "0 1 2 3 4 5 6 7 8 9 10 11 12", "8", "flake.exe", "-%M - -o %O -p %P"));
                encoders.Add(new CUEToolsUDC("takc", "tak", true, "0 1 2 2e 2m 3 3e 3m 4 4e 4m", "2", "takc.exe", "-e -p%M -overwrite - %O"));
                encoders.Add(new CUEToolsUDC("ffmpeg alac", "m4a", true, "", "", "ffmpeg.exe", "-i - -f ipod -acodec alac -y %O"));
                encoders.Add(new CUEToolsUDC("VBR (lame.exe)", "mp3", false, "V9 V8 V7 V6 V5 V4 V3 V2 V1 V0", "V2", "lame.exe", "--vbr-new -%M - %O"));
				encoders.Add(new CUEToolsUDC("CBR (lame.exe)", "mp3", false, "96 128 192 256 320", "256", "lame.exe", "-m s -q 0 -b %M --noreplaygain - %O"));
                encoders.Add(new CUEToolsUDC("oggenc", "ogg", false, "-1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8", "3", "oggenc.exe", "-q %M - -o %O"));
                encoders.Add(new CUEToolsUDC("nero aac", "m4a", false, "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9", "0.4", "neroAacEnc.exe", "-q %M -if - -of %O"));
                encoders.Add(new CUEToolsUDC("qaac tvbr", "m4a", false, "10 20 30 40 50 60 70 80 90 100 110 127", "80", "qaac.exe", "-s -V %M -q 2 - -o %O"));

                decoders.Add("takc", new CUEToolsUDC("takc", "tak", true, "", "", "takc.exe", "-d %I -"));
                decoders.Add("ffmpeg alac", new CUEToolsUDC("ffmpeg alac", "m4a", true, "", "", "ffmpeg.exe", "%I -f wav -"));
            }
            else
            {
                // !!!
            }

            formats = new Dictionary<string, CUEToolsFormat>();
            formats.Add("flac", new CUEToolsFormat("flac", CUEToolsTagger.TagLibSharp, true, false, true, true, true, encoders.GetDefault("flac", true), null, GetDefaultDecoder("flac")));
            formats.Add("wv", new CUEToolsFormat("wv", CUEToolsTagger.TagLibSharp, true, false, true, true, true, encoders.GetDefault("wv", true), null, GetDefaultDecoder("wv")));
            formats.Add("ape", new CUEToolsFormat("ape", CUEToolsTagger.TagLibSharp, true, false, false, true, true, encoders.GetDefault("ape", true), null, GetDefaultDecoder("ape")));
            formats.Add("tta", new CUEToolsFormat("tta", CUEToolsTagger.APEv2, true, false, false, false, true, encoders.GetDefault("tta", true), null, GetDefaultDecoder("tta")));
            formats.Add("wav", new CUEToolsFormat("wav", CUEToolsTagger.TagLibSharp, true, false, true, false, true, encoders.GetDefault("wav", true), null, GetDefaultDecoder("wav")));
            formats.Add("m4a", new CUEToolsFormat("m4a", CUEToolsTagger.TagLibSharp, true, true, false, false, true, encoders.GetDefault("m4a", true), encoders.GetDefault("m4a", false), GetDefaultDecoder("m4a")));
            formats.Add("tak", new CUEToolsFormat("tak", CUEToolsTagger.APEv2, true, false, true, true, true, encoders.GetDefault("tak", true), null, "takc"));
            formats.Add("mp3", new CUEToolsFormat("mp3", CUEToolsTagger.TagLibSharp, false, true, false, false, true, null, encoders.GetDefault("mp3", false), null));
            formats.Add("ogg", new CUEToolsFormat("ogg", CUEToolsTagger.TagLibSharp, false, true, false, false, true, null, encoders.GetDefault("ogg", false), null));

            scripts = new Dictionary<string, CUEToolsScript>();
            scripts.Add("default", new CUEToolsScript("default", true,
                new CUEAction[] { CUEAction.Verify, CUEAction.Encode },
                "return processor.Go();"));
            scripts.Add("only if found", new CUEToolsScript("only if found", true,
                new CUEAction[] { CUEAction.Verify },
@"if (processor.ArVerify.AccResult != HttpStatusCode.OK)
	return processor.WriteReport(); 
return processor.Go();"));
            scripts.Add("fix offset", new CUEToolsScript("fix offset", true,
                new CUEAction[] { CUEAction.Encode },
@"if (processor.ArVerify.AccResult != HttpStatusCode.OK)
    return processor.WriteReport(); 
processor.WriteOffset = 0;
processor.Action = CUEAction.Verify;
string status = processor.Go();
uint tracksMatch;
int bestOffset;
processor.FindBestOffset(processor.Config.fixOffsetMinimumConfidence, !processor.Config.fixOffsetToNearest, out tracksMatch, out bestOffset);
if (tracksMatch * 100 < processor.Config.fixOffsetMinimumTracksPercent * processor.TrackCount)
    return status;
processor.WriteOffset = bestOffset;
processor.Action = CUEAction.Encode;
//MessageBox.Show(null, processor.AccurateRipLog, " + "\"Done\"" + @"MessageBoxButtons.OK, MessageBoxIcon.Information);
return processor.Go();
"));
            scripts.Add("encode if verified", new CUEToolsScript("encode if verified", true,
                new CUEAction[] { CUEAction.Encode },
@"if (processor.ArVerify.AccResult != HttpStatusCode.OK)
    return processor.WriteReport();
processor.Action = CUEAction.Verify;
string status = processor.Go();
uint tracksMatch;
int bestOffset;
processor.FindBestOffset(processor.Config.encodeWhenConfidence, false, out tracksMatch, out bestOffset);
if (tracksMatch * 100 < processor.Config.encodeWhenPercent * processor.TrackCount || (processor.Config.encodeWhenZeroOffset && bestOffset != 0))
    return status;
processor.Action = CUEAction.Encode;
return processor.Go();
"));
            scripts.Add("repair", new CUEToolsScript("repair", true,
                new CUEAction[] { CUEAction.Encode },
@"
processor.UseCUEToolsDB();
processor.Action = CUEAction.Verify;
if (processor.CTDB.DBStatus != null)
	return CTDB.DBStatus;
processor.Go();
processor.CTDB.DoVerify();
if (!processor.CTDB.Verify.HasErrors)
	return ""nothing to fix"";
if (!processor.CTDB.Verify.CanRecover)
	return ""cannot fix"";
processor._useCUEToolsDBFix = true;
processor.Action = CUEAction.Encode;
return processor.Go();
"));
            defaultVerifyScript = "default";
            defaultEncodeScript = "default";
        }

        public void Save(SettingsWriter sw)
        {
            sw.Save("Version", 203);
            sw.Save("ArFixWhenConfidence", fixOffsetMinimumConfidence);
            sw.Save("ArFixWhenPercent", fixOffsetMinimumTracksPercent);
            sw.Save("ArEncodeWhenConfidence", encodeWhenConfidence);
            sw.Save("ArEncodeWhenPercent", encodeWhenPercent);
            sw.Save("ArEncodeWhenZeroOffset", encodeWhenZeroOffset);
            sw.Save("ArNoUnverifiedOutput", noUnverifiedOutput);
            sw.Save("ArFixOffset", fixOffset);
            sw.Save("ArWriteCRC", writeArTagsOnEncode);
            sw.Save("ArWriteLog", writeArLogOnConvert);
            sw.Save("ArWriteTagsOnVerify", writeArTagsOnVerify);
            sw.Save("ArWriteLogOnVerify", writeArLogOnVerify);

            sw.Save("PreserveHTOA", preserveHTOA);
            sw.Save("DetectGaps", detectGaps);            
            sw.Save("AutoCorrectFilenames", autoCorrectFilenames);
            sw.Save("KeepOriginalFilenames", keepOriginalFilenames);
            sw.Save("SingleFilenameFormat", singleFilenameFormat);
            sw.Save("TrackFilenameFormat", trackFilenameFormat);
            sw.Save("RemoveSpecialCharacters", removeSpecial);
            sw.Save("SpecialCharactersExceptions", specialExceptions);
            sw.Save("ReplaceSpaces", replaceSpaces);
            sw.Save("EmbedLog", embedLog);
            sw.Save("ExtractLog", extractLog);
            sw.Save("FillUpCUE", fillUpCUE);
            sw.Save("OverwriteCUEData", overwriteCUEData);
            sw.Save("FilenamesANSISafe", filenamesANSISafe);
            if (bruteForceDTL) sw.Save("BruteForceDTL", bruteForceDTL);
            sw.Save("CreateEACLOG", createEACLOG);
            sw.Save("DetectHDCD", detectHDCD);
            sw.Save("Wait750FramesForHDCD", wait750FramesForHDCD);
            sw.Save("DecodeHDCD", decodeHDCD);
            sw.Save("CreateM3U", createM3U);
            sw.Save("CreateCUEFileWhenEmbedded", createCUEFileWhenEmbedded);
            sw.Save("Truncate4608ExtraSamples", truncate4608ExtraSamples);
            sw.Save("LossyWAVQuality", lossyWAVQuality);
            sw.Save("DecodeHDCDToLossyWAV16", decodeHDCDtoLW16);
            sw.Save("DecodeHDCDTo24bit", decodeHDCDto24bit);
            sw.Save("OneInstance", oneInstance);
            sw.Save("CheckForUpdates", checkForUpdates);
            sw.Save("Language", language);

            sw.Save("SeparateDecodingThread", separateDecodingThread);

            sw.Save("WriteBasicTagsFromCUEData", writeBasicTagsFromCUEData);
            sw.Save("CopyBasicTags", copyBasicTags);
            sw.Save("CopyUnknownTags", copyUnknownTags);
            sw.Save("CopyAlbumArt", CopyAlbumArt);
            sw.Save("EmbedAlbumArt", embedAlbumArt);
            sw.Save("ExtractAlbumArt", extractAlbumArt);
            sw.Save("MaxAlbumArtSize", maxAlbumArtSize);

            sw.Save("ArLogToSourceFolder", arLogToSourceFolder);
            sw.Save("ArLogVerbose", arLogVerbose);
            sw.Save("FixOffsetToNearest", fixOffsetToNearest);

            sw.Save("ArLogFilenameFormat", ArLogFilenameFormat);
            sw.Save("AlArtFilenameFormat", AlArtFilenameFormat);

            using (TextWriter tw = new StringWriter())
            using (XmlWriter xw = XmlTextWriter.Create(tw, xmlEmptySettings))
            {
                CUEConfigAdvanced.serializer.Serialize(xw, advanced, xmlEmptyNamespaces);
                sw.SaveText("Advanced", tw.ToString());
            }

            int nEncoders = 0;
            foreach (CUEToolsUDC encoder in encoders)
            {
                sw.Save(string.Format("ExternalEncoder{0}Name", nEncoders), encoder.name);
                sw.Save(string.Format("ExternalEncoder{0}Modes", nEncoders), encoder.supported_modes);
                sw.Save(string.Format("ExternalEncoder{0}Mode", nEncoders), encoder.default_mode);
                if (encoder.path != null)
                {
                    sw.Save(string.Format("ExternalEncoder{0}Extension", nEncoders), encoder.extension);
                    sw.Save(string.Format("ExternalEncoder{0}Path", nEncoders), encoder.path);
                    sw.Save(string.Format("ExternalEncoder{0}Lossless", nEncoders), encoder.lossless);
                    sw.Save(string.Format("ExternalEncoder{0}Parameters", nEncoders), encoder.parameters);
                }
                else
                {
                    if (encoder.settingsSerializer != null)
                    {
                        using (TextWriter tw = new StringWriter())
                        using (XmlWriter xw = XmlTextWriter.Create(tw, xmlEmptySettings))
                        {
                            encoder.settingsSerializer.Serialize(xw, encoder.settings, xmlEmptyNamespaces);
                            sw.SaveText(string.Format("ExternalEncoder{0}Parameters", nEncoders), tw.ToString());
                        }
                    }
                }
                nEncoders++;
            }
            sw.Save("ExternalEncoders", nEncoders);

            int nDecoders = 0;
            foreach (KeyValuePair<string, CUEToolsUDC> decoder in decoders)
                if (decoder.Value.path != null)
                {
                    sw.Save(string.Format("ExternalDecoder{0}Name", nDecoders), decoder.Key);
                    sw.Save(string.Format("ExternalDecoder{0}Extension", nDecoders), decoder.Value.extension);
                    sw.Save(string.Format("ExternalDecoder{0}Path", nDecoders), decoder.Value.path);
                    sw.Save(string.Format("ExternalDecoder{0}Parameters", nDecoders), decoder.Value.parameters);
                    nDecoders++;
                }
            sw.Save("ExternalDecoders", nDecoders);

            int nFormats = 0;
            foreach (KeyValuePair<string, CUEToolsFormat> format in formats)
            {
                sw.Save(string.Format("CustomFormat{0}Name", nFormats), format.Key);
                sw.Save(string.Format("CustomFormat{0}EncoderLossless", nFormats), format.Value.encoderLossless == null ? "" : format.Value.encoderLossless.Name);
                sw.Save(string.Format("CustomFormat{0}EncoderLossy", nFormats), format.Value.encoderLossy == null ? "" : format.Value.encoderLossy.Name);
                sw.Save(string.Format("CustomFormat{0}Decoder", nFormats), format.Value.decoder);
                sw.Save(string.Format("CustomFormat{0}Tagger", nFormats), (int)format.Value.tagger);
                sw.Save(string.Format("CustomFormat{0}AllowLossless", nFormats), format.Value.allowLossless);
                sw.Save(string.Format("CustomFormat{0}AllowLossy", nFormats), format.Value.allowLossy);
                sw.Save(string.Format("CustomFormat{0}AllowLossyWAV", nFormats), format.Value.allowLossyWAV);
                sw.Save(string.Format("CustomFormat{0}AllowEmbed", nFormats), format.Value.allowEmbed);
                nFormats++;
            }
            sw.Save("CustomFormats", nFormats);

            int nScripts = 0;
            foreach (KeyValuePair<string, CUEToolsScript> script in scripts)
            {
                sw.Save(string.Format("CustomScript{0}Name", nScripts), script.Key);
                sw.SaveText(string.Format("CustomScript{0}Code", nScripts), script.Value.code);
                int nCondition = 0;
                foreach (CUEAction action in script.Value.conditions)
                    sw.Save(string.Format("CustomScript{0}Condition{1}", nScripts, nCondition++), (int)action);
                sw.Save(string.Format("CustomScript{0}Conditions", nScripts), nCondition);
                nScripts++;
            }
            sw.Save("CustomScripts", nScripts);
            sw.Save("DefaultVerifyScript", defaultVerifyScript);
            sw.Save("DefaultVerifyAndConvertScript", defaultEncodeScript);

            sw.Save("GapsHandling", (int)gapsHandling);
        }

        public void Load(SettingsReader sr)
        {
            int version = sr.LoadInt32("Version", null, null) ?? 202;

            fixOffsetMinimumConfidence = sr.LoadUInt32("ArFixWhenConfidence", 1, 1000) ?? 2;
            fixOffsetMinimumTracksPercent = sr.LoadUInt32("ArFixWhenPercent", 1, 100) ?? 51;
            encodeWhenConfidence = sr.LoadUInt32("ArEncodeWhenConfidence", 1, 1000) ?? 2;
            encodeWhenPercent = sr.LoadUInt32("ArEncodeWhenPercent", 1, 100) ?? 100;
            encodeWhenZeroOffset = sr.LoadBoolean("ArEncodeWhenZeroOffset") ?? false;
            noUnverifiedOutput = sr.LoadBoolean("ArNoUnverifiedOutput") ?? false;
            fixOffset = sr.LoadBoolean("ArFixOffset") ?? false;
            writeArTagsOnEncode = sr.LoadBoolean("ArWriteCRC") ?? writeArTagsOnEncode;
            writeArLogOnConvert = sr.LoadBoolean("ArWriteLog") ?? true;
            writeArTagsOnVerify = sr.LoadBoolean("ArWriteTagsOnVerify") ?? false;
            writeArLogOnVerify = sr.LoadBoolean("ArWriteLogOnVerify") ?? false;

            preserveHTOA = sr.LoadBoolean("PreserveHTOA") ?? true;
            detectGaps = sr.LoadBoolean("DetectGaps") ?? true;
            autoCorrectFilenames = sr.LoadBoolean("AutoCorrectFilenames") ?? true;
            keepOriginalFilenames = sr.LoadBoolean("KeepOriginalFilenames") ?? false;
            singleFilenameFormat = sr.Load("SingleFilenameFormat") ?? singleFilenameFormat;
            trackFilenameFormat = sr.Load("TrackFilenameFormat") ?? trackFilenameFormat;
            removeSpecial = sr.LoadBoolean("RemoveSpecialCharacters") ?? false;
            specialExceptions = sr.Load("SpecialCharactersExceptions") ?? "-()";
            replaceSpaces = sr.LoadBoolean("ReplaceSpaces") ?? false;
            embedLog = sr.LoadBoolean("EmbedLog") ?? true;
            extractLog = sr.LoadBoolean("ExtractLog") ?? true;
            fillUpCUE = sr.LoadBoolean("FillUpCUE") ?? true;
            overwriteCUEData = sr.LoadBoolean("OverwriteCUEData") ?? false;
            filenamesANSISafe = sr.LoadBoolean("FilenamesANSISafe") ?? true;
            bruteForceDTL = sr.LoadBoolean("BruteForceDTL") ?? false;
            createEACLOG = sr.LoadBoolean("CreateEACLOG") ?? createEACLOG;
            detectHDCD = sr.LoadBoolean("DetectHDCD") ?? true;
            wait750FramesForHDCD = sr.LoadBoolean("Wait750FramesForHDCD") ?? true;
            decodeHDCD = sr.LoadBoolean("DecodeHDCD") ?? false;
            createM3U = sr.LoadBoolean("CreateM3U") ?? false;
            createCUEFileWhenEmbedded = sr.LoadBoolean("CreateCUEFileWhenEmbedded") ?? true;
            truncate4608ExtraSamples = sr.LoadBoolean("Truncate4608ExtraSamples") ?? true;
            lossyWAVQuality = sr.LoadInt32("LossyWAVQuality", 0, 10) ?? 5;
            decodeHDCDtoLW16 = sr.LoadBoolean("DecodeHDCDToLossyWAV16") ?? false;
            decodeHDCDto24bit = sr.LoadBoolean("DecodeHDCDTo24bit") ?? true;

            oneInstance = sr.LoadBoolean("OneInstance") ?? true;
            checkForUpdates = sr.LoadBoolean("CheckForUpdates") ?? true;

            writeBasicTagsFromCUEData = sr.LoadBoolean("WriteBasicTagsFromCUEData") ?? true;
            copyBasicTags = sr.LoadBoolean("CopyBasicTags") ?? true;
            copyUnknownTags = sr.LoadBoolean("CopyUnknownTags") ?? true;
            CopyAlbumArt = sr.LoadBoolean("CopyAlbumArt") ?? true;
            embedAlbumArt = sr.LoadBoolean("EmbedAlbumArt") ?? true;
            extractAlbumArt = sr.LoadBoolean("ExtractAlbumArt") ?? true;
            maxAlbumArtSize = sr.LoadInt32("MaxAlbumArtSize", 100, 10000) ?? maxAlbumArtSize;

            arLogToSourceFolder = sr.LoadBoolean("ArLogToSourceFolder") ?? arLogToSourceFolder;
            arLogVerbose = sr.LoadBoolean("ArLogVerbose") ?? arLogVerbose;
            fixOffsetToNearest = sr.LoadBoolean("FixOffsetToNearest") ?? fixOffsetToNearest;
            ArLogFilenameFormat = sr.Load("ArLogFilenameFormat") ?? ArLogFilenameFormat;
            AlArtFilenameFormat = sr.Load("AlArtFilenameFormat") ?? AlArtFilenameFormat;

            separateDecodingThread = sr.LoadBoolean("SeparateDecodingThread") ?? separateDecodingThread;

            try
            {
                using (TextReader reader = new StringReader(sr.Load("Advanced")))
                    advanced = CUEConfigAdvanced.serializer.Deserialize(reader) as CUEConfigAdvanced;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(ex.Message);
            }

            int totalEncoders = sr.LoadInt32("ExternalEncoders", 0, null) ?? 0;
            for (int nEncoders = 0; nEncoders < totalEncoders; nEncoders++)
            {
                string name = sr.Load(string.Format("ExternalEncoder{0}Name", nEncoders));
                string extension = sr.Load(string.Format("ExternalEncoder{0}Extension", nEncoders));
                string path = sr.Load(string.Format("ExternalEncoder{0}Path", nEncoders));
                string parameters = sr.Load(string.Format("ExternalEncoder{0}Parameters", nEncoders));
                bool lossless = sr.LoadBoolean(string.Format("ExternalEncoder{0}Lossless", nEncoders)) ?? true;
                string supported_modes = sr.Load(string.Format("ExternalEncoder{0}Modes", nEncoders)) ?? "";
                string default_mode = sr.Load(string.Format("ExternalEncoder{0}Mode", nEncoders)) ?? "";
                CUEToolsUDC encoder;
                if (name == null) continue;
                if (!encoders.TryGetValue(name, out encoder))
                {
                    if (path == null || parameters == null || extension == null) continue;
                    encoders.Add(new CUEToolsUDC(name, extension, lossless, supported_modes, default_mode, path, parameters));
                }
                else if (version == 203)
                {
                    if (encoder.path != null)
                    {
                        if (path == null || parameters == null || extension == null) continue;
                        encoder.extension = extension;
                        encoder.path = path;
                        encoder.lossless = lossless;
                        encoder.parameters = parameters;
                    }
                    else
                    {
                        if (encoder.settingsSerializer != null && parameters != "")
                            try
                            {
                                using (TextReader reader = new StringReader(parameters))
                                    encoder.settings = encoder.settingsSerializer.Deserialize(reader);
                            }
                            catch
                            {
                            }
                    }
                    encoder.supported_modes = supported_modes;
                    encoder.default_mode = default_mode;
                }
            }

            int totalDecoders = sr.LoadInt32("ExternalDecoders", 0, null) ?? 0;
            for (int nDecoders = 0; nDecoders < totalDecoders; nDecoders++)
            {
                string name = sr.Load(string.Format("ExternalDecoder{0}Name", nDecoders));
                string extension = sr.Load(string.Format("ExternalDecoder{0}Extension", nDecoders));
                string path = sr.Load(string.Format("ExternalDecoder{0}Path", nDecoders));
                string parameters = sr.Load(string.Format("ExternalDecoder{0}Parameters", nDecoders));
                CUEToolsUDC decoder;
                if (!decoders.TryGetValue(name, out decoder))
                    decoders.Add(name, new CUEToolsUDC(name, extension, true, "", "", path, parameters));
                else
                {
                    decoder.extension = extension;
                    decoder.path = path;
                    decoder.parameters = parameters;
                }
            }

            int totalFormats = sr.LoadInt32("CustomFormats", 0, null) ?? 0;
            for (int nFormats = 0; nFormats < totalFormats; nFormats++)
            {
                string extension = sr.Load(string.Format("CustomFormat{0}Name", nFormats));
                string encoderLossless = sr.Load(string.Format("CustomFormat{0}EncoderLossless", nFormats)) ?? "";
                string encoderLossy = sr.Load(string.Format("CustomFormat{0}EncoderLossy", nFormats)) ?? "";
                string decoder = sr.Load(string.Format("CustomFormat{0}Decoder", nFormats));
                CUEToolsTagger tagger = (CUEToolsTagger)(sr.LoadInt32(string.Format("CustomFormat{0}Tagger", nFormats), 0, 2) ?? 0);
                bool allowLossless = sr.LoadBoolean(string.Format("CustomFormat{0}AllowLossless", nFormats)) ?? false;
                bool allowLossy = sr.LoadBoolean(string.Format("CustomFormat{0}AllowLossy", nFormats)) ?? false;
                bool allowLossyWav = sr.LoadBoolean(string.Format("CustomFormat{0}AllowLossyWAV", nFormats)) ?? false;
                bool allowEmbed = sr.LoadBoolean(string.Format("CustomFormat{0}AllowEmbed", nFormats)) ?? false;
                CUEToolsFormat format;
                CUEToolsUDC udcLossless, udcLossy;
                if (encoderLossless == "" || !encoders.TryGetValue(encoderLossless, out udcLossless))
					udcLossless = encoders.GetDefault(extension, true);
                if (encoderLossy == "" || !encoders.TryGetValue(encoderLossy, out udcLossy))
					udcLossy = encoders.GetDefault(extension, false);
                if (!formats.TryGetValue(extension, out format))
                    formats.Add(extension, new CUEToolsFormat(extension, tagger, allowLossless, allowLossy, allowLossyWav, allowEmbed, false, udcLossless, udcLossy, decoder));
                else
                {
                    format.encoderLossless = udcLossless;
                    format.encoderLossy = udcLossy;
                    format.decoder = decoder;
                    if (!format.builtin)
                    {
                        format.tagger = tagger;
                        format.allowLossless = allowLossless;
                        format.allowLossy = allowLossy;
                        format.allowLossyWAV = allowLossyWav;
                        format.allowEmbed = allowEmbed;
                    }
                }
            }

            int totalScripts = sr.LoadInt32("CustomScripts", 0, null) ?? 0;
            for (int nScripts = 0; nScripts < totalScripts; nScripts++)
            {
                string name = sr.Load(string.Format("CustomScript{0}Name", nScripts));
                string code = sr.Load(string.Format("CustomScript{0}Code", nScripts));
                List<CUEAction> conditions = new List<CUEAction>();
                int totalConditions = sr.LoadInt32(string.Format("CustomScript{0}Conditions", nScripts), 0, null) ?? 0;
                for (int nCondition = 0; nCondition < totalConditions; nCondition++)
                    conditions.Add((CUEAction)sr.LoadInt32(string.Format("CustomScript{0}Condition{1}", nScripts, nCondition), 0, null));
                CUEToolsScript script;
                if (!scripts.TryGetValue(name, out script))
                {
                    if (name != "submit")
                        scripts.Add(name, new CUEToolsScript(name, false, conditions, code));
                }
                else
                {
                    if (!script.builtin)
                    {
                        script.code = code;
                        script.conditions = conditions;
                    }
                }
            }

            defaultVerifyScript = sr.Load("DefaultVerifyScript") ?? "default";
            defaultEncodeScript = sr.Load("DefaultVerifyAndConvertScript") ?? "default";

            gapsHandling = (CUEStyle?)sr.LoadInt32("GapsHandling", null, null) ?? gapsHandling;

            language = sr.Load("Language") ?? Thread.CurrentThread.CurrentUICulture.Name;

            if (ArLogFilenameFormat.Contains("%F"))
                ArLogFilenameFormat = "%filename%.accurip";
            if (singleFilenameFormat.Contains("%F"))
                singleFilenameFormat = "%filename%";
            if (trackFilenameFormat.Contains("%N"))
                trackFilenameFormat = "%tracknumber%. %title%";
        }

        public string GetDefaultDecoder(string extension)
        {
            CUEToolsUDC result = null;
            foreach (KeyValuePair<string, CUEToolsUDC> decoder in decoders)
                if (decoder.Value.Extension == extension && (result == null || result.priority < decoder.Value.priority))
                    result = decoder.Value;
            return result == null ? null : result.Name;
        }

        public IWebProxy GetProxy()
        {
            IWebProxy proxy = null;
            switch (advanced.UseProxyMode)
            {
                case CUEConfigAdvanced.ProxyMode.System:
                    proxy = WebRequest.GetSystemWebProxy();
                    break;
                case CUEConfigAdvanced.ProxyMode.Custom:
                    proxy = new WebProxy(advanced.ProxyServer, advanced.ProxyPort);
                    if (advanced.ProxyUser != "")
                        proxy.Credentials = new NetworkCredential(advanced.ProxyUser, advanced.ProxyPassword);
                    break;
            }
            return proxy;
        }

        public string CleanseString(string s)
        {
            StringBuilder sb = new StringBuilder();
            char[] invalid = Path.GetInvalidFileNameChars();

            if (filenamesANSISafe)
                s = Encoding.Default.GetString(Encoding.Default.GetBytes(s));

            for (int i = 0; i < s.Length; i++)
            {
                char ch = s[i];
                if (filenamesANSISafe && removeSpecial && specialExceptions.IndexOf(ch) < 0 && !(
                    ((ch >= 'a') && (ch <= 'z')) ||
                    ((ch >= 'A') && (ch <= 'Z')) ||
                    ((ch >= '0') && (ch <= '9')) ||
                    (ch == ' ') || (ch == '_')))
                    ch = '_';
                if ((Array.IndexOf(invalid, ch) >= 0) || (replaceSpaces && ch == ' '))
                    sb.Append("_");
                else
                    sb.Append(ch);
            }

            return sb.ToString();
        }
    }
}
