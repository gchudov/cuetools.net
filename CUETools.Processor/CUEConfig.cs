using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Xml;
using CUETools.Codecs;
using CUETools.Processor.Settings;
using Newtonsoft.Json;
using System.Linq;

namespace CUETools.Processor
{
    public class CUEConfig
    {
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
        public bool ejectAfterRip;
        public bool disableEjectDisc;
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
        public bool decodeHDCDtoLW16;
        public bool decodeHDCDto24bit;
        public bool oneInstance;
        public bool checkForUpdates;
        public string language;
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
        public EncoderListViewModel Encoders => advanced.encodersViewModel;
        public DecoderListViewModel Decoders => advanced.decodersViewModel;
        public Dictionary<string, CUEToolsFormat> formats => advanced.formats;

        public CUEConfig()
            : base()
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
            ejectAfterRip = false;
            disableEjectDisc = false;
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
            advanced.Init();

            language = Thread.CurrentThread.CurrentUICulture.Name;

            scripts = new Dictionary<string, CUEToolsScript>();
            scripts.Add("default", new CUEToolsScript("default",
                new CUEAction[] { CUEAction.Verify, CUEAction.Encode }));
            scripts.Add("only if found", new CUEToolsScript("only if found",
                new CUEAction[] { CUEAction.Verify }));
            scripts.Add("fix offset", new CUEToolsScript("fix offset",
                new CUEAction[] { CUEAction.Encode }));
            scripts.Add("encode if verified", new CUEToolsScript("encode if verified",
                new CUEAction[] { CUEAction.Encode }));
            scripts.Add("repair", new CUEToolsScript("repair",
                new CUEAction[] { CUEAction.Encode }));
            defaultVerifyScript = "default";
            defaultEncodeScript = "default";
        }

        public CUEConfig(CUEConfig src)
        {
            fixOffsetMinimumConfidence = src.fixOffsetMinimumConfidence;
            fixOffsetMinimumTracksPercent = src.fixOffsetMinimumTracksPercent;
            encodeWhenConfidence = src.encodeWhenConfidence;
            encodeWhenPercent = src.encodeWhenPercent;
            encodeWhenZeroOffset = src.encodeWhenZeroOffset;
            fixOffset = src.fixOffset;
            noUnverifiedOutput = src.noUnverifiedOutput;
            writeArTagsOnEncode = src.writeArTagsOnEncode;
            writeArLogOnConvert = src.writeArLogOnConvert;
            writeArTagsOnVerify = src.writeArTagsOnVerify;
            writeArLogOnVerify = src.writeArLogOnVerify;

            autoCorrectFilenames = src.autoCorrectFilenames;
            preserveHTOA = src.preserveHTOA;
            ejectAfterRip = src.ejectAfterRip;
            disableEjectDisc = src.disableEjectDisc;
            detectGaps = src.detectGaps;
            keepOriginalFilenames = src.keepOriginalFilenames;
            trackFilenameFormat = src.trackFilenameFormat;
            singleFilenameFormat = src.singleFilenameFormat;
            removeSpecial = src.removeSpecial;
            specialExceptions = src.specialExceptions;
            replaceSpaces = src.replaceSpaces;
            embedLog = src.embedLog;
            extractLog = src.extractLog;
            fillUpCUE = src.fillUpCUE;
            overwriteCUEData = src.overwriteCUEData;
            filenamesANSISafe = src.filenamesANSISafe;
            bruteForceDTL = src.bruteForceDTL;
            createEACLOG = src.createEACLOG;
            detectHDCD = src.detectHDCD;
            wait750FramesForHDCD = src.wait750FramesForHDCD;
            decodeHDCD = src.decodeHDCD;
            createM3U = src.createM3U;
            createCUEFileWhenEmbedded = src.createCUEFileWhenEmbedded;
            truncate4608ExtraSamples = src.truncate4608ExtraSamples;
            decodeHDCDtoLW16 = src.decodeHDCDtoLW16;
            decodeHDCDto24bit = src.decodeHDCDto24bit;

            oneInstance = src.oneInstance;
            checkForUpdates = src.checkForUpdates;

            writeBasicTagsFromCUEData = src.writeBasicTagsFromCUEData;
            copyBasicTags = src.copyBasicTags;
            copyUnknownTags = src.copyUnknownTags;
            CopyAlbumArt = src.CopyAlbumArt;
            embedAlbumArt = src.embedAlbumArt;
            extractAlbumArt = src.extractAlbumArt;
            maxAlbumArtSize = src.maxAlbumArtSize;

            arLogToSourceFolder = src.arLogToSourceFolder;
            arLogVerbose = src.arLogVerbose;
            fixOffsetToNearest = src.fixOffsetToNearest;
            ArLogFilenameFormat = src.ArLogFilenameFormat;
            AlArtFilenameFormat = src.AlArtFilenameFormat;

            separateDecodingThread = src.separateDecodingThread;

            gapsHandling = src.gapsHandling;

            advanced = new CUEConfigAdvanced(src.advanced);

            language = src.language;

            scripts = new Dictionary<string, CUEToolsScript>();
            scripts.Add("default", new CUEToolsScript("default",
                new CUEAction[] { CUEAction.Verify, CUEAction.Encode }));
            scripts.Add("only if found", new CUEToolsScript("only if found",
                new CUEAction[] { CUEAction.Verify }));
            scripts.Add("fix offset", new CUEToolsScript("fix offset",
                new CUEAction[] { CUEAction.Encode }));
            scripts.Add("encode if verified", new CUEToolsScript("encode if verified",
                new CUEAction[] { CUEAction.Encode }));
            scripts.Add("repair", new CUEToolsScript("repair",
                new CUEAction[] { CUEAction.Encode }));

            defaultVerifyScript = src.defaultVerifyScript;
            defaultEncodeScript = src.defaultEncodeScript;
        }

        public void Save(SettingsWriter sw)
        {
            sw.Save("Version", 222);
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
            sw.Save("EjectAfterRip", ejectAfterRip);
            sw.Save("DisableEjectDisc", disableEjectDisc);
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

            sw.SaveText("Advanced", JsonConvert.SerializeObject(advanced,
                Newtonsoft.Json.Formatting.Indented,
                new JsonSerializerSettings
                {
                    DefaultValueHandling = DefaultValueHandling.IgnoreAndPopulate,
                    TypeNameHandling = TypeNameHandling.Auto,
                }));

            int nFormats = 0;
            foreach (KeyValuePair<string, CUEToolsFormat> format in formats)
            {
                sw.Save(string.Format("CustomFormat{0}Name", nFormats), format.Key);
                sw.Save(string.Format("CustomFormat{0}EncoderLossless", nFormats), format.Value.encoderLossless == null ? "" : format.Value.encoderLossless.Name);
                sw.Save(string.Format("CustomFormat{0}EncoderLossy", nFormats), format.Value.encoderLossy == null ? "" : format.Value.encoderLossy.Name);
                sw.Save(string.Format("CustomFormat{0}Decoder", nFormats), format.Value.decoder == null ? "" : format.Value.decoder.Name);
                sw.Save(string.Format("CustomFormat{0}Tagger", nFormats), (int)format.Value.tagger);
                sw.Save(string.Format("CustomFormat{0}AllowLossless", nFormats), format.Value.allowLossless);
                sw.Save(string.Format("CustomFormat{0}AllowLossy", nFormats), format.Value.allowLossy);
                sw.Save(string.Format("CustomFormat{0}AllowEmbed", nFormats), format.Value.allowEmbed);
                nFormats++;
            }
            sw.Save("CustomFormats", nFormats);

            int nScripts = 0;
            foreach (KeyValuePair<string, CUEToolsScript> script in scripts)
            {
                sw.Save(string.Format("CustomScript{0}Name", nScripts), script.Key);
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
            ejectAfterRip = sr.LoadBoolean("EjectAfterRip") ?? false;
            disableEjectDisc = sr.LoadBoolean("DisableEjectDisc") ?? false;
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

            var jsonConfig = sr.Load("Advanced");
            if (jsonConfig != null)
            {
                var backup = advanced;
                try
                {
                    var jsonObject = JsonConvert.DeserializeObject(jsonConfig,
                        typeof(CUEConfigAdvanced),
                        new JsonSerializerSettings
                        {
                            DefaultValueHandling = DefaultValueHandling.IgnoreAndPopulate,
                            TypeNameHandling = TypeNameHandling.Auto,
                            Error = (sender, ev) => {
                                System.Diagnostics.Trace.WriteLine(ev.ErrorContext.Error.ToString());
                                ev.ErrorContext.Handled = true;
                            }
                        });
                    if (jsonObject as CUEConfigAdvanced == null)
                        throw new Exception();
                    advanced = jsonObject as CUEConfigAdvanced;

                    // Add missing codecs
                    backup.encoders.Where(x => advanced.encoders
                        .FindAll(y => y.Extension == x.Extension && y.Lossless == x.Lossless && y.Name == x.Name).Count == 0)
                        .ToList().ForEach(x => advanced.encoders.Add(x));
                    backup.decoders.Where(x => advanced.decoders
                        .FindAll(y => y.Extension == x.Extension && y.Name == x.Name).Count == 0)
                        .ToList().ForEach(x => advanced.decoders.Add(x));

                    // Reset the ViewModel
                    advanced.encodersViewModel = new EncoderListViewModel(advanced.encoders);
                    advanced.decodersViewModel = new DecoderListViewModel(advanced.decoders);

                    // Reset the links in formats
                    foreach (var extension in formats.Keys)
                    {
                        var format = formats[extension];
                        AudioEncoderSettingsViewModel encoderLossless, encoderLossy;
                        AudioDecoderSettingsViewModel decoder;
                        if (format.encoderLossless == null || !Encoders.TryGetValue(extension, true, format.encoderLossless.Name, out encoderLossless))
                            encoderLossless = Encoders.GetDefault(extension, true);
                        if (format.encoderLossy == null || !Encoders.TryGetValue(extension, false, format.encoderLossy.Name, out encoderLossy))
                            encoderLossy = Encoders.GetDefault(extension, false);
                        if (format.decoder == null || !Decoders.TryGetValue(extension, format.decoder.Name, out decoder))
                            decoder = Decoders.GetDefault(extension);
                        format.encoderLossless = encoderLossless;
                        format.encoderLossy = encoderLossy;
                        format.decoder = decoder;
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(ex.Message);
                    advanced = backup;
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
                bool allowEmbed = sr.LoadBoolean(string.Format("CustomFormat{0}AllowEmbed", nFormats)) ?? false;
                CUEToolsFormat format;
                AudioEncoderSettingsViewModel udcLossless, udcLossy;
                AudioDecoderSettingsViewModel udcDecoder;
                if (encoderLossless == "" || !Encoders.TryGetValue(extension, true, encoderLossless, out udcLossless))
					udcLossless = Encoders.GetDefault(extension, true);
                if (encoderLossy == "" || !Encoders.TryGetValue(extension, false, encoderLossy, out udcLossy))
					udcLossy = Encoders.GetDefault(extension, false);
                if (decoder == "" || !Decoders.TryGetValue(extension, decoder, out udcDecoder))
                    udcDecoder = Decoders.GetDefault(extension);
                if (!formats.TryGetValue(extension, out format))
                    formats.Add(extension, new CUEToolsFormat(extension, tagger, allowLossless, allowLossy, allowEmbed, false, udcLossless, udcLossy, udcDecoder));
                else
                {
                    format.encoderLossless = udcLossless;
                    format.encoderLossy = udcLossy;
                    format.decoder = udcDecoder;
                    if (!format.builtin)
                    {
                        format.tagger = tagger;
                        format.allowLossless = allowLossless;
                        format.allowLossy = allowLossy;
                        format.allowEmbed = allowEmbed;
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
