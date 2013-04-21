using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using CSScriptLibrary;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.CTDB;
using CUETools.Codecs;
using CUETools.Compression;
using CUETools.Ripper;
using Freedb;

namespace CUETools.Processor
{
    public class CUESheet
    {
        #region Fields

        public readonly static string CUEToolsVersion = "2.1.5";

        private bool _stop, _pause;
        private List<CUELine> _attributes;
        private List<TrackInfo> _tracks;
        internal List<SourceInfo> _sources;
        private List<string> _sourcePaths, _trackFilenames;
        private string _htoaFilename, _singleFilename;
        private bool _hasHTOAFilename = false, _hasTrackFilenames = false, _hasSingleFilename = false, _appliedWriteOffset;
        private bool _hasEmbeddedCUESheet;
        private bool _paddedToFrame, _truncated4608, _usePregapForFirstTrackInSingleFile;
        private int _writeOffset;
        private CUEAction _action;
        private bool isUsingAccurateRip = false;
        private bool isUsingCUEToolsDB = false;
        private bool isUsingCUEToolsDBFix = false;
        private bool _processed = false;
        private uint? _minDataTrackLength;
        private string _accurateRipId;
        private string _eacLog;
        private string _defaultLog;
        private List<CUEToolsSourceFile> _logFiles;
        private string _inputPath, _inputDir;
        private string _outputPath;
        private string[] _destPaths;
        private TagLib.File _fileInfo;
        private const int _arOffsetRange = 5 * 588 - 1;
        private IAudioDest hdcdDecoder;
        private AudioEncoderType _audioEncoderType = AudioEncoderType.Lossless;
        private bool _outputLossyWAV = false;
        private string _outputFormat = "wav";
        private CUEStyle _outputStyle = CUEStyle.SingleFile;
        private CUEConfig _config;
        private string _cddbDiscIdTag;
        private bool _isCD;
        private string _ripperLog;
        private ICDRipper _ripper;
        private bool _isArchive;
        private List<string> _archiveContents;
        private string _archiveCUEpath;
        private ICompressionProvider _archive;
        private string _archivePassword;
        private CUEToolsProgressEventArgs _progress;
        private AccurateRipVerify _arVerify;
        private AccurateRipVerify _arTestVerify;
        private CUEToolsDB _CUEToolsDB;
        private CDImageLayout _toc;
        private string _arLogFileName, _alArtFileName;
        private List<TagLib.IPicture> _albumArt = new List<TagLib.IPicture>();
        private int _padding = 8192;
        private IWebProxy proxy;
        private CUEMetadata taglibMetadata;
        private CUEMetadata cueMetadata;
        private bool _useLocalDB;
        private CUEToolsLocalDB _localDB;

        #endregion

        #region Properties

        public int TrackCount
        {
            get { return _tracks.Count; }
        }

        public string OutputPath
        {
            get { return _outputPath; }
        }

        public string OutputDir
        {
            get
            {
                string outDir = Path.GetDirectoryName(_outputPath);
                return outDir == "" ? "." : outDir;
            }
        }

        public CDImageLayout TOC
        {
            get { return _toc; }
            set
            {
                _toc = new CDImageLayout(value);
                if (Tracks.Count == 0)
                {
                    for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
                    {
                        //_trackFilenames.Add(string.Format("{0:00}.wav", iTrack + 1));
                        _tracks.Add(new TrackInfo());
                    }
                }
            }
        }

        public List<TagLib.IPicture> AlbumArt
        {
            get { return _albumArt; }
        }

        public Image Cover
        {
            get
            {
                if (AlbumArt == null || AlbumArt.Count == 0)
                    return null;
                TagLib.IPicture picture = AlbumArt[0];
                using (MemoryStream imageStream = new MemoryStream(picture.Data.Data, 0, picture.Data.Count))
                    try { return Image.FromStream(imageStream); }
                    catch { }
                return null;
            }
        }

        public List<string> SourcePaths
        {
            get { return _sourcePaths; }
        }

        public string LOGContents
        {
            get { return _ripperLog; }
        }

        public string AccurateRipLog
        {
            get { return CUESheetLogWriter.GetAccurateRipLog(this); }
        }

        public bool IsUsingAccurateRip
        {
            get { return this.isUsingAccurateRip; }
        }

        public string[] DestPaths
        {
            get { return this._destPaths; }
        }

        public IAudioDest HDCDDecoder
        {
            get { return this.hdcdDecoder; }
        }

        public bool IsUsingCUEToolsDB
        {
            get { return this.isUsingCUEToolsDB; }
        }

        public bool IsUsingCUEToolsDBFix
        {
            get { return this.isUsingCUEToolsDBFix; }
        }

        public bool Truncated4608
        {
            get { return this._truncated4608; }
        }

        public string CDDBDiscIdTag
        {
            get { return this._cddbDiscIdTag; }
        }

        public string InputPath
        {
            get
            {
                return _inputPath;
            }
        }

        public AccurateRipVerify ArTestVerify
        {
            get
            {
                return _arTestVerify;
            }

            set
            {
                _arTestVerify = value;
            }
        }

        public AccurateRipVerify ArVerify
        {
            get
            {
                return _arVerify;
            }
        }

        public CUEToolsDB CTDB
        {
            get { return _CUEToolsDB; }
        }

        public ICDRipper CDRipper
        {
            get { return _ripper; }
            set { _ripper = value; }
        }

        public CUEMetadata Metadata
        {
            get { return cueMetadata; }
        }

        public List<CUELine> Attributes
        {
            get { return _attributes; }
        }

        public List<TrackInfo> Tracks
        {
            get { return _tracks; }
        }

        public bool HasHTOAFilename
        {
            get
            {
                return _hasHTOAFilename;
            }
        }

        public string HTOAFilename
        {
            get
            {
                return _htoaFilename;
            }
            set
            {
                _htoaFilename = value;
            }
        }

        public bool HasTrackFilenames
        {
            get
            {
                return _hasTrackFilenames;
            }
        }

        public List<string> TrackFilenames
        {
            get
            {
                return _trackFilenames;
            }
        }

        public bool HasSingleFilename
        {
            get
            {
                return _hasSingleFilename;
            }
        }

        public string SingleFilename
        {
            get
            {
                return _singleFilename;
            }
            set
            {
                _singleFilename = value;
            }
        }

        public string AccurateRipId
        {
            get
            {
                return _accurateRipId;
            }
        }

        public string ArLogFileName
        {
            get
            {
                return _arLogFileName;
            }
            set
            {
                _arLogFileName = value;
            }
        }

        public string AlArtFileName
        {
            get
            {
                return _alArtFileName;
            }
            set
            {
                _alArtFileName = value;
            }
        }

        public NameValueCollection Tags
        {
            get
            {
                TagLib.File fileInfo = _tracks[0]._fileInfo ?? _fileInfo;
                return fileInfo != null ? Tagging.Analyze(fileInfo) : null;
            }
        }

        public int WriteOffset
        {
            get
            {
                return _writeOffset;
            }
            set
            {
                if (_appliedWriteOffset)
                {
                    throw new Exception("Cannot change write offset after audio files have been written.");
                }
                _writeOffset = value;
            }
        }

        public bool PaddedToFrame
        {
            get
            {
                return _paddedToFrame;
            }
        }

        public uint DataTrackLength
        {
            get
            {
                if (!_toc[1].IsAudio)
                    return _toc[1].Length;
                else if (!_toc[_toc.TrackCount].IsAudio)
                    return _toc[_toc.TrackCount].Length;
                else
                    return 0U;
            }
            set
            {
                if (value == 0)
                    return;
                if (!_toc[1].IsAudio)
                {
                    // TODO: if track 2 has a pregap, we should adjust it!!!
                    for (int i = 2; i <= _toc.TrackCount; i++)
                    {
                        _toc[i].Start += value - _toc[1].Length;
                        for (int j = 0; j <= _toc[i].LastIndex; j++)
                            _toc[i][j].Start += value - _toc[1].Length;
                    }
                    _toc[1].Length = value;
                }
                else if (!_toc[_toc.TrackCount].IsAudio)
                {
                    //_toc[_toc.TrackCount].Start = tocFromLog[_toc.TrackCount].Start;
                    _toc[_toc.TrackCount].Length = value;
                    //_toc[_toc.TrackCount][0].Start = tocFromLog[_toc.TrackCount].Start;
                    //_toc[_toc.TrackCount][1].Start = tocFromLog[_toc.TrackCount].Start;
                }
                else
                    _toc.AddTrack(new CDTrack((uint)_toc.TrackCount + 1, _toc.Length + 152U * 75U, value, false, false));
            }
        }

        public string DataTrackLengthMSF
        {
            get
            {
                return CDImageLayout.TimeToString(DataTrackLength);
            }
            set
            {
                DataTrackLength = (uint)CDImageLayout.TimeFromString(value);
            }
        }

        public string PreGapLengthMSF
        {
            get
            {
                return CDImageLayout.TimeToString(_toc.Pregap);
            }
            set
            {
                PreGapLength = (uint)CDImageLayout.TimeFromString(value);
            }
        }

        public uint PreGapLength
        {
            get
            {
                return _toc.Pregap;
            }
            set
            {
                if (value == _toc.Pregap || value == 0)
                    return;
                if (!_toc[1].IsAudio)
                    throw new Exception("can't set pregap to a data track");
                if (value < _toc.Pregap)
                    throw new Exception("can't set negative pregap");
                uint offs = value - _toc.Pregap;
                for (int i = 1; i <= _toc.TrackCount; i++)
                {
                    _toc[i].Start += offs;
                    for (int j = 0; j <= _toc[i].LastIndex; j++)
                        _toc[i][j].Start += offs;
                }
                _toc[1][0].Start = 0;

                SourceInfo sourceInfo;
                sourceInfo.Path = null;
                sourceInfo.Offset = 0;
                sourceInfo.Length = offs * 588;
                _sources.Insert(0, sourceInfo);
            }
        }

        public bool UsePregapForFirstTrackInSingleFile
        {
            get
            {
                return _usePregapForFirstTrackInSingleFile;
            }
            set
            {
                _usePregapForFirstTrackInSingleFile = value;
            }
        }

        public CUEConfig Config
        {
            get
            {
                return _config;
            }
        }

        public CUEAction Action
        {
            get
            {
                return _action;
            }
            set
            {
                _action = value;
            }
        }

        public CUEStyle OutputStyle
        {
            get
            {
                return _outputStyle;
            }
            set
            {
                _outputStyle = value;
            }
        }

        public bool Processed
        {
            get
            {
                return _processed;
            }
        }

        public bool IsCD
        {
            get
            {
                return _isCD;
            }
        }

        public uint? MinDataTrackLength
        {
            get { return this._minDataTrackLength; }
        }

        #endregion

        #region Constructor

        public CUESheet(CUEConfig config)
        {
            _config = config;
            _progress = new CUEToolsProgressEventArgs();
            _progress.cueSheet = this;
            _attributes = new List<CUELine>();
            _tracks = new List<TrackInfo>();
            _trackFilenames = new List<string>();
            _toc = new CDImageLayout();
            _sources = new List<SourceInfo>();
            _sourcePaths = new List<string>();
            _stop = false;
            _pause = false;
            _outputPath = null;
            _paddedToFrame = false;
            _truncated4608 = false;
            _usePregapForFirstTrackInSingleFile = false;
            _action = CUEAction.Encode;
            _appliedWriteOffset = false;
            _minDataTrackLength = null;
            hdcdDecoder = null;
            _hasEmbeddedCUESheet = false;
            _isArchive = false;
            _isCD = false;
            _useLocalDB = false;
            proxy = _config.GetProxy();
        }

        #endregion

        #region Methods

        public void OpenCD(ICDRipper ripper)
        {
            _ripper = ripper;
            _toc = (CDImageLayout)_ripper.TOC.Clone();
            for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
            {
                _trackFilenames.Add(string.Format("{0:00}.wav", iTrack + 1));
                _tracks.Add(new TrackInfo());
            }
            cueMetadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
            _arVerify = new AccurateRipVerify(_toc, proxy);
            _isCD = true;
            SourceInfo cdInfo;
            cdInfo.Path = _ripper.ARName;
            cdInfo.Offset = 0;
            cdInfo.Length = _toc.AudioLength * 588;
            _sources.Add(cdInfo);
            // Causes memory leak, so had to disable!
            //_ripper.ReadProgress += new EventHandler<ReadProgressArgs>(CDReadProgress);
            _padding += TrackCount * 200;
            _padding += _config.embedLog ? 500 + TrackCount * 200 : 0;
        }

        public void Close()
        {
            if (_progress != null)
            {
                _progress.cueSheet = null;
                _progress = null;
            }
            if (_archive != null)
                _archive.Close();
            _archive = null;
            if (_ripper != null)
            {
                //_ripper.ReadProgress -= new EventHandler<ReadProgressArgs>(CDReadProgress);
                _ripper.Close();
            }
            _ripper = null;
        }

        public void CopyMetadata(CUEMetadata metadata)
        {
            if (this.cueMetadata == null)
                this.cueMetadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
            this.cueMetadata.CopyMetadata(metadata);
        }

        protected void ReportProgress(string status, double percent)
        {
            ShowProgress(status, percent, null, null);
        }

        public bool IsInLocalDB(string folder)
        {
            string norm = CUEToolsLocalDBEntry.NormalizePath(folder);
            if (norm.Length != 0 && norm[norm.Length - 1] != System.IO.Path.DirectorySeparatorChar) norm = norm + System.IO.Path.DirectorySeparatorChar;
            return _localDB.Find(item => item.InputPaths != null && item.InputPaths.Find(i => i.StartsWith(norm)) != null) != null;
        }

        public void ScanLocalDB(string folder)
        {
            // Delete missing files
            string norm = CUEToolsLocalDBEntry.NormalizePath(folder);
            if (norm.Length != 0 && norm[norm.Length - 1] != System.IO.Path.DirectorySeparatorChar) norm = norm + System.IO.Path.DirectorySeparatorChar;

            ReportProgress("Checking known files", 0.0);
            int oldi = 0, oldn = _localDB.Count;
            foreach (var item in _localDB.ToArray())
            {
                bool matches = false;
                oldi++;
                if (item.AudioPaths != null)
                {
                    foreach (var f in item.AudioPaths)
                        if (f.StartsWith(norm))
                        {
                            matches = true;
                            CheckStop();
                            if (!File.Exists(f))
                            {
                                _localDB.Remove(item);
                                _localDB.Dirty = true;
                                continue;
                            }
                        }
                }

                if (item.InputPaths != null)
                {
                    foreach (var f in item.InputPaths.ToArray())
                        if (f.StartsWith(norm))
                        {
                            CheckStop();
                            ReportProgress("Checking " + f, (double)oldi / oldn);
                            matches = true;
                            if (!File.Exists(f))
                            {
                                item.InputPaths.Remove(f);
                                _localDB.Dirty = true;
                            }
                            else
                            {
                                var cueSheet = new CUESheet(_config);
                                cueSheet.UseLocalDB(_localDB);
                                try
                                {
                                    cueSheet.Open(f);
                                    List<string> fullAudioPaths = cueSheet._sourcePaths == null ? null : cueSheet._sourcePaths.ConvertAll(p => CUEToolsLocalDBEntry.NormalizePath(p));
                                    if (!item.Equals(cueSheet._toc, fullAudioPaths))
                                    {
                                        item.InputPaths.Remove(f);
                                        _localDB.Dirty = true;
                                    }
                                }
                                catch (Exception)
                                {
                                    item.InputPaths.Remove(f);
                                    _localDB.Dirty = true;
                                }
                                cueSheet.Close();
                            }
                        }

                    if (matches && item.InputPaths.Count == 0)
                    {
                        _localDB.Remove(item);
                        _localDB.Dirty = true;
                        continue;
                    }
                }
            }

            // Add new files
            ReportProgress("Scanning for new files", 0.0);
            var results = new List<string>();

            int n = 2, j = 0;
            foreach (var fmt in _config.formats)
                if (fmt.Value.allowLossless)
                    n++;

            CheckStop();
            ReportProgress("Scanning *.cue", (double)(j++) / n);
            results.AddRange(Directory.GetFiles(folder, "*.cue", SearchOption.AllDirectories));

            CheckStop();
            ReportProgress("Scanning *.m3u", (double)(j++) / n);
            results.AddRange(Directory.GetFiles(folder, "*.m3u", SearchOption.AllDirectories));

            foreach (var fmt in _config.formats)
                if (fmt.Value.allowLossless)
                {
                    CheckStop();
                    ReportProgress("Scanning *." + fmt.Key, (double)(j++) / n);
                    results.AddRange(Directory.GetFiles(folder, "*." + fmt.Key, SearchOption.AllDirectories));
                }

            ReportProgress("Checking new files", 0.0);
            int i = 0;
            foreach (var result in results)
            {
                CheckStop();

                var path = CUEToolsLocalDBEntry.NormalizePath(result);
                var pathextension = Path.GetExtension(path).ToLower();
                bool skip = false;
                if (_localDB.Find(
                        item => item.HasPath(path) ||
                            (item.AudioPaths != null &&
                            item.AudioPaths.Count > 1 &&
                            item.AudioPaths.Contains(path))
                    ) != null)
                    skip = true;
                if (!skip && pathextension == ".m3u")
                {
                    var contents = new List<string>();
                    using (StreamReader m3u = new StreamReader(path))
                    {
                        do
                        {
                            string line = m3u.ReadLine();
                            if (line == null) break;
                            if (line == "" || line[0] == '#') continue;
                            //if (line.IndexOfAny(Path.GetInvalidPathChars()) >= 0) 
                            //    continue;
                            try
                            {
                                string extension = Path.GetExtension(line);
                                CUEToolsFormat fmt1;
                                if (!extension.StartsWith(".") || !_config.formats.TryGetValue(extension.ToLower().Substring(1), out fmt1) || !fmt1.allowLossless)
                                {
                                    skip = true;
                                    break;
                                }
                                string fullpath = CUEToolsLocalDBEntry.NormalizePath(Path.Combine(Path.GetDirectoryName(path), line));
                                if (!File.Exists(fullpath))
                                {
                                    skip = true;
                                    break;
                                }
                                contents.Add(fullpath);
                            }
                            catch
                            {
                                skip = true;
                                break;
                            }
                        } while (!skip);
                    }
                    if (!skip && _localDB.Find(item => item.EqualAudioPaths(contents)) != null)
                        skip = true;
                }
                if (!skip && pathextension != ".cue" && pathextension != ".m3u")
                {
                    if (_localDB.Find(item =>
                            item.AudioPaths != null &&
                            item.AudioPaths.Count == 1 &&
                            item.AudioPaths[0] == path
                        ) != null)
                    {
                        CUEToolsFormat fmt;
                        if (!pathextension.StartsWith(".") || !_config.formats.TryGetValue(pathextension.Substring(1), out fmt) || !fmt.allowLossless || !fmt.allowEmbed)
                            skip = true;
                        else
                        {
                            TagLib.File fileInfo;
                            TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                            TagLib.File.IFileAbstraction file = new TagLib.File.LocalFileAbstraction(path);
                            fileInfo = TagLib.File.Create(file);
                            NameValueCollection tags = Tagging.Analyze(fileInfo);
                            if (tags.Get("CUESHEET") == null)
                                skip = true;
                        }
                    }
                }
                if (skip)
                {
                    ReportProgress("Skipping " + path, (double)(i++) / results.Count);
                }
                else
                {
                    ReportProgress("Checking " + path, (double)(i++) / results.Count);
                    var cueSheet = new CUESheet(_config);
                    cueSheet.UseLocalDB(_localDB);
                    //cueSheet.PasswordRequired += new EventHandler<CompressionPasswordRequiredEventArgs>(PasswordRequired);
                    //cueSheet.CUEToolsProgress += new EventHandler<CUEToolsProgressEventArgs>(SetStatus);
                    //cueSheet.CUEToolsSelection += new EventHandler<CUEToolsSelectionEventArgs>(MakeSelection);
                    try
                    {
                        cueSheet.Open(path);
                        cueSheet.OpenLocalDBEntry();
                    }
                    catch (Exception)
                    {
                    }
                    cueSheet.Close();
                }
            }
            _localDB.Save();
        }

        public List<object> LookupAlbumInfo(bool useCache, bool useCUE, bool useCTDB, CTDBMetadataSearch metadataSearch)
        {
            List<object> Releases = new List<object>();

            CUEMetadata dbmeta = null;

            if (useCache && _localDB != null)
            {
                List<string> fullAudioPaths = this.SourcePaths.ConvertAll(p => CUEToolsLocalDBEntry.NormalizePath(p));
                var myEntry = _localDB.Find(e => e.Equals(this.TOC, fullAudioPaths));
                if (myEntry != null)
                    dbmeta = myEntry.Metadata;
            }

            if (dbmeta != null)
                Releases.Add(new CUEMetadataEntry(dbmeta, TOC, "local"));

            //if (useCache)
            //{
            //    try
            //    {
            //        CUEMetadata cache = CUEMetadata.Load(TOC.TOCID);
            //        if (cache != null)
            //            Releases.Add(new CUEMetadataEntry(cache, TOC, "local"));
            //    }
            //    catch (Exception ex)
            //    {
            //        System.Diagnostics.Trace.WriteLine(ex.Message);
            //    }
            //}

            if (useCUE)
            {
                if (dbmeta == null || !dbmeta.Contains(cueMetadata))
                {
                    if (cueMetadata.Contains(taglibMetadata) || !taglibMetadata.Contains(cueMetadata))
                        Releases.Add(new CUEMetadataEntry(new CUEMetadata(cueMetadata), TOC, "cue"));
                }
                if (dbmeta == null || !dbmeta.Contains(taglibMetadata))
                {
                    if (!cueMetadata.Contains(taglibMetadata))
                        Releases.Add(new CUEMetadataEntry(new CUEMetadata(taglibMetadata), TOC, "tags"));
                }
            }

            if (useCache && _localDB != null)
            {
                foreach (var entry in _localDB)
                    if (entry.DiscID == TOC.TOCID && entry.Metadata != null && (dbmeta == null || !dbmeta.Contains(entry.Metadata)))
                        Releases.Add(new CUEMetadataEntry(entry.Metadata, TOC, "local"));
            }

            bool ctdbFound = false;
            if (useCTDB)
            {
                ShowProgress("Looking up album via CTDB...", 0.0, null, null);
                var ctdb = new CUEToolsDB(TOC, proxy);
                ctdb.ContactDB(_config.advanced.CTDBServer, "CUETools " + CUEToolsVersion, null, false, false, metadataSearch);
                foreach (var meta in ctdb.Metadata)
                {
                    CUEMetadata metadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
                    metadata.FillFromCtdb(meta, TOC.FirstAudio - 1);
                    CDImageLayout toc = TOC; //  TocFromCDEntry(meta);
                    Releases.Add(new CUEMetadataEntry(metadata, toc, meta.source));
                    ctdbFound = true;
                }
            }

            if (!ctdbFound && metadataSearch == CTDBMetadataSearch.Extensive)
            {
                ShowProgress("Looking up album via Freedb...", 0.0, null, null);

                FreedbHelper m_freedb = new FreedbHelper();
                m_freedb.Proxy = proxy;
                m_freedb.UserName = _config.advanced.FreedbUser;
                m_freedb.Hostname = _config.advanced.FreedbDomain;
                m_freedb.ClientName = "CUETools";
                m_freedb.Version = CUEToolsVersion;
                m_freedb.SetDefaultSiteAddress("freedb.org");

                QueryResult queryResult;
                QueryResultCollection coll;
                string code = string.Empty;
                try
                {
                    CDEntry cdEntry = null;
                    code = m_freedb.Query(AccurateRipVerify.CalculateCDDBQuery(_toc), out queryResult, out coll);
                    if (code == FreedbHelper.ResponseCodes.CODE_200)
                    {
                        ShowProgress("Looking up album via Freedb... " + queryResult.Discid, 0.5, null, null);
                        code = m_freedb.Read(queryResult, out cdEntry);
                        if (code == FreedbHelper.ResponseCodes.CODE_210)
                        {
                            CUEMetadata metadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
                            metadata.FillFromFreedb(cdEntry, TOC.FirstAudio - 1);
                            CDImageLayout toc = TocFromCDEntry(cdEntry);
                            Releases.Add(new CUEMetadataEntry(metadata, toc, "freedb"));
                        }
                    }
                    else
                        if (code == FreedbHelper.ResponseCodes.CODE_210 ||
                            code == FreedbHelper.ResponseCodes.CODE_211)
                        {
                            int i = 0;
                            foreach (QueryResult qr in coll)
                            {
                                ShowProgress("Looking up album via freedb... " + qr.Discid, (++i + 0.0) / coll.Count, null, null);
                                CheckStop();
                                code = m_freedb.Read(qr, out cdEntry);
                                if (code == FreedbHelper.ResponseCodes.CODE_210)
                                {
                                    CUEMetadata metadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
                                    metadata.FillFromFreedb(cdEntry, TOC.FirstAudio - 1);
                                    CDImageLayout toc = TocFromCDEntry(cdEntry);
                                    Releases.Add(new CUEMetadataEntry(metadata, toc, "freedb"));
                                }
                            }
                        }
                }
                catch (Exception ex)
                {
                    if (ex is StopException)
                        throw ex;
                }
            }

            ShowProgress("", 0, null, null);
            return Releases;
        }

        public CDImageLayout TocFromCDEntry(CDEntry cdEntry)
        {
            CDImageLayout tocFromCDEntry = new CDImageLayout();
            for (int i = 0; i < cdEntry.Tracks.Count; i++)
            {
                if (i >= _toc.TrackCount)
                    break;
                tocFromCDEntry.AddTrack(new CDTrack((uint)i + 1,
                    (uint)cdEntry.Tracks[i].FrameOffset - 150,
                    (i + 1 < cdEntry.Tracks.Count) ? (uint)(cdEntry.Tracks[i + 1].FrameOffset - cdEntry.Tracks[i].FrameOffset) : _toc[i + 1].Length,
                    _toc[i + 1].IsAudio,
                    false/*preEmphasis*/));
            }
            if (tocFromCDEntry.TrackCount > 0 && tocFromCDEntry[1].IsAudio)
                tocFromCDEntry[1][0].Start = 0;
            return tocFromCDEntry;
        }

        public void Open(string pathIn)
        {
            _inputPath = pathIn;
            _inputDir = Path.GetDirectoryName(_inputPath) ?? _inputPath;
            if (_inputDir == "") _inputDir = ".";
            if (_inputDir == pathIn && CUEProcessorPlugins.ripper != null)
            {
                ICDRipper ripper = Activator.CreateInstance(CUEProcessorPlugins.ripper) as ICDRipper;
                try
                {
                    ripper.Open(pathIn[0]);
                    if (ripper.TOC.AudioTracks > 0)
                    {
                        OpenCD(ripper);
                        int driveOffset;
                        if (!AccurateRipVerify.FindDriveReadOffset(_ripper.ARName, out driveOffset))
                            throw new Exception("Failed to find drive read offset for drive" + _ripper.ARName);
                        _ripper.DriveOffset = driveOffset;
                        //LookupAlbumInfo();
                        return;
                    }
                }
                catch
                {
                    ripper.Dispose();
                    _ripper = null;
                    throw;
                }
            }

            TextReader sr;

            if (Directory.Exists(pathIn))
                throw new Exception("is a directory");
            //{
            //    if (cueDir + Path.DirectorySeparatorChar != pathIn && cueDir != pathIn)
            //        throw new Exception("Input directory must end on path separator character.");
            //    string cueSheet = null;
            //    string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a", "*.tta" };
            //    for (i = 0; i < audioExts.Length && cueSheet == null; i++)
            //        cueSheet = CUESheet.CreateDummyCUESheet(pathIn, audioExts[i]);
            //    if (_config.udc1Extension != null && cueSheet == null)
            //        cueSheet = CUESheet.CreateDummyCUESheet(pathIn, "*." + _config.udc1Extension);
            //    if (cueSheet == null)
            //        throw new Exception("Input directory doesn't contain supported audio files.");
            //    sr = new StringReader(cueSheet);

            //    List<CUEToolsSourceFile> logFiles = new List<CUEToolsSourceFile>();
            //    foreach (string logPath in Directory.GetFiles(pathIn, "*.log"))
            //        logFiles.Add(new CUEToolsSourceFile(logPath, new StreamReader(logPath, CUESheet.Encoding)));
            //    CUEToolsSourceFile selectedLogFile = ChooseFile(logFiles, null, false);
            //    _eacLog = selectedLogFile != null ? selectedLogFile.contents : null;
            //} 
            else if (CUEProcessorPlugins.arcp_fmt.Contains(Path.GetExtension(pathIn).ToLower().Trim('.')))
            {
                _archive = null;
                foreach (Type type in CUEProcessorPlugins.arcp)
                {
                    CompressionProviderClass archclass = Attribute.GetCustomAttribute(type, typeof(CompressionProviderClass)) as CompressionProviderClass;
                    if (archclass.Extension == Path.GetExtension(pathIn).ToLower().Trim('.'))
                    {
                        _archive = Activator.CreateInstance(type, pathIn) as ICompressionProvider;
                        break;
                    }
                }
                if (_archive == null)
                    throw new Exception("archive type not supported.");
                _isArchive = true;
                _archiveContents = new List<string>();
                _archive.PasswordRequired += new EventHandler<CompressionPasswordRequiredEventArgs>(unzip_PasswordRequired);
                _archive.ExtractionProgress += new EventHandler<CompressionExtractionProgressEventArgs>(unzip_ExtractionProgress);
                foreach (string f in _archive.Contents)
                    _archiveContents.Add(f);

                _logFiles = new List<CUEToolsSourceFile>();
                List<CUEToolsSourceFile> cueFiles = new List<CUEToolsSourceFile>();
                foreach (string s in _archiveContents)
                {
                    if (Path.GetExtension(s).ToLower() == ".cue" || Path.GetExtension(s).ToLower() == ".log")
                    {
                        Stream archiveStream = OpenArchive(s, false);
                        CUEToolsSourceFile sourceFile = new CUEToolsSourceFile(s, new StreamReader(archiveStream, CUESheet.Encoding));
                        archiveStream.Close();
                        if (Path.GetExtension(s).ToLower() == ".cue")
                            cueFiles.Add(sourceFile);
                        else
                            _logFiles.Add(sourceFile);
                    }
                }
                CUEToolsSourceFile selectedCUEFile = ChooseFile(cueFiles, null, true);
                if (selectedCUEFile == null || selectedCUEFile.contents == "")
                    throw new Exception("Input archive doesn't contain a usable cue sheet.");
                _defaultLog = Path.GetFileNameWithoutExtension(selectedCUEFile.path);
                _archiveCUEpath = Path.GetDirectoryName(selectedCUEFile.path);
                string cueText = selectedCUEFile.contents;
                if (_config.autoCorrectFilenames)
                {
                    string extension;
                    cueText = CorrectAudioFilenames(_config, _archiveCUEpath, cueText, false, _archiveContents, out extension);
                }
                sr = new StringReader(cueText);
                if (_logFiles.Count == 1)
                    _eacLog = _logFiles[0].contents;
            }
            else if (Path.GetExtension(pathIn).ToLower() == ".cue")
            {
                if (_config.autoCorrectFilenames)
                    sr = new StringReader(CorrectAudioFilenames(_config, pathIn, false));
                else
                    sr = new StreamReader(pathIn, CUESheet.Encoding);

                _logFiles = new List<CUEToolsSourceFile>();
                _defaultLog = Path.GetFileNameWithoutExtension(pathIn);
                foreach (string logPath in Directory.GetFiles(_inputDir, "*.log"))
                    try { _logFiles.Add(new CUEToolsSourceFile(logPath, new StreamReader(logPath, CUESheet.Encoding))); }
                    catch { }
            }
            else if (Path.GetExtension(pathIn).ToLower() == ".m3u")
            {
                string cueSheet = CUESheet.CreateDummyCUESheet(_config, pathIn);
                sr = new StringReader(cueSheet);
                _logFiles = new List<CUEToolsSourceFile>();
                _defaultLog = Path.GetFileNameWithoutExtension(pathIn);
                foreach (string logPath in Directory.GetFiles(_inputDir, "*.log"))
                    try { _logFiles.Add(new CUEToolsSourceFile(logPath, new StreamReader(logPath, CUESheet.Encoding))); }
                    catch { }
            }
            else
            {
                string extension = Path.GetExtension(pathIn).ToLower();
                sr = null;
                CUEToolsFormat fmt;
                if (!extension.StartsWith(".") || !_config.formats.TryGetValue(extension.Substring(1), out fmt) || !fmt.allowLossless)
                    throw new Exception("Unknown input format.");
                if (fmt.allowEmbed)
                {
                    string cuesheetTag = null;
                    TagLib.File fileInfo;
                    GetSampleLength(pathIn, out fileInfo);
                    NameValueCollection tags = Tagging.Analyze(fileInfo);
                    cuesheetTag = tags.Get("CUESHEET");
                    _accurateRipId = tags.Get("ACCURATERIPID");
                    _eacLog = tags.Get("LOG");
                    if (_eacLog == null) _eacLog = tags.Get("LOGFILE");
                    if (_eacLog == null) _eacLog = tags.Get("EACLOG");
                    if (cuesheetTag != null)
                    {
                        sr = new StringReader(cuesheetTag);
                        _hasEmbeddedCUESheet = true;
                    }
                }
                if (!_hasEmbeddedCUESheet)
                {
                    string cueSheet = CUESheet.CreateDummyCUESheet(_config, pathIn);
                    if (cueSheet == null)
                        throw new Exception("Input file doesn't seem to contain a cue sheet or be part of an album.");
                    sr = new StringReader(cueSheet);
                    _logFiles = new List<CUEToolsSourceFile>();
                    foreach (string logPath in Directory.GetFiles(_inputDir, "*.log"))
                        try { _logFiles.Add(new CUEToolsSourceFile(logPath, new StreamReader(logPath, CUESheet.Encoding))); }
                        catch { }
                }
            }

            OpenCUE(sr);
        }

        public void OpenCUE(TextReader sr)
        {
            string pathAudio = null;
            string lineStr, command, fileType;
            bool fileIsBinary = false;
            int timeRelativeToFileStart, absoluteFileStartTime = 0;
            int fileTimeLengthSamples = 0, fileTimeLengthFrames = 0, i;
            TagLib.File _trackFileInfo = null;
            bool seenFirstFileIndex = false;
            bool isAudioTrack = true;
            List<IndexInfo> indexes = new List<IndexInfo>();
            IndexInfo indexInfo;
            SourceInfo sourceInfo;
            TrackInfo trackInfo = null;
            int trackNumber = 0;

            using (sr)
            {
                while ((lineStr = sr.ReadLine()) != null)
                {
                    CUELine line = new CUELine(lineStr);
                    if (line.Params.Count > 0)
                    {
                        command = line.Params[0].ToUpper();

                        if (command == "FILE")
                        {
                            fileType = line.Params[2].ToUpper();
                            fileIsBinary = (fileType == "BINARY") || (fileType == "MOTOROLA");
                            if (fileIsBinary)
                            {
                                if (!_hasEmbeddedCUESheet && _sourcePaths.Count == 0)
                                {
                                    try
                                    {
                                        if (_isArchive)
                                            pathAudio = FileLocator.LocateFile(_archiveCUEpath, line.Params[1], _archiveContents);
                                        else
                                            pathAudio = FileLocator.LocateFile(_inputDir, line.Params[1], null);
                                        fileIsBinary = (pathAudio == null);
                                    }
                                    catch { }
                                }
                            }
                            if (!fileIsBinary)
                            {
                                if (_sourcePaths.Count != 0 && !seenFirstFileIndex)
                                    throw new Exception("Double FILE in CUE sheet: \"" + line.Params[1] + "\".");
                                if (!_hasEmbeddedCUESheet)
                                {
                                    if (_isArchive)
                                        pathAudio = FileLocator.LocateFile(_archiveCUEpath, line.Params[1], _archiveContents);
                                    else
                                        pathAudio = FileLocator.LocateFile(_inputDir, line.Params[1], null);
                                }
                                else
                                {
                                    pathAudio = _inputPath;
                                    if (_sourcePaths.Count > 0)
                                        throw new Exception("Extra file in embedded CUE sheet: \"" + line.Params[1] + "\".");
                                }

                                if (pathAudio == null)
                                {
                                    throw new Exception("Unable to locate file \"" + line.Params[1] + "\".");
                                    //fileTimeLengthFrames = 75 * 60 * 70;;
                                    //fileTimeLengthSamples = fileTimeLengthFrames * 588;
                                    //if (_hasEmbeddedCUESheet)
                                    //    _fileInfo = null;
                                    //else
                                    //    _trackFileInfo = null;
                                }
                                else
                                {
                                    // Wierd case: audio file after data track with only index 00 specified.
                                    if (!isAudioTrack && _sourcePaths.Count == 0 && indexes.Count > 0 && indexes[indexes.Count - 1].Index == 0)
                                    {
                                        indexInfo.Track = indexes[indexes.Count - 1].Track;
                                        indexInfo.Index = 1;
                                        indexInfo.Time = indexes[indexes.Count - 1].Time + 150;
                                        indexes.Add(indexInfo);
                                        absoluteFileStartTime += 150;
                                    }

                                    TagLib.File fileInfo;
                                    _sourcePaths.Add(pathAudio);
                                    absoluteFileStartTime += fileTimeLengthFrames;
                                    fileTimeLengthSamples = GetSampleLength(pathAudio, out fileInfo);
                                    if ((fileTimeLengthSamples % 588) == 492 && _config.truncate4608ExtraSamples)
                                    {
                                        _truncated4608 = true;
                                        fileTimeLengthSamples -= 4608;
                                    }
                                    fileTimeLengthFrames = (int)((fileTimeLengthSamples + 587) / 588);
                                    if (_hasEmbeddedCUESheet)
                                        _fileInfo = fileInfo;
                                    else
                                        _trackFileInfo = fileInfo;
                                }
                                seenFirstFileIndex = false;
                            }
                        }
                        else if (command == "TRACK")
                        {
                            isAudioTrack = line.Params[2].ToUpper() == "AUDIO";
                            trackNumber = int.Parse(line.Params[1]);
                            if (trackNumber != _toc.TrackCount + 1)
                                throw new Exception("Invalid track number");
                            // Disabled this check: fails on Headcandy test image
                            //if (isAudioTrack && _sourcePaths.Count == 0)
                            //    throw new Exception("No FILE seen before TRACK");
                            _toc.AddTrack(new CDTrack((uint)trackNumber, 0, 0, isAudioTrack, false));
                            if (isAudioTrack)
                            {
                                trackInfo = new TrackInfo();
                                _tracks.Add(trackInfo);
                            }
                        }
                        else if (command == "INDEX")
                        {
                            timeRelativeToFileStart = CDImageLayout.TimeFromString(line.Params[2]);
                            if (!seenFirstFileIndex)
                            {
                                if (timeRelativeToFileStart != 0)
                                    throw new Exception("First index must start at file beginning.");
                                seenFirstFileIndex = true;
                                if (isAudioTrack)
                                {
                                    if (_tracks.Count > 0 && _trackFileInfo != null)
                                        _tracks[_tracks.Count - 1]._fileInfo = _trackFileInfo;
                                    _trackFileInfo = null;
                                    sourceInfo.Path = pathAudio;
                                    sourceInfo.Offset = 0;
                                    sourceInfo.Length = (uint)fileTimeLengthSamples;
                                    _sources.Add(sourceInfo);
                                    if ((fileTimeLengthSamples % 588) != 0)
                                    {
                                        sourceInfo.Path = null;
                                        sourceInfo.Offset = 0;
                                        sourceInfo.Length = (uint)((fileTimeLengthFrames * 588) - fileTimeLengthSamples);
                                        _sources.Add(sourceInfo);
                                        _paddedToFrame = true;
                                    }
                                }
                            }
                            else
                            {
                                if (fileIsBinary)
                                {
                                    fileTimeLengthFrames = timeRelativeToFileStart + 150;
                                    sourceInfo.Path = null;
                                    sourceInfo.Offset = 0;
                                    sourceInfo.Length = 150 * 588;
                                    _sources.Add(sourceInfo);
                                    //throw new Exception("unexpected BINARY directive");
                                }
                                else
                                {
                                    if (timeRelativeToFileStart > fileTimeLengthFrames)
                                        throw new Exception(string.Format("TRACK {0} INDEX {1} is at {2}, which is past {3} - the end of source file {4}", trackNumber, line.Params[1], CDImageLayout.TimeToString((uint)timeRelativeToFileStart), CDImageLayout.TimeToString((uint)fileTimeLengthFrames), pathAudio));
                                }
                            }
                            indexInfo.Track = trackNumber;
                            indexInfo.Index = Int32.Parse(line.Params[1]);
                            indexInfo.Time = absoluteFileStartTime + timeRelativeToFileStart;
                            indexes.Add(indexInfo);
                        }
                        else if (!isAudioTrack)
                        {
                            // Ignore lines belonging to data tracks
                        }
                        else if (command == "PREGAP")
                        {
                            if (seenFirstFileIndex)
                                throw new Exception("Pregap must occur at the beginning of a file.");
                            int pregapLength = CDImageLayout.TimeFromString(line.Params[1]);
                            indexInfo.Track = trackNumber;
                            indexInfo.Index = 0;
                            indexInfo.Time = absoluteFileStartTime;
                            indexes.Add(indexInfo);
                            sourceInfo.Path = null;
                            sourceInfo.Offset = 0;
                            sourceInfo.Length = (uint)pregapLength * 588;
                            _sources.Add(sourceInfo);
                            absoluteFileStartTime += pregapLength;
                        }
                        else if (command == "POSTGAP")
                        {
                            throw new Exception("POSTGAP command isn't supported.");
                        }
                        //else if ((command == "REM") &&
                        //    (line.Params.Count >= 3) &&
                        //    (line.Params[1].Length >= 10) &&
                        //    (line.Params[1].Substring(0, 10).ToUpper() == "REPLAYGAIN"))
                        //{
                        //    // Remove ReplayGain lines
                        //}
                        else if ((command == "REM") &&
                           (line.Params.Count == 3) &&
                           (line.Params[1].ToUpper() == "ACCURATERIPID"))
                        {
                            _accurateRipId = line.Params[2];
                        }
                        //else if ((command == "REM") &&
                        //   (line.Params.Count == 3) &&
                        //   (line.Params[1].ToUpper() == "SHORTEN"))
                        //{
                        //    fileTimeLengthFrames -= General.TimeFromString(line.Params[2]);
                        //}							
                        //else if ((command == "REM") &&
                        //   (line.Params.Count == 3) &&
                        //   (line.Params[1].ToUpper() == "LENGTHEN"))
                        //{
                        //    fileTimeLengthFrames += General.TimeFromString(line.Params[2]);
                        //}							
                        else
                        {
                            if (trackInfo != null)
                            {
                                trackInfo.Attributes.Add(line);
                            }
                            else
                            {
                                if (line.Params.Count > 2 && !line.IsQuoted[1] &&
                                    (line.Params[0].ToUpper() == "TITLE" || line.Params[0].ToUpper() == "ARTIST" ||
                                    (line.Params[0].ToUpper() == "REM" && (line.Params[1].ToUpper() == "GENRE" || line.Params[1].ToUpper() == "COMMENT") && line.Params.Count > 3 && !line.IsQuoted[2])))
                                {
                                    CUELine modline = new CUELine();
                                    int nParams = line.Params[0].ToUpper() == "REM" ? 2 : 1;
                                    for (int iParam = 0; iParam < nParams; iParam++)
                                    {
                                        modline.Params.Add(line.Params[iParam]);
                                        modline.IsQuoted.Add(false);
                                    }
                                    string s = line.Params[nParams];
                                    for (int iParam = nParams + 1; iParam < line.Params.Count; iParam++)
                                        s += " " + line.Params[iParam];
                                    modline.Params.Add(s);
                                    modline.IsQuoted.Add(true);
                                    line = modline;
                                }
                                _attributes.Add(line);
                            }
                        }
                    }
                }
                sr.Close();
            }

            if (_tracks.Count == 0)
                throw new Exception("File must contain at least one audio track.");

            // Add dummy index 01 for data track
            if (!_toc[_toc.TrackCount].IsAudio && indexes[indexes.Count - 1].Index == 0)
            {
                fileTimeLengthFrames += 152 * 75;
                indexInfo.Track = trackNumber;
                indexInfo.Index = 1;
                indexInfo.Time = absoluteFileStartTime + fileTimeLengthFrames;
                indexes.Add(indexInfo);
            }

            // Add dummy track for calculation purposes
            indexInfo.Track = trackNumber + 1;
            indexInfo.Index = 1;
            indexInfo.Time = absoluteFileStartTime + fileTimeLengthFrames;
            indexes.Add(indexInfo);

            // Calculate the length of each index
            for (i = 0; i < indexes.Count - 1; i++)
            {
                if (indexes[i + 1].Time - indexes[i].Time < 0)
                    throw new Exception("Indexes must be in chronological order.");
                if ((indexes[i + 1].Track != indexes[i].Track || indexes[i + 1].Index != indexes[i].Index + 1) &&
                    (indexes[i + 1].Track != indexes[i].Track + 1 || indexes[i].Index < 1 || indexes[i + 1].Index > 1))
                    throw new Exception("Indexes must be in chronological order.");
                if (indexes[i].Index == 1 && (i == 0 || indexes[i - 1].Index != 0))
                    _toc[indexes[i].Track].AddIndex(new CDTrackIndex(0U, (uint)indexes[i].Time));
                _toc[indexes[i].Track].AddIndex(new CDTrackIndex((uint)indexes[i].Index, (uint)indexes[i].Time));
            }

            // Calculate the length of each track
            for (int iTrack = 1; iTrack <= _toc.TrackCount; iTrack++)
            {
                _toc[iTrack].Start = _toc[iTrack][1].Start;
                _toc[iTrack].Length = iTrack == _toc.TrackCount
                    ? (uint)indexes[indexes.Count - 1].Time - _toc[iTrack].Start
                    : _toc[iTrack + 1].IsAudio
                        ? _toc[iTrack + 1][1].Start - _toc[iTrack].Start
                        : _toc[iTrack + 1][0].Start - _toc[iTrack].Start;

            }

            // Store the audio filenames, generating generic names if necessary
            _hasSingleFilename = _sourcePaths.Count == 1;
            _singleFilename = _hasSingleFilename ? Path.GetFileName(_sourcePaths[0]) :
                "Range.wav";

            _hasHTOAFilename = (_sourcePaths.Count == (TrackCount + 1));
            _htoaFilename = _hasHTOAFilename ? Path.GetFileName(_sourcePaths[0]) : "00.wav";

            _hasTrackFilenames = !_hasEmbeddedCUESheet && !_hasSingleFilename && (_sourcePaths.Count == TrackCount || _hasHTOAFilename);
            for (i = 0; i < TrackCount; i++)
            {
                _trackFilenames.Add(_hasTrackFilenames ? Path.GetFileName(
                    _sourcePaths[i + (_hasHTOAFilename ? 1 : 0)]) : String.Format("{0:00}.wav", i + 1));
            }
            if (!_hasEmbeddedCUESheet && _hasSingleFilename)
            {
                _fileInfo = _tracks[0]._fileInfo;
                _tracks[0]._fileInfo = null;
            }
            taglibMetadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
            taglibMetadata.Artist = GetCommonTag(file => file.Tag.JoinedAlbumArtists) ?? GetCommonTag(file => file.Tag.JoinedPerformers) ?? "";
            taglibMetadata.Title = GetCommonTag(file => file.Tag.Album) ?? "";
            taglibMetadata.Year = GetCommonTag(file => file.Tag.Year != 0 ? file.Tag.Year.ToString() : null) ?? "";
            taglibMetadata.Genre = GetCommonTag(file => file.Tag.JoinedGenres) ?? "";
            taglibMetadata.Comment = GetCommonTag(file => file.Tag.Comment) ?? "";
            taglibMetadata.TotalDiscs = GetCommonTag(file => file.Tag.DiscCount != 0 ? file.Tag.DiscCount.ToString() : null) ?? "";
            taglibMetadata.DiscNumber = GetCommonTag(file => file.Tag.Disc != 0 ? file.Tag.Disc.ToString() : null) ?? "";
			taglibMetadata.ReleaseDate = GetCommonTag(file => file.Tag.ReleaseDate) ?? "";
			taglibMetadata.Country = GetCommonTag(file => file.Tag.MusicBrainzReleaseCountry) ?? "";
			taglibMetadata.Label = GetCommonTag(file => file.Tag.Publisher) ?? "";
			taglibMetadata.LabelNo = GetCommonTag(file => file.Tag.CatalogNo) ?? "";
			taglibMetadata.DiscName = GetCommonTag(file => file.Tag.DiscSubtitle) ?? "";
            for (i = 0; i < TrackCount; i++)
            {
                TrackInfo track = _tracks[i];
                taglibMetadata.Tracks[i].Artist = (_hasTrackFilenames && track._fileInfo != null ? track._fileInfo.Tag.JoinedPerformers :
                    _hasEmbeddedCUESheet && _fileInfo != null ? Tagging.TagListToSingleValue(Tagging.GetMiscTag(_fileInfo, String.Format("cue_track{0:00}_ARTIST", i + 1))) :
                    null) ?? "";
                taglibMetadata.Tracks[i].Title = (_hasTrackFilenames && track._fileInfo != null ? track._fileInfo.Tag.Title :
                    _hasEmbeddedCUESheet && _fileInfo != null ? Tagging.TagListToSingleValue(Tagging.GetMiscTag(_fileInfo, String.Format("cue_track{0:00}_TITLE", i + 1))) :
                    null) ?? "";
				taglibMetadata.Tracks[i].Comment = (_hasTrackFilenames && track._fileInfo != null ? track._fileInfo.Tag.Title :
					_hasEmbeddedCUESheet && _fileInfo != null ? Tagging.TagListToSingleValue(Tagging.GetMiscTag(_fileInfo, String.Format("cue_track{0:00}_COMMENT", i + 1))) :
					null) ?? "";
			}

            cueMetadata = new CUEMetadata(TOC.TOCID, (int)TOC.AudioTracks);
            cueMetadata.Artist = General.GetCUELine(_attributes, "PERFORMER").Trim();
            cueMetadata.Title = General.GetCUELine(_attributes, "TITLE").Trim();
            cueMetadata.Barcode = General.GetCUELine(_attributes, "CATALOG");
            cueMetadata.Year = General.GetCUELine(_attributes, "REM", "DATE");
            cueMetadata.DiscNumber = General.GetCUELine(_attributes, "REM", "DISCNUMBER");
            cueMetadata.TotalDiscs = General.GetCUELine(_attributes, "REM", "TOTALDISCS");
            cueMetadata.Genre = General.GetCUELine(_attributes, "REM", "GENRE");
            cueMetadata.Comment = General.GetCUELine(_attributes, "REM", "COMMENT");
			cueMetadata.ReleaseDate = General.GetCUELine(_attributes, "REM", "RELEASEDATE");
			cueMetadata.Country = General.GetCUELine(_attributes, "REM", "COUNTRY");
			cueMetadata.Label = General.GetCUELine(_attributes, "REM", "LABEL");
			cueMetadata.LabelNo = General.GetCUELine(_attributes, "REM", "CATALOGNUMBER");
			cueMetadata.DiscName = General.GetCUELine(_attributes, "REM", "DISCSUBTITLE");
            for (i = 0; i < Tracks.Count; i++)
            {
                cueMetadata.Tracks[i].Artist = General.GetCUELine(Tracks[i].Attributes, "PERFORMER").Trim();
                cueMetadata.Tracks[i].Title = General.GetCUELine(Tracks[i].Attributes, "TITLE").Trim();
                cueMetadata.Tracks[i].ISRC = General.GetCUELine(Tracks[i].Attributes, "ISRC");
            }
            // Now, TOC.TOCID might change!!!

            if (_config.fillUpCUE)
            {
                cueMetadata.Merge(taglibMetadata, _config.overwriteCUEData);
                for (i = 0; i < TrackCount; i++)
                {
                    if (cueMetadata.Tracks[i].Title == "" && _hasTrackFilenames)
                        cueMetadata.Tracks[i].Title = Path.GetFileNameWithoutExtension(_trackFilenames[i]).TrimStart(" .-_0123456789".ToCharArray());
                }
            }

            CUELine cddbDiscIdLine = General.FindCUELine(_attributes, "REM", "DISCID");
            _cddbDiscIdTag = cddbDiscIdLine != null && cddbDiscIdLine.Params.Count == 3 ? cddbDiscIdLine.Params[2] : null;
            if (_cddbDiscIdTag == null)
                _cddbDiscIdTag = GetCommonMiscTag("DISCID");

            if (_accurateRipId == null)
                _accurateRipId = GetCommonMiscTag("ACCURATERIPID");

            if (_eacLog == null && _logFiles != null && _logFiles.Count > 0)
            {
                foreach (CUEToolsSourceFile sf in _logFiles)
                {
                    CDImageLayout tocFromLog1 = LogToTocParser.LogToToc(this._toc, sf.contents);
                    if (tocFromLog1 != null && tocFromLog1.TOCID == _toc.TOCID)
                    {
                        if (_eacLog == null)
                            _eacLog = sf.contents;
                        else
                        {
                            _eacLog = null;
                            break;
                        }
                    }
                }
            }

            if (_eacLog == null && _logFiles != null && _logFiles.Count > 0)
            {
                CUEToolsSourceFile selectedLogFile = ChooseFile(_logFiles, _defaultLog, false);
                _eacLog = selectedLogFile != null ? selectedLogFile.contents : null;
            }

            CDImageLayout tocFromLog = _eacLog == null ? null : LogToTocParser.LogToToc(this._toc, _eacLog);

            if (tocFromLog == null)
            {
                string tocPath = Path.ChangeExtension(InputPath, ".toc");
                if (File.Exists(tocPath))
                {
                    tocFromLog = LogToTocParser.LogToToc(this._toc, new StreamReader(tocPath, CUESheet.Encoding).ReadToEnd());
                }
            }

            if (tocFromLog == null)
            {
                var tocTag = GetCommonMiscTag("CDTOC");
                if (tocTag != null)
                {
                    tocFromLog = CDImageLayout.FromTag(tocTag);
                    if (tocFromLog != null && tocFromLog.TOCID != _toc.TOCID)
                        tocFromLog = null;
                }
            }

            // use pregaps from log
            if (tocFromLog != null)
            {
                int trNo;
                for (trNo = 0; trNo < tocFromLog.AudioTracks && trNo < _toc.AudioTracks; trNo++)
                {
                    if (_toc[_toc.FirstAudio + trNo].Pregap < tocFromLog[tocFromLog.FirstAudio + trNo].Pregap)
                    {
                        if (_toc.FirstAudio + trNo == 1)
                            PreGapLength = tocFromLog[tocFromLog.FirstAudio + trNo].Pregap;
                        else
                            _toc[_toc.FirstAudio + trNo].Pregap = tocFromLog[tocFromLog.FirstAudio + trNo].Pregap;
                    }
                }
                //if (_toc[_toc.FirstAudio].Length > tocFromLog[tocFromLog.FirstAudio].Length)
                //{
                //    uint offs = _toc[_toc.FirstAudio].Length - tocFromLog[tocFromLog.FirstAudio].Length;
                //    _toc[_toc.FirstAudio].Length -= offs;

                //    sourceInfo = _sources[srcNo];
                //    sourceInfo.Length -= offs * 588;
                //    _sources[srcNo] = sourceInfo;
                //    for (i = _toc.FirstAudio + 1; i <= _toc.TrackCount; i++)
                //    {
                //        _toc[i].Start -= offs;
                //        for (int j = 0; j <= _toc[i].LastIndex; j++)
                //            if (i != _toc.FirstAudio + 1 || j != 0 || _toc[i][0].Start == _toc[i][1].Start)
                //                _toc[i][j].Start -= offs;
                //    }
                //}
                //for (trNo = 1; trNo < tocFromLog.AudioTracks && trNo < _toc.AudioTracks; trNo++)
                //{
                //    srcNo ++;
                //    if (_toc[_toc.FirstAudio + trNo].Length > tocFromLog[tocFromLog.FirstAudio + trNo].Length)
                //    {
                //        uint offs = _toc[_toc.FirstAudio + trNo].Length - tocFromLog[tocFromLog.FirstAudio + trNo].Length;
                //        _toc[_toc.FirstAudio + trNo].Length -= offs;
                //        sourceInfo = _sources[srcNo];
                //        sourceInfo.Length -= offs * 588;
                //        _sources[srcNo] = sourceInfo;
                //        for (i = _toc.FirstAudio + trNo + 1; i <= _toc.TrackCount; i++)
                //        {
                //            _toc[i].Start -= offs;
                //            for (int j = 0; j <= _toc[i].LastIndex; j++)
                //                if (i != _toc.FirstAudio + trNo + 1 || j != 0 || _toc[i][0].Start == _toc[i][1].Start)
                //                    _toc[i][j].Start -= offs;
                //        }
                //    }
                //}
            }

            // use data track length from log
            if (tocFromLog != null)
            {
                if (tocFromLog.AudioTracks == _toc.AudioTracks
                    && tocFromLog.TrackCount == tocFromLog.AudioTracks + 1
                    && !tocFromLog[tocFromLog.TrackCount].IsAudio)
                {
                    DataTrackLength = tocFromLog[tocFromLog.TrackCount].Length;
                    _toc[_toc.TrackCount].Start = tocFromLog[_toc.TrackCount].Start;
                    _toc[_toc.TrackCount][0].Start = tocFromLog[_toc.TrackCount].Start;
                    _toc[_toc.TrackCount][1].Start = tocFromLog[_toc.TrackCount].Start;
                }

                if (_toc.TrackCount == _toc.AudioTracks
                    && tocFromLog.TrackCount == tocFromLog.AudioTracks
                    && tocFromLog.TrackCount > _toc.TrackCount)
                {
                    int dtracks = tocFromLog.TrackCount - _toc.TrackCount;
                    var toc2 = new CDImageLayout(tocFromLog);
                    for (int iTrack = 1; iTrack <= dtracks; iTrack++)
                        toc2[iTrack].IsAudio = false;
                    toc2.FirstAudio += dtracks;
                    toc2.AudioTracks -= (uint)dtracks;
                    if (toc2.TOCID == _toc.TOCID)
                        tocFromLog = toc2;
                }

                if (tocFromLog.AudioTracks == _toc.AudioTracks
//                    && tocFromLog.TOCID == _toc.TOCID
                    && _toc.TrackCount == _toc.AudioTracks
                    && tocFromLog.FirstAudio > 1
                    && tocFromLog.TrackCount == tocFromLog.FirstAudio + tocFromLog.AudioTracks - 1)
                {
                    for (int iTrack = 1; iTrack < tocFromLog.FirstAudio; iTrack++)
                        _toc.InsertTrack(new CDTrack((uint)iTrack, 0, 0, false, false));
                }

                if (tocFromLog.AudioTracks == _toc.AudioTracks
                    && tocFromLog.TrackCount == _toc.TrackCount
                    && tocFromLog.FirstAudio == _toc.FirstAudio
                    && tocFromLog.TrackCount == tocFromLog.FirstAudio + tocFromLog.AudioTracks - 1)
                {
                    //DataTrackLength = tocFromLog[1].Length;
                    uint delta = tocFromLog[_toc.FirstAudio].Start - _toc[_toc.FirstAudio].Start;
                    for (int itr = 1; itr < _toc.FirstAudio; itr++)
                    {
                        _toc[itr].Start = tocFromLog[itr].Start;
                        _toc[itr].Length = tocFromLog[itr].Length;
                    }
                    for (int itr = _toc.FirstAudio; itr <= _toc.TrackCount; itr++)
                    {
                        _toc[itr].Start += delta;
                        for (int j = 0; j <= _toc[itr].LastIndex; j++)
                            _toc[itr][j].Start += delta;
                    }
                }
            }

            // use data track length range from cddbId
            if (DataTrackLength == 0 && _cddbDiscIdTag != null)
            {
                uint cddbDiscIdNum;
                if (uint.TryParse(_cddbDiscIdTag, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out cddbDiscIdNum) && (cddbDiscIdNum & 0xff) == _toc.AudioTracks + 1)
                {
                    if (_toc.TrackCount == _toc.AudioTracks)
                        _toc.AddTrack(new CDTrack((uint)_toc.TrackCount + 1, _toc.Length + 152 * 75, 0, false, false));
                    uint lengthFromTag = ((cddbDiscIdNum >> 8) & 0xffff);
                    _minDataTrackLength = (lengthFromTag + _toc[1].Start / 75) * 75 - _toc.Length;
                }
            }

            _arVerify = new AccurateRipVerify(_toc, proxy);

            if (_eacLog != null)
            {
                sr = new StringReader(_eacLog);
                bool isEACLog = false;
                int trNo = 1;
                while ((lineStr = sr.ReadLine()) != null)
                {
                    if (isEACLog && trNo <= TrackCount)
                    {
                        string[] s = { "Copy CRC ", "CRC копии" };
                        string[] s1 = { "CRC" };
                        string[] n = lineStr.Split(s, StringSplitOptions.None);
                        uint crc;
                        if (n.Length == 2 && uint.TryParse(n[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out crc))
                            _arVerify.CRCLOG(trNo++, crc);
                        else if (n.Length == 1)
                        {
                            n = lineStr.Split(s1, StringSplitOptions.None);
                            if (n.Length == 2 && n[0].Trim() == "" && uint.TryParse(n[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out crc))
                                _arVerify.CRCLOG(trNo++, crc);
                        }
                    }
                    else
                        if (lineStr.StartsWith("Exact Audio Copy")
                            || lineStr.StartsWith("EAC extraction logfile"))
                            isEACLog = true;
                }
                if (trNo == 2)
                {
                    _arVerify.CRCLOG(0, _arVerify.CRCLOG(1));
                    if (TrackCount > 1)
                        _arVerify.CRCLOG(1, 0);
                }
            }

            LoadAlbumArt(_tracks[0]._fileInfo ?? _fileInfo);
            ResizeAlbumArt();
            if (_config.embedAlbumArt || _config.CopyAlbumArt)
                _albumArt.ForEach(t => _padding += _albumArt[0].Data.Count);
            if (_config.embedLog && _eacLog != null)
                _padding += _eacLog.Length;

            cueMetadata.Id = TOC.TOCID;
            taglibMetadata.Id = TOC.TOCID;
            // TODO: It should also be set when assigning a DataTrack!!!
        }

        public void UseCUEToolsDB(string userAgent, string driveName, bool fuzzy, CTDBMetadataSearch metadataSearch)
        {
            ShowProgress((string)"Contacting CUETools database...", 0, null, null);

            _CUEToolsDB = new CUEToolsDB(_toc, proxy);
            _CUEToolsDB.UploadHelper.onProgress += new EventHandler<Krystalware.UploadHelper.UploadProgressEventArgs>(UploadProgress);
            _CUEToolsDB.ContactDB(_config.advanced.CTDBServer, userAgent, driveName, true, fuzzy, metadataSearch);

            if (!_isCD && !_toc[_toc.TrackCount].IsAudio && DataTrackLength == 0)
                foreach (DBEntry e in _CUEToolsDB.Entries)
                    if (e.toc.TrackCount == _toc.TrackCount && e.toc.AudioLength == _toc.AudioLength && !e.toc[e.toc.TrackCount].IsAudio)
                    {
                        DataTrackLength = e.toc[e.toc.TrackCount].Length;
                        break;
                    }

            ShowProgress("", 0.0, null, null);
            isUsingCUEToolsDB = true;
        }

        public void UseAccurateRip()
        {
            ShowProgress((string)"Contacting AccurateRip database...", 0, null, null);
            if (!_toc[_toc.TrackCount].IsAudio && DataTrackLength == 0 && _minDataTrackLength.HasValue && _accurateRipId == null && _config.bruteForceDTL)
            {
                uint minDTL = _minDataTrackLength.Value;
                CDImageLayout toc2 = new CDImageLayout(_toc);
                for (uint dtl = minDTL; dtl < minDTL + 75; dtl++)
                {
                    toc2[toc2.TrackCount].Length = dtl;
                    _arVerify.ContactAccurateRip(AccurateRipVerify.CalculateAccurateRipId(toc2));
                    if (_arVerify.ExceptionStatus == WebExceptionStatus.Success)
                    {
                        DataTrackLength = dtl;
                        break;
                    }
                    if (_arVerify.ExceptionStatus != WebExceptionStatus.ProtocolError ||
                        _arVerify.ResponseStatus != HttpStatusCode.NotFound)
                        break;
                    ShowProgress((string)"Contacting AccurateRip database...", (dtl - minDTL) / 75.0, null, null);
                    CheckStop();
                }
            }
            else
            {
                _arVerify.ContactAccurateRip(_accurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(_toc));
            }
            isUsingAccurateRip = true;
        }

        public static Encoding Encoding
        {
            get
            {
                return Encoding.Default;
            }
        }

        internal CUEToolsSourceFile ChooseFile(List<CUEToolsSourceFile> sourceFiles, string defaultFileName, bool quietIfSingle)
        {
            if (sourceFiles.Count <= 0)
                return null;

            if (defaultFileName != null)
            {
                CUEToolsSourceFile defaultFile = null;
                foreach (CUEToolsSourceFile file in sourceFiles)
                    if (Path.GetFileNameWithoutExtension(file.path).ToLower() == defaultFileName.ToLower())
                    {
                        if (defaultFile != null)
                        {
                            defaultFile = null;
                            break;
                        }
                        defaultFile = file;
                    }
                if (defaultFile != null)
                    return defaultFile;
            }

            if (quietIfSingle && sourceFiles.Count == 1)
                return sourceFiles[0];

            if (CUEToolsSelection == null)
                return null;

            CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
            e.choices = sourceFiles.ToArray();
            CUEToolsSelection(this, e);
            if (e.selection == -1)
                return null;

            return sourceFiles[e.selection];
        }

        internal Stream OpenArchive(string fileName, bool showProgress)
        {
            if (_archive == null)
                throw new Exception("Unknown archive type.");
            return _archive.Decompress(fileName);
        }

        private void ShowProgress(string status, double percent, string input, string output)
        {
            if (this.CUEToolsProgress == null)
                return;
            _progress.status = status;
            _progress.percent = percent;
            _progress.offset = 0;
            _progress.input = input;
            _progress.output = output;
            this.CUEToolsProgress(this, _progress);
        }

        private void ShowProgress(string status, int diskOffset, int diskLength, string input, string output)
        {
            if (this.CUEToolsProgress == null)
                return;
            _progress.status = status;
            _progress.percent = (double)diskOffset / diskLength;
            _progress.offset = diskOffset;
            _progress.input = input;
            _progress.output = output;
            this.CUEToolsProgress(this, _progress);
        }

        private void UploadProgress(object sender, Krystalware.UploadHelper.UploadProgressEventArgs e)
        {
            CheckStop();
            if (this.CUEToolsProgress == null)
                return;
            _progress.percent = e.percent;
            _progress.offset = 0;
            _progress.status = e.uri;
            this.CUEToolsProgress(this, _progress);
        }

        //private void CDReadProgress(object sender, ReadProgressArgs e)
        //{
        //    CheckStop();
        //    if (this.CUEToolsProgress == null)
        //        return;
        //    ICDRipper audioSource = (ICDRipper)sender;
        //    int processed = e.Position - e.PassStart;
        //    TimeSpan elapsed = DateTime.Now - e.PassTime;
        //    double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;
        //    _progress.percentDisk = (double)(e.PassStart + (processed + e.Pass * (e.PassEnd - e.PassStart)) / (audioSource.CorrectionQuality + 1)) / audioSource.TOC.AudioLength;
        //    _progress.percentTrck = (double) (e.Position - e.PassStart) / (e.PassEnd - e.PassStart);
        //    _progress.offset = 0;
        //    _progress.status = string.Format("Ripping @{0:00.00}x {1}", speed, e.Pass > 0 ? " (Retry " + e.Pass.ToString() + ")" : "");
        //    this.CUEToolsProgress(this, _progress);
        //}

        private void unzip_ExtractionProgress(object sender, CompressionExtractionProgressEventArgs e)
        {
            CheckStop();
            if (this.CUEToolsProgress == null)
                return;
            _progress.percent = e.PercentComplete / 100;
            this.CUEToolsProgress(this, _progress);
        }

        private void unzip_PasswordRequired(object sender, CompressionPasswordRequiredEventArgs e)
        {
            if (_archivePassword != null)
            {
                e.ContinueOperation = true;
                e.Password = _archivePassword;
                return;
            }
            if (this.PasswordRequired != null)
            {
                this.PasswordRequired(this, e);
                if (e.ContinueOperation && e.Password != "")
                {
                    _archivePassword = e.Password;
                    return;
                }
            }
            throw new IOException("Password is required for extraction.");
        }

        public delegate string GetStringTagProvider(TagLib.File file);

        public string GetCommonTag(GetStringTagProvider provider)
        {
            if (_hasEmbeddedCUESheet || _hasSingleFilename)
                return _fileInfo == null ? null : General.EmptyStringToNull(provider(_fileInfo));
            if (_hasTrackFilenames)
            {
                string tagValue = null;
                bool commonValue = true;
                for (int i = 0; i < TrackCount; i++)
                {
                    TrackInfo track = _tracks[i];
                    string newValue = track._fileInfo == null ? null :
                        General.EmptyStringToNull(provider(track._fileInfo));
                    if (tagValue == null)
                        tagValue = newValue;
                    else
                        commonValue = (newValue == null || tagValue == newValue);
                }
                return commonValue ? tagValue : null;
            }
            return null;
        }

        public string GetCommonMiscTag(string tagName)
        {
            return GetCommonTag(delegate(TagLib.File file) { return Tagging.TagListToSingleValue(Tagging.GetMiscTag(file, tagName)); });
        }

        private static bool IsCDROM(string pathIn)
        {
            return pathIn.Length == 3 && pathIn.Substring(1) == ":\\" && new DriveInfo(pathIn).DriveType == DriveType.CDRom;
        }

        public string GenerateUniqueOutputPath(string format, string ext, CUEAction action, string pathIn)
        {
            return GenerateUniqueOutputPath(_config, format, ext, action, new NameValueCollection(), pathIn, this);
        }

        public static string GenerateUniqueOutputPath(CUEConfig _config, string format, string ext, CUEAction action, NameValueCollection vars, string pathIn, CUESheet cueSheet)
        {
            if (pathIn == "" || (pathIn == null && action != CUEAction.Encode) || (pathIn != null && !IsCDROM(pathIn) && !File.Exists(pathIn) && !Directory.Exists(pathIn)))
                return String.Empty;
            if (action == CUEAction.Verify && _config.arLogToSourceFolder)
                return Path.ChangeExtension(pathIn, ".cue");
            if (action == CUEAction.CreateDummyCUE)
                return Path.ChangeExtension(pathIn, ".cue");
            if (action == CUEAction.CorrectFilenames)
                return pathIn;

            if (_config.detectHDCD && _config.decodeHDCD && (!ext.StartsWith(".lossy.") || !_config.decodeHDCDtoLW16))
            {
                if (_config.decodeHDCDto24bit)
                    ext = ".24bit" + ext;
                else
                    ext = ".20bit" + ext;
            }

            if (pathIn != null)
            {
                vars.Add("path", pathIn);
                try
                {
                    vars.Add("filename", Path.GetFileNameWithoutExtension(pathIn));
                    vars.Add("filename_ext", Path.GetFileName(pathIn));
                    vars.Add("directoryname", General.EmptyStringToNull(Path.GetDirectoryName(pathIn)));
                }
                catch { }
            }
            vars.Add("music", Environment.GetFolderPath(Environment.SpecialFolder.MyMusic));
            string artist = cueSheet == null ? "Artist" : cueSheet.Metadata.Artist == "" ? "Unknown Artist" : cueSheet.Metadata.Artist;
            string album = cueSheet == null ? "Album" : cueSheet.Metadata.Title == "" ? "Unknown Title" : cueSheet.Metadata.Title;
            vars.Add("artist", General.EmptyStringToNull(_config.CleanseString(artist)));
            vars.Add("album", General.EmptyStringToNull(_config.CleanseString(album)));

            if (cueSheet != null)
            {
                vars.Add("year", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.Year)));
                vars.Add("barcode", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.Barcode)));
                vars.Add("label", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.Label)));
                vars.Add("labelno", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.LabelNo)));
                vars.Add("labelandnumber", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.LabelAndNumber)));
                vars.Add("country", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.Country)));
                vars.Add("releasedate", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.ReleaseDate)));
                vars.Add("discname", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.DiscName)));
                vars.Add("discnumber", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.DiscNumber01)));
                vars.Add("totaldiscs", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.TotalDiscs)));
                vars.Add("releasedateandlabel", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.ReleaseDateAndLabel)));
                vars.Add("discnumberandname", General.EmptyStringToNull(_config.CleanseString(cueSheet.Metadata.DiscNumberAndName.Replace("/", " of "))));
                NameValueCollection tags = cueSheet.Tags;
                if (tags != null)
                    foreach (string tag in tags.AllKeys)
                    {
                        string key = tag.ToLower();
                        string val = tags[tag];
                        if (vars.Get(key) == null && val != null && val != "")
                            vars.Add(key, _config.CleanseString(val));
                    }
            }

            vars.Add("unique", null);

            string outputPath = "";
            for (int maxLen = 255; maxLen >= 8; maxLen--)
            {
                outputPath = General.ReplaceMultiple(format, vars, "unique", pathOut => File.Exists(Path.ChangeExtension(pathOut, ext)), maxLen);
                if (outputPath == "" || outputPath == null)
                    return "";
                try { outputPath = Path.ChangeExtension(outputPath, ext); }
                catch { return ""; }
                if (outputPath.Length < 255)
                    return outputPath;
            }
            return outputPath;
        }

        private bool CheckIfFileExists(string output)
        {
            return File.Exists(Path.Combine(OutputDir, output));
        }

        public void GenerateFilenames(AudioEncoderType audioEncoderType, string format, string outputPath)
        {
            _audioEncoderType = audioEncoderType;
            _outputLossyWAV = format.StartsWith("lossy.");
            _outputFormat = format;
            _outputPath = outputPath;

            string extension = "." + format;
            string filename;
            int iTrack;

            NameValueCollection vars = new NameValueCollection();
            vars.Add("unique", null);
            vars.Add("album artist", General.EmptyStringToNull(_config.CleanseString(Metadata.Artist)));
            vars.Add("artist", General.EmptyStringToNull(_config.CleanseString(Metadata.Artist)));
            vars.Add("album", General.EmptyStringToNull(_config.CleanseString(Metadata.Title)));
            vars.Add("year", General.EmptyStringToNull(_config.CleanseString(Metadata.Year)));
            vars.Add("catalog", General.EmptyStringToNull(_config.CleanseString(Metadata.Barcode)));
            vars.Add("discnumber", General.EmptyStringToNull(_config.CleanseString(Metadata.DiscNumber01)));
            vars.Add("totaldiscs", General.EmptyStringToNull(_config.CleanseString(Metadata.TotalDiscs)));
            vars.Add("filename", Path.GetFileNameWithoutExtension(outputPath));
            vars.Add("tracknumber", null);
            vars.Add("title", null);

            if (_config.detectHDCD && _config.decodeHDCD && (!_outputLossyWAV || !_config.decodeHDCDtoLW16))
            {
                if (_config.decodeHDCDto24bit)
                    extension = ".24bit" + extension;
                else
                    extension = ".20bit" + extension;
            }

			for (int maxLen = 255; maxLen >= 8; maxLen--)
			{
				ArLogFileName = General.ReplaceMultiple(_config.ArLogFilenameFormat, vars, "unique", CheckIfFileExists, maxLen);
				if (ArLogFileName == "" || ArLogFileName == null)
				{
					ArLogFileName = "ar.log";
					break;
				}
				if (Path.Combine(OutputDir, ArLogFileName).Length < 255)
					break;
			}

            AlArtFileName = General.ReplaceMultiple(_config.AlArtFilenameFormat, vars, "unique", CheckIfFileExists, -1)
                ?? "folder.jpg";

            if (OutputStyle == CUEStyle.SingleFileWithCUE)
                SingleFilename = Path.ChangeExtension(Path.GetFileName(outputPath), extension);
            else if (_config.keepOriginalFilenames && HasSingleFilename)
                SingleFilename = Path.ChangeExtension(SingleFilename, extension);
            else
                SingleFilename = (General.ReplaceMultiple(_config.singleFilenameFormat, vars, -1) ?? "range") + extension;

            for (iTrack = -1; iTrack < TrackCount; iTrack++)
            {
                bool htoa = (iTrack == -1);

                if (_config.keepOriginalFilenames && htoa && HasHTOAFilename)
                {
                    HTOAFilename = Path.ChangeExtension(HTOAFilename, extension);
                }
                else if (_config.keepOriginalFilenames && !htoa && HasTrackFilenames)
                {
                    TrackFilenames[iTrack] = Path.ChangeExtension(
                        TrackFilenames[iTrack], extension);
                }
                else
                {
                    string trackStr = htoa ? "00" : String.Format("{0:00}", iTrack + 1);
                    string artist = Metadata.Tracks[htoa ? 0 : iTrack].Artist;
                    string title = htoa ? "(HTOA)" : Metadata.Tracks[iTrack].Title;

                    vars["tracknumber"] = trackStr;
                    vars["artist"] = General.EmptyStringToNull(_config.CleanseString(artist)) ?? vars["album artist"];
                    vars["title"] = General.EmptyStringToNull(_config.CleanseString(title));

                    filename = "";
                    for (int maxLen = 255; maxLen >= 8; maxLen--)
                    {
                        filename = General.ReplaceMultiple(_config.trackFilenameFormat, vars, maxLen);
                        if (filename == "" || filename == null)
                        {
                            filename = vars["tracknumber"];
                            break;
                        }
                        if (OutputDir.Length + filename.Length < 255)
                            break;
                    }

                    filename = filename + extension;

                    if (htoa)
                        HTOAFilename = filename;
                    else
                        TrackFilenames[iTrack] = filename;
                }
            }

            if (OutputStyle == CUEStyle.SingleFile || OutputStyle == CUEStyle.SingleFileWithCUE)
            {
                _destPaths = new string[1];
                _destPaths[0] = Path.Combine(OutputDir, _singleFilename);
            }
            else
            {
                bool htoaToFile = ((OutputStyle == CUEStyle.GapsAppended) && _config.preserveHTOA &&
                    (_toc.Pregap != 0));
                _destPaths = new string[TrackCount + (htoaToFile ? 1 : 0)];
                if (htoaToFile)
                    _destPaths[0] = Path.Combine(OutputDir, _htoaFilename);
                for (int i = 0; i < TrackCount; i++)
                    _destPaths[i + (htoaToFile ? 1 : 0)] = Path.Combine(OutputDir, _trackFilenames[i]);
            }
        }

        public List<string> OutputExists()
        {
            List<string> outputExists = new List<string>();
            bool outputCUE = Action == CUEAction.Encode && (OutputStyle != CUEStyle.SingleFileWithCUE || _config.createCUEFileWhenEmbedded);
            bool outputAudio = Action == CUEAction.Encode && _audioEncoderType != AudioEncoderType.NoAudio;
            if (outputCUE)
                outputExists.Add(_outputPath);
            if (isUsingAccurateRip && (
                (Action == CUEAction.Encode && _config.writeArLogOnConvert) ||
                (Action == CUEAction.Verify && _config.writeArLogOnVerify)))
                outputExists.Add(Path.Combine(OutputDir, ArLogFileName));
            if (outputAudio)
            {
                if (_config.extractAlbumArt && AlbumArt != null && AlbumArt.Count != 0)
                    outputExists.Add(Path.Combine(OutputDir, AlArtFileName));
                if (OutputStyle == CUEStyle.SingleFile || OutputStyle == CUEStyle.SingleFileWithCUE)
                    outputExists.Add(Path.Combine(OutputDir, SingleFilename));
                else
                {
                    if (OutputStyle == CUEStyle.GapsAppended && _config.preserveHTOA)
                        outputExists.Add(Path.Combine(OutputDir, HTOAFilename));
                    for (int i = 0; i < TrackCount; i++)
                        outputExists.Add(Path.Combine(OutputDir, TrackFilenames[i]));
                }
            }
            outputExists.RemoveAll(path => !File.Exists(path));
            return outputExists;
        }

        private int GetSampleLength(string path, out TagLib.File fileInfo)
        {
            ShowProgress("Analyzing input file...", 0.0, path, null);

            if (Path.GetExtension(path).ToLower() == ".dummy" || Path.GetExtension(path).ToLower() == ".bin")
            {
                fileInfo = null;
            }
            else
            {
                TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                TagLib.File.IFileAbstraction file = _isArchive
                    ? (TagLib.File.IFileAbstraction)new ArchiveFileAbstraction(this, path)
                    : (TagLib.File.IFileAbstraction)new TagLib.File.LocalFileAbstraction(path);
                fileInfo = TagLib.File.Create(file);
                if (fileInfo.Properties.AudioSampleCount > 0)
                    return (int)fileInfo.Properties.AudioSampleCount;
            }

            IAudioSource audioSource = AudioReadWrite.GetAudioSource(path, _isArchive ? OpenArchive(path, true) : null, _config);
            try
            {
                if (!audioSource.PCM.IsRedBook)
                    throw new Exception("Audio format is not Red Book PCM.");
                if (audioSource.Length <= 0)
                {
                    AudioBuffer buff = new AudioBuffer(audioSource, 0x10000);
                    while (audioSource.Read(buff, -1) != 0)
                        CheckStop();
                }
                if (audioSource.Length <= 0 ||
                    audioSource.Length >= Int32.MaxValue)
                    throw new Exception("Audio file length is unknown or invalid.");
                return (int)audioSource.Length;
            }
            finally
            {
                audioSource.Close();
            }
        }

        public static void WriteText(string path, string text)
        {
            bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(text)) != text;
            var encoding = utf8Required ? Encoding.UTF8 : CUESheet.Encoding;
            using (StreamWriter sw1 = new StreamWriter(path, false, encoding))
                sw1.Write(text);
        }

        public bool PrintErrors(StringWriter logWriter, uint tr_start, uint len)
        {
            uint tr_end = (len + 74) / 75;
            int errCount = 0;
            for (uint iSecond = 0; iSecond < tr_end; iSecond++)
            {
                uint sec_start = tr_start + iSecond * 75;
                uint sec_end = Math.Min(sec_start + 74, tr_start + len - 1);
                bool fError = false;
                for (uint iSector = sec_start; iSector <= sec_end; iSector++)
                    if (_ripper.Errors[(int)iSector - (int)_toc[_toc.FirstAudio][0].Start])
                        fError = true;
                if (fError)
                {
                    uint end = tr_end - 1;
                    for (uint jSecond = iSecond + 1; jSecond < tr_end; jSecond++)
                    {
                        uint jsec_start = tr_start + jSecond * 75;
                        uint jsec_end = Math.Min(jsec_start + 74, tr_start + len - 1);
                        bool jfError = false;
                        for (uint jSector = jsec_start; jSector <= jsec_end; jSector++)
                            if (_ripper.Errors[(int)jSector - (int)_toc[_toc.FirstAudio][0].Start])
                                jfError = true;
                        if (!jfError)
                        {
                            end = jSecond - 1;
                            break;
                        }
                    }
                    if (errCount == 0)
                        logWriter.WriteLine();
                    if (errCount++ > 20)
                        break;
                    //"Suspicious position 0:02:20"
                    //"   Suspicious position 0:02:23 - 0:02:24"
                    string s1 = CDImageLayout.TimeToString("0:{0:00}:{1:00}", iSecond * 75);
                    string s2 = CDImageLayout.TimeToString("0:{0:00}:{1:00}", end * 75);
                    if (iSecond == end)
                        logWriter.WriteLine("     Suspicious position {0}", s1);
                    else
                        logWriter.WriteLine("     Suspicious position {0} - {1}", s1, s2);
                    iSecond = end + 1;
                }
            }
            return errCount > 0;
        }

        public void CreateRipperLOG()
        {
            if (!_isCD || _ripper == null)
                return;

            _ripperLog = _config.createEACLOG ?
                CUESheetLogWriter.GetExactAudioCopyLog(this) :
                CUESheetLogWriter.GetRipperLog(this);
        }

        public string GetM3UContents(CUEStyle style)
        {
            StringWriter sw = new StringWriter();
            if (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0)
                WriteLine(sw, 0, _htoaFilename);
            for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                WriteLine(sw, 0, _trackFilenames[iTrack]);
            sw.Close();
            return sw.ToString();
        }

        public string GetCUESheetContents()
        {
            CUEStyle style = _hasEmbeddedCUESheet ? CUEStyle.SingleFile
                : _hasSingleFilename ? CUEStyle.SingleFileWithCUE
                : CUEStyle.GapsAppended;
            bool htoaToFile = _hasHTOAFilename;
            return GetCUESheetContents(style, htoaToFile);
        }

        public string GetCUESheetContents(CUEStyle style)
        {
            return GetCUESheetContents(style, (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0));
        }

        public string GetCUESheetContents(CUEStyle style, bool htoaToFile)
        {
            StringWriter sw = new StringWriter();
            int i, iTrack, iIndex;

            uint timeRelativeToFileStart = 0;

            General.SetCUELine(_attributes, "PERFORMER", Metadata.Artist, true);
            General.SetCUELine(_attributes, "TITLE", Metadata.Title, true);
            General.SetCUELine(_attributes, "CATALOG", Metadata.Barcode, false);
            General.SetCUELine(_attributes, "REM", "DATE", Metadata.Year, false);
            General.SetCUELine(_attributes, "REM", "DISCNUMBER", Metadata.DiscNumber, false);
            General.SetCUELine(_attributes, "REM", "TOTALDISCS", Metadata.TotalDiscs, false);
            General.SetCUELine(_attributes, "REM", "GENRE", Metadata.Genre, true);
            General.SetCUELine(_attributes, "REM", "COMMENT", Metadata.Comment, true);
            for (i = 0; i < Tracks.Count; i++)
            {
                General.SetCUELine(Tracks[i].Attributes, "PERFORMER", Metadata.Tracks[i].Artist, true);
                General.SetCUELine(Tracks[i].Attributes, "TITLE", Metadata.Tracks[i].Title, true);
                General.SetCUELine(Tracks[i].Attributes, "ISRC", Metadata.Tracks[i].ISRC, false);
            }

            using (sw)
            {
                if (_config.writeArTagsOnEncode)
                    WriteLine(sw, 0, "REM ACCURATERIPID " + (_accurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(_toc)));

                for (i = 0; i < _attributes.Count; i++)
                    WriteLine(sw, 0, _attributes[i]);

                if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
                    WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _singleFilename));

                if (htoaToFile)
                    WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _htoaFilename));

                for (iTrack = 0; iTrack < TrackCount; iTrack++)
                {
                    if ((style == CUEStyle.GapsPrepended) ||
                        (style == CUEStyle.GapsLeftOut) ||
                        ((style == CUEStyle.GapsAppended) &&
                        ((_toc[_toc.FirstAudio + iTrack].Pregap == 0) || ((iTrack == 0) && !htoaToFile))))
                    {
                        WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
                        timeRelativeToFileStart = 0;
                    }

                    WriteLine(sw, 1, String.Format("TRACK {0:00} AUDIO", iTrack + 1));
                    for (i = 0; i < _tracks[iTrack].Attributes.Count; i++)
                        WriteLine(sw, 2, _tracks[iTrack].Attributes[i]);

                    if (_toc[_toc.FirstAudio + iTrack].Pregap != 0)
                    {
                        if (((style == CUEStyle.GapsLeftOut) ||
                            ((style == CUEStyle.GapsAppended) && (iTrack == 0) && !htoaToFile) ||
                            ((style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE) && (iTrack == 0) && _usePregapForFirstTrackInSingleFile)))
                            WriteLine(sw, 2, "PREGAP " + CDImageLayout.TimeToString(_toc[_toc.FirstAudio + iTrack].Pregap));
                        else
                        {
                            WriteLine(sw, 2, String.Format("INDEX 00 {0}", CDImageLayout.TimeToString(timeRelativeToFileStart)));
                            timeRelativeToFileStart += _toc[_toc.FirstAudio + iTrack].Pregap;
                            if (style == CUEStyle.GapsAppended)
                            {
                                WriteLine(sw, 0, String.Format("FILE \"{0}\" WAVE", _trackFilenames[iTrack]));
                                timeRelativeToFileStart = 0;
                            }
                        }
                    }
                    for (iIndex = 1; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
                    {
                        WriteLine(sw, 2, String.Format("INDEX {0:00} {1}", iIndex, CDImageLayout.TimeToString(timeRelativeToFileStart)));
                        timeRelativeToFileStart += _toc.IndexLength(_toc.FirstAudio + iTrack, iIndex);
                    }
                }
            }
            sw.Close();
            return sw.ToString();
        }

        public void GenerateCTDBLog(TextWriter sw)
        {
            sw.WriteLine("[CTDB TOCID: {0}] {1}.", _toc.TOCID, _CUEToolsDB.DBStatus ?? "found");
            if (!_processed)
                return;
            if (_CUEToolsDB.SubStatus != null)
                sw.WriteLine("CUETools DB: {0}.", _CUEToolsDB.SubStatus);
            _CUEToolsDB.GenerateLog(sw, _config.advanced.DetailedCTDBLog);
        }

        public string GenerateAccurateRipStatus()
        {
            string prefix = "";
            if (hdcdDecoder != null && string.Format("{0:s}", hdcdDecoder) != "")
                prefix += string.Format("{0:s}", hdcdDecoder);
            if (isUsingAccurateRip)
            {
                if (prefix != "") prefix += ", ";
                prefix += "AR: ";
                if (_arVerify.ARStatus != null)
                    prefix += _arVerify.ARStatus;
                else
                {
                    uint tracksMatch = 0;
                    int bestOffset = 0;
                    FindBestOffset(1, false, out tracksMatch, out bestOffset);
                    if (bestOffset != 0)
                        prefix += string.Format("offset {0}, ", bestOffset);
                    if (_arVerify.WorstConfidence() > 0)
                        prefix += string.Format("rip accurate ({0}/{1})", _arVerify.WorstConfidence(), _arVerify.WorstTotal());
                    else
                        prefix += string.Format("rip not accurate ({0}/{1})", 0, _arVerify.WorstTotal());
                }
            }
            if (!isUsingCUEToolsDBFix && isUsingCUEToolsDB)
            {
                if (prefix != "") prefix += ", ";
                prefix += "CTDB: " + CTDB.Status;
            }
            if (_isCD && _ripper.ErrorsCount > 0)
            {
                if (prefix != "") prefix += ", ";
                prefix += "ripper found " + _ripper.ErrorsCount + " suspicious sectors";
            }
            if (prefix == "")
                prefix += "done";
            return prefix;
        }

        public void GenerateAccurateRipTagsForTrack(NameValueCollection tags, int bestOffset, int iTrack, string prefix)
        {
            tags.Add(String.Format("{0}ACCURATERIPCRC", prefix), String.Format("{0:x8}", _arVerify.CRC(iTrack, 0)));
            tags.Add(String.Format("{0}AccurateRipDiscId", prefix), String.Format("{0:000}-{1}-{2:00}", TrackCount, _accurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(_toc), iTrack + 1));
            tags.Add(String.Format("{0}ACCURATERIPCOUNT", prefix), String.Format("{0}", _arVerify.Confidence(iTrack, 0)));
            tags.Add(String.Format("{0}ACCURATERIPCOUNTALLOFFSETS", prefix), String.Format("{0}", _arVerify.SumConfidence(iTrack)));
            tags.Add(String.Format("{0}ACCURATERIPTOTAL", prefix), String.Format("{0}", _arVerify.Total(iTrack)));
            if (bestOffset != 0)
                tags.Add(String.Format("{0}ACCURATERIPCOUNTWITHOFFSET", prefix), String.Format("{0}", _arVerify.Confidence(iTrack, bestOffset)));
        }

        public void GenerateAccurateRipTags(NameValueCollection tags, int bestOffset, int iTrack)
        {
            tags.Add("ACCURATERIPID", _accurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(_toc));
            if (bestOffset != 0)
                tags.Add("ACCURATERIPOFFSET", String.Format("{1}{0}", bestOffset, bestOffset > 0 ? "+" : ""));
            if (iTrack != -1)
                GenerateAccurateRipTagsForTrack(tags, bestOffset, iTrack, "");
            else
                for (iTrack = 0; iTrack < TrackCount; iTrack++)
                {
                    GenerateAccurateRipTagsForTrack(tags, bestOffset, iTrack,
                        String.Format("cue_track{0:00}_", iTrack + 1));
                }
        }

        public void CleanupTags(NameValueCollection tags, string substring)
        {
            string[] keys = tags.AllKeys;
            for (int i = 0; i < keys.Length; i++)
                if (keys[i].ToUpper().Contains(substring))
                    tags.Remove(keys[i]);
        }

        public void FindBestOffset(uint minConfidence, bool optimizeConfidence, out uint outTracksMatch, out int outBestOffset)
        {
            uint bestTracksMatch = 0;
            uint bestConfidence = 0;
            int bestOffset = 0;

            for (int offset = -_arOffsetRange; offset <= _arOffsetRange; offset++)
            {
                uint tracksMatch = 0;
                uint sumConfidence = 0;

                for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                {
                    uint confidence = 0;

                    for (int di = 0; di < (int)_arVerify.AccDisks.Count; di++)
                        if (_arVerify.CRC(iTrack, offset) == _arVerify.AccDisks[di].tracks[iTrack].CRC
                          || offset == 0 && _arVerify.CRCV2(iTrack) == _arVerify.AccDisks[di].tracks[iTrack].CRC)
                            confidence += _arVerify.AccDisks[di].tracks[iTrack].count;

                    if (confidence >= minConfidence)
                        tracksMatch++;

                    sumConfidence += confidence;
                }

                if (tracksMatch > bestTracksMatch
                    || (tracksMatch == bestTracksMatch && optimizeConfidence && sumConfidence > bestConfidence)
                    || (tracksMatch == bestTracksMatch && optimizeConfidence && sumConfidence == bestConfidence && Math.Abs(offset) < Math.Abs(bestOffset))
                    || (tracksMatch == bestTracksMatch && !optimizeConfidence && Math.Abs(offset) < Math.Abs(bestOffset))
                    )
                {
                    bestTracksMatch = tracksMatch;
                    bestConfidence = sumConfidence;
                    bestOffset = offset;
                }
            }
            outBestOffset = bestOffset;
            outTracksMatch = bestTracksMatch;
        }

        public void TestBeforeCopy()
        {
            if (!_isCD)
                throw new Exception("Not a cd");

            this.DetectGaps();

            _arTestVerify = new AccurateRipVerify(_toc, proxy);
            var buff = new AudioBuffer(AudioPCMConfig.RedBook, 0x10000);
            while (_ripper.Read(buff, -1) != 0)
            {
                _arTestVerify.Write(buff);
            }

            _ripper.Position = 0;
        }

        public string Go()
        {
            int[] destLengths;
            bool htoaToFile = ((OutputStyle == CUEStyle.GapsAppended) && _config.preserveHTOA &&
                (_toc.Pregap != 0));

            if (_isCD)
                DetectGaps();

            if (_usePregapForFirstTrackInSingleFile)
                throw new Exception("UsePregapForFirstTrackInSingleFile is not supported for writing audio files.");

            if (_action != CUEAction.Verify)
                for (int i = 0; i < _destPaths.Length; i++)
                    for (int j = 0; j < _sourcePaths.Count; j++)
                        if (_destPaths[i].ToLower() == _sourcePaths[j].ToLower())
                            throw new Exception("Source and destination audio file paths cannot be the same.");

            destLengths = CalculateAudioFileLengths(OutputStyle);

            // Lookup();

            if (_action != CUEAction.Verify)
            {
                if (!Directory.Exists(OutputDir))
                    Directory.CreateDirectory(OutputDir);
            }

            if (_action == CUEAction.Encode)
            {
                string cueContents = GetCUESheetContents(OutputStyle);
                if (_config.createEACLOG && _isCD)
                    cueContents = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(cueContents));
                if (OutputStyle == CUEStyle.SingleFileWithCUE && _config.createCUEFileWhenEmbedded)
                    WriteText(Path.ChangeExtension(_outputPath, ".cue"), cueContents);
                else
                    WriteText(_outputPath, cueContents);
            }

            if (_action == CUEAction.Verify)
                VerifyAudio();
            //				WriteAudioFilesPass(OutputDir, OutputStyle, destLengths, htoaToFile, _action == CUEAction.Verify);
            else if (_audioEncoderType != AudioEncoderType.NoAudio)
                WriteAudioFilesPass(OutputDir, OutputStyle, destLengths, htoaToFile, _action == CUEAction.Verify);

            if (isUsingCUEToolsDB && !isUsingCUEToolsDBFix)
            {
                if (_isCD) _CUEToolsDB.ContactDB(true, true, CTDBMetadataSearch.None);
                _CUEToolsDB.DoVerify();
            }

            _processed = true;

            CreateRipperLOG();

            if (_action == CUEAction.Verify && _useLocalDB)
            {
                var now = DateTime.Now;
                var entry = OpenLocalDBEntry();
                entry.Status = this.GenerateAccurateRipStatus();
                entry.ARConfidence = isUsingAccurateRip ? _arVerify.WorstConfidence() : 0;
                entry.CTDBConfidence = isUsingCUEToolsDB && !isUsingCUEToolsDBFix ? CTDB.Confidence : 0;
                entry.Log = AccurateRipLog;
                entry.VerificationDate =
                    isUsingAccurateRip &&
                    (_arVerify.ExceptionStatus == WebExceptionStatus.Success ||
                      (_arVerify.ExceptionStatus == WebExceptionStatus.ProtocolError &&
                        _arVerify.ResponseStatus == HttpStatusCode.NotFound
                      )
                    ) ? now : DateTime.MinValue;
                entry.CTDBVerificationDate =
                    isUsingCUEToolsDB &&
                    !isUsingCUEToolsDBFix &&
                    (CTDB.QueryExceptionStatus == WebExceptionStatus.Success ||
                      (CTDB.QueryExceptionStatus == WebExceptionStatus.ProtocolError &&
                        CTDB.QueryResponseStatus == HttpStatusCode.NotFound
                      )
                    ) ? now : DateTime.MinValue;
                entry.OffsetSafeCRC = _arVerify.OffsetSafeCRC;
            }

            if (_action == CUEAction.Encode)
            {
                uint tracksMatch = 0;
                int bestOffset = 0;

                if (isUsingAccurateRip &&
                    _config.writeArTagsOnEncode &&
                    _arVerify.ExceptionStatus == WebExceptionStatus.Success)
                    FindBestOffset(1, true, out tracksMatch, out bestOffset);

                if (_config.createEACLOG && _ripperLog != null)
                    _ripperLog = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(_ripperLog));

                if (_ripperLog != null)
                    WriteText(Path.ChangeExtension(_outputPath, ".log"), _ripperLog);
                else
                    if (_eacLog != null && _config.extractLog)
                        WriteText(Path.ChangeExtension(_outputPath, ".log"), _eacLog);

                if (_audioEncoderType != AudioEncoderType.NoAudio && _config.extractAlbumArt)
                    ExtractAlbumArt();

                bool fNeedAlbumArtist = false;
                for (int iTrack = 1; iTrack < TrackCount; iTrack++)
                    if (Metadata.Tracks[iTrack].Artist != Metadata.Tracks[0].Artist)
                        fNeedAlbumArtist = true;

                if (OutputStyle == CUEStyle.SingleFileWithCUE || OutputStyle == CUEStyle.SingleFile)
                {
                    if (_audioEncoderType != AudioEncoderType.NoAudio)
                    {
                        NameValueCollection tags = GenerateAlbumTags(bestOffset, OutputStyle == CUEStyle.SingleFileWithCUE, _ripperLog ?? _eacLog);
                        TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                        TagLib.File fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(_destPaths[0]));
                        if (Tagging.UpdateTags(fileInfo, tags, _config))
                        {
                            TagLib.File sourceFileInfo = _tracks[0]._fileInfo ?? _fileInfo;

                            // first, use cue sheet information
                            if (_config.writeBasicTagsFromCUEData)
                            {
                                uint temp;
                                if (fileInfo.Tag.Album == null && Metadata.Title != "")
                                    fileInfo.Tag.Album = Metadata.Title;
                                if (fNeedAlbumArtist && fileInfo.Tag.AlbumArtists.Length == 0 && Metadata.Artist != "")
                                    fileInfo.Tag.AlbumArtists = new string[] { Metadata.Artist };
                                if (!fNeedAlbumArtist && fileInfo.Tag.Performers.Length == 0 && Metadata.Artist != "")
                                    fileInfo.Tag.Performers = new string[] { Metadata.Artist };
                                if (fileInfo.Tag.Genres.Length == 0 && Metadata.Genre != "")
                                    fileInfo.Tag.Genres = new string[] { Metadata.Genre };
                                if (fileInfo.Tag.DiscCount == 0 && Metadata.TotalDiscs != "" && uint.TryParse(Metadata.TotalDiscs, out temp))
                                    fileInfo.Tag.DiscCount = temp;
                                if (fileInfo.Tag.Disc == 0 && Metadata.DiscNumber != "" && uint.TryParse(Metadata.DiscNumber, out temp))
                                    fileInfo.Tag.Disc = temp;
                                if (fileInfo.Tag.Year == 0 && Metadata.Year != "" && uint.TryParse(Metadata.Year, out temp))
                                    fileInfo.Tag.Year = temp;
                                if (fileInfo.Tag.Comment == null && Metadata.Comment != "")
                                    fileInfo.Tag.Comment = Metadata.Comment;
								if (fileInfo.Tag.ReleaseDate == null && Metadata.ReleaseDate != "")
									fileInfo.Tag.ReleaseDate = Metadata.ReleaseDate;
								if (fileInfo.Tag.MusicBrainzReleaseCountry == null && Metadata.Country != "")
									fileInfo.Tag.MusicBrainzReleaseCountry = Metadata.Country;
								if (fileInfo.Tag.Publisher == null && Metadata.Label != "")
									fileInfo.Tag.Publisher = Metadata.Label;
								if (fileInfo.Tag.CatalogNo == null && Metadata.LabelNo != "")
									fileInfo.Tag.CatalogNo = Metadata.LabelNo;
								if (fileInfo.Tag.DiscSubtitle == null && Metadata.DiscName != "")
									fileInfo.Tag.DiscSubtitle = Metadata.DiscName;
							}

                            // fill up missing information from tags
                            if (_config.copyBasicTags && sourceFileInfo != null)
                            {
                                if (fileInfo.Tag.DiscCount == 0)
                                    fileInfo.Tag.DiscCount = sourceFileInfo.Tag.DiscCount; // TODO: GetCommonTag?
                                if (fileInfo.Tag.Disc == 0)
                                    fileInfo.Tag.Disc = sourceFileInfo.Tag.Disc;
                                //fileInfo.Tag.Performers = sourceFileInfo.Tag.Performers;
                                if (fileInfo.Tag.Album == null)
                                    fileInfo.Tag.Album = sourceFileInfo.Tag.Album;
                                if (fileInfo.Tag.Performers.Length == 0)
                                    fileInfo.Tag.Performers = sourceFileInfo.Tag.Performers;
                                if (fileInfo.Tag.AlbumArtists.Length == 0)
                                    fileInfo.Tag.AlbumArtists = sourceFileInfo.Tag.AlbumArtists;
                                if (fileInfo.Tag.Genres.Length == 0)
                                    fileInfo.Tag.Genres = sourceFileInfo.Tag.Genres;
                                if (fileInfo.Tag.Year == 0)
                                    fileInfo.Tag.Year = sourceFileInfo.Tag.Year;
                                if (fileInfo.Tag.Comment == null)
                                    fileInfo.Tag.Comment = sourceFileInfo.Tag.Comment;
								if (fileInfo.Tag.ReleaseDate == null)
									fileInfo.Tag.ReleaseDate = sourceFileInfo.Tag.ReleaseDate;
								if (fileInfo.Tag.MusicBrainzReleaseCountry == null)
									fileInfo.Tag.MusicBrainzReleaseCountry = sourceFileInfo.Tag.MusicBrainzReleaseCountry;
								if (fileInfo.Tag.Publisher == null)
									fileInfo.Tag.Publisher = sourceFileInfo.Tag.Publisher;
								if (fileInfo.Tag.CatalogNo == null)
									fileInfo.Tag.CatalogNo = sourceFileInfo.Tag.CatalogNo;
								if (fileInfo.Tag.DiscSubtitle == null)
									fileInfo.Tag.DiscSubtitle = sourceFileInfo.Tag.DiscSubtitle;
							}

                            if ((_config.embedAlbumArt || _config.CopyAlbumArt) && _albumArt.Count > 0)
                                fileInfo.Tag.Pictures = _albumArt.ToArray();

                            fileInfo.Save();
                        }
                    }
                }
                else
                {
                    if (_config.createM3U)
                        WriteText(Path.ChangeExtension(_outputPath, ".m3u"), GetM3UContents(OutputStyle));
                    if (_audioEncoderType != AudioEncoderType.NoAudio)
                        for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                        {
                            string path = _destPaths[iTrack + (htoaToFile ? 1 : 0)];
                            NameValueCollection tags = GenerateTrackTags(iTrack, bestOffset);
                            TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                            TagLib.File fileInfo = TagLib.File.Create(new TagLib.File.LocalFileAbstraction(path));
                            if (Tagging.UpdateTags(fileInfo, tags, _config))
                            {
                                TagLib.File sourceFileInfo = _tracks[iTrack]._fileInfo ?? _fileInfo;

                                if (_config.writeBasicTagsFromCUEData)
                                {
                                    uint temp;
                                    fileInfo.Tag.TrackCount = (uint)TrackCount;
                                    fileInfo.Tag.Track = (uint)iTrack + 1;
                                    if (fileInfo.Tag.Title == null && Metadata.Tracks[iTrack].Title != "")
                                        fileInfo.Tag.Title = Metadata.Tracks[iTrack].Title;
                                    if (fileInfo.Tag.Album == null && Metadata.Title != "")
                                        fileInfo.Tag.Album = Metadata.Title;
                                    if (fileInfo.Tag.Performers.Length == 0 && Metadata.Tracks[iTrack].Artist != "")
                                        fileInfo.Tag.Performers = new string[] { Metadata.Tracks[iTrack].Artist };
                                    if (fileInfo.Tag.Performers.Length == 0 && Metadata.Artist != "")
                                        fileInfo.Tag.Performers = new string[] { Metadata.Artist };
                                    if (fNeedAlbumArtist && fileInfo.Tag.AlbumArtists.Length == 0 && Metadata.Artist != "")
                                        fileInfo.Tag.AlbumArtists = new string[] { Metadata.Artist };
                                    if (fileInfo.Tag.Genres.Length == 0 && Metadata.Genre != "")
                                        fileInfo.Tag.Genres = new string[] { Metadata.Genre };
                                    if (fileInfo.Tag.DiscCount == 0 && Metadata.TotalDiscs != "" && uint.TryParse(Metadata.TotalDiscs, out temp))
                                        fileInfo.Tag.DiscCount = temp;
                                    if (fileInfo.Tag.Disc == 0 && Metadata.DiscNumber != "" && uint.TryParse(Metadata.DiscNumber, out temp))
                                        fileInfo.Tag.Disc = temp;
                                    if (fileInfo.Tag.Year == 0 && Metadata.Year != "" && uint.TryParse(Metadata.Year, out temp))
                                        fileInfo.Tag.Year = temp;
                                    if (fileInfo.Tag.Comment == null && Metadata.Comment != "")
                                        fileInfo.Tag.Comment = Metadata.Comment;
									if (fileInfo.Tag.ReleaseDate == null && Metadata.ReleaseDate != "")
										fileInfo.Tag.ReleaseDate = Metadata.ReleaseDate;
									if (fileInfo.Tag.MusicBrainzReleaseCountry == null && Metadata.Country != "")
										fileInfo.Tag.MusicBrainzReleaseCountry = Metadata.Country;
									if (fileInfo.Tag.Publisher == null && Metadata.Label != "")
										fileInfo.Tag.Publisher = Metadata.Label;
									if (fileInfo.Tag.CatalogNo == null && Metadata.LabelNo != "")
										fileInfo.Tag.CatalogNo = Metadata.LabelNo;
									if (fileInfo.Tag.DiscSubtitle == null && Metadata.DiscName != "")
										fileInfo.Tag.DiscSubtitle = Metadata.DiscName;
								}

                                if (_config.copyBasicTags && sourceFileInfo != null)
                                {
                                    if (fileInfo.Tag.Title == null && _tracks[iTrack]._fileInfo != null)
                                        fileInfo.Tag.Title = _tracks[iTrack]._fileInfo.Tag.Title;
                                    if (fileInfo.Tag.DiscCount == 0)
                                        fileInfo.Tag.DiscCount = sourceFileInfo.Tag.DiscCount;
                                    if (fileInfo.Tag.Disc == 0)
                                        fileInfo.Tag.Disc = sourceFileInfo.Tag.Disc;
                                    if (fileInfo.Tag.Performers.Length == 0)
                                        fileInfo.Tag.Performers = sourceFileInfo.Tag.Performers;
                                    if (fileInfo.Tag.AlbumArtists.Length == 0)
                                        fileInfo.Tag.AlbumArtists = sourceFileInfo.Tag.AlbumArtists;
                                    if (fileInfo.Tag.Album == null)
                                        fileInfo.Tag.Album = sourceFileInfo.Tag.Album;
                                    if (fileInfo.Tag.Year == 0)
                                        fileInfo.Tag.Year = sourceFileInfo.Tag.Year;
                                    if (fileInfo.Tag.Genres.Length == 0)
                                        fileInfo.Tag.Genres = sourceFileInfo.Tag.Genres;
                                    if (fileInfo.Tag.Comment == null)
                                        fileInfo.Tag.Comment = sourceFileInfo.Tag.Comment;
									if (fileInfo.Tag.ReleaseDate == null)
										fileInfo.Tag.ReleaseDate = sourceFileInfo.Tag.ReleaseDate;
									if (fileInfo.Tag.MusicBrainzReleaseCountry == null)
										fileInfo.Tag.MusicBrainzReleaseCountry = sourceFileInfo.Tag.MusicBrainzReleaseCountry;
									if (fileInfo.Tag.Publisher == null)
										fileInfo.Tag.Publisher = sourceFileInfo.Tag.Publisher;
									if (fileInfo.Tag.CatalogNo == null)
										fileInfo.Tag.CatalogNo = sourceFileInfo.Tag.CatalogNo;
									if (fileInfo.Tag.DiscSubtitle == null)
										fileInfo.Tag.DiscSubtitle = sourceFileInfo.Tag.DiscSubtitle;
								}

                                if ((_config.embedAlbumArt || _config.CopyAlbumArt) && _albumArt.Count > 0)
                                    fileInfo.Tag.Pictures = _albumArt.ToArray();

                                fileInfo.Save();
                            }
                        }
                }
            }

            return WriteReport();
        }

        public CUEToolsLocalDBEntry OpenLocalDBEntry()
        {
            if (!_useLocalDB)
                return null;

            string path = CUEToolsLocalDBEntry.NormalizePath(InputPath);
            CUEToolsLocalDBEntry entry = _localDB.Lookup(_toc, _sourcePaths);
            if (entry.InputPaths == null)
                entry.InputPaths = new List<string>();
            if (!entry.InputPaths.Contains(path))
                entry.InputPaths.Add(path);
            if (entry.Metadata == null)
                entry.Metadata = new CUEMetadata(cueMetadata);
            _localDB.Dirty = true;
            return entry;
        }

        private static Bitmap resizeImage(Image imgToResize, Size size)
        {
            int sourceWidth = imgToResize.Width;
            int sourceHeight = imgToResize.Height;

            float nPercent = 0;
            float nPercentW = 0;
            float nPercentH = 0;

            nPercentW = ((float)size.Width / (float)sourceWidth);
            nPercentH = ((float)size.Height / (float)sourceHeight);

            if (nPercentH < nPercentW)
                nPercent = nPercentH;
            else
                nPercent = nPercentW;

            int destWidth = (int)(sourceWidth * nPercent);
            int destHeight = (int)(sourceHeight * nPercent);

            Bitmap b = new Bitmap(destWidth, destHeight);
            Graphics g = Graphics.FromImage((Image)b);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;

            g.DrawImage(imgToResize, 0, 0, destWidth, destHeight);
            g.Dispose();

            return b;
        }

        public void ExtractAlbumArt()
        {
            if (!_config.extractAlbumArt || _albumArt.Count == 0)
                return;

            string imgPath = Path.Combine(OutputDir, AlArtFileName);
            foreach (TagLib.IPicture picture in _albumArt)
                using (FileStream file = new FileStream(imgPath, FileMode.Create, FileAccess.Write, FileShare.Read))
                {
                    file.Write(picture.Data.Data, 0, picture.Data.Count);
                    return;
                }
        }

        public void LoadAlbumArt(TagLib.File fileInfo)
        {
            if ((_config.extractAlbumArt || _config.CopyAlbumArt) && fileInfo != null)
                foreach (TagLib.IPicture picture in fileInfo.Tag.Pictures)
                    if (picture.Type == TagLib.PictureType.FrontCover)
                        if (picture.MimeType == "image/jpeg")
                        {
                            _albumArt.Add(picture);
                            return;
                        }
            if ((_config.extractAlbumArt || _config.embedAlbumArt) && !_isCD)
            {
                foreach (string tpl in _config.advanced.CoverArtFiles)
                {
                    string name = tpl.Replace("%album%", Metadata.Title).Replace("%artist%", Metadata.Artist);
                    string imgPath = Path.Combine(_isArchive ? _archiveCUEpath : _inputDir, name);
                    bool exists = _isArchive ? _archiveContents.Contains(imgPath) : File.Exists(imgPath);
                    if (exists)
                    {
                        TagLib.File.IFileAbstraction file = _isArchive
                            ? (TagLib.File.IFileAbstraction)new ArchiveFileAbstraction(this, imgPath)
                            : (TagLib.File.IFileAbstraction)new TagLib.File.LocalFileAbstraction(imgPath);
                        TagLib.Picture pic = new TagLib.Picture(file);
                        pic.Description = name;
                        _albumArt.Add(pic);
                        return;
                    }
                }

                if (!_isArchive && _config.advanced.CoverArtSearchSubdirs)
                {
                    List<string> allfiles = new List<string>(Directory.GetFiles(_inputDir, "*.jpg", SearchOption.AllDirectories));
                    // TODO: archive case
                    foreach (string tpl in _config.advanced.CoverArtFiles)
                    {
                        string name = tpl.Replace("%album%", Metadata.Title).Replace("%artist%", Metadata.Artist);
                        List<string> matching = allfiles.FindAll(s => Path.GetFileName(s) == name);
                        if (matching.Count == 1)
                        {
                            string imgPath = matching[0];
                            TagLib.File.IFileAbstraction file = _isArchive
                                ? (TagLib.File.IFileAbstraction)new ArchiveFileAbstraction(this, imgPath)
                                : (TagLib.File.IFileAbstraction)new TagLib.File.LocalFileAbstraction(imgPath);
                            TagLib.Picture pic = new TagLib.Picture(file);
                            pic.Description = name;
                            _albumArt.Add(pic);
                            return;
                        }
                    }

                    if (CUEToolsSelection != null
                       && ((Action == CUEAction.Encode && allfiles.Count < 32)
                         || (Action != CUEAction.Encode && allfiles.Count < 2)
                         )
                      )
                    {
                        foreach (string imgPath in allfiles)
                        {
                            TagLib.Picture pic = new TagLib.Picture(imgPath);
                            if (imgPath.StartsWith(_inputDir))
                                pic.Description = imgPath.Substring(_inputDir.Length).Trim(Path.DirectorySeparatorChar);
                            else
                                pic.Description = Path.GetFileName(imgPath);
                            _albumArt.Add(pic);
                        }
                        if (_albumArt.Count > 0)
                        {
                            CUEToolsSelectionEventArgs e = new CUEToolsSelectionEventArgs();
                            e.choices = _albumArt.ToArray();
                            CUEToolsSelection(this, e);
                            TagLib.IPicture selected = e.selection == -1 ? null : _albumArt[e.selection];
                            _albumArt.RemoveAll(t => t != selected);
                        }
                    }
                }
            }
        }

        public void AddAlbumArt(byte[] encoded)
        {
            var data = new TagLib.ByteVector(encoded);
            var picture = new TagLib.Picture(data);
            picture.Type = TagLib.PictureType.FrontCover;
            _albumArt.Add(picture);
        }

        public void ResizeAlbumArt()
        {
            if (_albumArt == null)
                return;
            foreach (TagLib.IPicture picture in _albumArt)
                using (MemoryStream imageStream = new MemoryStream(picture.Data.Data, 0, picture.Data.Count))
                    try
                    {
                        using (Image img = Image.FromStream(imageStream))
                            if (img.Width > _config.maxAlbumArtSize || img.Height > _config.maxAlbumArtSize)
                            {
                                using (Bitmap small = resizeImage(img, new Size(_config.maxAlbumArtSize, _config.maxAlbumArtSize)))
                                using (MemoryStream encoded = new MemoryStream())
                                {
                                    //System.Drawing.Imaging.EncoderParameters encoderParams = new EncoderParameters(1);
                                    //encoderParams.Param[0] = new System.Drawing.Imaging.EncoderParameter(Encoder.Quality, quality);
                                    small.Save(encoded, System.Drawing.Imaging.ImageFormat.Jpeg);
                                    picture.Data = new TagLib.ByteVector(encoded.ToArray());
                                    picture.MimeType = "image/jpeg";
                                }
                            }
                    }
                    catch
                    {
                    }
        }

        public string WriteReport()
        {
            if (isUsingAccurateRip)
            {
                ShowProgress((string)"Generating AccurateRip report...", 0, null, null);
                if (_action == CUEAction.Verify && _config.writeArTagsOnVerify && _writeOffset == 0 && !_isArchive && !_isCD)
                {
                    uint tracksMatch;
                    int bestOffset;
                    FindBestOffset(1, true, out tracksMatch, out bestOffset);

                    if (_hasEmbeddedCUESheet)
                    {
                        if (_fileInfo is TagLib.Flac.File)
                        {
                            NameValueCollection tags = Tagging.Analyze(_fileInfo);
                            CleanupTags(tags, "ACCURATERIP");
                            GenerateAccurateRipTags(tags, bestOffset, -1);
                            if (Tagging.UpdateTags(_fileInfo, tags, _config))
                                _fileInfo.Save();
                        }
                    }
                    else if (_hasTrackFilenames)
                    {
                        for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                            if (_tracks[iTrack]._fileInfo is TagLib.Flac.File)
                            {
                                NameValueCollection tags = Tagging.Analyze(_tracks[iTrack]._fileInfo);
                                CleanupTags(tags, "ACCURATERIP");
                                GenerateAccurateRipTags(tags, bestOffset, iTrack);
                                if (Tagging.UpdateTags(_tracks[iTrack]._fileInfo, tags, _config))
                                    _tracks[iTrack]._fileInfo.Save();
                            }
                    }
                }

                if ((_action != CUEAction.Verify && _config.writeArLogOnConvert) ||
                    (_action == CUEAction.Verify && _config.writeArLogOnVerify))
                {
                    if (!Directory.Exists(OutputDir))
                        Directory.CreateDirectory(OutputDir);

                    using (StreamWriter sw = new StreamWriter(Path.Combine(OutputDir, ArLogFileName), false, CUESheet.Encoding))
                    {
                        CUESheetLogWriter.WriteAccurateRipLog(this, sw);
                    }
                }
                if (_config.advanced.CreateTOC)
                {
                    if (!Directory.Exists(OutputDir))
                        Directory.CreateDirectory(OutputDir);
                    WriteText(Path.ChangeExtension(_outputPath, ".toc"), CUESheetLogWriter.GetTOCContents(this));
                }
            }
            return GenerateAccurateRipStatus();
        }

        private NameValueCollection GenerateTrackTags(int iTrack, int bestOffset)
        {
            NameValueCollection destTags = new NameValueCollection();

            if (_config.copyUnknownTags)
            {
                if (_hasEmbeddedCUESheet)
                {
                    if (_fileInfo != null)
                    {
                        string trackPrefix = String.Format("cue_track{0:00}_", iTrack + 1);
                        NameValueCollection albumTags = Tagging.Analyze(_fileInfo);
                        foreach (string key in albumTags.AllKeys)
                        {
                            if (key.ToLower().StartsWith(trackPrefix)
                                || !key.ToLower().StartsWith("cue_track"))
                            {
                                string name = key.ToLower().StartsWith(trackPrefix) ?
                                    key.Substring(trackPrefix.Length) : key;
                                string[] values = albumTags.GetValues(key);
                                for (int j = 0; j < values.Length; j++)
                                    destTags.Add(name, values[j]);
                            }
                        }
                    }
                }
                else if (_hasTrackFilenames)
                {
                    if (_tracks[iTrack]._fileInfo != null)
                        destTags.Add(Tagging.Analyze(_tracks[iTrack]._fileInfo));
                }
                else if (_hasSingleFilename)
                {
                    // TODO?
                }

                // these will be set explicitely
                destTags.Remove("ARTIST");
                destTags.Remove("TITLE");
                destTags.Remove("ALBUM");
                destTags.Remove("ALBUMARTIST");
                destTags.Remove("ALBUM ARTIST");
                destTags.Remove("DATE");
                destTags.Remove("GENRE");
                destTags.Remove("COMMENT");
                destTags.Remove("TRACKNUMBER");
                destTags.Remove("TRACKTOTAL");
                destTags.Remove("TOTALTRACKS");
                destTags.Remove("DISCNUMBER");
                destTags.Remove("DISCTOTAL");
                destTags.Remove("TOTALDISCS");

                destTags.Remove("LOG");
                destTags.Remove("LOGFILE");
                destTags.Remove("EACLOG");

                // these are not valid
                destTags.Remove("CUESHEET");
                CleanupTags(destTags, "ACCURATERIP");
                //CleanupTags(destTags, "REPLAYGAIN");
            }

            if (_config.writeArTagsOnEncode &&
                _action == CUEAction.Encode &&
                isUsingAccurateRip &&
                _arVerify.ExceptionStatus == WebExceptionStatus.Success)
                GenerateAccurateRipTags(destTags, bestOffset, iTrack);
            
            if (_config.advanced.WriteCDTOCTag)
                destTags.Add("CDTOC", _toc.TAG);

            return destTags;
        }

        private NameValueCollection GenerateAlbumTags(int bestOffset, bool fWithCUE, string logContents)
        {
            NameValueCollection destTags = new NameValueCollection();

            if (_config.copyUnknownTags)
            {
                if (_hasEmbeddedCUESheet || _hasSingleFilename)
                {
                    if (_fileInfo != null)
                        destTags.Add(Tagging.Analyze(_fileInfo));
                    if (!fWithCUE)
                        CleanupTags(destTags, "CUE_TRACK");
                }
                else if (_hasTrackFilenames)
                {
                    for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                    {
                        if (_tracks[iTrack]._fileInfo == null) continue;
                        NameValueCollection trackTags = Tagging.Analyze(_tracks[iTrack]._fileInfo);
                        foreach (string key in trackTags.AllKeys)
                        {
                            string singleValue = GetCommonMiscTag(key);
                            if (singleValue != null)
                            {
                                if (destTags.Get(key) == null)
                                    destTags.Add(key, singleValue);
                            }
                            else if (fWithCUE && key.ToUpper() != "TRACKNUMBER" && key.ToUpper() != "TITLE" && key.ToUpper() != "ARTIST")
                            {
                                string[] values = trackTags.GetValues(key);
                                for (int j = 0; j < values.Length; j++)
                                    destTags.Add(String.Format("cue_track{0:00}_{1}", iTrack + 1, key), values[j]);
                            }
                        }
                    }
                }

                // these will be set explicitely
                destTags.Remove("ARTIST");
                destTags.Remove("TITLE");
                destTags.Remove("ALBUM");
                destTags.Remove("ALBUMARTIST");
                destTags.Remove("ALBUM ARTIST");
                destTags.Remove("DATE");
                destTags.Remove("GENRE");
                destTags.Remove("COMMENT");
                destTags.Remove("TRACKNUMBER");
                destTags.Remove("TRACKTOTAL");
                destTags.Remove("TOTALTRACKS");
                destTags.Remove("DISCNUMBER");
                destTags.Remove("DISCTOTAL");
                destTags.Remove("TOTALDISCS");
                destTags.Remove("LOG");
                destTags.Remove("LOGFILE");
                destTags.Remove("EACLOG");

                // these are not valid
                CleanupTags(destTags, "ACCURATERIP");
                //CleanupTags(destTags, "REPLAYGAIN");

                destTags.Remove("CUESHEET");
            }

            if (fWithCUE)
                destTags.Add("CUESHEET", GetCUESheetContents(CUEStyle.SingleFileWithCUE));

            if (_config.embedLog && logContents != null)
                destTags.Add("LOG", logContents);

            if (fWithCUE &&
                _config.writeArTagsOnEncode &&
                _action == CUEAction.Encode &&
                isUsingAccurateRip &&
                _arVerify.ExceptionStatus == WebExceptionStatus.Success)
                GenerateAccurateRipTags(destTags, bestOffset, -1);

            return destTags;
        }

        //public IAudioSource OpenSource(long position)
        //{
        //    int pos = 0;
        //    for (int iSource = 0; iSource < _sources.Count; iSource++)
        //    {
        //        if (position >= pos && position < pos + (int)_sources[iSource].Length)
        //        {
        //            IAudioSource audioSource = GetAudioSource(iSource);
        //            audioSource.Position = position - pos;
        //            return audioSource;
        //        }
        //        pos += (int)_sources[iSource].Length;
        //    }
        //    return null;
        //}

        internal void ApplyWriteOffset()
        {
            if (_writeOffset == 0)
                return;

            int absOffset = Math.Abs(_writeOffset);
            SourceInfo sourceInfo;

            sourceInfo.Path = null;
            sourceInfo.Offset = 0;
            sourceInfo.Length = (uint)absOffset;

            if (_writeOffset < 0)
            {
                _sources.Insert(0, sourceInfo);

                int last = _sources.Count - 1;
                while (absOffset >= _sources[last].Length)
                {
                    absOffset -= (int)_sources[last].Length;
                    _sources.RemoveAt(last--);
                }
                sourceInfo = _sources[last];
                sourceInfo.Length -= (uint)absOffset;
                _sources[last] = sourceInfo;
            }
            else
            {
                _sources.Add(sourceInfo);

                while (absOffset >= _sources[0].Length)
                {
                    absOffset -= (int)_sources[0].Length;
                    _sources.RemoveAt(0);
                }
                sourceInfo = _sources[0];
                sourceInfo.Offset += (uint)absOffset;
                sourceInfo.Length -= (uint)absOffset;
                _sources[0] = sourceInfo;
            }

            _appliedWriteOffset = true;
        }

        public void WriteAudioFilesPass(string dir, CUEStyle style, int[] destLengths, bool htoaToFile, bool noOutput)
        {
            int iTrack, iIndex;
            AudioBuffer sampleBuffer = new AudioBuffer(AudioPCMConfig.RedBook, 0x10000);
            TrackInfo track;
            IAudioSource audioSource = null;
            IAudioDest audioDest = null;
            bool discardOutput;
            int iSource = -1;
            int iDest = -1;
            int samplesRemSource = 0;

            ApplyWriteOffset();

            int destBPS = 16;
            hdcdDecoder = null;
            if (_config.detectHDCD && CUEProcessorPlugins.hdcd != null)
            {
                // currently broken verifyThenConvert on HDCD detection!!!! need to check for HDCD results higher
                try
                {
                    destBPS = ((_outputLossyWAV && _config.decodeHDCDtoLW16) || !_config.decodeHDCDto24bit) ? 20 : 24;
                    hdcdDecoder = Activator.CreateInstance(CUEProcessorPlugins.hdcd, 2, 44100, destBPS, _config.decodeHDCD) as IAudioDest;
                }
                catch { }
                if (hdcdDecoder == null || !_config.decodeHDCD)
                    destBPS = 16;
            }

            if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
            {
                iDest++;
                if (_isCD && style == CUEStyle.SingleFileWithCUE)
                    _padding += Encoding.UTF8.GetByteCount(GetCUESheetContents(style));
                audioDest = GetAudioDest(_destPaths[iDest], destLengths[iDest], destBPS, _padding, noOutput);
            }

            int currentOffset = 0, previousOffset = 0;
            int trackLength = (int)_toc.Pregap * 588;
            int diskLength = 588 * (int)_toc.AudioLength;
            int diskOffset = 0;

            // we init AR before CTDB so that CTDB gets inited with correct TOC
            if (isUsingAccurateRip || isUsingCUEToolsDB)
                _arVerify.Init(_toc);
            if (isUsingCUEToolsDB && !isUsingCUEToolsDBFix)
            {
                _CUEToolsDB.TOC = _toc; // This might be unnecessary, because they point to the same structure - if we modify _toc, _CUEToolsDB.TOC gets updated. Unless we set cueSheet.TOC...
                _CUEToolsDB.Init(_arVerify);
            }

            ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", 0, 0, noOutput ? "Verifying" : "Writing"), 0.0, null, null);

#if !DEBUG
            try
#endif
            {
                for (iTrack = 0; iTrack < TrackCount; iTrack++)
                {
                    track = _tracks[iTrack];

                    if ((style == CUEStyle.GapsPrepended) || (style == CUEStyle.GapsLeftOut))
                    {
                        iDest++;
                        if (hdcdDecoder != null)
                            (hdcdDecoder as IAudioFilter).AudioDest = null;
                        if (audioDest != null)
                            audioDest.Close();
                        audioDest = GetAudioDest(_destPaths[iDest], destLengths[iDest], destBPS, _padding, noOutput);
                    }

                    for (iIndex = 0; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
                    {
                        int samplesRemIndex = (int)_toc.IndexLength(_toc.FirstAudio + iTrack, iIndex) * 588;

                        if (iIndex == 1)
                        {
                            previousOffset = currentOffset;
                            currentOffset = 0;
                            trackLength = (int)_toc[_toc.FirstAudio + iTrack].Length * 588;
                        }

                        if ((style == CUEStyle.GapsAppended) && (iIndex == 1))
                        {
                            if (hdcdDecoder != null)
                                (hdcdDecoder as IAudioFilter).AudioDest = null;
                            if (audioDest != null)
                                audioDest.Close();
                            iDest++;
                            audioDest = GetAudioDest(_destPaths[iDest], destLengths[iDest], destBPS, _padding, noOutput);
                        }

                        if ((style == CUEStyle.GapsAppended) && (iIndex == 0) && (iTrack == 0))
                        {
                            discardOutput = !htoaToFile;
                            if (htoaToFile)
                            {
                                iDest++;
                                audioDest = GetAudioDest(_destPaths[iDest], destLengths[iDest], destBPS, _padding, noOutput);
                            }
                        }
                        else if ((style == CUEStyle.GapsLeftOut) && (iIndex == 0))
                        {
                            discardOutput = true;
                        }
                        else
                        {
                            discardOutput = false;
                        }

                        while (samplesRemIndex != 0)
                        {
                            if (samplesRemSource == 0)
                            {
                                //                                if (_isCD && audioSource != null && audioSource is CDDriveReader)
                                //                                    updatedTOC = ((CDDriveReader)audioSource).TOC;
                                if (audioSource != null) audioSource.Close();
                                audioSource = GetAudioSource(++iSource, _config.separateDecodingThread || _isCD);
                                samplesRemSource = (int)_sources[iSource].Length;
                            }

                            int copyCount = Math.Min(samplesRemIndex, samplesRemSource);

                            if (trackLength > 0 && !_isCD)
                            {
                                double trackPercent = (double)currentOffset / trackLength;
                                ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", iIndex > 0 ? iTrack + 1 : iTrack, (int)(100 * trackPercent),
                                    noOutput ? "Verifying" : "Writing"), (int)diskOffset, (int)diskLength,
                                    _isCD ? string.Format("{0}: {1:00} - {2}", audioSource.Path, iTrack + 1, Metadata.Tracks[iTrack].Title) : audioSource.Path, discardOutput ? null : audioDest.Path);
                            }

                            copyCount = audioSource.Read(sampleBuffer, copyCount);
                            if (copyCount == 0)
                                throw new Exception("Unexpected end of file");
                            if (isUsingCUEToolsDB && isUsingCUEToolsDBFix)
                                _CUEToolsDB.SelectedEntry.repair.Write(sampleBuffer);
                            // we use AR after CTDB fix, so that we can verify what we fixed
                            if (isUsingAccurateRip || isUsingCUEToolsDB)
                                _arVerify.Write(sampleBuffer);
                            if (!discardOutput)
                            {
                                if (!_config.detectHDCD || !_config.decodeHDCD)
                                    audioDest.Write(sampleBuffer);
                                if (_config.detectHDCD && hdcdDecoder != null)
                                {
                                    if (_config.wait750FramesForHDCD && diskOffset > 750 * 588 && string.Format("{0:s}", hdcdDecoder) == "")
                                    {
                                        (hdcdDecoder as IAudioFilter).AudioDest = null;
                                        hdcdDecoder = null;
                                        if (_config.decodeHDCD)
                                        {
                                            audioSource.Close();
                                            audioDest.Delete();
                                            throw new Exception("HDCD not detected.");
                                        }
                                    }
                                    else
                                    {
                                        if (_config.decodeHDCD)
                                            (hdcdDecoder as IAudioFilter).AudioDest = (discardOutput || noOutput) ? null : audioDest;
                                        hdcdDecoder.Write(sampleBuffer);
                                    }
                                }
                            }

                            currentOffset += copyCount;
                            diskOffset += copyCount;
                            samplesRemIndex -= copyCount;
                            samplesRemSource -= copyCount;

                            CheckStop();
                        }
                    }
                }
                if (hdcdDecoder != null)
                    (hdcdDecoder as IAudioFilter).AudioDest = null;
                hdcdDecoder = null;
                if (audioSource != null)
                    audioSource.Close();
                audioSource = null;
                if (audioDest != null)
                    audioDest.Close();
                audioDest = null;
            }
#if !DEBUG
            catch (Exception ex)
            {
                if (hdcdDecoder != null)
                    (hdcdDecoder as IAudioFilter).AudioDest = null;
                hdcdDecoder = null;
                if (audioSource != null)
                    try { audioSource.Close(); } catch { }
                audioSource = null;
                if (audioDest != null)
                    try { audioDest.Delete(); } catch { }
                audioDest = null;
                throw ex;
            }
#endif
        }

        public void VerifyAudio()
        {
            ApplyWriteOffset();

            hdcdDecoder = null;

            // we init AR before CTDB so that CTDB gets inited with correct TOC
            if (isUsingAccurateRip || isUsingCUEToolsDB)
                _arVerify.Init(_toc);
            if (isUsingCUEToolsDB && !isUsingCUEToolsDBFix)
            {
                _CUEToolsDB.TOC = _toc;
                _CUEToolsDB.Init(_arVerify);
            }

            ShowProgress(String.Format("Verifying ({0:00}%)...", 0), 0.0, null, null);

            AudioBuffer sampleBuffer = new AudioBuffer(AudioPCMConfig.RedBook, 0x10000);

            List<CUEToolsVerifyTask> tasks = new List<CUEToolsVerifyTask>();
            // also make sure all sources are seekable!!!
            // use overlapped io with large buffers?
            // ar.verify in each thread?
            int nThreads = 1;// _isCD || !_config.separateDecodingThread || isUsingCUEToolsDB || _config.detectHDCD ? 1 : Environment.ProcessorCount;

            int diskLength = 588 * (int)_toc.AudioLength;
            tasks.Add(new CUEToolsVerifyTask(this, 0, diskLength / nThreads, _arVerify));
            for (int iThread = 1; iThread < nThreads; iThread++)
                tasks.Add(new CUEToolsVerifyTask(this, iThread * diskLength / nThreads, (iThread + 1) * diskLength / nThreads));

#if !DEBUG
            try
#endif
            {
                int lastProgress = -588 * 75;
                int diskOffset = 0;
                int sourcesActive;
                do
                {
                    sourcesActive = 0;
                    for (int iSource = 0; iSource < tasks.Count; iSource++)
                    {
                        CUEToolsVerifyTask task = tasks[iSource];
                        if (task.Remaining == 0)
                            continue;
                        sourcesActive++;
                        if (tasks.Count == 1 && task.source.Position - lastProgress >= 588 * 75)
                        {
                            lastProgress = (int)task.source.Position;
                            int pos = 0;
                            int trackStart = 0;
                            int trackLength = (int)_toc.Pregap * 588;
                            for (int iTrack = 0; iTrack < TrackCount; iTrack++)
                                for (int iIndex = 0; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
                                {
                                    int indexLen = (int)_toc.IndexLength(_toc.FirstAudio + iTrack, iIndex) * 588;
                                    if (iIndex == 1)
                                    {
                                        trackStart = pos;
                                        trackLength = (int)_toc[_toc.FirstAudio + iTrack].Length * 588;
                                    }
                                    if (task.source.Position < pos + indexLen)
                                    {
                                        if (trackLength > 0 && !_isCD)
                                        {
                                            double trackPercent = (double)(task.source.Position - trackStart) / trackLength;
                                            ShowProgress(String.Format("{2} track {0:00} ({1:00}%)...", iIndex > 0 ? iTrack + 1 : iTrack, (int)(100 * trackPercent),
                                                "Verifying"), diskOffset, diskLength, task.source.Path, null);
                                        }
                                        iTrack = TrackCount;
                                        break;
                                    }
                                    pos += indexLen;
                                }
                        }
                        else if (tasks.Count > 1)
                        {
                            ShowProgress(String.Format("Verifying ({0:00}%)...", (uint)(100.0 * diskOffset / diskLength)),
                                diskOffset, diskLength, InputPath, null);
                        }

                        int copyCount = task.Step(sampleBuffer);
                        if (copyCount == 0)
                            throw new Exception("Unexpected end of file");
                        diskOffset += copyCount;

                        CheckStop();
                    }
                } while (sourcesActive > 0);
            }
#if !DEBUG
            catch (Exception ex)
            {
                tasks.ForEach(t => t.TryClose());
                tasks.Clear();
                throw ex;
            }
#endif
            hdcdDecoder = tasks[0].hdcd;
            for (int iThread = 1; iThread < nThreads; iThread++)
                tasks[0].Combine(tasks[iThread]);
            tasks.ForEach(t => t.Close());
            tasks.Clear();
        }

        private void DetectGaps()
        {
            if (!_isCD)
                throw new Exception("not a CD");

            if (_config.detectGaps)
            {
                try { _ripper.DetectGaps(); }
                catch (Exception ex)
                {
                    if (ex is StopException)
                        throw ex;
                }
            }

            if (!_ripper.GapsDetected)
                return;

            _toc = (CDImageLayout)_ripper.TOC.Clone();
            if (_toc.Barcode != null)
                Metadata.Barcode = _toc.Barcode;
            for (int iTrack = 0; iTrack < _toc.AudioTracks; iTrack++)
            {
                if (_toc[_toc.FirstAudio + iTrack].ISRC != null)
                    Metadata.Tracks[iTrack].ISRC = _toc[_toc.FirstAudio + iTrack].ISRC;
                //General.SetCUELine(_tracks[iTrack].Attributes, "ISRC", _toc[_toc.FirstAudio + iTrack].ISRC, false);
                if (_toc[_toc.FirstAudio + iTrack].DCP || _toc[_toc.FirstAudio + iTrack].PreEmphasis)
                    _tracks[iTrack].Attributes.Add(new CUELine("FLAGS" + (_toc[_toc.FirstAudio + iTrack].PreEmphasis ? " PRE" : "") + (_toc[_toc.FirstAudio + iTrack].DCP ? " DCP" : "")));
            }
        }

        public static string CreateDummyCUESheet(CUEConfig _config, string pathIn)
        {
            pathIn = Path.GetFullPath(pathIn);
            List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, Path.GetDirectoryName(pathIn));
            FileGroupInfo fileGroup = fileGroups.Find(f => f.type == FileGroupInfoType.TrackFiles && f.Contains(pathIn)) ??
                fileGroups.Find(f => f.type == FileGroupInfoType.FileWithCUE && f.Contains(pathIn)) ??
                fileGroups.Find(f => f.type == FileGroupInfoType.M3UFile && f.Contains(pathIn));
            return fileGroup == null ? null : CreateDummyCUESheet(_config, fileGroup);
        }

        public static string CreateDummyCUESheet(CUEConfig _config, FileGroupInfo fileGroup)
        {
            if (fileGroup.type == FileGroupInfoType.FileWithCUE)
            {
                TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                TagLib.File.IFileAbstraction fileAbsraction = new TagLib.File.LocalFileAbstraction(fileGroup.main.FullName);
                TagLib.File fileInfo = TagLib.File.Create(fileAbsraction);
                return Tagging.Analyze(fileInfo).Get("CUESHEET");
            }

            StringWriter sw = new StringWriter();
            sw.WriteLine(String.Format("REM COMMENT \"CUETools generated dummy CUE sheet\""));
            int trackNo = 0;
            foreach (FileSystemInfo file in fileGroup.files)
            {
                string name = file.Name;
                if (fileGroup.type == FileGroupInfoType.M3UFile
                    && Path.GetDirectoryName(file.FullName) != Path.GetDirectoryName(fileGroup.main.FullName)
                    && Path.GetDirectoryName(file.FullName).StartsWith(Path.GetDirectoryName(fileGroup.main.FullName)))
                {
                    name = file.FullName.Substring(Path.GetDirectoryName(fileGroup.main.FullName).Length + 1);
                }
                sw.WriteLine(String.Format("FILE \"{0}\" WAVE", name));
                sw.WriteLine(String.Format("  TRACK {0:00} AUDIO", ++trackNo));
                sw.WriteLine(String.Format("    INDEX 01 00:00:00"));
            }
            sw.Close();
            return sw.ToString();
        }

        public static string CorrectAudioFilenames(CUEConfig _config, string path, bool always)
        {
            StreamReader sr = new StreamReader(path, CUESheet.Encoding);
            string cue = sr.ReadToEnd();
            sr.Close();
            string extension;
            return CorrectAudioFilenames(_config, Path.GetDirectoryName(path), cue, always, null, out extension);
        }

        public static string CorrectAudioFilenames(CUEConfig _config, string dir, string cue, bool always, List<string> files, out string extension)
        {
            List<string> lines = new List<string>();
            List<int> filePos = new List<int>();
            List<string> origFiles = new List<string>();
            bool foundAll = true;
            string[] audioFiles = null;
            string lineStr;
            CUELine line;
            int i;
            string CDDBID = "";
            //bool isBinary = false;

            using (StringReader sr = new StringReader(cue))
            {
                while ((lineStr = sr.ReadLine()) != null)
                {
                    lines.Add(lineStr);
                    line = new CUELine(lineStr);
                    if ((line.Params.Count == 3) && (line.Params[0].ToUpper() == "FILE"))
                    {
                        string fileType = line.Params[2].ToUpper();
                        if (fileType == "MOTOROLA")
                            continue;
                        if (fileType == "BINARY")
                            continue;
                        //{
                        //    if (filePos.Count > 0)
                        //        continue;
                        //    isBinary = true;
                        //}
                        //else
                        //{
                        //    if (isBinary)
                        //    {
                        //        filePos.Clear();
                        //        origFiles.Clear();
                        //        foundAll = false;
                        //        isBinary = false;
                        //    }
                        //}
                        filePos.Add(lines.Count - 1);
                        origFiles.Add(line.Params[1]);
                        foundAll &= (FileLocator.LocateFile(dir, line.Params[1], files) != null);
                    }
                    if (line.Params.Count == 3 && line.Params[0].ToUpper() == "REM" && line.Params[1].ToUpper() == "DISCID")
                        CDDBID = line.Params[2].ToLower();
                }
                sr.Close();
            }


            extension = null;
            if (foundAll && !always)
                return cue;

            foundAll = false;

            foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
            {
                List<string> newFiles = new List<string>();
                for (int j = 0; j < origFiles.Count; j++)
                {
                    string newFilename = Path.ChangeExtension(Path.GetFileName(origFiles[j]), "." + format.Key);
                    string locatedFilename = FileLocator.LocateFile(dir, newFilename, files);
                    if (locatedFilename != null)
                        newFiles.Add(locatedFilename);
                }
                if (newFiles.Count == origFiles.Count)
                {
                    audioFiles = newFiles.ToArray();
                    extension = format.Key;
                    foundAll = true;
                    break;
                }
            }

            if (!foundAll && files == null)
            {
                List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, dir == "" ? "." : dir);

                // Choose filegroup by track count
                List<FileGroupInfo>
                    matching = fileGroups.FindAll(f => f.type == FileGroupInfoType.TrackFiles && f.files.Count == filePos.Count);
                // If ambiguous, use DISCID
                if (matching.Count > 1)
                    matching = fileGroups.FindAll(f => f.type == FileGroupInfoType.TrackFiles && f.files.Count == filePos.Count && f.TOC != null && AccurateRipVerify.CalculateCDDBId(f.TOC).ToLower() == CDDBID);
                if (matching.Count == 1)
                {
                    audioFiles = matching[0].files.ConvertAll<string>(info => info.FullName).ToArray();
                    // No need to sort - hopefully already sorted by ScanFolder
                    extension = matching[0].main.Extension.ToLower().TrimStart('.');
                    foundAll = true;
                }

                if (!foundAll && filePos.Count == 1)
                    foreach (FileGroupInfo fileGroup in fileGroups)
                    {
                        if (fileGroup.type == FileGroupInfoType.FileWithCUE && fileGroup.TOC != null)
                        {
                            CDImageLayout toc = CUE2TOC(cue, (int)fileGroup.TOC.AudioLength);
                            if (toc == null || toc.TrackOffsets != fileGroup.TOC.TrackOffsets)
                                continue;
                            if (foundAll)
                            {
                                foundAll = false;
                                break;
                            }
                            audioFiles = new string[] { fileGroup.main.FullName };
                            extension = fileGroup.main.Extension.ToLower().TrimStart('.');
                            foundAll = true;
                        }
                    }
            }

            // Use old-fashioned way if dealing with archive (files != null)
            // or with single file (filePos.Count == 1).
            // In other cases we use CUESheet.ScanFolder, which
            // is better at sorting and separating albums,
            // but doesn't support archives and single files yet.
            if (!foundAll)// && (files != null || filePos.Count == 1))
                foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
                {
                    if (files == null)
                        audioFiles = Directory.GetFiles(dir == "" ? "." : dir, "*." + format.Key);
                    else
                        audioFiles = files.FindAll(s => Path.GetDirectoryName(s) == dir && Path.GetExtension(s).ToLower() == "." + format.Key).ToArray();
                    if (audioFiles.Length == filePos.Count)
                    {
                        Array.Sort(audioFiles, FileGroupInfo.CompareTrackNames);
                        extension = format.Key;
                        foundAll = true;
                        break;
                    }
                }

            if (!foundAll)
                throw new Exception("unable to locate the audio files");

            for (i = 0; i < filePos.Count; i++)
                lines[filePos[i]] = "FILE \"" + Path.GetFileName(audioFiles[i]) + "\" WAVE";

            using (StringWriter sw = new StringWriter())
            {
                for (i = 0; i < lines.Count; i++)
                {
                    sw.WriteLine(lines[i]);
                }
                return sw.ToString();
            }
        }

        private int[] CalculateAudioFileLengths(CUEStyle style)
        {
            int iTrack, iIndex, iFile;
            int[] fileLengths;
            bool htoaToFile = (style == CUEStyle.GapsAppended && _config.preserveHTOA && _toc.Pregap != 0);
            bool discardOutput;

            if (style == CUEStyle.SingleFile || style == CUEStyle.SingleFileWithCUE)
            {
                fileLengths = new int[1];
                iFile = 0;
            }
            else
            {
                fileLengths = new int[TrackCount + (htoaToFile ? 1 : 0)];
                iFile = -1;
            }

            for (iTrack = 0; iTrack < TrackCount; iTrack++)
            {
                if (style == CUEStyle.GapsPrepended || style == CUEStyle.GapsLeftOut)
                    iFile++;

                for (iIndex = 0; iIndex <= _toc[_toc.FirstAudio + iTrack].LastIndex; iIndex++)
                {
                    if (style == CUEStyle.GapsAppended && (iIndex == 1 || (iIndex == 0 && iTrack == 0 && htoaToFile)))
                        iFile++;

                    if (style == CUEStyle.GapsAppended && iIndex == 0 && iTrack == 0)
                        discardOutput = !htoaToFile;
                    else
                        discardOutput = (style == CUEStyle.GapsLeftOut && iIndex == 0);

                    if (!discardOutput)
                        fileLengths[iFile] += (int)_toc.IndexLength(_toc.FirstAudio + iTrack, iIndex) * 588;
                }
            }

            return fileLengths;
        }

        public void CheckStop()
        {
            lock (this)
            {
                if (_stop)
                    throw new StopException();
                if (_pause)
                {
                    ShowProgress("Paused...", 0, null, null);
                    Monitor.Wait(this);
                }
            }
        }

        public void Stop()
        {
            lock (this)
            {
                if (_pause)
                {
                    _pause = false;
                    Monitor.Pulse(this);
                }
                _stop = true;
            }
        }

        public void Pause()
        {
            lock (this)
            {
                if (_pause)
                {
                    _pause = false;
                    Monitor.Pulse(this);
                }
                else
                {
                    _pause = true;
                }
            }
        }

        private IAudioDest GetAudioDest(string path, int finalSampleCount, int bps, int padding, bool noOutput)
        {
            var pcm = new AudioPCMConfig(bps, 2, 44100);
            string extension = Path.GetExtension(path).ToLower();
            return AudioReadWrite.GetAudioDest(noOutput ? AudioEncoderType.NoAudio : _audioEncoderType, path, pcm, finalSampleCount, padding, extension, _config);
        }

        internal IAudioSource GetAudioSource(int sourceIndex, bool pipe)
        {
            SourceInfo sourceInfo = _sources[sourceIndex];
            IAudioSource audioSource;

            if (sourceInfo.Path == null)
            {
                audioSource = new SilenceGenerator(sourceInfo.Offset + sourceInfo.Length);
            }
            else
            {
                if (_isCD)
                {
                    _ripper.Position = 0;
                    //audioSource = _ripper;
                    if (pipe)
                        audioSource = new AudioPipe(_ripper, 0x100000, false, ThreadPriority.Highest);
                    else
                        audioSource = _ripper;
                }
                else
                    if (_isArchive)
                        audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, OpenArchive(sourceInfo.Path, false), _config);
                    else
                        audioSource = AudioReadWrite.GetAudioSource(sourceInfo.Path, null, _config);
            }

            if (sourceInfo.Offset != 0)
            {
                try
                {
                    audioSource.Position = sourceInfo.Offset;
                }
                catch(Exception ex)
                {
                    audioSource.Close();
                    throw ex;
                }
            }

            //if (!(audioSource is AudioPipe) && !(audioSource is UserDefinedReader) && _config.separateDecodingThread)
            if (!(audioSource is AudioPipe) && pipe)
                audioSource = new AudioPipe(audioSource, 0x10000);

            return audioSource;
        }

        private void WriteLine(TextWriter sw, int level, CUELine line)
        {
            WriteLine(sw, level, line.ToString());
        }

        private void WriteLine(TextWriter sw, int level, string line)
        {
            sw.Write(new string(' ', level * 2));
            sw.WriteLine(line);
        }

        public static CDImageLayout CUE2TOC(string cue, int fileTimeLengthFrames)
        {
            CDImageLayout toc = new CDImageLayout();
            bool seenFirstFileIndex = false;
            int absoluteFileStartTime = 0;
            int trackStart = -1;
            try
            {
                using (TextReader sr = new StringReader(cue))
                {
                    string lineStr;
                    while ((lineStr = sr.ReadLine()) != null)
                    {
                        CUELine line = new CUELine(lineStr);
                        if (line.Params.Count > 0)
                        {
                            string command = line.Params[0].ToUpper();

                            if (command == "TRACK")
                            {
                                if (line.Params[2].ToUpper() != "AUDIO")
                                    return null;
                            }
                            else if (command == "INDEX")
                            {
                                int index = int.Parse(line.Params[1]);
                                int timeRelativeToFileStart = CDImageLayout.TimeFromString(line.Params[2]);
                                if (!seenFirstFileIndex)
                                {
                                    if (timeRelativeToFileStart != 0)
                                        return null;
                                    seenFirstFileIndex = true;
                                }
                                else
                                {
                                    if (timeRelativeToFileStart > fileTimeLengthFrames)
                                        return null;
                                    if (Int32.TryParse(line.Params[1], out index) && index == 1 && trackStart >= 0)
                                        toc.AddTrack(new CDTrack((uint)toc.TrackCount + 1, (uint)trackStart, (uint)(absoluteFileStartTime + timeRelativeToFileStart - trackStart), true, false));
                                }
                                if (index == 1)
                                    trackStart = absoluteFileStartTime + timeRelativeToFileStart;
                            }
                            else if (command == "PREGAP")
                            {
                                if (seenFirstFileIndex)
                                    return null;
                                int pregapLength = CDImageLayout.TimeFromString(line.Params[1]);
                                absoluteFileStartTime += pregapLength;
                            }
                        }
                    }
                    sr.Close();
                }
            }
            catch
            {
                return null;
            }
            toc.AddTrack(new CDTrack((uint)toc.TrackCount + 1, (uint)trackStart, (uint)(absoluteFileStartTime + fileTimeLengthFrames - trackStart), true, false));
            toc[1][0].Start = 0;
            return toc;
        }

        public static List<FileGroupInfo> ScanFolder(CUEConfig _config, string path)
        {
            DirectoryInfo dir = new DirectoryInfo(path);
            return ScanFolder(_config, dir.GetFileSystemInfos());
        }

        public static List<FileGroupInfo> ScanFolder(CUEConfig _config, IEnumerable<FileSystemInfo> files)
        {
            List<FileGroupInfo> fileGroups = new List<FileGroupInfo>();
            foreach (FileSystemInfo file in files)
            {
                // file.Refresh();
                // file.Attributes returns -1 for long paths!!!
                if ((file.Attributes & FileAttributes.Hidden) != 0)
                    continue;
                if ((file.Attributes & FileAttributes.Directory) != 0)
                {
                    // foreach (FileSystemInfo subfile in ((DirectoryInfo)e.file).GetFileSystemInfos())
                    // if (IsVisible(subfile))
                    // {
                    //     e.isExpandable = true;
                    //  break;
                    // }
                    fileGroups.Add(new FileGroupInfo(file, FileGroupInfoType.Folder));
                    continue;
                }
                string ext = file.Extension.ToLower();
                if (ext == ".cue")
                {
                    fileGroups.Add(new FileGroupInfo(file, FileGroupInfoType.CUESheetFile));
                    continue;
                }
                if (ext == ".m3u")
                {
                    FileGroupInfo m3uGroup = new FileGroupInfo(file, FileGroupInfoType.M3UFile);
                    using (StreamReader m3u = new StreamReader(file.FullName))
                    {
                        do
                        {
                            string line = m3u.ReadLine();
                            if (line == null) break;
                            if (line == "" || line[0] == '#') continue;
                            //if (line.IndexOfAny(Path.GetInvalidPathChars()) >= 0) 
                            //    continue;
                            try
                            {
                                line = Path.Combine(Path.GetDirectoryName(file.FullName), line);
                                if (File.Exists(line))
                                {
                                    FileInfo f = new FileInfo(line);
                                    CUEToolsFormat fmt1;
                                    if (!f.Extension.StartsWith(".") || !_config.formats.TryGetValue(f.Extension.ToLower().Substring(1), out fmt1) || !fmt1.allowLossless)
                                        throw new Exception("not lossless");
                                    m3uGroup.files.Add(f);
                                    continue;
                                }
                            }
                            catch { }
                            m3uGroup = null;
                            break;
                        } while (true);
                    };
                    if (m3uGroup != null)
                        fileGroups.Add(m3uGroup);
                    continue;
                }
                if (ext == ".zip")
                {
                    fileGroups.Add(new FileGroupInfo(file, FileGroupInfoType.Archive));
                    //try
                    //{
                    //    using (ICSharpCode.SharpZipLib.Zip.ZipFile unzip = new ICSharpCode.SharpZipLib.Zip.ZipFile(file.FullName))
                    //    {
                    //        foreach (ICSharpCode.SharpZipLib.Zip.ZipEntry entry in unzip)
                    //        {
                    //            if (entry.IsFile && Path.GetExtension(entry.Name).ToLower() == ".cue")
                    //            {
                    //                e.node.Nodes.Add(fileSystemTreeView1.NewNode(file, false));
                    //                break;
                    //            }

                    //        }
                    //        unzip.Close();
                    //    }
                    //}
                    //catch
                    //{
                    //}
                    continue;
                }
                if (ext == ".rar")
                {
                    fileGroups.Add(new FileGroupInfo(file, FileGroupInfoType.Archive));
                    continue;
                }
                CUEToolsFormat fmt;
                if (ext.StartsWith(".") && _config.formats.TryGetValue(ext.Substring(1), out fmt) && fmt.allowLossless)
                {
                    uint disc = 0;
                    uint number = 0;
                    string album = null;
                    string cueFound = null;
                    CDImageLayout tocFound = null;
                    TimeSpan dur = TimeSpan.Zero;
                    TagLib.UserDefined.AdditionalFileTypes.Config = _config;
                    TagLib.File.IFileAbstraction fileAbsraction = new TagLib.File.LocalFileAbstraction(file.FullName);
                    try
                    {
                        TagLib.File fileInfo = TagLib.File.Create(fileAbsraction);
                        disc = fileInfo.Tag.Disc;
                        album = fileInfo.Tag.Album;
                        number = fileInfo.Tag.Track;
                        dur = fileInfo.Properties.Duration;
                        var tags = Tagging.Analyze(fileInfo);
                        cueFound = fmt.allowEmbed ? tags.Get("CUESHEET") : null;
                        var toc = tags.Get("CDTOC");
                        if (toc != null) tocFound = CDImageLayout.FromTag(toc);
                    }
                    catch { }
                    if (cueFound != null)
                    {
                        FileGroupInfo group = new FileGroupInfo(file, FileGroupInfoType.FileWithCUE);
                        if (dur != TimeSpan.Zero)
                            group.TOC = CUE2TOC(cueFound, (int)((dur.TotalMilliseconds * 75 + 500) / 1000));
                        fileGroups.Add(group);
                        continue;
                    }
                    disc = Math.Min(50, Math.Max(1, disc));
                    FileGroupInfo groupFound = null;
                    foreach (FileGroupInfo fileGroup in fileGroups)
                    {
                        if (fileGroup.type == FileGroupInfoType.TrackFiles
                            && fileGroup.discNo == disc
                            && fileGroup.album == album
                            && (fileGroup.TOC == null ? "" : fileGroup.TOC.ToString()) == (tocFound == null ? "" : tocFound.ToString())
                            && fileGroup.main.Extension.ToLower() == ext)
                        {
                            groupFound = fileGroup;
                            break;
                        }
                    }
                    if (groupFound == null)
                    {
                        groupFound = new FileGroupInfo(file, FileGroupInfoType.TrackFiles);
                        groupFound.discNo = disc;
                        groupFound.album = album;
                        groupFound.TOC = tocFound;
                        groupFound.durations = new Dictionary<FileSystemInfo, TimeSpan>();
                        fileGroups.Add(groupFound);
                    }
                    groupFound.files.Add(file);
                    if (number > 0) groupFound.numbers.Add(file, number);
                    if (dur != TimeSpan.Zero) groupFound.durations.Add(file, dur);
                }
            }
            fileGroups.RemoveAll(group => group.type == FileGroupInfoType.TrackFiles && (
                (group.TOC == null && group.files.Count < 2) ||
                (group.TOC != null && group.TOC.AudioTracks != group.files.Count)));
            // tracks must be sorted according to tracknumer (or filename if missing)
            foreach (FileGroupInfo group in fileGroups)
                if (group.type == FileGroupInfoType.TrackFiles)
                {
                    group.files.Sort(group.Compare());
                    group.numbers = null;
                    group.TOC = new CDImageLayout();
                    foreach (FileSystemInfo f in group.files)
                    {
                        if (!group.durations.ContainsKey(f))
                        {
                            group.TOC = null;
                            break;
                        }
                        uint len = (uint)((group.durations[f].TotalMilliseconds * 75 + 500) / 1000);
                        group.TOC.AddTrack(new CDTrack((uint)group.TOC.TrackCount + 1, group.TOC.Length, len, true, false));
                    }
                }
            fileGroups.Sort(FileGroupInfo.Compare);
            return fileGroups;
        }

        public void UseLocalDB(CUEToolsLocalDB db)
        {
            _useLocalDB = true;
            _localDB = db;
        }

        public string ExecuteScript(CUEToolsScript script)
        {
            if (!script.builtin)
                return ExecuteScript(script.code);

            switch (script.name)
            {
                case "default":
                    return Go();
                case "only if found":
                    return ArVerify.ExceptionStatus != WebExceptionStatus.Success ? WriteReport() : Go();
                case "repair":
                    {
                        UseCUEToolsDB("CUETools " + CUEToolsVersion, null, true, CTDBMetadataSearch.None);
                        Action = CUEAction.Verify;
                        if (CTDB.DBStatus != null)
                            return CTDB.DBStatus;
                        bool useAR = isUsingAccurateRip;
                        isUsingAccurateRip = true;
                        Go();
                        isUsingAccurateRip = useAR;
                        List<CUEToolsSourceFile> choices = new List<CUEToolsSourceFile>();
                        foreach (DBEntry entry in CTDB.Entries)
                            if (!entry.hasErrors || entry.canRecover)
                            {
                                StringBuilder sb = new StringBuilder();
                                if (entry.hasErrors)
                                {
                                    sb.AppendFormat("Affected positions:\n");
                                    for (int sec = 0; sec < entry.repair.AffectedSectorArray.Length; sec++)
                                        if (entry.repair.AffectedSectorArray[sec])
                                            sb.AppendFormat("{0}\n", CDImageLayout.TimeToString((uint)sec));
                                }
                                CUEToolsSourceFile choice = new CUEToolsSourceFile(entry.Status, new StringReader(sb.ToString()));
                                choice.data = entry;
                                choices.Add(choice);
                            }
                        CUEToolsSourceFile selectedEntry = ChooseFile(choices, null, true);
                        if (selectedEntry == null)
                            return CTDB.Status;
                        CTDB.SelectedEntry = (DBEntry)selectedEntry.data;
                        if (!CTDB.SelectedEntry.hasErrors)
                            return CTDB.Status;
                        isUsingCUEToolsDBFix = true;
                        Action = CUEAction.Encode;
                        return Go();
                    }
                case "fix offset":
                    {
                        if (ArVerify.ExceptionStatus != WebExceptionStatus.Success)
                            return WriteReport();

                        WriteOffset = 0;
                        Action = CUEAction.Verify;
                        string status = Go();

                        uint tracksMatch;
                        int bestOffset;
                        FindBestOffset(Config.fixOffsetMinimumConfidence, !Config.fixOffsetToNearest, out tracksMatch, out bestOffset);
                        if (tracksMatch * 100 >= Config.fixOffsetMinimumTracksPercent * TrackCount)
                        {
                            WriteOffset = bestOffset;
                            Action = CUEAction.Encode;
                            status = Go();
                        }
                        return status;
                    }

                case "encode if verified":
                    {
                        if (ArVerify.ExceptionStatus != WebExceptionStatus.Success)
                            return WriteReport();

                        Action = CUEAction.Verify;
                        string status = Go();

                        uint tracksMatch;
                        int bestOffset;
                        FindBestOffset(Config.encodeWhenConfidence, false, out tracksMatch, out bestOffset);
                        if (tracksMatch * 100 >= Config.encodeWhenPercent * TrackCount && (!_config.encodeWhenZeroOffset || bestOffset == 0))
                        {
                            Action = CUEAction.Encode;
                            status = Go();
                        }
                        return status;
                    }
            }

            return "internal error";
        }

        public string ExecuteScript(string script)
        {
            AsmHelper helper = CompileScript(script);
            return (string)helper.Invoke("*.Execute", this);
        }

        public static AsmHelper CompileScript(string script)
        {
            //CSScript.GlobalSettings.InMemoryAsssembly = true;
            //CSScript.GlobalSettings.HideAutoGeneratedFiles =
            //CSScript.CacheEnabled = false;
            return new AsmHelper(CSScript.LoadCode("using System; using System.Windows.Forms; using System.Net; using CUETools.Processor; using CUETools.Codecs; using CUETools.AccurateRip; public class Script { "
                + "public static string Execute(CUESheet processor) { \r\n"
                + script
                + "\r\n } "
                + " }", null, true));
        }

        public static bool TryCompileScript(string script)
        {
            AsmHelper helper = CompileScript(script);
            return helper != null;
        }

        #endregion

        #region Events

        public event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;
        public event EventHandler<CUEToolsProgressEventArgs> CUEToolsProgress;
        public event EventHandler<CUEToolsSelectionEventArgs> CUEToolsSelection;

        #endregion
    }
}
