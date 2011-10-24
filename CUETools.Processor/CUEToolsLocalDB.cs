using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Xml.Serialization;
using CUETools.CDImage;
using CUETools.Processor.Settings;

namespace CUETools.Processor
{
    [Serializable]
    public class CUEToolsLocalDB : List<CUEToolsLocalDBEntry>
    {
        private static XmlSerializer serializer = new XmlSerializer(typeof(CUEToolsLocalDB));

        public bool Dirty
        {
            get;
            set;
        }

        public static List<CUEToolsLocalDBEntry>[] Group(List<CUEToolsLocalDBEntry> items, Converter<CUEToolsLocalDBEntry, string> convert, Comparison<List<CUEToolsLocalDBEntry>> compare)
        {
            var results = new Dictionary<string, List<CUEToolsLocalDBEntry>>(items.Count);
            foreach (var item in items)
            {
                var key = convert(item);
                if (key != null)
                {
                    if (!results.ContainsKey(key))
                        results[key] = new List<CUEToolsLocalDBEntry>();
                    results[key].Add(item);
                }
            }

            var groups = new List<CUEToolsLocalDBEntry>[results.Count];
            results.Values.CopyTo(groups, 0);
            if (compare != null)
                Array.Sort(groups, (a, b) => compare(a, b));
            else
            {
                var keys = new string[results.Count];
                results.Keys.CopyTo(keys, 0);
                Array.Sort(keys, groups);
            }
            return groups;
        }

        public CUEToolsLocalDBEntry Lookup(string inputPath)
        {
            return this.Find(e => e.HasPath(inputPath));
        }

        public CUEToolsLocalDBEntry Lookup(CDImageLayout layout, List<string> audioPaths)
        {
            List<string> fullAudioPaths = audioPaths == null ? null : audioPaths.ConvertAll(p => CUEToolsLocalDBEntry.NormalizePath(p));
            var entry = this.Find(e => e.Equals(layout, fullAudioPaths));
            if (entry == null)
            {
                entry = new CUEToolsLocalDBEntry();
                entry.TrackCount = layout.TrackCount;
                entry.AudioTracks = (int)layout.AudioTracks;
                entry.FirstAudio = layout.FirstAudio;
                entry.TrackOffsets = layout.TrackOffsets;
                entry.DiscID = layout.TOCID;
                entry.AudioPaths = fullAudioPaths;
                this.Add(entry);
                this.Dirty = true;
            }
            return entry;
        }

        public static string LocalDBPath
        {
            get
            {
                return Path.Combine(SettingsShared.GetProfileDir("CUE Tools", System.Windows.Forms.Application.ExecutablePath), "LocalDB.xml.z");
            }
        }

        public void Save()
        {
            if (!this.Dirty) return;
            string tempPath = LocalDBPath + "." + DateTime.Now.Ticks.ToString() + ".tmp";
            using (var fileStream = new FileStream(tempPath, FileMode.CreateNew))
            using (var deflateStream = new DeflateStream(fileStream, CompressionMode.Compress))
            using (TextWriter writer = new StreamWriter(deflateStream))
                serializer.Serialize(writer, this);
            File.Delete(LocalDBPath);
            File.Move(tempPath, LocalDBPath);
            this.Dirty = false;
        }

        public static CUEToolsLocalDB Load()
        {
            if (!File.Exists(LocalDBPath))
                return new CUEToolsLocalDB();
            using (var fileStream = new FileStream(LocalDBPath, FileMode.Open))
            using (var deflateStream = new DeflateStream(fileStream, CompressionMode.Decompress))
                return serializer.Deserialize(deflateStream) as CUEToolsLocalDB;
        }
    }
}
