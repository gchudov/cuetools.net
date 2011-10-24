using System;
using System.Collections.Generic;
using System.Xml.Serialization;
using CUETools.AccurateRip;
using CUETools.CDImage;

namespace CUETools.Processor
{
    [Serializable]
    public class CUEToolsLocalDBEntry
    {
        public CUEToolsLocalDBEntry()
        {
        }

        public string DiscID { get; set; }

        public OffsetSafeCRCRecord OffsetSafeCRC { get; set; }

        public CUEMetadata Metadata { get; set; }

        public List<string> InputPaths { get; set; }

        public List<string> AudioPaths { get; set; }

        public int TrackCount { get; set; }

        public int AudioTracks { get; set; }

        public int FirstAudio { get; set; }

        public string TrackOffsets { get; set; }

        public uint ARConfidence { get; set; }

        public string Status { get; set; }

        public string Log { get; set; }

        public DateTime VerificationDate { get; set; }

        public DateTime CTDBVerificationDate { get; set; }

        public int CTDBConfidence { get; set; }

        [XmlIgnore]
        public string Path
        {
            get
            {
                return InputPaths == null || InputPaths.Count < 1 ? null : InputPaths[0];
            }
        }

        static public string NormalizePath(string path)
        {
            if (System.Environment.OSVersion.Platform != System.PlatformID.Unix)
                return System.IO.Path.GetFullPath(path).ToLower();
            else
                return System.IO.Path.GetFullPath(path);
        }

        public bool HasPath(string inputPath)
        {
            string norm = CUEToolsLocalDBEntry.NormalizePath(inputPath);
            return this.InputPaths != null && this.InputPaths.Find(i => i == norm) != null;
        }

        public bool EqualAudioPaths(List<string> fullAudioPaths)
        {
            int count1 = this.AudioPaths == null ? 0 : this.AudioPaths.Count;
            int count2 = fullAudioPaths == null ? 0 : fullAudioPaths.Count;
            if (count1 == count2)
            {
                bool equals = true;
                for (int i = 0; i < count1; i++)
                    equals &= this.AudioPaths[i] == fullAudioPaths[i];
                return equals;
            }
            return false;
        }

        public bool EqualLayouts(CDImageLayout layout)
        {
            return this.TrackCount == layout.TrackCount
                && this.AudioTracks == layout.AudioTracks
                && this.FirstAudio == layout.FirstAudio
                && this.TrackOffsets == layout.TrackOffsets;
        }

        public bool Equals(CDImageLayout layout, List<string> fullAudioPaths)
        {
            return EqualLayouts(layout) && EqualAudioPaths(fullAudioPaths);
        }
    }
}
