using System;
using System.Collections.Generic;
using System.IO;
using CUETools.CDImage;

namespace CUETools.Processor
{
    public class FileGroupInfo
    {
        public List<FileSystemInfo> files;
        public Dictionary<FileSystemInfo, uint> numbers;
        public Dictionary<FileSystemInfo, TimeSpan> durations;
        public FileSystemInfo main;
        public FileGroupInfoType type;
        public CDImageLayout TOC;
        public uint discNo;
        public string album;

        public override string ToString()
        {
            switch (type)
            {
                case FileGroupInfoType.TrackFiles:
                    return (album == null ? main.Name :
                        album + (discNo > 0 ? string.Format(" (disc {0})", discNo) : "")) + ": " + files.Count.ToString() + " files";
            }
            return main.Name;
        }

        public FileGroupInfo(FileSystemInfo _main, FileGroupInfoType _type)
        {
            main = _main;
            type = _type;
            files = new List<FileSystemInfo>();
            numbers = new Dictionary<FileSystemInfo, uint>();
        }

        public static long IntPrefix(ref string a)
        {
            long na = 0;
            string sa = a;
            sa = sa.TrimStart(' ', '_');
            if (!(sa.Length > 0 && sa[0] >= '0' && sa[0] <= '9'))
                return -1;
            while (sa.Length > 0 && sa[0] >= '0' && sa[0] <= '9')
            {
                na = 10 * na + (sa[0] - '0');
                sa = sa.Substring(1);
            }
            a = sa.TrimStart(' ', '_');
            return na;
        }

        public static int Compare(FileGroupInfo a, FileGroupInfo b)
        {
            if (a.type == b.type)
                return CompareTrackNames(a.main.FullName, b.main.FullName);
            return Comparer<FileGroupInfoType>.Default.Compare(a.type, b.type);
        }

        public Comparison<FileSystemInfo> Compare()
        {
            if (files.Find(f => !numbers.ContainsKey(f)) == null)
                return (a, b) => Comparer<uint>.Default.Compare(numbers[a], numbers[b]);
            return CompareTrackNames;
        }

        public static int CompareTrackNames(FileSystemInfo a, FileSystemInfo b)
        {
            return CompareTrackNames(a.FullName, b.FullName);
        }

        public static int CompareTrackNames(string a, string b)
        {
            while (a.Length > 0 && b.Length > 0 && a[0] == b[0])
            {
                a = a.Substring(1);
                b = b.Substring(1);
            }
            long na = IntPrefix(ref a);
            long nb = IntPrefix(ref b);
            if (na != nb)
                return Comparer<long>.Default.Compare(na, nb);
            if (na < 0)
                return Comparer<string>.Default.Compare(a, b);
            return CompareTrackNames(a, b);
        }

        public bool Contains(string pathIn)
        {
            if (type != FileGroupInfoType.TrackFiles)
                return main.FullName.ToLower() == pathIn.ToLower();
            return null != files.Find(file => file.FullName.ToLower() == pathIn.ToLower());
        }
    }
}
