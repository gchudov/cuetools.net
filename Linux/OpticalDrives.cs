// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.IO;

namespace CUETools.Ripper
{
    public static class OpticalDrives
    {
        public static string[] Available()
        {
            var drives = new List<string>();
            if (!Directory.Exists("/sys/class/block")) return drives.ToArray();
            foreach (string entry in Directory.EnumerateDirectories("/sys/class/block", "sr*"))
            {
                string path = "/dev/" + Path.GetFileName(entry);
                if (IsOptical(path)) drives.Add(path);
            }
            drives.Sort(StringComparer.Ordinal);
            return drives.ToArray();
        }

        public static string Resolve(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("An optical drive is required.");
            path = Path.GetFullPath(path);
            var target = File.ResolveLinkTarget(path, true);
            if (target != null) path = target.FullName;
            if (!IsOptical(path)) throw new IOException("Not a Linux optical drive: " + path);
            return path;
        }

        private static bool IsOptical(string path)
        {
            if (!path.StartsWith("/dev/sr", StringComparison.Ordinal)) return false;
            string number = path.Substring(7);
            if (number.Length == 0) return false;
            foreach (char c in number) if (c < '0' || c > '9') return false;
            try
            {
                return File.Exists(path) && File.ReadAllText("/sys/class/block/" + Path.GetFileName(path) + "/device/type").Trim() == "5";
            }
            catch (IOException) { return false; }
            catch (UnauthorizedAccessException) { return false; }
        }
    }
}
