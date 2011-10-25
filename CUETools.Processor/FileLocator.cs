using System.Collections.Generic;
using System.IO;

namespace CUETools.Processor
{
    class FileLocator
    {
        public static string LocateFile(string dir, string file, List<string> contents)
        {
            List<string> dirList, fileList;
            string altDir;

            dirList = new List<string>();
            fileList = new List<string>();
            altDir = Path.GetDirectoryName(file);
            file = Path.GetFileName(file);

            dirList.Add(dir);
            if (altDir.Length != 0)
            {
                dirList.Add(Path.IsPathRooted(altDir) ? altDir : Path.Combine(dir, altDir));
            }

            fileList.Add(file);
            fileList.Add(file.Replace(' ', '_'));
            fileList.Add(file.Replace('_', ' '));

            for (int iDir = 0; iDir < dirList.Count; iDir++)
            {
                for (int iFile = 0; iFile < fileList.Count; iFile++)
                {
                    string path = Path.Combine(dirList[iDir], fileList[iFile]);
                    if (contents == null && System.IO.File.Exists(path))
                        return path;
                    if (contents != null)
                    {
                        List<string> matching = contents.FindAll(s => s.ToLower().Replace('/', Path.DirectorySeparatorChar) ==
                            path.ToLower().Replace('/', Path.DirectorySeparatorChar));
                        if (matching.Count == 1)
                            return matching[0];
                    }
                }
            }

            return null;
        }
    }
}
