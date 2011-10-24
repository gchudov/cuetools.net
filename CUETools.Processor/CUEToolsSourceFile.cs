using System.IO;

namespace CUETools.Processor
{
    public class CUEToolsSourceFile
    {
        public string path;
        public string contents;
        public object data;

        public CUEToolsSourceFile(string _path, TextReader reader)
        {
            path = _path;
            contents = reader.ReadToEnd();
            reader.Close();
        }
    }
}
