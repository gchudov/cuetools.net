using System.IO;

namespace CUETools.Processor
{
    public class ArchiveFileAbstraction : TagLib.File.IFileAbstraction
    {
        private string name;
        private CUESheet _cueSheet;

        public string Name
        {
            get { return name; }
        }

        public Stream ReadStream
        {
            get { return _cueSheet.OpenArchive(Name, true); }
        }

        public Stream WriteStream
        {
            get { return null; }
        }

        public ArchiveFileAbstraction(CUESheet cueSheet, string file)
        {
            name = file;
            _cueSheet = cueSheet;
        }

        public void CloseStream(Stream stream)
        {
            stream.Close();
        }
    }
}
