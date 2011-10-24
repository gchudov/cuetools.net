using System.Collections.Generic;

namespace CUETools.Processor
{
    public class TrackInfo
    {
        private List<CUELine> _attributes;
        public TagLib.File _fileInfo;

        public List<CUELine> Attributes
        {
            get
            {
                return _attributes;
            }
        }

        public TrackInfo()
        {
            _attributes = new List<CUELine>();
            _fileInfo = null;
        }
    }
}
