using System.IO;
using CUEControls;
using CUETools.Processor;

namespace JDP
{
    public class FileSystemTreeNodeLocalDBEntry : FileSystemTreeNode
    {
        private string m_input_path;

        public bool ShowArtist { get; set; }
        public bool ShowYear { get; set; }
        public CUEToolsLocalDBEntry Item { get; private set; }
        public override string Path
        {
            get
            {
                return m_input_path ?? Item.Path;
            }
        }

        public override string DisplayName
        {
            get
            {
                if (m_input_path != null && File.Exists(m_input_path))
                    return icon_mgr.GetDisplayName(new FileInfo(m_input_path));
                return
                      (string.IsNullOrEmpty(Item.Metadata.Artist) || !ShowArtist ? "" : Item.Metadata.Artist + " - ")
                    + Item.Metadata.Title
                    + (string.IsNullOrEmpty(Item.Metadata.Year) || !ShowYear ? "" : " (" + Item.Metadata.Year + ")")
                    + (string.IsNullOrEmpty(Item.Metadata.DiscNumberAndTotal) ? "" : " [" + Item.Metadata.DiscNumberAndTotal + "]");
            }
        }

        public override int DisplayIcon
        {
            get
            {
                return icon_mgr.GetIconIndex(m_input_path ?? (Item.AudioPaths == null || Item.AudioPaths.Count == 0 ? "*.wav" : Item.AudioPaths[0]));
            }
        }

        public FileSystemTreeNodeLocalDBEntry(IIconManager icon_mgr, CUEToolsLocalDBEntry item, bool showArtist, bool showYear, string inputPath)
            : base(icon_mgr, inputPath == null && item.InputPaths != null && item.InputPaths.Count > 1)
        {
            this.Item = item;
            this.m_input_path = inputPath;
            this.ShowArtist = showArtist;
            this.ShowYear = showYear;
            this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
            this.Text = this.DisplayName;
            //// Choose state from m_state_image_list
            //if (item.InputPaths.Find(path => Path.GetExtension(path).ToLower() == ".cue") != null)
            //    album.StateImageKey = "cue";
            //else
            //    album.StateImageKey = "blank";
        }

        public override void DoExpand()
        {
            if (Item.InputPaths != null)
                foreach (var path in Item.InputPaths)
                    this.Nodes.Add(new FileSystemTreeNodeLocalDBEntry(icon_mgr, Item, ShowArtist, ShowYear, path));
        }
    }
}
