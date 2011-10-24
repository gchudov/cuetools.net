using System;
using System.Collections.Generic;
using CUETools.Processor;

namespace JDP
{
    public class FileSystemTreeNodeLocalDBGroup : FileSystemTreeNodeLocalDBFolder
    {
        private int m_icon;
        private string m_name;

        public bool ShowArtist { get; set; }
        public bool ShowYear { get; set; }

        public override string Path
        {
            get { return null; }
        }

        public override string DisplayName
        {
            get { return m_name; }
        }

        public override int DisplayIcon
        {
            get { return m_icon; }
        }

        public FileSystemTreeNodeLocalDBGroup(CUEControls.IIconManager icon_mgr, List<CUEToolsLocalDBEntry> group, bool showArtist, bool showYear, int icon, string name)
            : base(icon_mgr)
        {
            this.Group = group;
            this.m_icon = icon;
            this.m_name = name;
            this.ShowArtist = showArtist;
            this.ShowYear = showYear;
            this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
            this.Text = this.DisplayName;
        }

        private static int Compare(List<CUEToolsLocalDBEntry> a, List<CUEToolsLocalDBEntry> b)
        {
            int diff = FileSystemTreeNodeLocalDBCollision.GetGroupType(a) - FileSystemTreeNodeLocalDBCollision.GetGroupType(b);
            return diff != 0 ? diff :
                String.Compare(
                a[0].Metadata.Artist + " - " + a[0].Metadata.Title + " - " + a[0].Metadata.DiscNumberAndTotal,
                b[0].Metadata.Artist + " - " + b[0].Metadata.Title + " - " + b[0].Metadata.DiscNumberAndTotal);
        }

        public override void DoExpand()
        {
            var byDiscId = CUEToolsLocalDB.Group(Group, i => i.DiscID, (a, b) => Compare(a, b));
            foreach (var group in byDiscId)
            {
                if (group.Count > 1)
                    this.Nodes.Add(new FileSystemTreeNodeLocalDBCollision(icon_mgr, group, ShowArtist, ShowYear));
                else
                    this.Nodes.Add(new FileSystemTreeNodeLocalDBEntry(icon_mgr, group[0], ShowArtist, ShowYear, null));
            }
        }
    }
}
