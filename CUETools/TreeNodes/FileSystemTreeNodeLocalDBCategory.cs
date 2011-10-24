using System;
using System.Collections.Generic;
using CUETools.Processor;

namespace JDP
{
    public class FileSystemTreeNodeLocalDBCategory : FileSystemTreeNodeLocalDBFolder
    {
        private Converter<CUEToolsLocalDBEntry, string> m_converter_key;
        private Converter<CUEToolsLocalDBEntry, string> m_converter_name;
        private Converter<CUEToolsLocalDBEntry, int> m_converter_icon;
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

        public FileSystemTreeNodeLocalDBCategory(CUEControls.IIconManager icon_mgr, List<CUEToolsLocalDBEntry> group, bool showArtist, bool showYear, int icon, string name, Converter<CUEToolsLocalDBEntry, string> converter_key, Converter<CUEToolsLocalDBEntry, string> converter_name, Converter<CUEToolsLocalDBEntry, int> converter_icon)
            : base(icon_mgr)
        {
            this.Group = group;
            this.m_converter_key = converter_key;
            this.m_converter_name = converter_name ?? converter_key;
            this.m_converter_icon = converter_icon ?? (i => m_icon);
            this.m_icon = icon;
            this.m_name = name;
            this.ShowArtist = showArtist;
            this.ShowYear = showYear;
            this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
            this.Text = this.DisplayName;
        }

        public override void DoExpand()
        {
            foreach (var group in CUEToolsLocalDB.Group(Group, m_converter_key, null))
                this.Nodes.Add(new FileSystemTreeNodeLocalDBGroup(icon_mgr, group, ShowArtist, ShowYear, m_converter_icon(group[0]), m_converter_name(group[0])));
        }
    }
}
