using System.Collections.Generic;
using CUETools.Processor;

namespace JDP
{
    public class FileSystemTreeNodeLocalDBCollision : FileSystemTreeNodeLocalDBFolder
    {
        public enum GroupType
        {
            Unverified = 0,
            Different = 1,
            Offsetted = 2,
            Equal = 3,
            Single = 4
        }

        public bool ShowArtist { get; set; }
        public bool ShowYear { get; set; }
        public override string Path
        {
            get
            {
                return null;
            }
        }

        public override string DisplayName
        {
            get
            {
                var artistItem = Group.Find(i => !string.IsNullOrEmpty(i.Metadata.Artist) && ShowArtist);
                var titleItem = Group.Find(i => !string.IsNullOrEmpty(i.Metadata.Title));
                var yearItem = Group.Find(i => !string.IsNullOrEmpty(i.Metadata.Year) && ShowYear);
                var discItem = Group.Find(i => !string.IsNullOrEmpty(i.Metadata.DiscNumberAndTotal));
                return
                      (artistItem == null ? "" : artistItem.Metadata.Artist + " - ")
                    + (titleItem == null ? "" : titleItem.Metadata.Title)
                    + (yearItem == null ? "" : " (" + yearItem.Metadata.Year + ")")
                    + (discItem == null ? "" : " [" + discItem.Metadata.DiscNumberAndTotal + "]");
            }
        }

        public override int DisplayIcon
        {
            get
            {
                return icon_mgr.GetIconIndex(GroupTypeToIconTag(GetGroupType(Group)));
            }
        }

        public FileSystemTreeNodeLocalDBCollision(CUEControls.IIconManager icon_mgr, List<CUEToolsLocalDBEntry> group, bool showArtist, bool showYear)
            : base(icon_mgr)
        {
            this.Group = group;
            this.ShowArtist = showArtist;
            this.ShowYear = showYear;
            this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
            this.Text = this.DisplayName;
        }

        internal static string GroupTypeToIconTag(GroupType type)
        {
            return type == GroupType.Equal ? ".#picture"
                : type == GroupType.Offsetted ? ".#pictures"
                : type == GroupType.Different ? ".#images"
                : type == GroupType.Unverified ? ".#images_question"
                : ".#puzzle";
        }

        internal static string GroupTypeToDescription(GroupType type)
        {
            return type == GroupType.Equal ? "Identical clones"
                : type == GroupType.Offsetted ? "Offsetted clones"
                : type == GroupType.Different ? "Mismatching clones"
                : type == GroupType.Unverified ? "Not yet verified clones"
                : "Unique";
        }

        internal static GroupType GetGroupType(List<CUEToolsLocalDBEntry> group)
        {
            if (group.Count < 2)
                return GroupType.Single;
            if (!group.TrueForAll(i => i.OffsetSafeCRC != null))
                return GroupType.Unverified;
            if (!group.TrueForAll(i => i.OffsetSafeCRC.DifferByOffset(group[0].OffsetSafeCRC)))
                return GroupType.Different;
            if (!group.TrueForAll(i => i.OffsetSafeCRC == group[0].OffsetSafeCRC))
                return GroupType.Offsetted;
            return GroupType.Equal;
        }

        public override void DoExpand()
        {
            foreach (var item in Group)
                this.Nodes.Add(new FileSystemTreeNodeLocalDBEntry(icon_mgr, item, ShowArtist, ShowYear, null));
        }
    }
}
