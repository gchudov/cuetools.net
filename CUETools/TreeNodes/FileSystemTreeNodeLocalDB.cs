using System;
using System.Collections.Generic;
using CUETools.Processor;

namespace JDP
{
    public class FileSystemTreeNodeLocalDB : FileSystemTreeNodeLocalDBFolder
    {
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
                return "Local DB";
            }
        }

        public override int DisplayIcon
        {
            get
            {
                return icon_mgr.GetIconIndex(".#puzzle");
            }
        }

        public FileSystemTreeNodeLocalDB(CUEControls.IIconManager icon_mgr, List<CUEToolsLocalDBEntry> group)
            : base(icon_mgr)
        {
            this.Group = group;
            this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
            this.Text = this.DisplayName;
        }

        public override void DoExpand()
        {
            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, true, true, icon_mgr.GetIconIndex(".#puzzle"), "By Uniqueness",
                    i => ((int)FileSystemTreeNodeLocalDBCollision.GetGroupType(this.Group.FindAll(j => j.DiscID == i.DiscID))).ToString(),
                    i => FileSystemTreeNodeLocalDBCollision.GroupTypeToDescription(FileSystemTreeNodeLocalDBCollision.GetGroupType(this.Group.FindAll(j => j.DiscID == i.DiscID))),
                    i => icon_mgr.GetIconIndex(FileSystemTreeNodeLocalDBCollision.GroupTypeToIconTag(FileSystemTreeNodeLocalDBCollision.GetGroupType(this.Group.FindAll(j => j.DiscID == i.DiscID)))))); //converter_icon

            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, true, true, icon_mgr.GetIconIndex(".flac"), "By Format",
                    i => i.AudioPaths == null || i.AudioPaths.Count == 0 ? null : System.IO.Path.GetExtension(i.AudioPaths[0]).ToLower(),
                    null,
                    i => icon_mgr.GetIconIndex(i.AudioPaths[0])));

            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, false, true, icon_mgr.GetIconIndex(".#users"), "By Artist",
                    i => i.Metadata.Artist, null, null));

            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, true, false, icon_mgr.GetIconIndex(".#calendar"), "By Release Date",
                    i => i.Metadata.Year, null, null));

            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, true, true, icon_mgr.GetIconIndex(".#alarm_clock"), "By Verification Date",
                    i =>
                        i.VerificationDate == DateTime.MinValue ? "0" :
                        i.VerificationDate.AddHours(1) > DateTime.Now ? "1" :
                        i.VerificationDate.AddDays(1) > DateTime.Now ? "2" :
                        i.VerificationDate.AddDays(7) > DateTime.Now ? "3" :
                        i.VerificationDate.AddDays(31) > DateTime.Now ? "4" :
                        i.VerificationDate.AddDays(365) > DateTime.Now ? "5" :
                        "6",
                    i =>
                        i.VerificationDate == DateTime.MinValue ? "never" :
                        i.VerificationDate.AddHours(1) > DateTime.Now ? "this hour" :
                        i.VerificationDate.AddDays(1) > DateTime.Now ? "this day" :
                        i.VerificationDate.AddDays(7) > DateTime.Now ? "this week" :
                        i.VerificationDate.AddDays(31) > DateTime.Now ? "this month" :
                        i.VerificationDate.AddDays(365) > DateTime.Now ? "this year" :
                        "more than a year ago",
                    null));

            this.Nodes.Add(new FileSystemTreeNodeLocalDBCategory(
                icon_mgr, this.Group, true, true, icon_mgr.GetIconIndex(".#ar"), "By AccurateRip Confidence",
                    i =>
                        i.VerificationDate == DateTime.MinValue ? "00" :
                        i.ARConfidence == 0 ? "01" :
                        i.ARConfidence == 1 ? "02" :
                        i.ARConfidence == 2 ? "03" :
                        i.ARConfidence == 3 ? "04" :
                        i.ARConfidence < 5 ? "05" :
                        i.ARConfidence < 10 ? "06" :
                        i.ARConfidence < 20 ? "07" :
                        i.ARConfidence < 50 ? "08" :
                        i.ARConfidence < 100 ? "09" :
                        "10",
                    i =>
                        i.VerificationDate == DateTime.MinValue ? "?" :
                        i.ARConfidence == 0 ? "0" :
                        i.ARConfidence == 1 ? "1" :
                        i.ARConfidence == 2 ? "2" :
                        i.ARConfidence == 3 ? "3" :
                        i.ARConfidence < 5 ? "<   5" :
                        i.ARConfidence < 10 ? "<  10" :
                        i.ARConfidence < 20 ? "<  20" :
                        i.ARConfidence < 50 ? "<  50" :
                        i.ARConfidence < 100 ? "< 100" :
                        ">=100",
                    null));
        }
    }
}
