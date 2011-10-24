using System.Collections.Generic;
using System.Windows.Forms;
using CUEControls;
using CUETools.Processor;

namespace JDP
{
    public abstract class FileSystemTreeNodeLocalDBFolder : FileSystemTreeNode
    {
        public List<CUEToolsLocalDBEntry> Group { get; protected set; }

        public FileSystemTreeNodeLocalDBFolder(IIconManager icon_mgr)
            : base(icon_mgr, true)
        {
        }

        public void Purge(List<CUEToolsLocalDBEntry> entries)
        {
            foreach (TreeNode child in this.Nodes)
            {
                if (child is FileSystemTreeNodeLocalDBFolder)
                    (child as FileSystemTreeNodeLocalDBFolder).Purge(entries);
                if ((child is FileSystemTreeNodeLocalDBEntry && entries.Contains((child as FileSystemTreeNodeLocalDBEntry).Item))
                    || (child is FileSystemTreeNodeLocalDBGroup && (child as FileSystemTreeNodeLocalDBGroup).Group.Count == 0))
                    child.Remove();
            }

            this.Group.RemoveAll(item => entries.Contains(item));
        }
    }
}
