//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for Windows XP
// 
// Copyright (C) 2006 by Jack W. Griffin (butchg@comcast.net)
//
// This program is free software; you can redistribute it and/or modify 
// it under the terms of the GNU General Public License as published by 
// the Free Software Foundation; either version 2 of the License, or 
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
// for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program; if not, write to the 
//
// Free Software Foundation, Inc., 
// 59 Temple Place, Suite 330, 
// Boston, MA 02111-1307 USA
//
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;
using System.Windows.Forms.Design;
using System.IO;

namespace CUEControls
{
	#region Event Delegate Definitions

	/// <summary>
	/// Represents the method that will handle node attribute events
	/// </summary>
	public delegate void FileSystemTreeViewNodeExpandHandler(object sender, FileSystemTreeViewNodeExpandEventArgs e);
	#endregion

	public class DummyNode : TreeNode
	{
	}

	public abstract class FileSystemTreeNode : TreeNode
	{
		protected IIconManager icon_mgr;

		abstract public string Path
		{
			get;
		}

		abstract public string DisplayName
		{
			get;
		}

		abstract public int DisplayIcon
		{
			get;
		}

		public bool IsExpandable
		{
			get
			{
				return Nodes.Count == 1 && Nodes[0] is DummyNode;
			}

			set
			{
				if (value)
				{
					Nodes.Clear();
					Nodes.Add(new DummyNode());
				}
			}
		}

		public FileSystemTreeNode(IIconManager icon_mgr, bool expandable)
		{
			this.icon_mgr = icon_mgr;
			if (expandable)
				IsExpandable = true;
		}

		virtual public void DoExpand()
		{
			if (!Directory.Exists(this.Path))
				return;
			var info = new DirectoryInfo(this.Path);
			FileSystemTreeViewNodeExpandEventArgs e = new FileSystemTreeViewNodeExpandEventArgs();
			e.node = this;
			e.files = info.GetFileSystemInfos();
			(this.TreeView as FileSystemTreeView).OnDoExpand(e);
		}
	}

	public class FileSystemTreeNodeSpecialFolder : FileSystemTreeNode
	{
		public ExtraSpecialFolder Folder
		{
			get;
			private set;
		}

		public override string Path
		{
			get
			{
				return icon_mgr.GetFolderPath(Folder);
			}
		}

		public override string DisplayName
		{
			get
			{
				return icon_mgr.GetDisplayName(Folder);
			}
		}

		public override int DisplayIcon
		{
			get
			{
				return icon_mgr.GetIconIndex(Folder, !IsExpandable);
			}
		}

		public FileSystemTreeNodeSpecialFolder(IIconManager icon_mgr, ExtraSpecialFolder folder)
			: base(icon_mgr, true)
		{
			this.Folder = folder;
			this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
			this.Text = this.DisplayName;
		}

		public override void DoExpand()
		{
			switch (Folder)
			{
				case ExtraSpecialFolder.Desktop:
					foreach (ExtraSpecialFolder fldr in (this.TreeView as FileSystemTreeView).SpecialFolders)
						try { Nodes.Add(new FileSystemTreeNodeSpecialFolder(icon_mgr, fldr)); }
						catch { }
					break;
				case ExtraSpecialFolder.MyComputer:
					if (Path == "/")
						break;
					foreach (DriveInfo di in DriveInfo.GetDrives())
						try { Nodes.Add(new FileSystemTreeNodeFileSystemInfo(icon_mgr, new DirectoryInfo(di.Name))); }
						catch { }
					return;
			}
			base.DoExpand();
		}
	}

	public class FileSystemTreeNodeFileSystemInfo : FileSystemTreeNode
	{
		public FileSystemInfo File
		{
			get;
			private set;
		}

		public override string Path
		{
			get
			{
				return File.FullName;
			}
		}

		public override string DisplayName
		{
			get
			{
				return icon_mgr.GetDisplayName(File);
			}
		}

		public override int DisplayIcon
		{
			get
			{
				return icon_mgr.GetIconIndex(File, !IsExpandable);
			}
		}

		public FileSystemTreeNodeFileSystemInfo(IIconManager icon_mgr, FileSystemInfo file)
			: base(icon_mgr, (file.Attributes & FileAttributes.Directory) != 0)
		{
			this.File = file;
			this.SelectedImageIndex = this.ImageIndex = this.DisplayIcon;
			this.Text = this.DisplayName;
		}
	}

	public partial class FileSystemTreeView : TreeView
	{
		private IIconManager m_icon_mgr;
		private ExtraSpecialFolder[] m_extra_folders;

		#region Public event declarations
		/// <summary>
		/// Event that is raised when a new chunk of data has been extracted
		/// </summary>
		public event FileSystemTreeViewNodeExpandHandler NodeExpand;
		#endregion

		/// <summary>
		/// Create the file system tree view control
		/// </summary>
		public FileSystemTreeView()
		{
			m_extra_folders = new ExtraSpecialFolder[] { ExtraSpecialFolder.MyComputer, ExtraSpecialFolder.MyDocuments, ExtraSpecialFolder.CommonDocuments };
			InitializeComponent();
		}

		/// <summary>
		/// This property is the ICON manager for the icons
		/// </summary>
		[DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
		public IIconManager IconManager
		{
			get
			{
				return m_icon_mgr;
			}
			set
			{
				m_icon_mgr = value;
				if (m_icon_mgr != null)
				{
					ImageList = m_icon_mgr.ImageList;
					if (!DesignMode && Nodes.Count == 0)
					{
						Nodes.Add(NewNode(ExtraSpecialFolder.Desktop));
						Nodes[0].Expand();
						//Nodes[0].Nodes[0].Expand();
					}
				}
				//if (DesignMode)
				//{
				//    Nodes.Clear();
				//    Nodes.Add(new TreeNode("Desktop"));
				//    Nodes[0].Nodes.Add(new TreeNode("My Computer"));
				//    Nodes[0].Nodes[0].Nodes.Add(new TreeNode("Local Disk (C:)"));
				//    foreach (ExtraSpecialFolder fldr in m_extra_folders)
				//        Nodes[0].Nodes.Add(new TreeNode(fldr.ToString()));
				//    Nodes[0].Expand();
				//}
			}
		}

		/// <summary>
		/// The filesystem path corresponding to SelectedNode.
		/// </summary>
		[DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
		public string SelectedPath
		{
			set
			{
				if (!DesignMode && value != null)
				{
					TreeNode node = LookupNode(value);
					if (node != null)
						SelectedNode = node;
				}
			}
			get
			{
				if (!DesignMode && SelectedNode != null)
				{
					if (SelectedNode is FileSystemTreeNode)
						return (SelectedNode as FileSystemTreeNode).Path;
					if (SelectedNode.Tag is string)
						return (string)SelectedNode.Tag;
				}
				return null;				
			}
		}

		/// <summary>
		/// The special folder corresponding to SelectedNode.
		/// </summary>
		[DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
		public ExtraSpecialFolder SelectedFolder
		{
			set
			{
				if (!DesignMode && m_icon_mgr != null)
				{
					TreeNode node = LookupNode(m_icon_mgr.GetFolderPath(value));
					if (node != null)
						SelectedNode = node;
				}
			}
			get
			{
				if (!DesignMode && SelectedNode as FileSystemTreeNodeSpecialFolder != null)
					return (SelectedNode as FileSystemTreeNodeSpecialFolder).Folder;
				return ExtraSpecialFolder.Desktop;
			}
		}

		[Localizable(false)]
		[MergableProperty(false)]
		[DesignerSerializationVisibility(DesignerSerializationVisibility.Content)]
		public ExtraSpecialFolder[] SpecialFolders
		{
			get
			{
				return m_extra_folders;
			}
			set
			{
				m_extra_folders = value;
			}
		}

		private TreeNode LookupNode(TreeNodeCollection nodes, ExtraSpecialFolder tag)
		{
			foreach (TreeNode node in nodes)
				if (node is FileSystemTreeNodeSpecialFolder && (node as FileSystemTreeNodeSpecialFolder).Folder == tag)
					return node;
			return null;
		}

		public TreeNode LookupNode(string path)
		{
			path = Path.GetFullPath(path).ToUpper();

			TreeNode desktop = LookupNode(Nodes, ExtraSpecialFolder.Desktop);
			if (desktop == null)
				return null;
			if (!desktop.IsExpanded)
				desktop.Expand();

			TreeNode top = null;

			string specialPath = m_icon_mgr.GetFolderPath(ExtraSpecialFolder.Desktop);
			if (specialPath != null && path.StartsWith(specialPath.ToUpper()))
			{
				if (path == specialPath.ToUpper())
					return desktop;
				top = desktop;
			}

			foreach (TreeNode node in desktop.Nodes)
				if (node is FileSystemTreeNodeSpecialFolder)
				{
					specialPath = (node as FileSystemTreeNodeSpecialFolder).Path;
					if (specialPath != null && path.StartsWith(specialPath.ToUpper()) && (top as FileSystemTreeNodeSpecialFolder == null || (top as FileSystemTreeNodeSpecialFolder).Path.Length < specialPath.Length))
					{
						if (path == specialPath.ToUpper())
							return node;
						top = node;
					}
				}

			if (top == null)
			{
				TreeNode computer = LookupNode(desktop.Nodes, ExtraSpecialFolder.MyComputer);
				if (computer != null)
					top = computer;
			}

			bool found;
			do
			{
				if (!top.IsExpanded)
					top.Expand();
				found = false;
				foreach (TreeNode node in top.Nodes)
				{
					if (node is FileSystemTreeNodeFileSystemInfo)
					{
						string prefix = (node as FileSystemTreeNodeFileSystemInfo).File.FullName.ToUpper();
						if (path == prefix)
							return node;
						if (path.StartsWith(prefix) && (prefix.EndsWith(PathSeparator) || path.Substring(prefix.Length).StartsWith(PathSeparator)))
						{
							top = node;
							found = true;
							break;
						}
					}
				}
			} while (found);
			return null;
		}

		public TreeNode NewNode(ExtraSpecialFolder folder)
		{
			return new FileSystemTreeNodeSpecialFolder(m_icon_mgr, folder);
		}

		public TreeNode NewNode(FileSystemInfo file)
		{
			return new FileSystemTreeNodeFileSystemInfo(m_icon_mgr, file);
		}

		public TreeNode NewNode(string file)
		{
			if (File.Exists(file))
				return new FileSystemTreeNodeFileSystemInfo(m_icon_mgr, new FileInfo(file));
			var icon = m_icon_mgr.GetIconIndex(file);
			var res = new TreeNode(Path.GetFileNameWithoutExtension(file), icon, icon);
			return res;
		}

		/// <summary>
		/// Populate the contents of the node that is about to be expanded.
		/// </summary>
		/// <param name="e">the arguments giving the node to be expanded</param>
		protected override void OnBeforeExpand(TreeViewCancelEventArgs e)
		{
			FileSystemTreeNode node = e.Node as FileSystemTreeNode;
			if (node != null && node.IsExpandable)
			{
				BeginUpdate();
				node.Nodes.Clear();
				try
				{
					node.DoExpand();
					node.SelectedImageIndex = node.ImageIndex = node.DisplayIcon;
				}
				catch (Exception ex)
				{
					node.Text = node.DisplayName + " : " + ex.Message;
					node.IsExpandable = true;
					e.Cancel = true;
				}
				finally
				{
					EndUpdate();
				}
			}
			base.OnBeforeExpand(e);
		}

		/// <summary>
		/// Called after a node is collapsed, used to remove the contents of the node
		/// as node contents are generated when opened.
		/// </summary>
		/// <param name="e"></param>
		protected override void OnAfterCollapse(TreeViewEventArgs e)
		{
			FileSystemTreeNode node = e.Node as FileSystemTreeNode;
			if (node != null)
			{
				node.IsExpandable = true;
				node.SelectedImageIndex = node.ImageIndex = node.DisplayIcon;
			}
		}

		internal void OnDoExpand(FileSystemTreeViewNodeExpandEventArgs e)
		{
			if (NodeExpand != null)
				NodeExpand(this, e);
			else
			{
				foreach (FileSystemInfo file in e.files)
					if ((file.Attributes & FileAttributes.Hidden) == 0 && (file.Attributes & FileAttributes.Directory) != 0)
						e.node.Nodes.Add(NewNode(file));
			}
		}

		protected override void WndProc(ref Message m)
		{
			// Filter double-click messages
			if (CheckBoxes && m.Msg == 0x203) return;

			base.WndProc(ref m);
		}

		/// <summary>
		/// Called when the drag'n'drop operation begins
		/// </summary>
		/// <param name="e">the location of the event</param>
		protected override void OnItemDrag(ItemDragEventArgs e)
		{
			if (e.Item != null && e.Item is FileSystemTreeNode)
			{
				var item = e.Item as FileSystemTreeNode;
				if (item.Path != null && File.Exists(item.Path))
				{
					string[] arr = new string[1];
					arr[0] = item.Path;
					DataObject dobj = new DataObject(DataFormats.FileDrop, arr);
					DragDropEffects effects = DoDragDrop(dobj, DragDropEffects.All);
					return;
				}
			}

			base.OnItemDrag(e);
		}
	}

	public class FileSystemTreeViewNodeExpandEventArgs
	{
		public TreeNode node;
		public FileSystemInfo[] files;
	}

	[ToolStripItemDesignerAvailability(ToolStripItemDesignerAvailability.ToolStrip | ToolStripItemDesignerAvailability.StatusStrip)]
	public partial class ToolStripCheckedBox : ToolStripControlHost
	{
		public ToolStripCheckedBox()
			: base(new CheckBox())
		{
		}

		[DesignerSerializationVisibility(DesignerSerializationVisibility.Content)]
		public CheckBox MyCheckBox
		{
			get { return (CheckBox)this.Control; }
		}
	}
}