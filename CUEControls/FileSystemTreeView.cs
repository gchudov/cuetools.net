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
using System.IO;

namespace CUEControls
{
	#region Event Delegate Definitions

	/// <summary>
	/// Represents the method that will handle node attribute events
	/// </summary>
	public delegate void FileSystemTreeViewNodeExpandHandler(object sender, FileSystemTreeViewNodeExpandEventArgs e);
	#endregion

	public partial class FileSystemTreeView : TreeView
	{
		private const string DummyNodeText = "DUMMY";
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
					if (SelectedNode.Tag is FileSystemInfo) 
						return ((FileSystemInfo)SelectedNode.Tag).FullName;
					if (SelectedNode.Tag is ExtraSpecialFolder)
						return m_icon_mgr.GetFolderPath((ExtraSpecialFolder)SelectedNode.Tag);
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
				if (!DesignMode && SelectedNode != null)
					if (SelectedNode.Tag is ExtraSpecialFolder)
						return (ExtraSpecialFolder)SelectedNode.Tag;
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
				if (node.Tag is ExtraSpecialFolder && (ExtraSpecialFolder)node.Tag == tag)
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
				if (node.Tag is ExtraSpecialFolder)
				{
					specialPath = m_icon_mgr.GetFolderPath((ExtraSpecialFolder)node.Tag);
					if (specialPath != null && path.StartsWith(specialPath.ToUpper()) && (top == null || m_icon_mgr.GetFolderPath((ExtraSpecialFolder)top.Tag).Length < specialPath.Length))
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
					if (node.Tag is FileSystemInfo)
					{
						string prefix = ((FileSystemInfo)node.Tag).FullName.ToUpper();
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

		private bool IsDummy(TreeNode n)
		{
			return n.Nodes.Count == 1 && n.Nodes[0].Tag == null && n.Nodes[0].Text == DummyNodeText;
		}

		public TreeNode NewNode(ExtraSpecialFolder folder)
		{
			TreeNode node = new TreeNode();
			node.Tag = folder;
			node.ImageIndex = m_icon_mgr.GetIconIndex(folder, false);
			node.SelectedImageIndex = node.ImageIndex;
			node.Text = m_icon_mgr.GetDisplayName(folder);
			node.Nodes.Add(DummyNodeText);
			return node;
		}

		public TreeNode NewNode(FileSystemInfo file, bool expandable)
		{
			TreeNode node = new TreeNode();
			node.Tag = file;
			node.ImageIndex = m_icon_mgr.GetIconIndex(file, false);
			node.SelectedImageIndex = node.ImageIndex;
			node.Text = m_icon_mgr.GetDisplayName(file);
			if (expandable) node.Nodes.Add(DummyNodeText);
			return node;
		}

		/// <summary>
		/// Populate the contents of the node that is about to be expanded.
		/// </summary>
		/// <param name="e">the arguments giving the node to be expanded</param>
		protected override void OnBeforeExpand(TreeViewCancelEventArgs e)
		{
			if (!IsDummy(e.Node))
				return;

			e.Node.Nodes.Clear();
			try
			{
				ExpandDirInNode(e.Node);
				base.OnBeforeExpand(e);
			}
			catch (Exception ex)
			{
				e.Node.Text = (e.Node.Tag is FileSystemInfo ? m_icon_mgr.GetDisplayName((FileSystemInfo)e.Node.Tag) : 
					e.Node.Tag is ExtraSpecialFolder ? m_icon_mgr.GetDisplayName((ExtraSpecialFolder)e.Node.Tag) : ""
					) + " : " + ex.Message;
				e.Node.Nodes.Clear();
				e.Node.Nodes.Add(DummyNodeText);
				e.Cancel = true;
			}
		}

		/// <summary>
		/// Called after a node is collapsed, used to remove the contents of the node
		/// as node contents are generated when opened.
		/// </summary>
		/// <param name="e"></param>
		protected override void OnAfterCollapse(TreeViewEventArgs e)
		{
			e.Node.Nodes.Clear();

			// Add the dummy node
			e.Node.Nodes.Add(DummyNodeText);
			if (e.Node.Tag is ExtraSpecialFolder)
				e.Node.ImageIndex = m_icon_mgr.GetIconIndex((ExtraSpecialFolder)e.Node.Tag, false);
			else if (e.Node.Tag is DirectoryInfo)
				e.Node.ImageIndex = m_icon_mgr.GetIconIndex((DirectoryInfo)e.Node.Tag, false);
		}

		private void ExpandDirInNode(TreeNode node, DirectoryInfo info)
		{
			FileSystemTreeViewNodeExpandEventArgs e = new FileSystemTreeViewNodeExpandEventArgs();
			e.node = node;
			e.files = info.GetFileSystemInfos();
			if (NodeExpand != null)
				NodeExpand(this, e);
			else
			{
				foreach (FileSystemInfo file in e.files)
				{
					bool isExpandable = (file.Attributes & FileAttributes.Directory) != 0;
					if ((file.Attributes & FileAttributes.Hidden) == 0 && (file.Attributes & FileAttributes.Directory) != 0)
						node.Nodes.Add(NewNode(file, isExpandable));
				}
			}
		}

		private void ExpandDirInNode(TreeNode node, ExtraSpecialFolder path)
		{
			switch (path)
			{
				case ExtraSpecialFolder.Desktop:
					foreach (ExtraSpecialFolder fldr in m_extra_folders)
						try { node.Nodes.Add(NewNode(fldr)); }
						catch { }
					break;
				case ExtraSpecialFolder.MyComputer:
					if (m_icon_mgr.GetFolderPath(path) == "/")
						break;
					foreach (DriveInfo di in DriveInfo.GetDrives())
						try { node.Nodes.Add(NewNode(new DirectoryInfo(di.Name), true)); }
						catch { }
					return;
			}
			string dir = m_icon_mgr.GetFolderPath(path);
			if (dir != null && dir != "" && Directory.Exists(dir)) 
				ExpandDirInNode(node, new DirectoryInfo(dir));
		}

		private void ExpandDirInNode(TreeNode node)
		{
			try
			{
				BeginUpdate();
				if (node.Tag is ExtraSpecialFolder)
				{
					ExpandDirInNode(node, (ExtraSpecialFolder)node.Tag);
					node.ImageIndex = m_icon_mgr.GetIconIndex((ExtraSpecialFolder)node.Tag, true);
				}
				if (node.Tag is DirectoryInfo)
				{
					ExpandDirInNode(node, (DirectoryInfo)node.Tag);
					node.ImageIndex = m_icon_mgr.GetIconIndex((DirectoryInfo)node.Tag, true);
				}
			}
			finally
			{
				EndUpdate();
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
			if (e.Item != null && e.Item is TreeNode && (e.Item as TreeNode).Tag is FileSystemInfo)
			{
				string[] arr = new string[1];
				arr[0] = ((FileSystemInfo)((e.Item as TreeNode).Tag)).FullName;
				DataObject dobj = new DataObject(DataFormats.FileDrop, arr);
				DragDropEffects effects = DoDragDrop(dobj, DragDropEffects.All);
				return;
			}

			base.OnItemDrag(e);
		}
	}

	public class FileSystemTreeViewNodeExpandEventArgs
	{
		public TreeNode node;
		public FileSystemInfo[] files;
	}
}