using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUEControls;
using CUETools.Codecs;
using CUETools.Processor;

namespace CUEPlayer
{
	public partial class Browser : Form
	{
		private CUEConfig _config;

		public Browser()
		{
			InitializeComponent();
		}

		public void Init(frmCUEPlayer parent)
		{
			_config = parent.Config;
			MdiParent = parent;
			Show();
			fileSystemTreeView1.IconManager = parent.IconMgr;
			fileSystemTreeView1.SelectedFolder = ExtraSpecialFolder.MyMusic;
		}

		internal FileSystemTreeView TreeView
		{
			get
			{
				return fileSystemTreeView1;
			}
		}

		private void fileSystemTreeView1_NodeExpand(object sender, CUEControls.FileSystemTreeViewNodeExpandEventArgs e)
		{
			List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, e.files);
			foreach (FileGroupInfo fileGroup in fileGroups)
			{
				TreeNode node = fileSystemTreeView1.NewNode(fileGroup.main);
				if (fileGroup.type == FileGroupInfoType.TrackFiles)
					node.Text = node.Text + ": " + fileGroup.files.Count.ToString() + " files";
				e.node.Nodes.Add(node);
			}
		}

		private void Browser_Load(object sender, EventArgs e)
		{

		}
	}
}
