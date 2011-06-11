using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;
using CUEControls;
using CUETools.Codecs;
using CUETools.Processor;

namespace CUEPlayer
{
	public partial class Playlist : Form
	{
		private CUEConfig _config;
		private ShellIconMgr _icon_mgr;
		private DataSet1 dataSet;

		public Playlist()
		{
			InitializeComponent();
		}

		public void Init(frmCUEPlayer parent)
		{
			_config = parent.Config;
			dataSet = parent.DataSet;
			MdiParent = parent;
			Show();
			_icon_mgr = parent.IconMgr;
			listViewTracks.SmallImageList = _icon_mgr.ImageList;
			foreach (DataSet1.PlaylistRow row in dataSet.Playlist)
			{
				try
				{
					listViewTracks.Items.Add(ToItem(row));
				}
				catch (Exception ex)
				{
					Trace.WriteLine(ex.Message);
				}
			}
		}

		public ListView List
		{
			get
			{
				return listViewTracks;
			}
		}

		public ListViewItem ToItem(DataSet1.PlaylistRow row)
		{
			ListViewGroup in_group = null;
			string group_name = (row.artist ?? "") + " - " + (row.album ?? "");
			foreach (ListViewGroup group in listViewTracks.Groups)
			{
				if (group.Name == group_name)
				{
					in_group = group;
					break;
				}
			}
			if (in_group == null)
			{
				in_group = new ListViewGroup(group_name, group_name);
				listViewTracks.Groups.Add(in_group);
			}
			int iconIndex = _icon_mgr.GetIconIndex(new FileInfo(row.path), true);
			ListViewItem item = new ListViewItem(row.title, iconIndex);
			TimeSpan Length = TimeSpan.FromSeconds(row.length);
			string lenStr = string.Format("{0:d}.{1:d2}:{2:d2}:{3:d2}", Length.Days, Length.Hours, Length.Minutes, Length.Seconds).TrimStart('0', ':', '.');
			item.SubItems.Add(new ListViewItem.ListViewSubItem(item, lenStr));
			item.Group = in_group;
			item.Tag = row;
			return item;
		}

		private void exploreToolStripMenuItem_Click(object sender, EventArgs e)
		{
			if (listViewTracks.SelectedIndices.Count == 1)
			{
				int index = listViewTracks.SelectedIndices[0];
				string path = (listViewTracks.Items[index].Tag as DataSet1.PlaylistRow).path;
				(MdiParent as frmCUEPlayer).browser.TreeView.SelectedPath = path;
			}
		}

		private void removeToolStripMenuItem_Click(object sender, EventArgs e)
		{
			while (listViewTracks.SelectedIndices.Count > 0)
			{
				int index = listViewTracks.SelectedIndices[0];
				(listViewTracks.Items[index].Tag as DataSet1.PlaylistRow).Delete();
				listViewTracks.Items.RemoveAt(index);
			}
		}

		private void listViewTracks_DragDrop(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
				if (files.Length == 1)
				{
					string path = files[0];
					try
					{
						CUESheet cue = new CUESheet(_config);
						cue.Open(path);
						for (int iTrack = 0; iTrack < cue.TrackCount; iTrack++)
						{
							DataSet1.PlaylistRow row = dataSet.Playlist.AddPlaylistRow(
								path,
								cue.Metadata.Artist,
								cue.Metadata.Tracks[iTrack].Title,
								cue.Metadata.Title,
								(int)cue.TOC[cue.TOC.FirstAudio + iTrack].Length / 75,
								iTrack + 1);
							listViewTracks.Items.Add(ToItem(row));
						}
						cue.Close();
						return;
					}
					catch (Exception ex)
					{
						Trace.WriteLine(ex.Message);
					}

					FileInfo fi = new FileInfo(path);
					if (fi.Extension != ".cue")
					{
						DataSet1.PlaylistRow row = dataSet.Playlist.AddPlaylistRow(
							path,
							null, // cue.Artist,
							null, // cue.Tracks[iTrack].Title,
							null, // cue.Title,
							0, // (int)cue.TOC[cue.TOC.FirstAudio + iTrack].Length / 75,
							0);
						listViewTracks.Items.Add(ToItem(row));
					}
				}
			}
		}

		private void listViewTracks_DragOver(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				e.Effect = DragDropEffects.Copy;
			}
		}

		private void listViewTracks_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Delete)
				removeToolStripMenuItem_Click(sender, EventArgs.Empty);
		}

		private void listViewTracks_ItemDrag(object sender, ItemDragEventArgs e)
		{
			if (e.Item != null && e.Item is ListViewItem)
			{
				DataObject dobj = new DataObject(DataFormats.Serializable, listViewTracks.SelectedIndices);
				DragDropEffects effects = DoDragDrop(dobj, DragDropEffects.All);
				return;
			}
		}

		private void Playlist_Load(object sender, EventArgs e)
		{

		}
	}
}
