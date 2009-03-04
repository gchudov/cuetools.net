using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CDImage;
using CUETools.Processor;
using MusicBrainz;
using Freedb;

namespace JDP
{
	public partial class frmChoice : Form
	{
		public frmChoice()
		{
			InitializeComponent();
		}

		public CUESheet CUE;

		private void frmChoice_Load(object sender, EventArgs e)
		{
			button1.Select();
		}

		public IEnumerable<object> Choices
		{
			set
			{
				bool isCD = false;
				foreach(object i in value)
				{
					string text = "";
					int image = -1;
					if (i is string)
						text = i as string;
					else if (i is CUEToolsSourceFile)
					{
						text = (i as CUEToolsSourceFile).path;
						image = 0;
					}
					else if (i is MusicBrainz.Release)
					{
						MusicBrainz.Release release = i as MusicBrainz.Release;
						text = String.Format("{0}: {1} - {2}",
							release.GetEvents().Count > 0 ? release.GetEvents()[0].Date.Substring(0, 4) : "YYYY",
							release.GetArtist(),
							release.GetTitle());
						image = 2;
						isCD = true;
					}
					else if (i is Freedb.CDEntry)
					{
						CDEntry cdEntry = i as CDEntry;
						text = String.Format("{0}: {1} - {2}",
							cdEntry.Year,
							cdEntry.Artist,
							cdEntry.Title);
						image = 1;
						isCD = true;
					}
					ListViewItem item = new ListViewItem(text, image);
					item.Tag = i;
					listChoices.Items.Add(item);
				}
				if (isCD)
				{
					if (CUE == null)
						throw new Exception("selecting release information, but cue sheet has not been set");
					string text = String.Format("{0}: {1} - {2}",
						CUE.Year == "" ? "YYYY" : CUE.Year,
						CUE.Artist == "" ? "Unknown Artist" : CUE.Artist,
						CUE.Title == "" ? "Unknown Title" : CUE.Title);
					ListViewItem item = new ListViewItem(text, 3);
					item.Tag = CUE;
					listChoices.Items.Insert(0, item);
					textBox1.Hide();
					listTracks.Show();
					btnEdit.Show();
				}
				if (listChoices.Items.Count > 0)
					listChoices.Items[0].Selected = true;
			}
		}

		public int ChosenIndex
		{
			get
			{
				return listChoices.SelectedItems.Count > 0 ? listChoices.SelectedItems[0].Index : -1;
			}
		}

		public object ChosenObject
		{
			get
			{
				return listChoices.SelectedItems.Count > 0 ? listChoices.SelectedItems[0].Tag : null;
			}
		}

		private void frmChoice_FormClosing(object sender, FormClosingEventArgs e)
		{
			object item = ChosenObject;
			if (e.CloseReason != CloseReason.None || DialogResult != DialogResult.OK || item == null)
				return;
			if (item is MusicBrainz.Release)
				CUE.FillFromMusicBrainz((MusicBrainz.Release)item);
			else if (item is CDEntry)
				CUE.FillFromFreedb((CDEntry)item);
		}

		private void AutoResizeTracks()
		{
			listTracks.Columns[1].AutoResize(ColumnHeaderAutoResizeStyle.ColumnContent);
			listTracks.Columns[2].AutoResize(ColumnHeaderAutoResizeStyle.ColumnContent);
			listTracks.Columns[3].AutoResize(ColumnHeaderAutoResizeStyle.ColumnContent);
			int widthAvailable = listTracks.Width - listTracks.Columns[1].Width - listTracks.Columns[2].Width - listTracks.Columns[3].Width - listTracks.Padding.Horizontal - 24;
			if (listTracks.Columns[0].Width < widthAvailable)
				listTracks.Columns[0].Width = widthAvailable;
		}

		private void listChoices_SelectedIndexChanged(object sender, EventArgs e)
		{
			object item = ChosenObject;
			if (item != null && item is CUEToolsSourceFile)
			{
				textBox1.Text = (item as CUEToolsSourceFile).contents.Replace("\r\n", "\r").Replace("\r", "\r\n");
			}
			else if (item != null && item is MusicBrainz.Release)
			{
				MusicBrainz.Release release = item as MusicBrainz.Release;
				listTracks.Items.Clear();
				foreach (MusicBrainz.Track track in release.GetTracks())
				{
					listTracks.Items.Add(new ListViewItem(new string[] { 
						track.GetTitle(), 
						(listTracks.Items.Count + 1).ToString(),
						CUE == null ? "" : CUE.TOC[listTracks.Items.Count + 1].StartMSF,
						CUE == null ? "" : CUE.TOC[listTracks.Items.Count + 1].LengthMSF
					}));
				}
				AutoResizeTracks();
			}
			else if (item != null && item is CDEntry)
			{
				CDEntry cdEntry = item as CDEntry;
				
				listTracks.Items.Clear();
				for (int i = 0; i < cdEntry.Tracks.Count; i++)
				{
					listTracks.Items.Add(new ListViewItem(new string[] { 
						cdEntry.Tracks[i].Title, 						
						(i + 1).ToString(),
						CDImageLayout.TimeToString((uint)cdEntry.Tracks[i].FrameOffset - 150),
						CDImageLayout.TimeToString((i + 1 < cdEntry.Tracks.Count) ? (uint) (cdEntry.Tracks[i + 1].FrameOffset - cdEntry.Tracks[i].FrameOffset) :
							(CUE == null || i >= CUE.TOC.TrackCount) ? 0 : CUE.TOC[i + 1].Length)
					}));
				}
				AutoResizeTracks();
			}
			else if (item != null && item is CUESheet)
			{
				CUESheet cueSheet = item as CUESheet;
				listTracks.Items.Clear();
				foreach (TrackInfo track in cueSheet.Tracks)
				{
					listTracks.Items.Add(new ListViewItem(new string[] { 
						track.Title, 						
						(listTracks.Items.Count + 1).ToString(),
						CUE == null ? "" : CUE.TOC[listTracks.Items.Count + 1].StartMSF,
						CUE == null ? "" : CUE.TOC[listTracks.Items.Count + 1].LengthMSF
					}));
				}
				AutoResizeTracks();
			}
			else
			{
				listTracks.Items.Clear();
				textBox1.Text = "";
			}
		}

		private void btnEdit_Click(object sender, EventArgs e)
		{
			object item = ChosenObject;
			if (item == null || CUE == null)
				return;
			if (item is MusicBrainz.Release)
				CUE.FillFromMusicBrainz((MusicBrainz.Release)item);
			else if (item is CDEntry)
				CUE.FillFromFreedb((CDEntry)item);
			else if (!(item is CUESheet))
				return;
			listChoices.Items[0].Selected = true;
			listChoices.Items[0].Text = String.Format("{0}: {1} - {2}",
				CUE.Year == "" ? "YYYY" : CUE.Year,
				CUE.Artist == "" ? "Unknown Artist" : CUE.Artist,
				CUE.Title == "" ? "Unknown Title" : CUE.Title);
			frmProperties frm = new frmProperties();
			frm.CUE = CUE;
			if (frm.ShowDialog(this) == DialogResult.OK)
				listChoices.Items[0].Text = String.Format("{0}: {1} - {2}",
					CUE.Year == "" ? "YYYY" : CUE.Year,
					CUE.Artist == "" ? "Unknown Artist" : CUE.Artist,
					CUE.Title == "" ? "Unknown Title" : CUE.Title);
		}

		private void listTracks_DoubleClick(object sender, EventArgs e)
		{
			listTracks.FocusedItem.BeginEdit();
		}

		private void listTracks_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.F2)
				listTracks.FocusedItem.BeginEdit();
		}

		private void listTracks_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			object item = ChosenObject;
			if (item == null || !(item is CUESheet))
			{
				e.CancelEdit = true;
				return;
			}
		}

		private void listTracks_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			if (e.Label != null)
			{
				CUE.Tracks[e.Item].Title = e.Label;
			}
		}

		private void listTracks_PreviewKeyDown(object sender, PreviewKeyDownEventArgs e)
		{
			if (e.KeyCode == Keys.Enter)
			{
			    if (listTracks.FocusedItem.Index + 1 < listTracks.Items.Count) // && e.editing
			    {
			        listTracks.FocusedItem.Selected = false;
			        listTracks.FocusedItem = listTracks.Items[listTracks.FocusedItem.Index + 1];
			        listTracks.FocusedItem.Selected = true;
			        listTracks.FocusedItem.BeginEdit();
			    }
			}
		}
	}
}