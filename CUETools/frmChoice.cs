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

		private ListViewItem ToItem(object i)
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
				ReleaseInfo r = new ReleaseInfo(CUE, i as MusicBrainz.Release);
				text = r.Text;
				image = 2;
				i = r;
			}
			else if (i is Freedb.CDEntry)
			{
				ReleaseInfo r = new ReleaseInfo(CUE, i as Freedb.CDEntry);
				text = r.Text;
				image = 1;
				i = r;
			}
			ListViewItem item = new ListViewItem(text, image);
			item.Tag = i;
			return item;
		}

		public IEnumerable<object> Choices
		{
			set
			{
				foreach(object i in value)
				{
					ListViewItem item = ToItem(i);
					listChoices.Items.Add(item);
				}
				if (CUE != null)
				{
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
			if (e.CloseReason != CloseReason.None || DialogResult != DialogResult.OK || item == null || !(item is ReleaseInfo))
				return;
			(item as ReleaseInfo).FillCUE();
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
				textBox1.Text = (item as CUEToolsSourceFile).contents.Replace("\r\n", "\n").Replace("\r", "\n").Replace("\n", "\r\n");
				chkFixEncoding.Visible = false;
			}
			else if (item != null && item is ReleaseInfo)
			{
				ReleaseInfo r = (item as ReleaseInfo);
				listTracks.Items.Clear();
				if (r.musicbrainz != null)
					foreach (MusicBrainz.Track track in r.musicbrainz.GetTracks())
					{
						listTracks.Items.Add(new ListViewItem(new string[] { 
							track.GetTitle(), 
							(listTracks.Items.Count + 1).ToString(),
							CUE == null ? "" : CUE.TOC[listTracks.Items.Count + CUE.TOC.FirstAudio].StartMSF,
							CUE == null ? "" : CUE.TOC[listTracks.Items.Count + CUE.TOC.FirstAudio].LengthMSF
						}));
					}
				if (r.freedb != null)
					for (int i = 0; i < r.freedb.Tracks.Count; i++)
					{
						listTracks.Items.Add(new ListViewItem(new string[] { 
						r.freedb.Tracks[i].Title,
						(i + 1).ToString(),
						CDImageLayout.TimeToString((uint)r.freedb.Tracks[i].FrameOffset - 150),
						CDImageLayout.TimeToString((i + 1 < r.freedb.Tracks.Count) ? (uint) (r.freedb.Tracks[i + 1].FrameOffset - r.freedb.Tracks[i].FrameOffset) :
							(CUE == null || i >= CUE.TOC.TrackCount) ? 0 : CUE.TOC[i + CUE.TOC.FirstAudio].Length)
					}));
					}
				AutoResizeTracks();
				chkFixEncoding.Visible = r.freedb != null;
				chkFixEncoding.Checked = r.freedb_latin1 != null;
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
				chkFixEncoding.Visible = false;
			}
			else
			{
				listTracks.Items.Clear();
				chkFixEncoding.Visible = false;
				textBox1.Text = "";
			}
		}

		private void btnEdit_Click(object sender, EventArgs e)
		{
			object item = ChosenObject;
			if (item == null || CUE == null)
				return;
			if (item is ReleaseInfo)
				(item as ReleaseInfo).FillCUE();
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

		private void chkFixEncoding_CheckedChanged(object sender, EventArgs e)
		{
			if (listChoices.SelectedItems.Count > 0)
			{
				ListViewItem item = listChoices.Items[listChoices.SelectedItems[0].Index];
				if (item.Tag is ReleaseInfo)
				{
					ReleaseInfo r = item.Tag as ReleaseInfo;
					if ((r.freedb_latin1 == null) == chkFixEncoding.Checked)
					{
						r.FixEncoding();
						item.Text = r.Text;
						for (int i = 0; i < r.freedb.Tracks.Count; i++)
							listTracks.Items[i].Text = r.freedb.Tracks[i].Title;
					}
				}
			}
		}
	}

	sealed class ReleaseInfo
	{
		public Freedb.CDEntry freedb_latin1;
		public Freedb.CDEntry freedb;
		public MusicBrainz.Release musicbrainz;
		public CUESheet CUE;
		private Encoding iso;

		public ReleaseInfo(CUESheet cue, Freedb.CDEntry release)
		{
			CUE = cue;
			iso = Encoding.GetEncoding("iso-8859-1");
			freedb_latin1 = null;
			freedb = release;
		}

		public ReleaseInfo(CUESheet cue, MusicBrainz.Release release)
		{
			CUE = cue;
			iso = Encoding.GetEncoding("iso-8859-1");
			musicbrainz = release;
		}

		private string FixEncoding(string src)
		{
			return Encoding.Default.GetString(iso.GetBytes(src));
		}

		public string Text
		{
			get
			{
				if (musicbrainz != null)
					return string.Format("{0}: {1} - {2}",
						musicbrainz.GetEvents().Count > 0 ? musicbrainz.GetEvents()[0].Date.Substring(0, 4) : "YYYY",
						musicbrainz.GetArtist(),
						musicbrainz.GetTitle());
				if (freedb != null)
					return string.Format("{0}: {1} - {2}",
						freedb.Year,
						freedb.Artist,
						freedb.Title);
				return null;
			}
		}

		public void FixEncoding()
		{
			if (freedb == null)
				return;
			if (freedb_latin1 != null)
			{
				freedb = freedb_latin1;
				freedb_latin1 = null;
				return;
			}
			freedb_latin1 = freedb;
			freedb = new Freedb.CDEntry(freedb_latin1);
			freedb.Artist = FixEncoding(freedb.Artist);
			freedb.Title = FixEncoding(freedb.Title);
			foreach (Freedb.Track tr in freedb.Tracks)
				tr.Title = FixEncoding(tr.Title);
		}

		public void FillCUE()
		{
			if (musicbrainz != null)
				CUE.FillFromMusicBrainz(musicbrainz);
			else if (freedb != null)
				CUE.FillFromFreedb(freedb);
		}
	}
}