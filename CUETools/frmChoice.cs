using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.IO;
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
			buttonOk.Select();
		}

		private void AddItem(object i)
		{
			if (i is CUEToolsSourceFile)
			{
				CUEToolsSourceFile sf = i as CUEToolsSourceFile;
				ListViewItem item = new ListViewItem(sf.path, 0);
				item.Tag = sf;
				listChoices.Items.Add(item);
			}
			else if (i is TagLib.IPicture)
			{
				TagLib.IPicture pic = i as TagLib.IPicture;
				ListViewItem item = new ListViewItem(pic.Description, -1);
				item.Tag = pic;
				listChoices.Items.Add(item);
			}
			else if (i is MusicBrainz.Release)
			{
				ReleaseInfo r = new ReleaseInfo(CUE, i as MusicBrainz.Release);
				ListViewItem item = new ListViewItem(r.Text, 2);
				item.Tag = r;
				listChoices.Items.Add(item);
			}
			else if (i is Freedb.CDEntry)
			{
				ReleaseInfo r = new ReleaseInfo(CUE, i as Freedb.CDEntry);
				ListViewItem item = new ListViewItem(r.Text, 1);
				item.Tag = r;
				listChoices.Items.Add(item);

				// check if the entry contains non-iso characters,
				// and add a second one if it does
				ReleaseInfo r2 = new ReleaseInfo(CUE, i as Freedb.CDEntry);
				r2.FixEncoding();
				if (!r.Equals(r2))
				{
					item = new ListViewItem(r2.Text, 1);
					item.Tag = r2;
					listChoices.Items.Add(item);
				}
				return;
			}
			else if (i is CUESheet)
			{
				ReleaseInfo r = new ReleaseInfo(CUE);
				ListViewItem item = new ListViewItem(r.Text, 3);
				item.Tag = r;
				listChoices.Items.Add(item);
			}
			else
			{
				ListViewItem item = new ListViewItem(i.ToString(), -1);
				item.Tag = i;
				listChoices.Items.Add(item);
			}
		}

		public IEnumerable<object> Choices
		{
			set
			{
				if (CUE != null)
					AddItem(CUE);
				foreach (object i in value)
					AddItem(i);
				if (CUE != null)
				{
					textBox1.Hide();
					pictureBox1.Hide();
					listTracks.Show();
					listMetadata.Show();
					tableLayoutPanel1.SetRowSpan(listChoices, 3);
					tableLayoutPanel1.PerformLayout();
				}
				else
				{
					textBox1.Show();
					pictureBox1.Hide();
					listTracks.Hide();
					listMetadata.Hide();
					tableLayoutPanel1.SetRowSpan(textBox1, 4);
					//tableLayoutPanel1.RowStyles[2].Height = 0;
					//tableLayoutPanel1.RowStyles[3].Height = 0;
					tableLayoutPanel1.PerformLayout();
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

		private ReleaseInfo ChosenRelease
		{
			get
			{
				object o = ChosenObject;
				return o != null && o is ReleaseInfo ? o as ReleaseInfo : null;
			}
		}

		private ListViewItem ChosenItem
		{
			get
			{
				return listChoices.SelectedItems.Count > 0 ? listChoices.SelectedItems[0] : null;
			}
		}

		private void frmChoice_FormClosing(object sender, FormClosingEventArgs e)
		{
			ReleaseInfo ri = ChosenRelease;
			if (e.CloseReason != CloseReason.None || DialogResult != DialogResult.OK || ri == null || CUE == null)
				return;
			CUE.CopyMetadata(ri.metadata);
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
			if (item != null && item is TagLib.IPicture)
			{
				TagLib.IPicture picture = item as TagLib.IPicture;
				using (MemoryStream imageStream = new MemoryStream(picture.Data.Data, 0, picture.Data.Count))
					try { pictureBox1.Image = Image.FromStream(imageStream); }
					catch { }				
				textBox1.Hide();
				pictureBox1.Show();
				tableLayoutPanel1.SetRowSpan(pictureBox1, 4);
			}
			else if (item != null && item is CUEToolsSourceFile)
			{
				textBox1.Text = (item as CUEToolsSourceFile).contents.Replace("\r\n", "\n").Replace("\r", "\n").Replace("\n", "\r\n");
			}
			else if (item != null && item is ReleaseInfo)
			{
				ReleaseInfo r = (item as ReleaseInfo);
				listTracks.Items.Clear();
				foreach (TrackInfo track in r.metadata.Tracks)
				{
					listTracks.Items.Add(new ListViewItem(new string[] { 
						track.Title,
						(listTracks.Items.Count + 1).ToString(),
						r.metadata.TOC[listTracks.Items.Count + r.metadata.TOC.FirstAudio].StartMSF,
						r.metadata.TOC[listTracks.Items.Count + r.metadata.TOC.FirstAudio].LengthMSF
					}));
				}
				AutoResizeTracks();
				listMetadata.Items.Clear();
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Artist, "Artist" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Title, "Album" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Year, "Date" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Genre, "Genre" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.DiscNumber, "Disc Number" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.TotalDiscs, "Total Discs" }));
			}
			else
			{
				listMetadata.Items.Clear();
				listTracks.Items.Clear();
				textBox1.Text = "";
			}
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
			if (ChosenRelease == null)
			{
				e.CancelEdit = true;
				return;
			}
		}

		private void listTracks_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			ReleaseInfo ri = ChosenRelease;
			if (ri != null && e.Label != null)
				ri.metadata.Tracks[e.Item].Title = e.Label;
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

		private void listMetadata_DoubleClick(object sender, EventArgs e)
		{
			listMetadata.FocusedItem.BeginEdit();
		}

		private void listMetadata_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.F2)
				listMetadata.FocusedItem.BeginEdit();
		}

		private void listMetadata_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			ListViewItem item = ChosenItem;
			ReleaseInfo r = ChosenRelease;
			if (e.Label != null && item != null && r != null)
			{
				switch (e.Item)
				{
					case 0:
						foreach (TrackInfo track in r.metadata.Tracks)
							if (track.Artist == r.metadata.Artist)
								track.Artist = e.Label;
						r.metadata.Artist = e.Label;
						break;
					case 1: r.metadata.Title = e.Label; break;
					case 2: r.metadata.Year = e.Label; break;
					case 3: r.metadata.Genre = e.Label; break;
					case 4: r.metadata.DiscNumber = e.Label; break;
					case 5: r.metadata.TotalDiscs = e.Label; break;
				}
				item.Text = r.Text;
			}
		}

		private void pictureBox1_DoubleClick(object sender, EventArgs e)
		{
			pictureBox1.SizeMode = pictureBox1.SizeMode == PictureBoxSizeMode.Zoom ?
				PictureBoxSizeMode.CenterImage : PictureBoxSizeMode.Zoom;
		}
	}

	sealed class ReleaseInfo
	{
		public CUESheet metadata;

		public ReleaseInfo(CUESheet cue)
		{
			metadata = new CUESheet(cue.Config);
			metadata.TOC = cue.TOC;
			metadata.CopyMetadata(cue);
		}

		public ReleaseInfo(CUESheet cue, Freedb.CDEntry release)
		{
			metadata = new CUESheet(cue.Config);
			metadata.TOC = cue.TocFromCDEntry(release);
			metadata.FillFromFreedb(release);
		}

		public ReleaseInfo(CUESheet cue, MusicBrainz.Release release)
		{
			metadata = new CUESheet(cue.Config);
			metadata.TOC = cue.TOC;
			metadata.FillFromMusicBrainz(release);
		}

		private string FixEncoding(string src)
		{
			Encoding iso = Encoding.GetEncoding("iso-8859-1");
			return Encoding.Default.GetString(iso.GetBytes(src));
		}

		public string Text
		{
			get
			{
				return string.Format("{0}: {1} - {2}",
					metadata.Year == "" ? "YYYY" : metadata.Year,
					metadata.Artist == "" ? "Unknown Artist" : metadata.Artist,
					metadata.Title == "" ? "Unknown Title" : metadata.Title);
			}
		}

		public void FixEncoding()
		{
			metadata.Artist = FixEncoding(metadata.Artist);
			metadata.Title = FixEncoding(metadata.Title);
			foreach (TrackInfo track in metadata.Tracks)
			{
				track.Title = FixEncoding(track.Title);
				track.Artist = FixEncoding(track.Artist);
			}
		}

		public bool Equals(ReleaseInfo r)
		{
			bool equal = metadata.Title == r.metadata.Title && metadata.Artist == r.metadata.Artist;
			for (int t = 0; t < metadata.TrackCount; t++)
				if (r.metadata.Tracks[t].Title != metadata.Tracks[t].Title || r.metadata.Tracks[t].Artist != metadata.Tracks[t].Artist)
					equal = false;
			return equal;
		}
	}
}