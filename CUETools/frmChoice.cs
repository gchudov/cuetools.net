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

namespace JDP
{
	public partial class frmChoice : Form
	{
		public frmChoice()
		{
			InitializeComponent();
		}

		public CUESheet CUE;

		private bool freedb, ctdb;

		public void LookupAlbumInfo(bool freedb, bool ctdb, bool cache, bool cue)
		{
			this.freedb = freedb;
			this.ctdb = ctdb;
			var releases = CUE.LookupAlbumInfo(false, false, cache, cue);
			this.Choices = releases;
			if (freedb || ctdb)
				backgroundWorker1.RunWorkerAsync(null);
		}

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
			else if (i is CUEMetadataEntry)
			{
				CUEMetadataEntry entry = i as CUEMetadataEntry;
				ListViewItem item = new ListViewItem(entry.ToString(), entry.ImageKey);
				item.Tag = entry;
				listChoices.Items.Add(item);

				if (entry.ImageKey == "freedb")
				{
					// check if the entry contains non-iso characters,
					// and add a second one if it does
					CUEMetadata copy = new CUEMetadata(entry.metadata);
					if (copy.FreedbToEncoding())
					{
						entry = new CUEMetadataEntry(copy, entry.TOC, entry.ImageKey);
						item = new ListViewItem(entry.ToString(), entry.ImageKey);
						item.Tag = entry;
						listChoices.Items.Add(item);
					}
				}
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
				foreach (object i in value)
					AddItem(i);
				if (CUE != null)
				{
					textBox1.Hide();
					pictureBox1.Hide();
					tableLayoutPanelMeta.Show();
					tableLayoutPanel1.SetRowSpan(listChoices, 2);
					tableLayoutPanel1.PerformLayout();
				}
				else
				{
					textBox1.Show();
					pictureBox1.Hide();
					tableLayoutPanelMeta.Hide();
					tableLayoutPanel1.SetRowSpan(textBox1, 3);
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

		public CUEMetadataEntry ChosenRelease
		{
			get
			{
				object o = ChosenObject;
				return o != null && o is CUEMetadataEntry ? o as CUEMetadataEntry : null;
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
			if (backgroundWorker1.IsBusy)
			{
				e.Cancel = true;
				return;
			}
			CUEMetadataEntry ri = ChosenRelease;
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
				tableLayoutPanelMeta.Hide();
				tableLayoutPanel1.SetRowSpan(pictureBox1, 2);
			}
			else if (item != null && item is CUEToolsSourceFile)
			{
				textBox1.Text = (item as CUEToolsSourceFile).contents.Replace("\r\n", "\n").Replace("\r", "\n").Replace("\n", "\r\n");
			}
			else if (item != null && item is CUEMetadataEntry)
			{
				CUEMetadataEntry r = (item as CUEMetadataEntry);
				listTracks.Items.Clear();
				foreach (CUETrackMetadata track in r.metadata.Tracks)
				{
					listTracks.Items.Add(new ListViewItem(new string[] { 
						track.Title,
						(listTracks.Items.Count + 1).ToString(),
						r.TOC[listTracks.Items.Count + r.TOC.FirstAudio].StartMSF,
						r.TOC[listTracks.Items.Count + r.TOC.FirstAudio].LengthMSF
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
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.DiscName, "Disc Name" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Barcode, "Barcode" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.ReleaseDate, "Release Date" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Label, "Label" }));
				listMetadata.Items.Add(new ListViewItem(new string[] { r.metadata.Country, "Country" }));
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
			CUEMetadataEntry ri = ChosenRelease;
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
			CUEMetadataEntry r = ChosenRelease;
			if (e.Label != null && item != null && r != null)
			{
				switch (e.Item)
				{
					case 0:
						foreach (CUETrackMetadata track in r.metadata.Tracks)
							if (track.Artist == r.metadata.Artist)
								track.Artist = e.Label;
						r.metadata.Artist = e.Label;
						break;
					case 1: r.metadata.Title = e.Label; break;
					case 2: r.metadata.Year = e.Label; break;
					case 3: r.metadata.Genre = e.Label; break;
					case 4: r.metadata.DiscNumber = e.Label; break;
					case 5: r.metadata.TotalDiscs = e.Label; break;
					case 6: r.metadata.DiscName = e.Label; break;
					case 7: r.metadata.Barcode = e.Label; break;
					case 8: r.metadata.ReleaseDate = e.Label; break;
					case 9: r.metadata.Label = e.Label; break;
					case 10: r.metadata.Country = e.Label; break;
				}
				item.Text = r.ToString();
			}
		}

		private void pictureBox1_DoubleClick(object sender, EventArgs e)
		{
			pictureBox1.SizeMode = pictureBox1.SizeMode == PictureBoxSizeMode.Zoom ?
				PictureBoxSizeMode.CenterImage : PictureBoxSizeMode.Zoom;
		}

		private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
		{
			e.Result = CUE.LookupAlbumInfo(this.freedb, this.ctdb, false, false);
		}

		private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			foreach (object i in (e.Result as List<object>))
				AddItem(i);
		}
	}
}
