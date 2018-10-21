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
using CUETools.CTDB;

namespace JDP
{
    public partial class frmChoice : Form
    {
        public frmChoice()
        {
            InitializeComponent();
        }

        public CUESheet CUE;
        public CUEConfig config;

        private bool ctdb;
        private CTDBMetadataSearch metadataSearch;
        private enum CropAlign
        {
            None, TopLeft, BottomRight
        };
        private RotateFlipType imageRotation = RotateFlipType.RotateNoneFlipNone;
        private CropAlign cropAlign = CropAlign.None;
        private int trimEdge = 0;

        public void LookupAlbumInfo(bool cache, bool cue, bool ctdb, CTDBMetadataSearch metadataSearch)
        {
            this.ctdb = ctdb;
            this.metadataSearch = metadataSearch;
            var releases = CUE.LookupAlbumInfo(cache, cue, false, CTDBMetadataSearch.None);
            this.Choices = releases;
            if (ctdb || metadataSearch != CTDBMetadataSearch.None)
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
                    buttonMusicBrainz.Show();
                    buttonNavigateCTDB.Show();
                }
                else
                {
                    textBox1.Show();
                    pictureBox1.Hide();
                    tableLayoutPanelMeta.Hide();
                    tableLayoutPanel1.SetRowSpan(textBox1, 3);
                    tableLayoutPanel1.PerformLayout();
                    buttonMusicBrainz.Hide();
                    buttonNavigateCTDB.Hide();
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

            TagLib.IPicture picture = ChosenObject as TagLib.IPicture;
            if (e.CloseReason == CloseReason.None && DialogResult == DialogResult.OK
                && pictureBox1.Image != null && picture != null
                && (cropAlign != CropAlign.None || imageRotation != RotateFlipType.RotateNoneFlipNone || trimEdge != 0))
            {
                using (MemoryStream encoded = new MemoryStream())
                {
                    pictureBox1.Image.Save(encoded, System.Drawing.Imaging.ImageFormat.Jpeg);
                    picture.Data = new TagLib.ByteVector(encoded.ToArray());
                    picture.MimeType = "image/jpeg";
                }
            }

            CUEMetadataEntry ri = ChosenRelease;
            if (e.CloseReason != CloseReason.None || DialogResult != DialogResult.OK || ri == null || CUE == null)
                return;
            for (int i = 0; i < ri.metadata.Tracks.Count; i++)
            {
                if (ri.metadata.Tracks[i].ISRC == "")
                    ri.metadata.Tracks[i].ISRC = CUE.Metadata.Tracks[i].ISRC;
            }
            CUE.CopyMetadata(ri.metadata);

            // TODO: copy album art

        }

        private void AutoResizeList(ListView list, int mainCol)
        {
            list.SuspendLayout();
            int widthAvailable = list.ClientSize.Width - 2 * SystemInformation.BorderSize.Width - SystemInformation.VerticalScrollBarWidth;
            for (int i = 0; i < list.Columns.Count; i++)
                if (i != mainCol)
                {
                    list.Columns[i].AutoResize(ColumnHeaderAutoResizeStyle.ColumnContent);
                    widthAvailable -= list.Columns[i].Width + SystemInformation.BorderSize.Width;
                }
            if (list.Columns[mainCol].Width != widthAvailable)
                list.Columns[mainCol].Width = widthAvailable;
            list.ResumeLayout(false);
        }

        private void listChoices_SelectedIndexChanged(object sender, EventArgs e)
        {
            object item = ChosenObject;
            if (item != null && item is TagLib.IPicture)
            {
                imageRotation = RotateFlipType.RotateNoneFlipNone;
                cropAlign = CropAlign.None;
                trimEdge = 0;
                ResetPictureBox();
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
                dataGridViewTracks.SuspendLayout();
                dataGridViewTracks.Rows.Clear();
                foreach (CUETrackMetadata track in r.metadata.Tracks)
                {
                    int no = dataGridViewTracks.Rows.Count;
                    dataGridViewTracks.Rows.Add(
                        (no + 1).ToString(),
                        track.Title,
                        r.TOC[no + r.TOC.FirstAudio].StartMSF,
                        r.TOC[no + r.TOC.FirstAudio].LengthMSF
                    );
                }
                dataGridViewTracks.ResumeLayout();
                dataGridViewMetadata.Rows.Clear();
                dataGridViewMetadata.Rows.Add("Artist", r.metadata.Artist);
                dataGridViewMetadata.Rows.Add("Album", r.metadata.Title);
                dataGridViewMetadata.Rows.Add("Date", r.metadata.Year);
                dataGridViewMetadata.Rows.Add("Genre", r.metadata.Genre);
                dataGridViewMetadata.Rows.Add("Disc Number", r.metadata.DiscNumber);
                dataGridViewMetadata.Rows.Add("Total Discs", r.metadata.TotalDiscs);
                dataGridViewMetadata.Rows.Add("Disc Name", r.metadata.DiscName);
                dataGridViewMetadata.Rows.Add("Label", r.metadata.Label);
                dataGridViewMetadata.Rows.Add("Label#", r.metadata.LabelNo);
                dataGridViewMetadata.Rows.Add("Country", r.metadata.Country);
                dataGridViewMetadata.Rows.Add("Release Date", r.metadata.ReleaseDate);
                dataGridViewMetadata.Rows.Add("Barcode", r.metadata.Barcode);
                dataGridViewMetadata.Rows.Add("Comment", r.metadata.Comment);
                if (pictureBox1.ImageLocation == null)
                {
                    if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
                    pictureBox1.Image = null;
                }
                pictureBox1.ImageLocation = null;
                if (r.metadata.AlbumArt.Count > 0)
                {
                    pictureBox1.Show();
                    var image = r.metadata.AlbumArt.Find(x => x.primary) ?? r.metadata.AlbumArt[0];
                    pictureBox1.ImageLocation = image.uri150;
                } else
                {
                    pictureBox1.Hide();
                }
            }
            else
            {
                dataGridViewMetadata.Rows.Clear();
                dataGridViewTracks.Rows.Clear();
                textBox1.Text = "";
            }
        }

        private void pictureBox1_DoubleClick(object sender, EventArgs e)
        {
            pictureBox1.SizeMode = pictureBox1.SizeMode == PictureBoxSizeMode.Zoom ?
                PictureBoxSizeMode.CenterImage : PictureBoxSizeMode.Zoom;
        }

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            e.Result = CUE.LookupAlbumInfo(false, false, this.ctdb, this.metadataSearch);
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            foreach (object i in (e.Result as List<object>))
                AddItem(i);
        }

        private void frmChoice_Resize(object sender, EventArgs e)
        {
            AutoResizeList(listChoices, 0);
        }

        private void dataGridViewMetadata_EditingControlShowing(object sender, DataGridViewEditingControlShowingEventArgs e)
        {
            var te = e.Control as DataGridViewTextBoxEditingControl;
            if (te != null)
            {
                //				te.AutoCompleteMode = AutoCompleteMode.None;
                te.AutoCompleteMode = AutoCompleteMode.Suggest;
                te.AutoCompleteSource = AutoCompleteSource.CustomSource;
                te.AutoCompleteCustomSource.Clear();
                foreach (ListViewItem item in listChoices.Items)
                {
                    var r = item.Tag as CUEMetadataEntry;
                    switch (dataGridViewMetadata.CurrentCell.RowIndex)
                    {
                        case 0:
                            foreach (CUETrackMetadata track in r.metadata.Tracks)
                                te.AutoCompleteCustomSource.Add(track.Artist);
                            te.AutoCompleteCustomSource.Add(r.metadata.Artist);
                            break;
                        case 1: te.AutoCompleteCustomSource.Add(r.metadata.Title); break;
                        case 2: te.AutoCompleteCustomSource.Add(r.metadata.Year); break;
                        case 3: te.AutoCompleteCustomSource.Add(r.metadata.Genre); break;
                        case 4: te.AutoCompleteCustomSource.Add(r.metadata.DiscNumber); break;
                        case 5: te.AutoCompleteCustomSource.Add(r.metadata.TotalDiscs); break;
                        case 6: te.AutoCompleteCustomSource.Add(r.metadata.DiscName); break;
                        case 7: te.AutoCompleteCustomSource.Add(r.metadata.Label); break;
                        case 8: te.AutoCompleteCustomSource.Add(r.metadata.LabelNo); break;
                        case 9: te.AutoCompleteCustomSource.Add(r.metadata.Country); break;
                        case 10: te.AutoCompleteCustomSource.Add(r.metadata.ReleaseDate); break;
                        case 11: te.AutoCompleteCustomSource.Add(r.metadata.Barcode); break;
                        case 12: te.AutoCompleteCustomSource.Add(r.metadata.Comment); break;
                    }
                }
            }
        }

        private void dataGridViewMetadata_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            ListViewItem item = ChosenItem;
            CUEMetadataEntry r = ChosenRelease;
            var label = dataGridViewMetadata.Rows[e.RowIndex].Cells[1].Value as string ?? "";
            if (item != null && r != null)
            {
                switch (e.RowIndex)
                {
                    case 0:
                        foreach (CUETrackMetadata track in r.metadata.Tracks)
                            if (track.Artist == r.metadata.Artist)
                                track.Artist = label;
                        r.metadata.Artist = label;
                        break;
                    case 1: r.metadata.Title = label; break;
                    case 2: r.metadata.Year = label; break;
                    case 3: r.metadata.Genre = label; break;
                    case 4: r.metadata.DiscNumber = label; break;
                    case 5: r.metadata.TotalDiscs = label; break;
                    case 6: r.metadata.DiscName = label; break;
                    case 7: r.metadata.Label = label; break;
                    case 8: r.metadata.LabelNo = label; break;
                    case 9: r.metadata.Country = label; break;
                    case 10: r.metadata.ReleaseDate = label; break;
                    case 11: r.metadata.Barcode = label; break;
                    case 12: r.metadata.Comment = label; break;
                }
                item.Text = r.ToString();
            }
        }

        private void dataGridViewMetadata_KeyDown(object sender, KeyEventArgs e)
        {
            CUEMetadataEntry r = ChosenRelease;
            if (r != null)
            {
                if (e.KeyCode == Keys.Delete && e.Modifiers.HasFlag(Keys.Shift))
                {
                    var ee = new DataGridViewCellEventArgs(1, dataGridViewMetadata.CurrentCellAddress.Y);
                    Clipboard.SetText(dataGridViewMetadata.Rows[ee.RowIndex].Cells[1].Value as string);
                    dataGridViewMetadata.Rows[ee.RowIndex].Cells[1].Value = null;
                    dataGridViewMetadata_CellEndEdit(sender, ee);
                }
                else if (e.KeyCode == Keys.Delete)
                {
                    var ee = new DataGridViewCellEventArgs(1, dataGridViewMetadata.CurrentCellAddress.Y);
                    dataGridViewMetadata.Rows[ee.RowIndex].Cells[1].Value = null;
                    dataGridViewMetadata_CellEndEdit(sender, ee);
                }
                else if (e.KeyCode == Keys.Insert && e.Modifiers.HasFlag(Keys.Shift))
                {
                    var ee = new DataGridViewCellEventArgs(1, dataGridViewMetadata.CurrentCellAddress.Y);
                    dataGridViewMetadata.Rows[ee.RowIndex].Cells[1].Value = Clipboard.GetText();
                    dataGridViewMetadata_CellEndEdit(sender, ee);
                }
            }
        }

        private void dataGridViewTracks_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            CUEMetadataEntry ri = ChosenRelease;
            var label = dataGridViewTracks.Rows[e.RowIndex].Cells[e.ColumnIndex].Value as string;
            if (ri != null && label != null)
                ri.metadata.Tracks[e.RowIndex].Title = label;

        }

        private void frmChoice_KeyPress(object sender, KeyPressEventArgs e)
        {
            TagLib.IPicture picture = ChosenObject as TagLib.IPicture;
            if (e.KeyChar == 'r' && picture != null)
            {
                switch (imageRotation)
                {
                    case RotateFlipType.RotateNoneFlipNone:
                        imageRotation = RotateFlipType.Rotate90FlipNone;
                        break;
                    case RotateFlipType.Rotate90FlipNone:
                        imageRotation = RotateFlipType.Rotate180FlipNone;
                        break;
                    case RotateFlipType.Rotate180FlipNone:
                        imageRotation = RotateFlipType.Rotate270FlipNone;
                        break;
                    case RotateFlipType.Rotate270FlipNone:
                        imageRotation = RotateFlipType.RotateNoneFlipNone;
                        break;
                }
                ResetPictureBox();
                e.Handled = true;
                return;
            }

            if (e.KeyChar == 'c' && picture != null)
            {
                switch (cropAlign)
                {
                    case CropAlign.None:
                        cropAlign = CropAlign.TopLeft;
                        break;
                    case CropAlign.TopLeft:
                        cropAlign = CropAlign.BottomRight;
                        break;
                    case CropAlign.BottomRight:
                        cropAlign = CropAlign.None;
                        break;
                }
                ResetPictureBox();
                e.Handled = true;
                return;
            }

            if (e.KeyChar == 't' && picture != null)
            {
                if (trimEdge == 0) trimEdge = 1;
                else trimEdge <<= 1;
                if (trimEdge > 1024) trimEdge = 0;
                ResetPictureBox();
                e.Handled = true;
                return;
            }
        }

        private void ResetPictureBox()
        {
            TagLib.IPicture picture = ChosenObject as TagLib.IPicture;
            if (pictureBox1.Image != null) pictureBox1.Image.Dispose();
            pictureBox1.Image = null;
            pictureBox1.ImageLocation = null;
            using (MemoryStream imageStream = new MemoryStream(picture.Data.Data, 0, picture.Data.Count))
                try
                {
                    var image = Image.FromStream(imageStream);
                    if (imageRotation != RotateFlipType.RotateNoneFlipNone)
                        image.RotateFlip(imageRotation);
                    if (cropAlign == CropAlign.None && trimEdge == 0)
                    {
                        pictureBox1.Image = image;
                        return;
                    }

                    var width = image.Width;
                    var height = image.Height;
                    var sz = Math.Min(width, height);
                    Rectangle dstRect =
                        cropAlign == CropAlign.None ?
                        Rectangle.FromLTRB(0, 0, width - 2 * trimEdge, height - 2 * trimEdge) :
                        Rectangle.FromLTRB(0, 0, sz - 2 * trimEdge, sz - 2 * trimEdge);
                    Rectangle srcRect =
                        cropAlign == CropAlign.None ?
                        Rectangle.FromLTRB(trimEdge, trimEdge, width - trimEdge, height - trimEdge) :
                        cropAlign == CropAlign.TopLeft ?
                        Rectangle.FromLTRB(trimEdge, trimEdge, sz - trimEdge, sz - trimEdge) :
                        //cropAlign == CropAlign.BottomRight ?
                        Rectangle.FromLTRB(width - sz + trimEdge, height - sz + trimEdge, width - trimEdge, height - trimEdge);
                    var mode = System.Drawing.Drawing2D.InterpolationMode.Default;
                    if (dstRect.Width > config.maxAlbumArtSize || dstRect.Height > config.maxAlbumArtSize)
                    {
                        dstRect =
                            cropAlign != CropAlign.None ?
                            Rectangle.FromLTRB(0, 0, config.maxAlbumArtSize, config.maxAlbumArtSize) :
                            width > height ? Rectangle.FromLTRB(0, 0, config.maxAlbumArtSize, (height - 2 * trimEdge) * config.maxAlbumArtSize / (width - 2 * trimEdge))
                            : Rectangle.FromLTRB(0, 0, (width - 2 * trimEdge) * config.maxAlbumArtSize / (height - 2 * trimEdge), config.maxAlbumArtSize);
                        mode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    }

                    Bitmap b = new Bitmap(dstRect.Width, dstRect.Height);
                    pictureBox1.Image = b;
                    using (Graphics g = Graphics.FromImage((Image)b))
                    {
                        g.InterpolationMode = mode;
                        g.DrawImage(image, dstRect, srcRect, GraphicsUnit.Pixel);
                    }
                    image.Dispose();
                }
                catch { }
        }

        private void buttonMusicBrainz_Click(object sender, EventArgs e)
        {
            if (CUE == null) return;
            System.Diagnostics.Process.Start($"http://musicbrainz.org/bare/cdlookup.html?toc={CUE.TOC.MusicBrainzTOC}");
        }

        private void buttonNavigateCTDB_Click(object sender, EventArgs e)
        {
            if (CUE == null) return;
            System.Diagnostics.Process.Start($"http://{config.advanced.CTDBServer}/?tocid={CUE.TOC.TOCID}");
        }

        /*
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
         * */
    }
}
