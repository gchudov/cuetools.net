using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB.EACPlugin.Properties;
using System.Net;
using System.IO;
using AudioDataPlugIn;

namespace CUETools.CTDB.EACPlugin
{
    public partial class FormMetadata : Form
    {
        private CUEToolsDB ctdb;
        private string agent;
        private bool cdinfo, cover;
        private ImagePreview m_currently_selected;

        public FormMetadata(CUEToolsDB ctdb, string agent, bool cdinfo, bool cover)
        {
            this.ctdb = ctdb;
            this.agent = agent;
            this.cdinfo = cdinfo;
            this.cover = cover;
            this.InitializeComponent();
            this.Size = Options.MetadataWindowSize;
        }

        public CTDBResponseMeta Meta
        {
            get
            {
                return this.DialogResult != DialogResult.Cancel &&
                    listView1.SelectedItems.Count > 0 ?
                    listView1.SelectedItems[0].Tag as CTDBResponseMeta : null;
            }
        }

        public InternetImage Image
        {
            get
            {
                return this.DialogResult != DialogResult.Cancel &&
                    m_currently_selected != null ?
                    m_currently_selected.Image : null;
            }
        }

        private void FormMetadata_Load(object sender, EventArgs e)
        {
            this.Icon = Resources.ctdb;
            this.backgroundWorker1.RunWorkerAsync();
        }

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
#if DEBUG
            string server = "db.cuetools.net";
#else
            string server = null;
#endif
            this.ctdb.ContactDB(server, this.agent, null, false, false,
                AudioDataPlugIn.Options.MetadataSearch);
            if (this.cdinfo)
            {
                foreach (var metadata in ctdb.Metadata)
                {
                    backgroundWorker1.ReportProgress(0, metadata);
                }
            }
            var knownUrls = new List<string>();
            foreach (var metadata in ctdb.Metadata)
            {
                if (metadata.coverart == null || !this.cover)
                    continue;
                if (!this.cdinfo)
                {
                    backgroundWorker1.ReportProgress(0, metadata);
                }
                if (backgroundWorker1.CancellationPending)
                {
                    throw new Exception();
                }
                foreach (var coverart in metadata.coverart)
                {
                    var uri = Options.CoversSize == CTDBCoversSize.Large ?
                        coverart.uri : coverart.uri150 ?? coverart.uri;
                    if (knownUrls.Contains(uri) || 
                        (Options.CoversSearch == CTDBCoversSearch.Primary && !coverart.primary))
                        continue;
                    var ms = new MemoryStream();
                    if (!this.ctdb.FetchFile(uri, ms))
                        continue;
                    var img = new InternetImage();
                    img.URL = uri;
                    img.Data = ms.ToArray();
                    img.Image = new Bitmap(ms);
                    knownUrls.Add(uri);
                    backgroundWorker1.ReportProgress(0, img);
                }
            }
        }

        public void Form1_DoubleClick(object sender, MouseEventArgs e)
        {
            if (this.m_currently_selected != null && !this.cdinfo && !backgroundWorker1.IsBusy)
            {
                this.DialogResult = DialogResult.OK;
            }
        }

        public void Form1_MouseClick(object sender, MouseEventArgs e)
        {
            ImagePreview ssp = null;

            if (sender is ImagePreview)
            {
                ssp = sender as ImagePreview;
            }
            //else if (sender is LargeImage)
            //{
            //    if (m_current_control is ImagePreview)
            //    {
            //        ssp = m_current_control as ImagePreview;
            //    }
            //}
            else if (sender is Control)
            {
                Control cp = (sender as Control).Parent;
                if (cp != null)
                {
                    if (cp is ImagePreview)
                    {
                        ssp = cp as ImagePreview;
                    }
                }
            }
            if (ssp != null)
            {
                if (m_currently_selected != null)
                {
                    m_currently_selected.Selected = false;
                }
                ssp.Selected = true;
                m_currently_selected = ssp;
                //AcceptImage.Enabled = true;
            }
        }

        public void Form1_MouseMove(object sender, MouseEventArgs e)
        {
            //if (sender is Control)
            //{
            //    bool ison = false;
            //    Point screencoord = (sender as Control).PointToScreen(e.Location);
            //    Point clientcoord = ImagesPanel.PointToClient(screencoord);
            //    Control cp = ImagesPanel.GetChildAtPoint(clientcoord);
            //    ImagePreview ssp = null;
            //    if (cp is ImagePreview)
            //    {
            //        ssp = cp as ImagePreview;

            //        if (!ssp.IsMouseOverPanel(screencoord))
            //        {
            //            cp = null;
            //            ssp = null;
            //        }
            //    }
            //    label1.Text = "(" + (ison) + "/" + (cp != null) + "/" + cp + "-" + m_current_control + ")";
            //    if (cp != m_current_control)
            //    {
            //        m_current_control = cp;
            //        if (cp == null)
            //        {
            //            Capture = false;
            //            li.UnshowImage();
            //        }
            //        else
            //        {
            //            Point np = new Point(cp.Left + cp.Width / 2, cp.Top + cp.Height / 2);
            //            if (ssp != null)
            //            {
            //                if (ssp.Image != null)
            //                {
            //                    //Capture = true;
            //                    li.ShowImage(ssp.Image, ImagesPanel.PointToScreen(np));
            //                    //System.Threading.Thread.Sleep(5000);
            //                    li.Activate();
            //                    li.Focus();
            //                    //label2.Text = "" + li.ContainsFocus;
            //                }
            //            }
            //        }
            //    }

            //}
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            this.progressBar1.Visible = false;
            this.button1.Visible = true;
            this.button2.Visible = true;
            if (listView1.Items.Count == 0 && flowLayoutPanel1.Controls.Count == 0)
            {
                this.DialogResult = DialogResult.Cancel;
                return;
            }
            if (listView1.Items.Count > 0)
                listView1.Items[0].Selected = true;
            if (flowLayoutPanel1.Controls.Count > 0 && m_currently_selected == null)
            {
                m_currently_selected = flowLayoutPanel1.Controls[0] as ImagePreview;
                m_currently_selected.Selected = true;
            }
            if ((!this.cdinfo || listView1.Items.Count == 1) && (!this.cover || flowLayoutPanel1.Controls.Count == 1))
            {
                this.DialogResult = DialogResult.OK;
            }
        }

        private void listView1_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            if (this.cdinfo && !backgroundWorker1.IsBusy)
            {
                var ht = listView1.HitTest(e.Location);
                if (ht.Item != null)
                    this.DialogResult = DialogResult.OK;
            }
        }

        private void AddMeta(CTDBResponseMeta metadata)
        {
            uint td = 0, dn = 0;
            var disccount = metadata.disccount ?? "1";
            var discnumber = metadata.discnumber ?? "1";
            var discnumber01 = (uint.TryParse(disccount, out td) && uint.TryParse(discnumber, out dn) && td > 9 && dn > 0) ?
                string.Format("{0:00}", dn) : discnumber;
            var discnumberandtotal = disccount != "1" ? discnumber01 + "/" + disccount : (discnumber != "1" ? discnumber01 : "");
            var label = "";
            if (metadata.release != null)
                foreach (var r in metadata.release)
                    label = (label == "" ? "" : label + ": ") + (r.country ?? "") + (r.country != null && r.date != null ? " " : "") + (r.date ?? "");
            if (metadata.label != null)
                foreach (var l in metadata.label)
                    label = (label == "" ? "" : label + ": ") + (l.name ?? "") + (l.name != null && l.catno != null ? " " : "") + (l.catno ?? "");
            var text = string.Format("{0}{1} - {2}{3}{4}", metadata.year != null ? metadata.year + ": " : "",
                metadata.artist == null ? "Unknown Artist" : metadata.artist,
                metadata.album == "" ? "Unknown Title" : metadata.album,
                discnumberandtotal != "" ? " (disc " + discnumberandtotal + (metadata.discname != null ? ": " + metadata.discname : "") + ")" : "",
                label == "" ? "" : " (" + label + ")");
            var tip = new StringBuilder();
            var i = 0;
            if (metadata.track != null)
            {
                foreach (var tr in metadata.track)
                    tip.AppendFormat("{0}. {2}{1}\n", ++i, tr.name, ((tr.artist ?? metadata.artist) == metadata.artist) ? "" : tr.artist + " / ");
            }
            listView1.Items.Add(new ListViewItem(text) { Tag = metadata, ImageKey = metadata.source, ToolTipText = tip.ToString() });
            this.listView1.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
        }

        private static string FreedbToEncoding(Encoding iso, Encoding def, ref bool changed, ref bool error, string s)
        {
            if (s == null) 
                return null;
            try
            {
                string res = def.GetString(iso.GetBytes(s));
                changed |= res != s;
                return res;
            }
            catch // EncoderFallbackException, DecoderFallbackException
            {
                error = true;
            }
            return s;
        }

        private CTDBResponseMeta FreedbToEncoding(CTDBResponseMeta src)
        {
            if (src.source != "freedb")
                return null;
            Encoding iso = Encoding.GetEncoding("iso-8859-1", new EncoderExceptionFallback(), new DecoderExceptionFallback());
            Encoding def = Encoding.GetEncoding(Encoding.Default.CodePage, new EncoderExceptionFallback(), new DecoderExceptionFallback());
            bool different = false;
            bool error = false;
            var metadata = new CTDBResponseMeta(src);
            metadata.artist = FreedbToEncoding(iso, def, ref different, ref error, metadata.artist);
            metadata.album = FreedbToEncoding(iso, def, ref different, ref error, metadata.album);
            metadata.extra = FreedbToEncoding(iso, def, ref different, ref error, metadata.extra);
            if (metadata.track != null)
            {
                for (int i = 0; i < metadata.track.Length; i++)
                {
                    metadata.track[i].name = FreedbToEncoding(iso, def, ref different, ref error, metadata.track[i].name);
                    metadata.track[i].artist = FreedbToEncoding(iso, def, ref different, ref error, metadata.track[i].artist);
                    metadata.track[i].extra = FreedbToEncoding(iso, def, ref different, ref error, metadata.track[i].extra);
                }
            }
            return different && !error ? metadata : null;
        }

        private void backgroundWorker1_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (e.UserState is InternetImage)
            {
                var img = e.UserState as InternetImage;
                var preview = new ImagePreview(flowLayoutPanel1, Form1_MouseMove, Form1_MouseClick, Form1_DoubleClick, img);
                flowLayoutPanel1.Height = preview.Height + preview.Margin.Vertical;
                //flowLayoutPanel1.DoubleBuffered = true;
            }

            if (e.UserState is CTDBResponseMeta)
            {
                var metadata = e.UserState as CTDBResponseMeta;
                var freedbenc = FreedbToEncoding(metadata);
                AddMeta(metadata);
                if (freedbenc != null) AddMeta(freedbenc);
            }
        }

        private void FormMetadata_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (backgroundWorker1.IsBusy)
            {
                backgroundWorker1.CancelAsync();
                e.Cancel = true;
                return;
            }

            Options.MetadataWindowSize = this.Size;
        }
    }
}
