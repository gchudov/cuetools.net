using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB.EACPlugin.Properties;
using System.IO;

namespace CUETools.CTDB.EACPlugin
{
	public partial class FormSubmitParity : Form
	{
		private CUEToolsDB ctdb;
		private int confidence, quality;
		private string artist, title, agent, drivename;
        private CTDBResponse resp;

		public FormSubmitParity(CUEToolsDB ctdb, string agent, string drivename, int confidence, int quality, string artist, string title)
		{
			this.ctdb = ctdb;
			this.confidence = confidence;
			this.quality = quality;
			this.artist = artist;
			this.title = title;
            this.agent = agent;
            this.drivename = drivename;
            this.InitializeComponent();
		}

		private void FormMetadata_Load(object sender, EventArgs e)
		{
			this.Icon = Resources.ctdb;
			this.backgroundWorker1.RunWorkerAsync();
		}

		private void UploadProgress(object sender, Krystalware.UploadHelper.UploadProgressEventArgs e)
		{
			this.backgroundWorker1.ReportProgress((int)e.percent, e.uri);
		}

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            this.ctdb.UploadHelper.onProgress += UploadProgress;
            if (resp == null)
            {
#if DEBUG
                string server = "db.cuetools.net";
#else
                string server = null;
#endif
                this.ctdb.ContactDB(server, this.agent, this.drivename, true, true, CTDBMetadataSearch.None);
                this.ctdb.DoVerify();
                resp = this.ctdb.Submit(this.confidence, this.quality, this.artist, this.title, null);
            } else
            {
                var url = resp.updateurl;
                resp = null;
                var temp = Path.GetTempPath() + Path.GetFileName(url.Substring(url.LastIndexOf('/') + 1));
                bool ok = false;
                using (var stream = new FileStream(temp, FileMode.Create))
                    ok = this.ctdb.FetchFile(url, stream);
                if (ok)
                    System.Diagnostics.Process.Start(temp);
            }
            this.ctdb.UploadHelper.onProgress -= UploadProgress;
        }

		private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
            if (resp != null && resp.updateurl != null)
            {
                DialogResult mb = MessageBox.Show(this, (resp.updatemsg ?? "") + " Do you wish to download and install it?", "An updated version of CTDB plugin is available", MessageBoxButtons.OKCancel);
                if (mb == DialogResult.OK)
                {
                    this.backgroundWorker1.RunWorkerAsync();
                    return;
                }
            }
			this.DialogResult = DialogResult.OK;
		}

		private void backgroundWorker1_ProgressChanged(object sender, ProgressChangedEventArgs e)
		{
			this.progressBar1.Style = e.ProgressPercentage != 0 ? ProgressBarStyle.Continuous : ProgressBarStyle.Marquee;
			this.progressBar1.Value = Math.Max(0, Math.Min(100, e.ProgressPercentage));
            this.labelStatus.Text = e.UserState is string ? e.UserState as string : string.Empty;
		}

		private void FormSubmitParity_FormClosing(object sender, FormClosingEventArgs e)
		{
			if (this.backgroundWorker1.IsBusy)
			{
				e.Cancel = true;
				this.progressBar1.Style = ProgressBarStyle.Marquee;
				this.ctdb.CancelRequest();
			}
		}
	}
}
