using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB.EACPlugin.Properties;

namespace CUETools.CTDB.EACPlugin
{
	public partial class FormSubmitParity : Form
	{
		private CUEToolsDB ctdb;
		private int confidence, quality;
		private string artist, title, agent, drivename;

		public FormSubmitParity(CUEToolsDB ctdb, int confidence, int quality, string artist, string title)
		{
			this.ctdb = ctdb;
			this.confidence = confidence;
			this.quality = quality;
			this.artist = artist;
			this.title = title;
			this.InitializeComponent();
		}

		public FormSubmitParity(CUEToolsDB ctdb, string agent, string drivename)
		{
			this.ctdb = ctdb;
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
			if (this.agent != null)
			{
				this.ctdb.ContactDB(this.agent, this.drivename, false, false);
			}
			else
			{
				this.ctdb.DoVerify();
				this.ctdb.Submit(this.confidence, this.quality, this.artist, this.title);
			}
			this.ctdb.UploadHelper.onProgress -= UploadProgress;
		}

		private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			this.DialogResult = DialogResult.OK;
		}

		private void backgroundWorker1_ProgressChanged(object sender, ProgressChangedEventArgs e)
		{
			this.progressBar1.Style = e.ProgressPercentage != 0 ? ProgressBarStyle.Continuous : ProgressBarStyle.Marquee;
			this.progressBar1.Value = Math.Max(0, Math.Min(100, e.ProgressPercentage));
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
