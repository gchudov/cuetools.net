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
	public partial class FormMetadata : Form
	{
		private CUEToolsDB ctdb;
		private string agent;

		public FormMetadata(CUEToolsDB ctdb, string agent)
		{
			this.ctdb = ctdb;
			this.agent = agent;
			this.InitializeComponent();
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

		private void FormMetadata_Load(object sender, EventArgs e)
		{
			this.Icon = Resources.ctdb;
			this.backgroundWorker1.RunWorkerAsync();
		}

		private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
		{
			this.ctdb.ContactDB(null, this.agent, null, true, false);
		}

		private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			this.progressBar1.Visible = false;
			this.button1.Visible = true;
			this.button2.Visible = true;
			foreach (var metadata in ctdb.Metadata)
			{
				uint td = 0, dn = 0;
				var disccount = metadata.disccount ?? "1";
				var discnumber = metadata.discnumber ?? "1";
				var discnumber01 = (uint.TryParse(disccount, out td) && uint.TryParse(discnumber, out dn) && td > 9 && dn > 0) ?
					string.Format("{0:00}", dn) : discnumber;
				var discnumberandtotal = disccount != "1" ? discnumber01 + "/" + disccount : (discnumber != "1" ? discnumber01 : "");
				var label = metadata.country ?? "";
				if (metadata.label != null)
					foreach (var l in metadata.label)
						label = (label == "" ? "" : label + ": ") + (l.name ?? "") + (l.name != null && l.catno != null ? " " : "") + (l.catno ?? "");
				if (metadata.releasedate != null)
					label = (label == "" ? "" : label + ": ") + metadata.releasedate;
				var text = string.Format("{0}{1} - {2}{3}{4}", metadata.year != null ? metadata.year + ": " : "",
					metadata.artist == null ? "Unknown Artist" : metadata.artist,
					metadata.album == "" ? "Unknown Title" : metadata.album,
					discnumberandtotal != "" ? " (disc " + discnumberandtotal + (metadata.discname != null ? ": " + metadata.discname : "") + ")" : "",
					label == "" ? "" : " (" + label + ")");
				listView1.Items.Add(new ListViewItem(text) { Tag = metadata });
			}
			this.listView1.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
			if (listView1.Items.Count == 0)
			{
				this.DialogResult = DialogResult.Cancel;
				return;
			}
			listView1.Items[0].Selected = true;
			if (listView1.Items.Count == 1)
				this.DialogResult = DialogResult.OK;
		}

		private void listView1_MouseDoubleClick(object sender, MouseEventArgs e)
		{
			var ht = listView1.HitTest(e.Location);
			if (ht.Item != null)
				this.DialogResult = DialogResult.OK;
		}
	}
}
