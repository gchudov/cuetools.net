using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace CUETools.CTDB.EACPlugin
{
	public partial class FormMetadata : Form
	{
		private CUEToolsDB ctdb;
		private CTDBResponseMeta meta;
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
				return this.meta;
			}
		}

		private void FormMetadata_Load(object sender, EventArgs e)
		{
			this.backgroundWorker1.RunWorkerAsync();
		}

		private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
		{
			this.ctdb.ContactDB(this.agent, null, true, false);
		}

		private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			this.progressBar1.Visible = false;
			this.button1.Visible = true;
			CTDBResponseMeta bestMeta = null;
			foreach (var metadata in ctdb.Metadata)
			{
				bestMeta = metadata;
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
				var text = string.Format("{0}{1} - {2}{3}{4}", metadata.year != null ? metadata.year + ": " : "",
					metadata.artist == null ? "Unknown Artist" : metadata.artist,
					metadata.album == "" ? "Unknown Title" : metadata.album,
					discnumberandtotal != "" ? " (disc " + discnumberandtotal + (metadata.discname != null ? ": " + metadata.discname : "") + ")" : "",
					label == "" ? "" : " (" + label + ")");
				listView1.Items.Add(new ListViewItem(text) { Tag = metadata });
			}
			this.listView1.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
			this.meta = bestMeta;
			if (listView1.Items.Count < 2)
				this.Close();
		}

		private void listView1_MouseDoubleClick(object sender, MouseEventArgs e)
		{
			var ht = listView1.HitTest(e.Location);
			if (ht.Item != null)
			{
				meta = ht.Item.Tag as CTDBResponseMeta;
				this.Close();
			}
		}
	}
}
