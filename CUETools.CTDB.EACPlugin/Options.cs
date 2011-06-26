using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB.EACPlugin.Properties;
using CUETools.CTDB;

namespace AudioDataPlugIn
{
    public partial class Options : Form
    {
		public static CTDBPriority priorityMusicbrainz = CTDBPriority.High;
		public static CTDBPriority priorityFreedb = CTDBPriority.Medium;
		public static CTDBPriority priorityFreedbFuzzy = CTDBPriority.Low;		

        public Options()
        {
            InitializeComponent();
        }

		private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
		{
			System.Diagnostics.Process.Start(linkLabel1.Text);
		}

		private void Options_Load(object sender, EventArgs e)
		{
			this.Icon = Resources.ctdb;
			this.radioButtonMBHigh.Checked = priorityMusicbrainz == CTDBPriority.High;
			this.radioButtonMBMedium.Checked = priorityMusicbrainz == CTDBPriority.Medium;
			this.radioButtonMBLow.Checked = priorityMusicbrainz == CTDBPriority.Low;
			this.radioButtonMBNone.Checked = priorityMusicbrainz == CTDBPriority.None;
			this.radioButtonFDHigh.Checked = priorityFreedb == CTDBPriority.High;
			this.radioButtonFDMedium.Checked = priorityFreedb == CTDBPriority.Medium;
			this.radioButtonFDLow.Checked = priorityFreedb == CTDBPriority.Low;
			this.radioButtonFDNone.Checked = priorityFreedb == CTDBPriority.None;
			this.radioButtonFZHigh.Checked = priorityFreedbFuzzy == CTDBPriority.High;
			this.radioButtonFZMedium.Checked = priorityFreedbFuzzy == CTDBPriority.Medium;
			this.radioButtonFZLow.Checked = priorityFreedbFuzzy == CTDBPriority.Low;
			this.radioButtonFZNone.Checked = priorityFreedbFuzzy == CTDBPriority.None;
		}

		private void button2_Click(object sender, EventArgs e)
		{
			priorityMusicbrainz = this.radioButtonMBHigh.Checked ? CTDBPriority.High
				: this.radioButtonMBMedium.Checked ? CTDBPriority.Medium
				: this.radioButtonMBLow.Checked ? CTDBPriority.Low
				: CTDBPriority.None;
			priorityFreedb = this.radioButtonFDHigh.Checked ? CTDBPriority.High
				: this.radioButtonFDMedium.Checked ? CTDBPriority.Medium
				: this.radioButtonFDLow.Checked ? CTDBPriority.Low
				: CTDBPriority.None;
			priorityFreedbFuzzy = this.radioButtonFZHigh.Checked ? CTDBPriority.High
				: this.radioButtonFZMedium.Checked ? CTDBPriority.Medium
				: this.radioButtonFZLow.Checked ? CTDBPriority.Low
				: CTDBPriority.None;
		}
    }
}
