using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.CTDB.EACPlugin.Properties;
using CUETools.CTDB;
using Microsoft.Win32;

namespace AudioDataPlugIn
{
    public partial class Options : Form
    {
		private static CTDBMetadataSearch? metadataSearch = null;
		private static string optionsKey = @"SOFTWARE\CUETools\EACPugin";
		public static CTDBMetadataSearch MetadataSearch
		{
			get
			{
				if (!metadataSearch.HasValue)
				{
					try
					{
						using (var key = Registry.CurrentUser.OpenSubKey(optionsKey, false))
						{
							var val = key.GetValue("MetadataSearch") as string;
							if (val == "Default") metadataSearch = CTDBMetadataSearch.Default;
							if (val == "Fast") metadataSearch = CTDBMetadataSearch.Fast;
							if (val == "Extensive") metadataSearch = CTDBMetadataSearch.Extensive;
						}
					}
					catch (Exception ex)
					{
					}
				}

				return metadataSearch ?? CTDBMetadataSearch.Default;
			}

			set
			{
				using (var key = Registry.CurrentUser.CreateSubKey(optionsKey))
				{
					key.SetValue("MetadataSearch", value.ToString());
				}

				metadataSearch = value;
			}
		}

        public Options()
        {
            this.InitializeComponent();
        }

		private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
		{
			System.Diagnostics.Process.Start(linkLabel1.Text);
		}

		private void Options_Load(object sender, EventArgs e)
		{
			this.Icon = Resources.ctdb;
			this.radioButtonMBExtensive.Checked = MetadataSearch == CTDBMetadataSearch.Extensive;
			this.radioButtonMBDefault.Checked = MetadataSearch == CTDBMetadataSearch.Default;
			this.radioButtonMBFast.Checked = MetadataSearch == CTDBMetadataSearch.Fast;
		}

		private void button2_Click(object sender, EventArgs e)
		{
			Options.MetadataSearch = this.radioButtonMBExtensive.Checked ? CTDBMetadataSearch.Extensive
				: this.radioButtonMBDefault.Checked ? CTDBMetadataSearch.Default
				: this.radioButtonMBFast.Checked ? CTDBMetadataSearch.Fast
				: CTDBMetadataSearch.None;
		}
    }
}
