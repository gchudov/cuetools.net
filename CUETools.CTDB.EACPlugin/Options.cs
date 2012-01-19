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
using System.Runtime.Serialization.Formatters.Binary;

namespace AudioDataPlugIn
{
    public enum CTDBCoversSearch
    {
        Large,
        Small,
        None
    }

    public partial class Options : Form
    {
		private static CTDBMetadataSearch? metadataSearch = null;
        private static CTDBCoversSearch? coversSearch = null;
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
                            metadataSearch = (CTDBMetadataSearch)Enum.Parse(typeof(CTDBMetadataSearch), val);
						}
					}
					catch (Exception)
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

        public static CTDBCoversSearch CoversSearch
        {
            get
            {
                if (!coversSearch.HasValue)
                {
                    try
                    {
                        using (var key = Registry.CurrentUser.OpenSubKey(optionsKey, false))
                        {
                            var val = key.GetValue("CoversSearch") as string;
                            coversSearch = (CTDBCoversSearch)Enum.Parse(typeof(CTDBCoversSearch), val);
                        }
                    }
                    catch (Exception)
                    {
                    }
                }

                return coversSearch ?? CTDBCoversSearch.Small;
            }

            set
            {
                using (var key = Registry.CurrentUser.CreateSubKey(optionsKey))
                {
                    key.SetValue("CoversSearch", value.ToString());
                }

                coversSearch = value;
            }
        }

        public static Size MetadataWindowSize
        {
            get
            {
                try
                {
                    using (var key = Registry.CurrentUser.OpenSubKey(optionsKey, false))
                    {
                        var val = key.GetValue("MetadataWindowSize") as string;
                        return (Size)TypeDescriptor.GetConverter(typeof(Size)).ConvertFromInvariantString(val);
                    }
                }
                catch (Exception)
                {
                }

                return new Size();
            }

            set
            {
                using (var key = Registry.CurrentUser.CreateSubKey(optionsKey))
                {
                    var val = TypeDescriptor.GetConverter(value.GetType()).ConvertToInvariantString(value);
                    key.SetValue("MetadataWindowSize", val);
                }
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
            this.radioButtonCoversLarge.Checked = CoversSearch == CTDBCoversSearch.Large;
            this.radioButtonCoversSmall.Checked = CoversSearch == CTDBCoversSearch.Small;
            this.radioButtonCoversNone.Checked = CoversSearch == CTDBCoversSearch.None;
		}

		private void button2_Click(object sender, EventArgs e)
		{
			Options.MetadataSearch = this.radioButtonMBExtensive.Checked ? CTDBMetadataSearch.Extensive
				: this.radioButtonMBDefault.Checked ? CTDBMetadataSearch.Default
				: this.radioButtonMBFast.Checked ? CTDBMetadataSearch.Fast
				: CTDBMetadataSearch.None;
            Options.CoversSearch = this.radioButtonCoversLarge.Checked ? CTDBCoversSearch.Large
                : this.radioButtonCoversSmall.Checked ? CTDBCoversSearch.Small
                : CTDBCoversSearch.None;
        }
    }
}
