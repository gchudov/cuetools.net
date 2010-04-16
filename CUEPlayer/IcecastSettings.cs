using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.Codecs.Icecast;

namespace CUEPlayer
{
	public partial class IcecastSettings : Form
	{
		public IcecastSettings(IcecastSettingsData data)
		{
			InitializeComponent();
			icecastSettingsDataBindingSource.DataSource = data;
		}

		private void IcecastSettings_Load(object sender, EventArgs e)
		{
			
		}
	}
}
