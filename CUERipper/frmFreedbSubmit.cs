using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace CUERipper
{
	public partial class frmFreedbSubmit : Form
	{
		public frmFreedbSubmit()
		{
			InitializeComponent();
			Data = new frmFreedbSubmitData();
		}

		public frmFreedbSubmitData Data { get; set; }

		private void frmFreedbSubmit_Load(object sender, EventArgs e)
		{
			frmFreedbSubmitDataBindingSource.DataSource = Data;
			Text += " (" + Data.SiteAddress + ")";
		}
	}

	public class frmFreedbSubmitData
	{
		public frmFreedbSubmitData() { Categories = new BindingList<string>(); }
		public BindingList<string> Categories { get; set; }
		public string Category { get; set; }
		public string User { get; set; }
		public string Domain { get; set; }
		public string SiteAddress { get; set; }
	}
}
