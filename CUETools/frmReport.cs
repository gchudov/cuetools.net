using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace JDP
{
	public partial class frmReport : Form
	{
		public frmReport()
		{
			InitializeComponent();
		}
		public string Message {
			get { return txtReport.Text; }
			set { txtReport.Text = value; }
		}
	}
}