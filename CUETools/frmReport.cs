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
			txtReport.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
		}

		public string Message {
			get { return txtReport.Text; }
			set { txtReport.Text = value; }
		}
	}
}