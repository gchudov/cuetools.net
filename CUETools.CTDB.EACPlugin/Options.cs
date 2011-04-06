using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

 
namespace AudioDataPlugIn
{
    public partial class Options : Form
    {
        public Options()
        {
            InitializeComponent();
        }

		private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
		{
			System.Diagnostics.Process.Start(linkLabel1.Text);
		}
    }
}
