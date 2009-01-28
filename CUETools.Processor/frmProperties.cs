using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.Processor;

namespace CUETools.Processor
{
	public partial class frmProperties : Form
	{
		public frmProperties()
		{
			InitializeComponent();
		}

		private void frmProperties_Load(object sender, EventArgs e)
		{
			textArtist.Text = _cueSheet.Artist;
			textTitle.Text = _cueSheet.Title;
			textYear.Text = _cueSheet.Year;
			textGenre.Text = _cueSheet.Genre;
			textCatalog.Text = _cueSheet.Catalog;
		}

		public CUESheet CUE
		{
			get
			{
				return _cueSheet;
			}
			set
			{
				_cueSheet = value;
			}
		}

		CUESheet _cueSheet;

		private void button1_Click(object sender, EventArgs e)
		{
			_cueSheet.Artist = textArtist.Text;
			_cueSheet.Title = textTitle.Text;
			_cueSheet.Year = textYear.Text;
			_cueSheet.Genre = textGenre.Text;
			_cueSheet.Catalog = textCatalog.Text;
		}
	}
}