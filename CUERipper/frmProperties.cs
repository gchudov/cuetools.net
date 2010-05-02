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
			textArtist.Text = Metadata.Artist;
			textTitle.Text = Metadata.Title;
			textYear.Text = Metadata.Year;
			textGenre.Text = Metadata.Genre;
			textCatalog.Text = Metadata.Catalog;
			textBoxDiscNumber.Text = Metadata.DiscNumber;
			textBoxTotalDiscs.Text = Metadata.TotalDiscs;
		}

		public CUEMetadata Metadata { get; set; }

		private void button1_Click(object sender, EventArgs e)
		{
			Metadata.Tracks.ForEach(track => track.Artist = track.Artist == Metadata.Artist ? textArtist.Text : track.Artist);
			Metadata.Artist = textArtist.Text;
			Metadata.Title = textTitle.Text;
			Metadata.Year = textYear.Text;
			Metadata.Genre = textGenre.Text;
			Metadata.Catalog = textCatalog.Text;
			Metadata.DiscNumber = textBoxDiscNumber.Text;
			Metadata.TotalDiscs = textBoxTotalDiscs.Text;
		}
	}
}