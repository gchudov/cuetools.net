using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.Processor;
using MusicBrainz;
using Freedb;

namespace JDP
{
	public partial class frmChoice : Form
	{
		public frmChoice()
		{
			InitializeComponent();
		}

		public CUESheet CUE;

		private void comboRelease_DrawItem(object sender, DrawItemEventArgs e)
		{
			e.DrawBackground();
			StringFormat format = new StringFormat();
			format.FormatFlags = StringFormatFlags.NoClip;
			format.Alignment = StringAlignment.Near;
			if (e.Index >= 0 && e.Index < comboRelease.Items.Count)
			{
				string text = null;
				Bitmap ImageToDraw = null;
				if (comboRelease.Items[e.Index] is string)
				{
					text = (string) comboRelease.Items[e.Index];
					//comboRelease.GetItemText(comboRelease.Items[e.Index]);
				}
				else if (comboRelease.Items[e.Index] is MusicBrainz.Release)
				{
					ImageToDraw = Properties.Resources.musicbrainz;
					MusicBrainz.Release release = (MusicBrainz.Release) comboRelease.Items[e.Index];
					text = String.Format("{0}{1} - {2}", 
						release.GetEvents().Count > 0 ? release.GetEvents()[0].Date.Substring(0, 4) + ": " : "",
						release.GetArtist(),
						release.GetTitle());
				}
				else if (comboRelease.Items[e.Index] is CDEntry)
				{
					ImageToDraw = Properties.Resources.freedb;
					CDEntry cdEntry = (CDEntry)comboRelease.Items[e.Index];
					text = String.Format("{0}: {1} - {2}",
						cdEntry.Year,
						cdEntry.Artist,
						cdEntry.Title);
				}
				if (ImageToDraw != null)
					e.Graphics.DrawImage(ImageToDraw, new Rectangle(e.Bounds.X, e.Bounds.Y, e.Bounds.Height, e.Bounds.Height));
					//e.Graphics.DrawImage(ImageToDraw, new Rectangle(e.Bounds.X + e.Bounds.Width - ImageToDraw.Width, e.Bounds.Y, ImageToDraw.Width, e.Bounds.Height));
				if (text != null)
					e.Graphics.DrawString(text, e.Font, new SolidBrush(e.ForeColor), new RectangleF((float)e.Bounds.X + e.Bounds.Height, (float)e.Bounds.Y, (float)(e.Bounds.Width - e.Bounds.Height), (float)e.Bounds.Height), format);
			}
			//e.DrawFocusRectangle();
		}

		private void frmChoice_Load(object sender, EventArgs e)
		{
			button1.Select();
		}

		private void frmChoice_FormClosing(object sender, FormClosingEventArgs e)
		{
			if (e.CloseReason != CloseReason.None || DialogResult != DialogResult.OK)
				return;
			if (comboRelease.SelectedItem != null && comboRelease.SelectedItem is MusicBrainz.Release)
				CUE.FillFromMusicBrainz((MusicBrainz.Release)comboRelease.SelectedItem);
			else
				if (comboRelease.SelectedItem != null && comboRelease.SelectedItem is CDEntry)
					CUE.FillFromFreedb((CDEntry)comboRelease.SelectedItem);
				else
					return;
			frmProperties frm = new frmProperties();
			frm.CUE = CUE;
			if (frm.ShowDialog(this) != DialogResult.OK)
				e.Cancel = true;
		}
	}
}