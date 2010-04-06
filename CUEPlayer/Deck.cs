using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CUETools.Codecs;
using CUETools.Processor;
using CUETools.DSP.Mixer;

namespace CUEPlayer
{
	public partial class Deck : Form
	{
		IAudioSource playingSource = null;
		CUESheet playingCue = null;
		int playingRow = -1;
		long playingStart = 0;
		long playingFinish = 0;
		Thread playThread;
		int iSource;
		MixingSource mixer;
		MixingWriter writer;
		Deck nextDeck;
		bool needUpdate = false;

		public Deck(int iSource, string suffix)
		{
			InitializeComponent();
			this.iSource = iSource;
			if (suffix != null)
				Text += " " + suffix;
			//mediaSliderA.FlyOutInfo += new MediaSlider.MediaSlider.FlyOutInfoDelegate(mediaSliderA_FlyOutInfo);
		}

		public void Init(frmCUEPlayer parent, Deck nextDeck)
		{
			MdiParent = parent;
			mixer = (parent as frmCUEPlayer).Mixer;
			writer = new MixingWriter(mixer, iSource);
			this.nextDeck = nextDeck;
			Show();
		}

		//void mediaSliderA_FlyOutInfo(ref string data)
		//{
		//    TimeSpan ts = TimeSpan.FromSeconds(mediaSliderA.Value / 44100.0);
		//    data = ts.ToString();
		//}

		internal int PlayingOffset
		{
			set
			{
				if (!mediaSlider.Capture) mediaSlider.Value = value;
			}
		}

		internal void UpdateDeck()
		{
			if (needUpdate)
			{
				needUpdate = false;

				DataSet1 dataSet = (MdiParent as frmCUEPlayer).DataSet;

				mediaSlider.Maximum = (int)(playingFinish - playingStart);
				mediaSlider.Value = 0;
				textBoxArtist.Text = playingRow < 0 ? "" : dataSet.Playlist[playingRow].artist;
				textBoxAlbum.Text = playingRow < 0 ? "" : dataSet.Playlist[playingRow].album;
				textBoxTitle.Text = playingRow < 0 ? "" : dataSet.Playlist[playingRow].title;
				textBoxDuration.Text = "";
				pictureBox.Image = playingCue != null ? playingCue.Cover : pictureBox.InitialImage;

				if (nextDeck != null && nextDeck.playingSource == null && playingRow >= 0 && playingRow < dataSet.Playlist.Rows.Count - 1)
				{
					nextDeck.LoadDeck(playingRow + 1);
				}
			}
			mediaSlider.Enabled = playingSource != null;
			if (playingSource != null)
				mediaSlider.Value = (int)(playingSource.Position - playingStart);
		}

		private void mediaSliderA_ValueChanged(object sender, EventArgs e)
		{
			if (mediaSlider.Maximum == 1) return;
			TimeSpan len1 = TimeSpan.FromSeconds(mediaSlider.Maximum / 44100.0);
			TimeSpan len2 = TimeSpan.FromSeconds(mediaSlider.Value / 44100.0);
			string lenStr1 = string.Format("{0:d}.{1:d2}:{2:d2}:{3:d2}", len1.Days, len1.Hours, len1.Minutes, len1.Seconds).TrimStart('0', ':', '.');
			string lenStr2 = string.Format("{0:d}.{1:d2}:{2:d2}:{3:d2}", len2.Days, len2.Hours, len2.Minutes, len2.Seconds).TrimStart('0', ':', '.');
			lenStr1 = "0:00".Substring(0, Math.Max(0, 4 - lenStr1.Length)) + lenStr1;
			lenStr2 = "0:00".Substring(0, Math.Max(0, 4 - lenStr2.Length)) + lenStr2;
			textBoxDuration.Text = lenStr2 + " / " + lenStr1;
		}

		private int seekTo = -1;
		private bool stopNow = false;

		private void PlayThread()
		{
			AudioBuffer buff = new AudioBuffer(playingSource.PCM, 0x2000);

			try
			{
				do
				{
					if (playingSource == null)
						writer.Pause();
					else
					{
						if (seekTo >= 0 && playingStart + seekTo < playingFinish)
						{
							playingSource.Position = playingStart + seekTo;
							seekTo = -1;
						}
						if (playingSource.Position == playingFinish || stopNow || seekTo == (int)(playingFinish - playingStart))
						{
							seekTo = -1;
							playingSource.Close();
							playingSource = null;
							if (playingCue != null)
							{
								playingCue.Close();
								playingCue = null;
							}
							playingFinish = 0;
							playingStart = 0;
							playingRow = -1;
							if (stopNow || nextDeck == null || nextDeck.playingSource == null)
							{
								writer.Flush();
								stopNow = false;
								mixer.BufferPlaying(iSource, false);
								needUpdate = true;
								playThread = null;
								return;
							}
							playingSource = nextDeck.playingSource;
							playingCue = nextDeck.playingCue;
							playingStart = nextDeck.playingStart;
							playingFinish = nextDeck.playingFinish;
							playingRow = nextDeck.playingRow;
							needUpdate = true;
							nextDeck.playingSource = null;
							nextDeck.playingCue = null;
							nextDeck.playingStart = 0;
							nextDeck.playingFinish = 0;
							nextDeck.playingRow = -1;
							nextDeck.needUpdate = true;
						}
						playingSource.Read(buff, Math.Min(buff.Size, (int)(playingFinish - playingSource.Position)));
						writer.Write(buff);
					}
				} while (true);
			}
			catch (Exception ex)
			{
			}
			if (playingCue != null)
			{
				playingCue.Close();
				playingCue = null;
			}
			if (playingSource != null)
			{
				playingSource.Close();
				playingSource = null;
			}
			playThread = null;
		}

		internal void LoadDeck(int row)
		{
			CUEConfig _config = (MdiParent as frmCUEPlayer).Config;
			DataSet1 dataSet = (MdiParent as frmCUEPlayer).DataSet;
			Playlist playlist = (MdiParent as frmCUEPlayer).wndPlaylist;
			string path = dataSet.Playlist[row].path;
			int track = dataSet.Playlist[row].track;

			try
			{
				playingCue = new CUESheet(_config);
				playingCue.Open(path);
				playingSource = new CUESheetAudio(playingCue);
				playingSource.Position = (long)playingCue.TOC[track].Start * 588;
				playingSource = new AudioPipe(playingSource, 0x2000);
				playingStart = playingSource.Position;
				playingFinish = playingStart + (long)playingCue.TOC[track].Length * 588;
				playingRow = row;
				//playlist.List.Items[playingRow].BackColor = Color.AliceBlue;
				needUpdate = true;
				UpdateDeck();
			}
			catch (Exception ex)
			{
				playingStart = playingFinish = 0;
				playingCue = null;
				playingSource = null;
				return;
			}
		}

		private void buttonPlay_Click(object sender, EventArgs e)
		{
			if (playingSource == null)
			{
				Playlist playlist = (MdiParent as frmCUEPlayer).wndPlaylist;
				LoadDeck(playlist.List.SelectedIndices[0]);
			}
			mixer.BufferPlaying(iSource, true);
			if (playThread == null)
			{
				playThread = new Thread(PlayThread);
				playThread.Priority = ThreadPriority.AboveNormal;
				playThread.IsBackground = true;
				playThread.Name = Text;
				playThread.Start();
			}
		}

		private void buttonStop_Click(object sender, EventArgs e)
		{
			if (playThread != null)
			{
				stopNow = true;
				playThread.Join();
			}
			else
			{
				if (playingSource != null)
				{
					playingSource.Close();
					playingSource = null;
				}
				if (playingCue != null)
				{
					playingCue.Close();
					playingCue = null;
				}
				playingFinish = 0;
				playingStart = 0;
				playingRow = -1;
				needUpdate = true;
				UpdateDeck();
			}
		}

		private void buttonPause_Click(object sender, EventArgs e)
		{
			mixer.BufferPlaying(iSource, false);
		}

		private void timer1_Tick(object sender, EventArgs e)
		{
			UpdateDeck();
		}

		private void mediaSlider_Scrolled(object sender, EventArgs e)
		{
			if (playThread != null)
			{
				seekTo = mediaSlider.Value;
			}
			else
			{
				if (playingSource != null)
					playingSource.Position = playingStart + mediaSlider.Value;
			}
		}

		private void mediaSliderVolume_Scrolled(object sender, EventArgs e)
		{
			writer.Volume = mediaSliderVolume.Value / 100.0f;
		}

		private void Deck_DragOver(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.Serializable))
			{
				e.Effect = DragDropEffects.Copy;
			}
		}

		private void Deck_DragDrop(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.Serializable))
			{
				ListView.SelectedIndexCollection indexes = 
					e.Data.GetData(DataFormats.Serializable) as ListView.SelectedIndexCollection;
				if (playThread == null && indexes != null)
				{
					LoadDeck(indexes[0]);
				}
			}
		}

		private void buttonNext_Click(object sender, EventArgs e)
		{
			seekTo = (int)(playingFinish - playingStart);
		}

		private void buttonRewind_Click(object sender, EventArgs e)
		{
			if (playThread != null)
			{
				seekTo = 0;
			}
			else if (playingSource != null)
			{
				playingSource.Position = playingStart;
			}
		}
	}
}
