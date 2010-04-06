using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using System.Diagnostics;
using NAudio.CoreAudioApi;
using CUEControls;
using CUETools.Codecs;
using CUETools.Codecs.CoreAudio;
using CUETools.DSP.Mixer;
using CUETools.DSP.Resampler;
using CUETools.Processor;

namespace CUEPlayer
{
	public partial class frmCUEPlayer : Form
	{
		private ShellIconMgr _icon_mgr;
		private CUEConfig _config;
		private IWavePlayer _player;
		DataSet1TableAdapters.PlaylistTableAdapter adapterPlayList = new DataSet1TableAdapters.PlaylistTableAdapter();
		private DataSet1 dataSet = new DataSet1();
		private Thread mixThread;
		private MixingSource _mixer;
		private SOXResampler _resampler;

		internal Playlist wndPlaylist
		{
			get
			{
				return playlist;
			}
		}

		internal DataSet1 DataSet
		{
			get
			{
				return dataSet;
			}
		}

		internal CUEConfig Config
		{
			get
			{
				return _config;
			}
		}

		internal ShellIconMgr IconMgr
		{
			get
			{
				return _icon_mgr;
			}
		}

		public MixingSource Mixer
		{
			get
			{
				return _mixer;
			}
		}

		public frmCUEPlayer()
		{
			InitializeComponent();
			_icon_mgr = new ShellIconMgr();
			_config = new CUEConfig();
			_config.separateDecodingThread = false;
		}

		internal Deck deckA = new Deck(0, "A");
		internal Deck deckB = new Deck(1, "B");
		internal Output outputA = new Output();
		internal Browser browser = new Browser();
		internal Playlist playlist = new Playlist();

		private void frmCUEPlayer_Load(object sender, EventArgs e)
		{
			int delay = 100;
			AudioPCMConfig mixerPCM = AudioPCMConfig.RedBook;

			//System.Data.SqlServerCe.SqlCeDataAdapter ad = new System.Data.SqlServerCe.SqlCeDataAdapter();
			//ad.SelectCommand = new System.Data.SqlServerCe.SqlCeCommand("SELECT * FROM Playlist WHERE track=1", adapterPlayList.Connection);
			//ad.Fill(dataSet.Playlist);
			adapterPlayList.Fill(dataSet.Playlist);

			_mixer = new MixingSource(mixerPCM, delay, 2);

			outputA.Init(this);
			browser.Init(this);
			playlist.Init(this);
			deckB.Init(this, null);
			deckA.Init(this, deckB);
			//LayoutMdi(MdiLayout.TileHorizontal);
			
			browser.Location = new Point(0, 0);
			browser.Height = ClientRectangle.Height - 5;
			playlist.Location = new Point(browser.Location.X + browser.Width, 0);
			playlist.Height = ClientRectangle.Height - 5;
			deckA.Location = new Point(playlist.Location.X + playlist.Width, 0);
			deckB.Location = new Point(playlist.Location.X + playlist.Width, deckA.Height);
			outputA.Location = new Point(deckA.Location.X + deckA.Width, 0);

			try
			{
				_player = new WasapiOut(outputA.Device, NAudio.CoreAudioApi.AudioClientShareMode.Shared, true, delay, new AudioPCMConfig(32, 2, 44100));
			}
			catch
			{
				_player = null;
			}
			if (_player == null)
			{
				try
				{
					_player = new WasapiOut(outputA.Device, NAudio.CoreAudioApi.AudioClientShareMode.Shared, true, delay, new AudioPCMConfig(32, 2, 48000));
					SOXResamplerConfig cfg;
					cfg.quality = SOXResamplerQuality.Very;
					cfg.phase = 50;
					cfg.allow_aliasing = false;
					cfg.bandwidth = 0;
					_resampler = new SOXResampler(mixerPCM, _player.PCM, cfg);
				}
				catch(Exception ex)
				{
					_player = null;
					Trace.WriteLine(ex.Message);
				}
			}
			if (_player != null)
			{
				_player.Play();

				mixThread = new Thread(MixThread);
				mixThread.Priority = ThreadPriority.AboveNormal;
				mixThread.IsBackground = true;
				mixThread.Name = "Mixer";
				mixThread.Start();
			}
		}

		Thread playThread;

		int playingRow;

		private void PlayThread()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0x2000);
			IAudioSource playingSource = null;
			CUESheet playingCue = null;
			long playingOffs = 0;
			long playingFin = 0;

			try
			{
				do
				{
					// End of playlist entry or source file
					if (playingSource != null && (playingOffs == playingFin || playingSource.Remaining == 0))
					{
						this.Invoke((MethodInvoker)delegate()
						{
							playlist.List.Items[playingRow].BackColor = Color.White;
						});
						playingRow++;
						playingOffs = 0;
						if (playingRow >= dataSet.Playlist.Rows.Count)
							break;
						string path = dataSet.Playlist[playingRow].path;
						int track = dataSet.Playlist[playingRow].track;

						if (playingCue == null ||
							playingSource == null ||
							playingCue.InputPath != path ||
							playingSource.Position != (long)playingCue.TOC[track].Start * 588)
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
						}
						else
						{
							playingFin = (long)playingCue.TOC[track].Length * 588;
							this.Invoke((MethodInvoker)delegate()
							{
								playlist.List.Items[playingRow].BackColor = Color.AliceBlue;
								//deckA.UpdateDeck(dataSet, playingRow, playingCue, playingFin);
							});
						}
					}
					// Open it
					if (playingSource == null)
					{
						string path = dataSet.Playlist[playingRow].path;
						int track = dataSet.Playlist[playingRow].track;

						try
						{
							playingCue = new CUESheet(_config);
							playingCue.Open(path);
							playingSource = new CUESheetAudio(playingCue);
							playingSource.Position = (long)playingCue.TOC[track].Start * 588 + playingOffs;
							playingSource = new AudioPipe(playingSource, 0x2000);
							playingFin = (long)playingCue.TOC[track].Length * 588;
							this.Invoke((MethodInvoker)delegate()
							{
								playlist.List.Items[playingRow].BackColor = Color.AliceBlue;
								//deckA.UpdateDeck(dataSet, playingRow, playingCue, playingFin);
							});
						}
						catch (Exception ex)
						{
							// skip it
							playingOffs = playingFin = 0;
							continue;
						}
					}
					playingSource.Read(buff, (int)(playingFin - playingOffs));

					this.Invoke((MethodInvoker)delegate()
					{
						deckA.PlayingOffset = (int)playingOffs;
					});

					_player.Write(buff);
					playingOffs += buff.Length;
				} while (_player.PlaybackState != PlaybackState.Stopped);
			}
			catch (Exception ex)
			{
				// Can't invoke while joining

				//if (playingRow < dataSet.Playlist.Rows.Count)
				//    this.Invoke((MethodInvoker)delegate()
				//    {
				//        listViewTracks.Items[playingRow].BackColor = Color.White;
				//    });
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
		}

		internal void buttonPlay_Click(object sender, EventArgs e)
		{
			try
			{
				_player.Stop();
				if (playThread != null)
				{
					playThread.Join();
					playThread = null;
				}

				if (playlist.List.SelectedIndices.Count < 0)
					return;

				playingRow = playlist.List.SelectedIndices[0];
				playlist.List.Items[playingRow].BackColor = Color.AliceBlue;

				_player.Play();

				playThread = new Thread(new ThreadStart(PlayThread));
				playThread.Priority = ThreadPriority.AboveNormal;
				playThread.IsBackground = true;
				playThread.Start();
			}
			catch (Exception ex)
			{
				Trace.WriteLine(ex.Message);
			}
		}

		internal void buttonStop_Click(object sender, EventArgs e)
		{
			try
			{
				_player.Stop();
				if (playThread != null)
				{
					playThread.Join();
					playThread = null;
				}
			}
			catch (Exception ex)
			{
				Trace.WriteLine(ex.Message);
			}
		}

		internal void buttonPause_Click(object sender, EventArgs e)
		{
			_player.Pause();
		}

		private void frmCUEPlayer_FormClosing(object sender, FormClosingEventArgs e)
		{
			try
			{
				int rowsAffected = adapterPlayList.Update(dataSet.Playlist);
			}
			catch (Exception ex)
			{
				System.Diagnostics.Trace.WriteLine(ex.Message);
			}
		}

		private void MixThread()
		{
			AudioBuffer result = new AudioBuffer(
				new AudioPCMConfig(_player.PCM.BitsPerSample, _player.PCM.ChannelCount, _mixer.PCM.SampleRate), _mixer.BufferSize);
			AudioBuffer resampled = _resampler == null ? null : new AudioBuffer(_player.PCM, _mixer.BufferSize * 2 * _mixer.PCM.SampleRate / _player.PCM.SampleRate);
			while (true)
			{
				//Trace.WriteLine(string.Format("Read"));
				_mixer.Read(result, -1);
				if (_resampler == null)
					_player.Write(result);
				else
				{
					//Trace.WriteLine(string.Format("Flow: {0}", result.Length));
					_resampler.Flow(result, resampled);
					//Trace.WriteLine(string.Format("Play: {0}", resampled.Length));
					if (resampled.Length != 0)
						_player.Write(resampled);
				} 
			}
		}
	}
}
