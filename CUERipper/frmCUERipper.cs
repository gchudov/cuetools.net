using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Processor;
using CUETools.Ripper.SCSI;
using MusicBrainz;

namespace CUERipper
{
	public partial class frmCUERipper : Form
	{
		private Thread _workThread = null;
		private CDDriveReader _reader = null;
		private StartStop _startStop;
		private CUEConfig _config;
		private OutputAudioFormat _format;
		private CUEStyle _style;
		private CUESheet _cueSheet;

		public frmCUERipper()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_startStop = new StartStop();
		}

		private void frmCUERipper_Load(object sender, EventArgs e)
		{
			foreach(char drive in CDDriveReader.DrivesAvailable())
			{
				CDDriveReader reader = new CDDriveReader();
				if (reader.Open(drive))
					comboDrives.Items.Add(reader);
			}
			if (comboDrives.Items.Count == 0)
				comboDrives.Items.Add("No CD drives found");
			comboDrives.SelectedIndex = 0;
			comboLossless.SelectedIndex = 0;
			comboCodec.SelectedIndex = 0;
			comboImage.SelectedIndex = 0;
		}

		private void SetupControls ()
		{
			bool running = _workThread != null;
			listTracks.Enabled =
			comboDrives.Enabled =
			comboRelease.Enabled =
			comboCodec.Enabled = 
			comboImage.Enabled = 
			comboLossless.Enabled = !running;
			buttonPause.Visible = buttonPause.Enabled = buttonAbort.Visible = buttonAbort.Enabled = running;
			buttonGo.Visible = buttonGo.Enabled = !running;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			toolStripProgressBar2.Value = 0;
		}

		private void CDReadProgress(object sender, ReadProgressArgs e)
		{
			CDDriveReader audioSource = (CDDriveReader)sender;
			lock (_startStop)
			{
				if (_startStop._stop)
				{
					_startStop._stop = false;
					_startStop._pause = false;
					throw new StopException();
				}
				if (_startStop._pause)
				{
					this.BeginInvoke((MethodInvoker)delegate()
					{
						toolStripStatusLabel1.Text = "Paused...";
					});
					Monitor.Wait(_startStop);
				}
			}
			int processed = e.Position - e.PassStart;
			TimeSpan elapsed = DateTime.Now - e.PassTime;
			double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;

			double percentDisk = (double)(e.PassStart + (processed + e.Pass * (e.PassEnd - e.PassStart)) / (audioSource.CorrectionQuality + 1)) / audioSource.TOC.AudioLength;
			double percentTrck = (double)(e.Position - e.PassStart) / (e.PassEnd - e.PassStart);
			string status = string.Format("Ripping @{0:00.00}x {1}", speed, e.Pass > 0 ? " (Retry " + e.Pass.ToString() + ")" : "");

			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = status;
				toolStripProgressBar1.Value = Math.Max(0, Math.Min(100, (int)(percentTrck * 100)));
				toolStripProgressBar2.Value = Math.Max(0, Math.Min(100, (int)(percentDisk * 100)));
			});
		}

		private void Rip(object o)
		{
			CDDriveReader audioSource = (CDDriveReader)o;
			audioSource.ReadProgress += new EventHandler<ReadProgressArgs>(CDReadProgress);
			int[,] buff = new int[audioSource.BestBlockSize, audioSource.ChannelCount];
			AccurateRipVerify arVerify = new AccurateRipVerify(audioSource.TOC);
			string ArId = AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC);

			arVerify.ContactAccurateRip(ArId);

			IAudioDest audioDest = null;

			try
			{
				audioSource.Position = 0;
				if (_style == CUEStyle.SingleFile || _style == CUEStyle.SingleFileWithCUE)
					audioDest = AudioReadWrite.GetAudioDest(_cueSheet.SingleFilename, (long)audioSource.Length, _config);

				for (int iTrack = 0; iTrack <= audioSource.TOC.AudioTracks; iTrack++)
				{
					uint samplesRemTrack = (iTrack > 0 ? audioSource.TOC[iTrack].Length : audioSource.TOC.Pregap) * 588;
					if (_style == CUEStyle.GapsAppended)
					{
						if (audioDest != null)
							audioDest.Close();
						audioDest = AudioReadWrite.GetAudioDest(iTrack > 0 ? _cueSheet.TrackFilenames[iTrack - 1] : _cueSheet.HTOAFilename, (long)samplesRemTrack, _config);
					}
					while (samplesRemTrack > 0)
					{
						uint toRead = Math.Min((uint)buff.GetLength(0), (uint)samplesRemTrack);
						uint samplesRead = audioSource.Read(buff, toRead);
						if (samplesRead != toRead)
							throw new Exception("samples read != samples requested");
						arVerify.Write(buff, samplesRead);
						audioDest.Write(buff, samplesRead);
						samplesRemTrack -= samplesRead;
					}
				}

				if (audioDest != null)
					audioDest.Close();
				audioDest = null;
			}
			catch (StopException)
			{
				if (audioDest != null) 
					try { audioDest.Close(); }
					catch { };
				audioDest = null;
			}
			catch (Exception ex)
			{
				if (audioDest != null)
					try { audioDest.Close(); }
					catch { };
				audioDest = null;
				this.Invoke((MethodInvoker)delegate()
				{
					string message = "Exception";
					for (Exception e = ex; e != null; e = e.InnerException)
						message += ": " + e.Message;
					DialogResult dlgRes = MessageBox.Show(this, message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				});
			}
			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
			});
		}

		private void buttonGo_Click(object sender, EventArgs e)
		{
			if (_reader == null)
				return;
			
			_config.lossyWAVHybrid = comboLossless.SelectedIndex == 1; // _cueSheet.Config?
			_config.singleFilenameFormat = "%D - %C.cue";
			_format = (string)comboCodec.SelectedItem == "wav" ? OutputAudioFormat.WAV :
				(string)comboCodec.SelectedItem == "flac" ? OutputAudioFormat.FLAC :
				(string)comboCodec.SelectedItem == "wv" ? OutputAudioFormat.WavPack :
				(string)comboCodec.SelectedItem == "ape" ? OutputAudioFormat.APE :
				OutputAudioFormat.NoAudio;
			_style = comboImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE :
				CUEStyle.GapsAppended;
			_cueSheet.GenerateFilenames(_format, comboLossless.SelectedIndex != 0, ".");

			_workThread = new Thread(Rip);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(_reader);			
		}

		private void buttonAbort_Click(object sender, EventArgs e)
		{
			lock (_startStop)
			{
				if (_startStop._pause)
				{
					_startStop._pause = false;
					Monitor.Pulse(_startStop);
				}
				_startStop._stop = true;
			}
		}

		private void buttonPause_Click(object sender, EventArgs e)
		{
			lock (_startStop)
			{
				if (_startStop._pause)
				{
					_startStop._pause = false;
					Monitor.Pulse(_startStop);
				}
				else
				{
					_startStop._pause = true;
				}
			}
		}

		private void comboRelease_Format(object sender, ListControlConvertEventArgs e)
		{
			if (e.ListItem is string)
				return;
			CUELine date = General.FindCUELine(((CUESheet)e.ListItem).Attributes, "REM", "DATE");
			e.Value = string.Format("{0}{1} - {2}", date != null ? date.Params[2] + ": " : "", ((CUESheet)e.ListItem).Artist, ((CUESheet)e.ListItem).Title);
		}

		private void comboRelease_SelectedIndexChanged(object sender, EventArgs e)
		{
			listTracks.Items.Clear();
			if (comboRelease.SelectedItem == null || comboRelease.SelectedItem is string)
				return;
			_cueSheet = (CUESheet)comboRelease.SelectedItem;
			for (int i = 1; i <= _reader.TOC.AudioTracks; i++)
				listTracks.Items.Add(new ListViewItem(new string[] { 
					_cueSheet.Tracks[i-1].Title, 
					_reader.TOC[i].Number.ToString(), 
					_reader.TOC[i].StartMSF, 
					_reader.TOC[i].LengthMSF }));
		}

		private void MusicBrainz_LookupProgress(object sender, XmlRequestEventArgs e)
		{
			//_progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
			//_progress.input = e.Uri.ToString();
			lock (_startStop)
			{
				if (_startStop._stop)
				{
					_startStop._stop = false;
					_startStop._pause = false;
					throw new StopException();
				}
				if (_startStop._pause)
				{
					this.BeginInvoke((MethodInvoker)delegate()
					{
						toolStripStatusLabel1.Text = "Paused...";
					});
					Monitor.Wait(_startStop);
				}
			}
			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = "Looking up album via MusicBrainz";
				toolStripProgressBar1.Value = 0;
				toolStripProgressBar2.Value = (100 + toolStripProgressBar2.Value) / 2;
			});
		}

		private void Lookup(object o)
		{
			CDDriveReader audioSource = (CDDriveReader)o;

			ReleaseQueryParameters p = new ReleaseQueryParameters();
			p.DiscId = audioSource.TOC.MusicBrainzId;
			Query<Release> results = Release.Query(p);
			MusicBrainzService.XmlRequest += new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			foreach (Release release in results)
			{
				release.GetEvents();
				release.GetTracks();
				
				this.BeginInvoke((MethodInvoker)delegate()
				{
					CUESheet cueSheet = new CUESheet(_config);
					cueSheet.OpenTOC(audioSource.TOC);
					cueSheet.Artist = release.GetArtist();
					cueSheet.Title = release.GetTitle();
					if (release.GetEvents().Count > 0)
						General.SetCUELine(cueSheet.Attributes, "REM", "DATE", release.GetEvents()[0].Date.Substring(0, 4), false);
					for (int i = 1; i <= audioSource.TOC.AudioTracks; i++)
					{
						Track track = release.GetTracks()[(int)audioSource.TOC[i].Number - 1];
						cueSheet.Tracks[i - 1].Title = track.GetTitle();
						cueSheet.Tracks[i - 1].Artist = track.GetArtist();
					}
					comboRelease.Items.Add(cueSheet);
				});
			}
			MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (comboRelease.Items.Count == 0)
				{
					CUESheet cueSheet = new CUESheet(_config);
					cueSheet.OpenTOC(audioSource.TOC); 
					comboRelease.Items.Add(cueSheet);
					// cueSheet.Tracks[i - 1].Title = "Track " + _reader.TOC[i].Number.ToString();
				}
			});
			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
				comboRelease.SelectedIndex = 0;
			});
		}

		private void comboDrives_SelectedIndexChanged(object sender, EventArgs e)
		{
			comboRelease.Items.Clear();
			listTracks.Items.Clear();
			if (comboDrives.SelectedItem is string)
				return;
			_reader = (CDDriveReader)comboDrives.SelectedItem;
			if (_reader.TOC.AudioTracks == 0)
			{
				comboRelease.Items.Add("No audio tracks");
				return;
			}
			comboRelease_SelectedIndexChanged(sender, e);
			_workThread = new Thread(Lookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(_reader);
		}

		private void listTracks_DoubleClick(object sender, EventArgs e)
		{
			listTracks.FocusedItem.BeginEdit();
		}

		private void listTracks_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.F2)
			{
				listTracks.FocusedItem.BeginEdit();
			}
		}

		private void listTracks_PreviewKeyDown(object sender, PreviewKeyDownEventArgs e)
		{
			if (e.KeyCode == Keys.Enter)
			{
				if (listTracks.FocusedItem.Index + 1 < listTracks.Items.Count)// && e.Label != null)
				{
					listTracks.FocusedItem.Selected = false;
					listTracks.FocusedItem = listTracks.Items[listTracks.FocusedItem.Index + 1];
					listTracks.FocusedItem.Selected = true;
					listTracks.FocusedItem.BeginEdit();
				}
			}
		}
	}

	public class StopException : Exception
	{
		public StopException()
			: base()
		{
		}
	}

	public class StartStop
	{
		public bool _stop, _pause;
		public StartStop()
		{
			_stop = false;
			_pause = false;
		}
	}
}
