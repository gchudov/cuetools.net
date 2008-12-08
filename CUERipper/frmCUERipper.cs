using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CUETools.Ripper.SCSI;
using CUETools.CDImage;
using MusicBrainz;

namespace CUERipper
{
	public partial class frmCUERipper : Form
	{
		private CDDriveReader _reader = null;
		private Thread _workThread = null;
		private StartStop _startStop;

		public frmCUERipper()
		{
			InitializeComponent();
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
			listTracks.Enabled = !running;
			comboDrives.Enabled = !running;
			comboRelease.Enabled = !running;
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

			try
			{
				audioSource.Position = 0;
				do
				{
					uint toRead = Math.Min((uint)buff.GetLength(0), (uint)audioSource.Remaining);
					uint samplesRead = audioSource.Read(buff, toRead);
					if (samplesRead == 0) break;
					if (samplesRead != toRead)
						throw new Exception("samples read != samples requested");
					//arVerify.Write(buff, samplesRead);
					//audioDest.Write(buff, samplesRead);
				} while (true);
			}
			catch (StopException)
			{
			}
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					string message = "Exception";
					for (Exception e = ex; e != null; e = e.InnerException)
						message += ": " + e.Message;
					DialogResult dlgRes = MessageBox.Show(this, message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				});
			}
			_workThread = null;
			SetupControls();
		}

		private void buttonGo_Click(object sender, EventArgs e)
		{
			if (_reader == null)
				return;
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
			ReadOnlyCollection<Event> events = ((Release)e.ListItem).GetEvents();
			string year = events.Count > 0 ? events[0].Date.Substring(0, 4) + ": " : "";
			e.Value = string.Format("{0}{1} - {2}", year, ((Release)e.ListItem).GetArtist(), ((Release)e.ListItem).GetTitle());
		}

		private void comboRelease_SelectedIndexChanged(object sender, EventArgs e)
		{
			listTracks.Items.Clear();
			if (comboRelease.SelectedItem == null || comboRelease.SelectedItem is string)
			{
				for (int i = 1; i <= _reader.TOC.AudioTracks; i++)
					listTracks.Items.Add(new ListViewItem(new string[] { _reader.TOC[i].Number.ToString(), "Track " + _reader.TOC[i].Number.ToString(), _reader.TOC[i].StartMSF, _reader.TOC[i].LengthMSF }));
				return;
			}
			Release release = (Release) comboRelease.SelectedItem;
			for (int i = 1; i <= _reader.TOC.AudioTracks; i++)
			{
				Track track = release.GetTracks()[(int)_reader.TOC[i].Number - 1];
				listTracks.Items.Add(new ListViewItem(new string[] { _reader.TOC[i].Number.ToString(), track.GetTitle(), _reader.TOC[i].StartMSF, _reader.TOC[i].LengthMSF }));
			}
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
			p.DiscId = _reader.TOC.MusicBrainzId;
			Query<Release> results = Release.Query(p);
			MusicBrainzService.XmlRequest += new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			foreach (Release release in results)
			{
				release.GetEvents();
				release.GetTracks();
				this.BeginInvoke((MethodInvoker)delegate()
				{
					comboRelease.Items.Add(release);
				});
			}
			MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (comboRelease.Items.Count == 0)
					comboRelease.Items.Add("MusicBrainz: not found");
			});
			_workThread = null;
			SetupControls();
			this.BeginInvoke((MethodInvoker)delegate()
			{
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
