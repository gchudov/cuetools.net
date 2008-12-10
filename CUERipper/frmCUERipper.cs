using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Configuration;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Processor;
using CUETools.Ripper.SCSI;
using MusicBrainz;
using Freedb;

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
		private string _pathOut;

		public frmCUERipper()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_startStop = new StartStop();
		}

		private void frmCUERipper_Load(object sender, EventArgs e)
		{
			//byte[] _subchannelBuffer0 = { 0x1, 0x01, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x02, 0x10, 0x75, 0x4E };
			//byte[] _subchannelBuffer1 = { 0x21, 0x01, 0x01, 0x00, 0x00, 0x11, 0x00, 0x00, 0x02, 0x11, 0xCF, 0x3E };
			//byte[] _subchannelBuffer2 = { 0x21, 0x01, 0x01, 0x00, 0x00, 0x12, 0x00, 0x00, 0x02, 0x12, 0x11, 0x8F };
			//Crc16Ccitt _crc = new Crc16Ccitt(InitialCrcValue.Zeros);
			//ushort crc0a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer0, 0, 10) ^ 0xffff);
			//ushort crc0b = (ushort)(_subchannelBuffer0[11] + (_subchannelBuffer0[10] << 8));
			//ushort crc1a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer1, 0, 10) ^ 0xffff);
			//ushort crc1b = (ushort)(_subchannelBuffer1[11] + (_subchannelBuffer1[10] << 8));
			//ushort crc2a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer2, 0, 10) ^ 0xffff);
			//ushort crc2b = (ushort)(_subchannelBuffer2[11] + (_subchannelBuffer2[10] << 8));
			//if (crc0a != crc0b || crc1a != crc1b || crc2a != crc2b)
			//{
			//}

			foreach(char drive in CDDriveReader.DrivesAvailable())
			{
				CDDriveReader reader = new CDDriveReader();
				if (reader.Open(drive))
				{
					int driveOffset;
					if (!AccurateRipVerify.FindDriveReadOffset(reader.ARName, out driveOffset))
						; //throw new Exception("Failed to find drive read offset for drive" + _ripper.ARName);
					reader.DriveOffset = driveOffset;
					comboDrives.Items.Add(reader);
				}
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

			CUESheet.WriteText(_pathOut, _cueSheet.CUESheetContents(_style));
			CUESheet.WriteText(Path.ChangeExtension(_pathOut, ".log"), _cueSheet.LOGContents());
			try
			{
				_cueSheet.WriteAudioFiles(".", _style);
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
			audioSource.ReadProgress -= new EventHandler<ReadProgressArgs>(CDReadProgress);
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

			_style = comboImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE :
				CUEStyle.GapsAppended;
			_pathOut = _config.CleanseString(_cueSheet.Artist) + " - " + 
				_config.CleanseString(_cueSheet.Title) + ".cue";
			_config.lossyWAVHybrid = comboLossless.SelectedIndex == 1; // _cueSheet.Config?
			if (_style == CUEStyle.SingleFileWithCUE)
				_cueSheet.SingleFilename = Path.GetFileName(_pathOut);
			_format = (string)comboCodec.SelectedItem == "wav" ? OutputAudioFormat.WAV :
				(string)comboCodec.SelectedItem == "flac" ? OutputAudioFormat.FLAC :
				(string)comboCodec.SelectedItem == "wv" ? OutputAudioFormat.WavPack :
				(string)comboCodec.SelectedItem == "ape" ? OutputAudioFormat.APE :
				OutputAudioFormat.NoAudio;
			_cueSheet.GenerateFilenames(_format, comboLossless.SelectedIndex != 0, _pathOut);

			_workThread = new Thread(Rip);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(_reader);			
		}

		private void buttonAbort_Click(object sender, EventArgs e)
		{
			_startStop.Stop();
		}

		private void buttonPause_Click(object sender, EventArgs e)
		{
			_startStop.Pause();
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

		private CUESheet CreateCUESheet(CDDriveReader audioSource, Release release, CDEntry cdEntry)
		{
			CUESheet cueSheet = new CUESheet(_config);
			cueSheet.OpenCD(audioSource);
			General.SetCUELine(cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);
			General.SetCUELine(cueSheet.Attributes, "REM", "COMMENT", CDDriveReader.RipperVersion(), true);
			if (release != null)
				cueSheet.FillFromMusicBrainz(release);
			else if (cdEntry != null)
			{
				cueSheet.Artist = cdEntry.Artist;
				cueSheet.Title = cdEntry.Title;
				General.SetCUELine(cueSheet.Attributes, "REM", "DATE", cdEntry.Year, false);
				General.SetCUELine(cueSheet.Attributes, "REM", "GENRE", cdEntry.Genre, true);
				for (int i = 0; i < audioSource.TOC.AudioTracks; i++)
					cueSheet.Tracks[i].Title = cdEntry.Tracks[i].Title;
			}
			else
			{
				cueSheet.Artist = "Unknown Artist";
				cueSheet.Title = "Unknown Title";
				for (int i = 0; i < audioSource.TOC.AudioTracks; i++)
					cueSheet.Tracks[i].Title = string.Format("Track {0:00}", i + 1);
			}
			cueSheet.AccurateRip = AccurateRipMode.VerifyAndConvert;
			cueSheet.ArVerify.ContactAccurateRip(AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC));
			return cueSheet;
		}

		private void Lookup(object o)
		{
			CDDriveReader audioSource = (CDDriveReader)o;

			ReleaseQueryParameters p = new ReleaseQueryParameters();
			p.DiscId = audioSource.TOC.MusicBrainzId;
			Query<Release> results = Release.Query(p);
			MusicBrainzService.XmlRequest += new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);
			try
			{
				foreach (Release release in results)
				{
					release.GetEvents();
					release.GetTracks();
					CUESheet cueSheet = CreateCUESheet(audioSource, release, null);
					this.BeginInvoke((MethodInvoker)delegate()
					{
						comboRelease.Items.Add(cueSheet);
					});
				}
			}
			catch (Exception)
			{
			}
			MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);


			FreedbHelper m_freedb = new FreedbHelper();

			m_freedb.UserName = "gchudov";
			m_freedb.Hostname = "gmail.com";
			m_freedb.ClientName = "CUERipper";
			m_freedb.Version = "1.0";
			m_freedb.SetDefaultSiteAddress(Properties.Settings.Default.MAIN_FREEDB_SITEADDRESS);
			
			QueryResult queryResult;
			QueryResultCollection coll;
			string code = string.Empty;
			try
			{
				code = m_freedb.Query(AccurateRipVerify.CalculateCDDBQuery(audioSource.TOC), out queryResult, out coll);
				if (code == FreedbHelper.ResponseCodes.CODE_200)
				{
					CDEntry cdEntry;
					code = m_freedb.Read(queryResult, out cdEntry);
					if (code == FreedbHelper.ResponseCodes.CODE_210)
					{
						CUESheet cueSheet = CreateCUESheet(audioSource, null, cdEntry);
						this.BeginInvoke((MethodInvoker)delegate()
						{
							comboRelease.Items.Add(cueSheet);
						});
					}
				}
				else
				if (code == FreedbHelper.ResponseCodes.CODE_210 ||
					code == FreedbHelper.ResponseCodes.CODE_211 )
				{
					foreach (QueryResult qr in coll)
					{
						CDEntry cdEntry;
						code = m_freedb.Read(qr, out cdEntry);
						if (code == FreedbHelper.ResponseCodes.CODE_210)
						{
							CUESheet cueSheet = CreateCUESheet(audioSource, null, cdEntry);
							this.BeginInvoke((MethodInvoker)delegate()
							{
								comboRelease.Items.Add(cueSheet);
							});
						}
					}
				}
			}
			catch (Exception)
			{
			}

			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (comboRelease.Items.Count == 0)
				{
					CUESheet cueSheet = CreateCUESheet(audioSource, null, null);
					comboRelease.Items.Add(cueSheet);
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

		private void listTracks_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUESheet cueSheet = (CUESheet)comboRelease.SelectedItem;
			if (e.Label != null)
				cueSheet.Tracks[e.Item].Title = e.Label;
		}

		private void editToolStripMenuItem_Click(object sender, EventArgs e)
		{
			CUESheet cueSheet = (CUESheet)comboRelease.SelectedItem;
			frmRelease frm = new frmRelease();
			frm.CUE = cueSheet;
			frm.ShowDialog();
			comboRelease.Items[comboRelease.SelectedIndex] = cueSheet;
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

		public void Stop()
		{
			lock (this)
			{
				if (_pause)
				{
					_pause = false;
					Monitor.Pulse(this);
				}
				_stop = true;
			}
		}

		public void Pause()
		{
			lock (this)
			{
				if (_pause)
				{
					_pause = false;
					Monitor.Pulse(this);
				}
				else
				{
					_pause = true;
				}
			}
		}
	}
}
