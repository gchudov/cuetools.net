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
		private string _format;
		private CUESheet _cueSheet;
		private string _pathOut;
		private string _pathFormat;
		string _defaultLosslessFormat, _defaultLossyFormat, _defaultHybridFormat;
		private CUEControls.ShellIconMgr m_icon_mgr;

		public frmCUERipper()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_startStop = new StartStop();
			m_icon_mgr = new CUEControls.ShellIconMgr();
		}

		//private byte toBCD(int val)
		//{
		//    return (byte)(((val / 10) << 4) + (val % 10));
		//}

		private void frmCUERipper_Load(object sender, EventArgs e)
		{
			//byte[] _subchannelBuffer0 = { 0x01, 0x01, 0x01, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x02, 0x0A, 0x4C, 0x43 };
			//byte[] _subchannelBuffer1 = { 0x21, 0x01, 0x01, 0x00, 0x00, 0x11, 0x00, 0x00, 0x02, 0x11, 0xCF, 0x3E };
			//byte[] _subchannelBuffer2 = { 0x21, 0x01, 0x01, 0x00, 0x00, 0x12, 0x00, 0x00, 0x02, 0x12, 0x11, 0x8F };

			//_subchannelBuffer0[3] = toBCD(_subchannelBuffer0[3]);
			//_subchannelBuffer0[4] = toBCD(_subchannelBuffer0[4]);
			//_subchannelBuffer0[5] = toBCD(_subchannelBuffer0[5]);
			//_subchannelBuffer0[7] = toBCD(_subchannelBuffer0[7]);
			//_subchannelBuffer0[8] = toBCD(_subchannelBuffer0[8]);
			//_subchannelBuffer0[9] = toBCD(_subchannelBuffer0[9]);

			//Crc16Ccitt _crc = new Crc16Ccitt(InitialCrcValue.Zeros);
			//ushort crc0a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer0, 0, 10) ^ 0xffff);
			//ushort crc0b = (ushort)(_subchannelBuffer0[11] + (_subchannelBuffer0[10] << 8));
			//ushort crc1a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer1, 0, 10) ^ 0xffff);
			//ushort crc1b = (ushort)(_subchannelBuffer1[11] + (_subchannelBuffer1[10] << 8));
			//ushort crc2a = (ushort)(_crc.ComputeChecksum(_subchannelBuffer2, 0, 10) ^ 0xffff);
			//ushort crc2b = (ushort)(_subchannelBuffer2[11] + (_subchannelBuffer2[10] << 8));
			//if (crc0a != crc0b) // || crc1a != crc1b || crc2a != crc2b)
			//{
			//}

			SettingsReader sr = new SettingsReader("CUERipper", "settings.txt", Application.ExecutablePath);
			_defaultLosslessFormat = sr.Load("DefaultLosslessFormat") ?? "flac";
			_defaultLossyFormat = sr.Load("DefaultLossyFormat") ?? "mp3";
			_defaultHybridFormat = sr.Load("DefaultHybridFormat") ?? "lossy.flac";
			_config.createEACLOG = sr.LoadBoolean("CreateEACLOG") ?? false;
			_config.preserveHTOA = sr.LoadBoolean("PreserveHTOA") ?? false;
			_config.createM3U = sr.LoadBoolean("CreateM3U") ?? true;
			_pathFormat = sr.Load("PathFormat") ?? "%music%\\%artist%\\[%year% - ]%album%\\%artist% - %album%.cue";
			checkBoxEACMode.Checked = _config.createEACLOG;
			SelectedOutputAudioType = (AudioEncoderType?)sr.LoadInt32("OutputAudioType", null, null) ?? AudioEncoderType.Lossless;
			comboBoxAudioFormat.SelectedIndex = sr.LoadInt32("ComboCodec", 0, comboBoxAudioFormat.Items.Count - 1) ?? 0;
			comboImage.SelectedIndex = sr.LoadInt32("ComboImage", 0, comboImage.Items.Count - 1) ?? 0;
			UpdateDrives();
		}

		#region private constants
		/// <summary>
		/// The window message of interest, device change
		/// </summary>
		const int WM_DEVICECHANGE = 0x0219;
		const ushort DBT_DEVICEARRIVAL = 0x8000;
		const ushort DBT_DEVICEREMOVECOMPLETE = 0x8004;
		const ushort DBT_DEVNODES_CHANGED = 0x0007;
		#endregion

		/// <summary>
		/// This method is called when a window message is processed by the dotnet application
		/// framework.  We override this method and look for the WM_DEVICECHANGE message. All
		/// messages are delivered to the base class for processing, but if the WM_DEVICECHANGE
		/// method is seen, we also alert any BWGBURN programs that the media in the drive may
		/// have changed.
		/// </summary>
		/// <param name="m">the windows message being processed</param>
		protected override void WndProc(ref Message m)
		{
			if (m.Msg == WM_DEVICECHANGE)
			{
				int val = m.WParam.ToInt32();
				if (val == DBT_DEVICEARRIVAL || val == DBT_DEVICEREMOVECOMPLETE)
					UpdateDrive();
				else if (val == DBT_DEVNODES_CHANGED)
					UpdateDrives();
			}
			base.WndProc(ref m);
		}

		private void UpdateDrives()
		{
			buttonGo.Enabled = false;
			comboDrives.Items.Clear();
			comboRelease.Items.Clear();
			listTracks.Items.Clear();
			foreach (char drive in CDDriveReader.DrivesAvailable())
			{
				CDDriveReader reader = new CDDriveReader();
				string arName = null;
				int driveOffset;
				try
				{
					reader.Open(drive);
					arName = reader.ARName;
				}
				catch (Exception ex)
				{
					try
					{
						arName = reader.ARName;
					}
					catch
					{
						comboDrives.Items.Add(drive + ": " + ex.Message);
						continue;
					}
				}
				if (!AccurateRipVerify.FindDriveReadOffset(arName, out driveOffset))
					; //throw new Exception("Failed to find drive read offset for drive" + _ripper.ARName);
				reader.DriveOffset = driveOffset;
				comboDrives.Items.Add(reader);
			}
			if (comboDrives.Items.Count == 0)
				comboDrives.Items.Add("No CD drives found");
			comboDrives.SelectedIndex = 0;
		}

		private void SetupControls ()
		{
			bool running = _workThread != null;
			listTracks.Enabled =
			comboDrives.Enabled =
			comboRelease.Enabled =
			groupBoxSettings.Enabled = !running;
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
				//Color color = ColorTranslator.FromWin32(e.ErrorsCount == 0 || e.Position == 0 ? (byte)0 : (byte)(Math.Log(1 + e.ErrorsCount / e.Position, 2) * 255));
				toolStripStatusLabel1.Text = status;
				toolStripProgressBar1.Value = Math.Max(0, Math.Min(100, (int)(percentTrck * 100)));
				//toolStripProgressBar1.ProgressBar.Style = ProgressBarStyle.Blocks;
				toolStripProgressBar2.Value = Math.Max(0, Math.Min(100, (int)(percentDisk * 100)));
				//if (e.ErrorsCount == 0 || e.Position == 0)
					//toolStripProgressBar3.Value = 0;
				//else
					//toolStripProgressBar3.Value = Math.Min(100, (int)(100 + Math.Log10(e.ErrorsCount / e.Position / 4 / 588) * 100));
			});
		}

		private void Rip(object o)
		{
			CDDriveReader audioSource = (CDDriveReader)o;
			audioSource.ReadProgress += new EventHandler<ReadProgressArgs>(CDReadProgress);
			audioSource.DriveOffset = (int)numericWriteOffset.Value;

			try
			{
				_cueSheet.Go();
				//CUESheet.WriteText(_pathOut, _cueSheet.CUESheetContents(_style));
				//CUESheet.WriteText(Path.ChangeExtension(_pathOut, ".log"), _cueSheet.LOGContents());
			}
			catch (StopException)
			{
			}
#if !DEBUG
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
#endif
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

			_format = (string)comboBoxAudioFormat.SelectedItem;
			_cueSheet.OutputStyle = comboImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE :
				CUEStyle.GapsAppended;
			_pathOut = _cueSheet.GenerateUniqueOutputPath(_pathFormat,
					_cueSheet.OutputStyle == CUEStyle.SingleFileWithCUE ? "." + _format : ".cue",
					CUEAction.Encode, null);
			if (_pathOut == "")
			{
				MessageBox.Show(this, "Output path generation failed", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			_cueSheet.GenerateFilenames(SelectedOutputAudioType, _format, _pathOut);

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
			ReleaseInfo r = (ReleaseInfo)(e.ListItem);
			e.Value = string.Format("{0}{1} - {2}", r.cueSheet.Year != "" ? r.cueSheet.Year + ": " : "", r.cueSheet.Artist, r.cueSheet.Title);
		}

		private void UpdateRelease()
		{
			listTracks.Items.Clear();
			if (comboRelease.SelectedItem == null || comboRelease.SelectedItem is string)
				return;
			_cueSheet = ((ReleaseInfo)comboRelease.SelectedItem).cueSheet;
			for (int i = 1; i <= _reader.TOC.TrackCount; i++)
			{
				listTracks.Items.Add(new ListViewItem(new string[] { 
					_reader.TOC[i].IsAudio ? _cueSheet.Tracks[i - _reader.TOC.FirstAudio].Title : "Data track",
					_reader.TOC[i].Number.ToString(), 
					_reader.TOC[i].StartMSF, 
					_reader.TOC[i].LengthMSF }));
			}
		}

		private void comboRelease_SelectedIndexChanged(object sender, EventArgs e)
		{
			UpdateRelease();
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
				toolStripStatusLabel1.Text = "Looking up album via " + (e == null ? "FreeDB" : "MusicBrainz");
				toolStripProgressBar1.Value = 0;
				toolStripProgressBar2.Value = (100 + 2 * toolStripProgressBar2.Value) / 3;
			});
		}

		private ReleaseInfo CreateCUESheet(CDDriveReader audioSource, Release release, CDEntry cdEntry)
		{
			ReleaseInfo r = new ReleaseInfo();
			r.cueSheet = new CUESheet(_config);
			r.cueSheet.OpenCD(audioSource);
			//r.cueSheet.WriteOffset = 
			General.SetCUELine(r.cueSheet.Attributes, "REM", "GENRE", "", true);
			General.SetCUELine(r.cueSheet.Attributes, "REM", "DATE", "", false);
			General.SetCUELine(r.cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);
			General.SetCUELine(r.cueSheet.Attributes, "REM", "COMMENT", _config.createEACLOG ? "ExactAudioCopy v0.99pb4" : CDDriveReader.RipperVersion(), true);
			if (release != null)
			{
				r.cueSheet.FillFromMusicBrainz(release);
				r.bitmap = Properties.Resources.musicbrainz;
			}
			else if (cdEntry != null)
			{
				r.cueSheet.FillFromFreedb(cdEntry);
				r.bitmap = Properties.Resources.freedb;
			}
			else
			{
				r.cueSheet.Artist = "Unknown Artist";
				r.cueSheet.Title = "Unknown Title";
				for (int i = 0; i < audioSource.TOC.AudioTracks; i++)
					r.cueSheet.Tracks[i].Title = string.Format("Track {0:00}", i + 1);
			}
			if (r.cueSheet.Genre == "") r.cueSheet.Genre = "";
			if (r.cueSheet.Year == "") r.cueSheet.Year = "";
			r.cueSheet.Action = CUEAction.Encode;
			r.cueSheet.UseAccurateRip();
			//r.cueSheet.ArVerify.ContactAccurateRip(AccurateRipVerify.CalculateAccurateRipId(audioSource.TOC));
			return r;
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
					ReleaseInfo r = CreateCUESheet(audioSource, release, null);
					this.BeginInvoke((MethodInvoker)delegate()
					{
						comboRelease.Items.Add(r);
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
				MusicBrainz_LookupProgress(this, null);
				code = m_freedb.Query(AccurateRipVerify.CalculateCDDBQuery(audioSource.TOC), out queryResult, out coll);
				if (code == FreedbHelper.ResponseCodes.CODE_200)
				{
					CDEntry cdEntry;
					MusicBrainz_LookupProgress(this, null);
					code = m_freedb.Read(queryResult, out cdEntry);
					if (code == FreedbHelper.ResponseCodes.CODE_210)
					{
						Encoding iso = Encoding.GetEncoding("iso-8859-1");
						ReleaseInfo r = CreateCUESheet(audioSource, null, cdEntry);
						this.BeginInvoke((MethodInvoker)delegate()
						{
							comboRelease.Items.Add(r);
						});
						if (Encoding.Default.GetString(iso.GetBytes(cdEntry.Title)) != cdEntry.Title)
						{
							cdEntry.Artist = Encoding.Default.GetString(iso.GetBytes(cdEntry.Artist));
							cdEntry.Title = Encoding.Default.GetString(iso.GetBytes(cdEntry.Title));
							for (int i = 0; i < cdEntry.Tracks.Count; i++)
								cdEntry.Tracks[i].Title = Encoding.Default.GetString(iso.GetBytes(cdEntry.Tracks[i].Title));
							r = CreateCUESheet(audioSource, null, cdEntry);
							this.BeginInvoke((MethodInvoker)delegate()
							{
								comboRelease.Items.Add(r);
							});
						}
					}
				}
				else
				if (code == FreedbHelper.ResponseCodes.CODE_210 ||
					code == FreedbHelper.ResponseCodes.CODE_211 )
				{
					foreach (QueryResult qr in coll)
					{
						CDEntry cdEntry;
						MusicBrainz_LookupProgress(this, null);
						code = m_freedb.Read(qr, out cdEntry);
						if (code == FreedbHelper.ResponseCodes.CODE_210)
						{
							ReleaseInfo r = CreateCUESheet(audioSource, null, cdEntry);
							this.BeginInvoke((MethodInvoker)delegate()
							{
								comboRelease.Items.Add(r);
							});
						}
					}
				}
			}
			catch (Exception)
			{
			}

			if (comboRelease.Items.Count == 0)
			{
				ReleaseInfo r = CreateCUESheet(audioSource, null, null);
				this.BeginInvoke((MethodInvoker)delegate()
				{
					comboRelease.Items.Add(r);
				});
			}
			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
				comboRelease.SelectedIndex = 0;
				CUESheet cueSheet = ((ReleaseInfo)comboRelease.SelectedItem).cueSheet;
				toolStripStatusAr.Visible = cueSheet.ArVerify.ARStatus == null;
				toolStripStatusAr.Text = cueSheet.ArVerify.ARStatus == null ? cueSheet.ArVerify.Total(0).ToString() : "?";
				toolStripStatusAr.ToolTipText = "AccurateRip: " + (cueSheet.ArVerify.ARStatus ?? "found") + ".";
			});
		}

		private void UpdateDrive()
		{
			toolStripStatusAr.Visible = false;
			buttonGo.Enabled = false;
			comboRelease.Items.Clear();
			listTracks.Items.Clear();
			if (comboDrives.SelectedItem is string)
			{
				_reader = null;
				return;
			}
			_reader = (CDDriveReader)comboDrives.SelectedItem;
			try
			{
				_reader.Open(_reader.Path[0]);
				numericWriteOffset.Value = _reader.DriveOffset;
			}
			catch (Exception ex)
			{
				numericWriteOffset.Value = _reader.DriveOffset;
				_reader.Close();
				comboRelease.Items.Add(ex.Message);
				comboRelease.SelectedIndex = 0;
				return;
			}
			if (_reader.TOC.AudioTracks == 0)
			{
				comboRelease.Items.Add("No audio tracks");
				comboRelease.SelectedIndex = 0;
				return;
			}
			UpdateRelease();
			_workThread = new Thread(Lookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(_reader);
		}

		private void comboDrives_SelectedIndexChanged(object sender, EventArgs e)
		{
			UpdateDrive();
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
			CUESheet cueSheet = ((ReleaseInfo)comboRelease.SelectedItem).cueSheet;
			if (e.Label != null && _reader.TOC[e.Item + 1].IsAudio)
				cueSheet.Tracks[e.Item].Title = e.Label;
			else
				e.CancelEdit = true;
		}

		private void editToolStripMenuItem_Click(object sender, EventArgs e)
		{
			CUESheet cueSheet = ((ReleaseInfo)comboRelease.SelectedItem).cueSheet;
			frmProperties frm = new frmProperties();
			frm.CUE = cueSheet;
			frm.ShowDialog();
		}

		private void comboRelease_DrawItem(object sender, DrawItemEventArgs e)
		{
			e.DrawBackground();
			StringFormat format = new StringFormat();
			format.FormatFlags = StringFormatFlags.NoClip;
			format.Alignment = StringAlignment.Near;
			if (e.Index >= 0 && e.Index < comboRelease.Items.Count)
			{
				string text = comboRelease.GetItemText(comboRelease.Items[e.Index]);
				if (comboRelease.Items[e.Index] is ReleaseInfo)
				{
					Bitmap ImageToDraw = ((ReleaseInfo)comboRelease.Items[e.Index]).bitmap;
					if (ImageToDraw != null)
						e.Graphics.DrawImage(ImageToDraw, new Rectangle(e.Bounds.X, e.Bounds.Y, e.Bounds.Height, e.Bounds.Height));
					//e.Graphics.DrawImage(ImageToDraw, new Rectangle(e.Bounds.X + e.Bounds.Width - ImageToDraw.Width, e.Bounds.Y, ImageToDraw.Width, e.Bounds.Height));
				}
				e.Graphics.DrawString(text, e.Font, new SolidBrush(e.ForeColor), new RectangleF((float)e.Bounds.X + e.Bounds.Height, (float)e.Bounds.Y, (float)(e.Bounds.Width - e.Bounds.Height), (float)e.Bounds.Height), format);
			}
			e.DrawFocusRectangle();
		}

		private void frmCUERipper_FormClosed(object sender, FormClosedEventArgs e)
		{
			SettingsWriter sw = new SettingsWriter("CUERipper", "settings.txt", Application.ExecutablePath);
			sw.Save("DefaultLosslessFormat", _defaultLosslessFormat);
			sw.Save("DefaultLossyFormat", _defaultLossyFormat);
			sw.Save("DefaultHybridFormat", _defaultHybridFormat);
			sw.Save("CreateEACLOG", _config.createEACLOG);
			sw.Save("PreserveHTOA", _config.preserveHTOA);
			sw.Save("CreateM3U", _config.createM3U);
			sw.Save("OutputAudioType", (int)SelectedOutputAudioType);
			sw.Save("ComboCodec", comboBoxAudioFormat.SelectedIndex);
			sw.Save("ComboImage", comboImage.SelectedIndex);
			sw.Save("PathFormat", _pathFormat);
			sw.Close();
		}

		private void listTracks_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			if (!_reader.TOC[e.Item + 1].IsAudio)
				e.CancelEdit = true;
		}

		private string SelectedOutputAudioFormat
		{
			get
			{
				return (string)(comboBoxAudioFormat.SelectedItem ?? "dummy");
			}
			set
			{
				comboBoxAudioFormat.SelectedItem = value;
			}
		}

		private CUEToolsFormat SelectedOutputAudioFmt
		{
			get
			{
				CUEToolsFormat fmt;
				if (comboBoxAudioFormat.SelectedItem == null)
					return null;
				string formatName = (string)comboBoxAudioFormat.SelectedItem;
				if (formatName.StartsWith("lossy."))
					formatName = formatName.Substring(6);
				return _config.formats.TryGetValue(formatName, out fmt) ? fmt : null;
			}
		}

		private AudioEncoderType SelectedOutputAudioType
		{
			get
			{
				return
					radioButtonAudioHybrid.Checked ? AudioEncoderType.Hybrid :
					radioButtonAudioLossy.Checked ? AudioEncoderType.Lossy :
					AudioEncoderType.Lossless;
			}
			set
			{
				switch (value)
				{
					case AudioEncoderType.Hybrid:
						radioButtonAudioHybrid.Checked = true;
						break;
					case AudioEncoderType.Lossy:
						radioButtonAudioLossy.Checked = true;
						break;
					default:
						radioButtonAudioLossless.Checked = true;
						break;
				}
			}
		}

		private void checkBoxEACMode_CheckedChanged(object sender, EventArgs e)
		{
			_config.createEACLOG = checkBoxEACMode.Checked;
		}

		private void radioButtonAudioLossless_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton && !((RadioButton)sender).Checked)
				return;
			//labelFormat.ImageKey = null;
			comboBoxAudioFormat.Items.Clear();
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (SelectedOutputAudioType == AudioEncoderType.Lossless && !format.Value.allowLossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Hybrid) // && format.Key != "wv") TODO!!!!!
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossy && !format.Value.allowLossy)
					continue;
				comboBoxAudioFormat.Items.Add(format.Key);
			}
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (!format.Value.allowLossyWAV)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
					continue;
				comboBoxAudioFormat.Items.Add("lossy." + format.Key);
			}
			switch (SelectedOutputAudioType)
			{
				case AudioEncoderType.Lossless:
					SelectedOutputAudioFormat = _defaultLosslessFormat;
					break;
				case AudioEncoderType.Lossy:
					SelectedOutputAudioFormat = _defaultLossyFormat;
					break;
				case AudioEncoderType.Hybrid:
					SelectedOutputAudioFormat = _defaultHybridFormat;
					break;
			}
		}

		private void comboBoxAudioFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			//labelFormat.ImageKey = SelectedOutputAudioFmt == null ? null : "." + SelectedOutputAudioFmt.extension;
			comboBoxEncoder.Items.Clear();
			if (SelectedOutputAudioFmt == null)
				return;

			foreach (CUEToolsUDC encoder in _config.encoders)
				if (encoder.extension == SelectedOutputAudioFmt.extension)
				{
					if (SelectedOutputAudioFormat.StartsWith("lossy.") && !encoder.lossless)
						continue;
					else if (SelectedOutputAudioType == AudioEncoderType.Lossless && !encoder.lossless)
						continue;
					else if (SelectedOutputAudioType == AudioEncoderType.Lossy && encoder.lossless)
						continue;
					comboBoxEncoder.Items.Add(encoder);
				}
			comboBoxEncoder.SelectedItem = SelectedOutputAudioFormat.StartsWith("lossy.") ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioType == AudioEncoderType.Lossless ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioFmt.encoderLossy;
			comboBoxEncoder.Enabled = true;
		}

		private void comboBoxEncoder_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (SelectedOutputAudioFormat == null)
				return;
			CUEToolsUDC encoder = comboBoxEncoder.SelectedItem as CUEToolsUDC;
			if (SelectedOutputAudioFormat.StartsWith("lossy."))
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else if (SelectedOutputAudioType == AudioEncoderType.Lossless)
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else
				SelectedOutputAudioFmt.encoderLossy = encoder;
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

	class ReleaseInfo
	{
		public CUESheet cueSheet;
		public Bitmap bitmap;
	}
}
