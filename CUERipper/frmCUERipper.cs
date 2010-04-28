using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Configuration;
using CUETools.AccurateRip;
using CUETools.CTDB;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Processor;
using CUETools.Ripper;
using CUEControls;
using MusicBrainz;
using Freedb;

namespace CUERipper
{
	public partial class frmCUERipper : Form
	{
		private Thread _workThread = null;
		private StartStop _startStop;
		private CUEConfig _config;
		private CUESheet cueSheet;
		private ReleaseInfo selectedRelease;
		private DriveInfo selectedDriveInfo;
		private string _pathOut;
		string _defaultLosslessFormat, _defaultLossyFormat, _defaultHybridFormat;
		private CUEControls.ShellIconMgr m_icon_mgr;

		public frmCUERipper()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_startStop = new StartStop();
			m_icon_mgr = new CUEControls.ShellIconMgr();
			m_icon_mgr.SetExtensionIcon(".flac", Properties.Resources.flac);
			m_icon_mgr.SetExtensionIcon(".wv", Properties.Resources.wv);
			m_icon_mgr.SetExtensionIcon(".ape", Properties.Resources.ape);
			m_icon_mgr.SetExtensionIcon(".tta", Properties.Resources.tta);
			m_icon_mgr.SetExtensionIcon(".wav", Properties.Resources.wave);
			m_icon_mgr.SetExtensionIcon(".mp3", Properties.Resources.mp3);
			m_icon_mgr.SetExtensionIcon(".m4a", Properties.Resources.ipod_sound);
			m_icon_mgr.SetExtensionIcon(".ogg", Properties.Resources.ogg);
		}

		string[] OutputPathUseTemplates = {
			"%music%\\%artist%\\[%year% - ]%album%\\%artist% - %album%.cue",
			"%music%\\%artist%\\[%year% - ]%album%[ - %edition%]$ifgreater($max(%discnumber%,%totaldiscs%),1, - cd %discnumber%,)[' ('%unique%')']\\%artist% - %album%[ - %edition%].cue"
		};

		private BindingList<string> cueStyles = new BindingList<string> { "image", "tracks" };
		//private BindingList<string> losslessOrNot = new BindingList<string> { "lossless", "lossy" };
		private BindingList<BNComboBoxItem<AudioEncoderType>> losslessOrNot = new BindingList<BNComboBoxItem<AudioEncoderType>> { 
			new BNComboBoxItem<AudioEncoderType>("lossless", "checked", AudioEncoderType.Lossless),
			new BNComboBoxItem<AudioEncoderType>("lossy", "unchecked", AudioEncoderType.Lossy) 
		};		
		private BindingList<ReleaseInfo> releases = new BindingList<ReleaseInfo>();
		private BindingList<DriveInfo> drives = new BindingList<DriveInfo>();
		private BindingList<FormatInfo> formats = new BindingList<FormatInfo>();
		private BindingList<CUEToolsUDC> encoders = new BindingList<CUEToolsUDC>();

		public BindingList<string> CUEStyles
		{
			get
			{
				return cueStyles;
			}
		}

		public BindingList<BNComboBoxItem<AudioEncoderType>> LosslessOrNot
		{
			get
			{
				return losslessOrNot;
			}
		}

		public BindingList<ReleaseInfo> Releases
		{
			get
			{
				return releases;
			}
		}

		public BindingList<DriveInfo> Drives
		{
			get
			{
				return drives;
			}
		}

		public BindingList<FormatInfo> Formats
		{
			get
			{
				return formats;
			}
		}

		public BindingList<CUEToolsUDC> Encoders
		{
			get
			{
				return encoders;
			}
		}

		private void frmCUERipper_Load(object sender, EventArgs e)
		{
			SettingsReader sr = new SettingsReader("CUERipper", "settings.txt", Application.ExecutablePath);
			_config.Load(sr);
			_defaultLosslessFormat = sr.Load("DefaultLosslessFormat") ?? "flac";
			_defaultLossyFormat = sr.Load("DefaultLossyFormat") ?? "mp3";
			_defaultHybridFormat = sr.Load("DefaultHybridFormat") ?? "lossy.flac";
			//_config.createEACLOG = sr.LoadBoolean("CreateEACLOG") ?? true;
			//_config.preserveHTOA = sr.LoadBoolean("PreserveHTOA") ?? false;
			//_config.createM3U = sr.LoadBoolean("CreateM3U") ?? true;

			bindingSourceCR.DataSource = this;
			bnComboBoxDrives.ImageList = m_icon_mgr.ImageList;
			bnComboBoxFormat.ImageList = m_icon_mgr.ImageList;
			SetupControls();

			int iFormat, nFormats = sr.LoadInt32("OutputPathUseTemplates", 0, 10) ?? 0;
			for (iFormat = 0; iFormat < OutputPathUseTemplates.Length; iFormat++)
				bnComboBoxOutputFormat.Items.Add(OutputPathUseTemplates[iFormat]);
			for (iFormat = nFormats - 1; iFormat >= 0; iFormat--)
				bnComboBoxOutputFormat.Items.Add(sr.Load(string.Format("OutputPathUseTemplate{0}", iFormat)) ?? "");

			bnComboBoxOutputFormat.Text = sr.Load("PathFormat") ?? "%music%\\%artist%\\[%year% - ]%album%\\%artist% - %album%.cue";
			checkBoxEACMode.Checked = _config.createEACLOG;
			SelectedOutputAudioType = (AudioEncoderType?)sr.LoadInt32("OutputAudioType", null, null) ?? AudioEncoderType.Lossless;
			bnComboBoxImage.SelectedIndex = sr.LoadInt32("ComboImage", 0, bnComboBoxImage.Items.Count - 1) ?? 0;
			trackBarSecureMode.Value = sr.LoadInt32("SecureMode", 0, trackBarSecureMode.Maximum - 1) ?? 1;
			trackBarSecureMode_Scroll(this, new EventArgs());
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
			if (m.Msg == WM_DEVICECHANGE && _workThread == null)
			{
				int val = m.WParam.ToInt32();
				if (val == DBT_DEVICEARRIVAL || val == DBT_DEVICEREMOVECOMPLETE)
					UpdateDrive();
				else if (val == DBT_DEVNODES_CHANGED)
					UpdateDrives();
			}
			base.WndProc(ref m);
		}

		private void DrivesLookup(object o)
		{
			// Lookup
			drives.RaiseListChangedEvents = false;
			foreach (char drive in CDDrivesList.DrivesAvailable())
			{
				ICDRipper reader = null;
				string arName = null;
				int driveOffset;
				try
				{
					this.BeginInvoke((MethodInvoker)delegate()
					{
						toolStripStatusLabel1.Text = Properties.Resources.DetectingDrives + ": " + drive + ":\\...";
					});
					reader = Activator.CreateInstance(CUEProcessorPlugins.ripper) as ICDRipper;
					reader.Open(drive);
					arName = reader.ARName;
					reader.Close();
				}
				catch (Exception ex)
				{
					try
					{
						arName = reader.ARName;
					}
					catch
					{
						drives.Add(new DriveInfo(m_icon_mgr, drive + ":\\", ex.Message));
						continue;
					}
				}
				if (!AccurateRipVerify.FindDriveReadOffset(arName, out driveOffset))
					; //throw new Exception("Failed to find drive read offset for drive" + _ripper.ARName);
				reader.DriveOffset = driveOffset;
				drives.Add(new DriveInfo(m_icon_mgr, drive + ":\\", reader));
			}
			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
				drives.RaiseListChangedEvents = true;
				drives.ResetBindings();
				if (drives.Count == 0)
					bnComboBoxDrives.Text = Properties.Resources.NoDrives;
			});
		}

		private void UpdateDrives()
		{
			buttonGo.Enabled = false;
			foreach (DriveInfo driveInfo in drives)
				if (driveInfo.drive != null)
					driveInfo.drive.Close();
			drives.Clear();
			listTracks.Items.Clear();
			releases.Clear();
			selectedRelease = null;
			selectedDriveInfo = null;
			bnComboBoxRelease.Text = "";

			if (CUEProcessorPlugins.ripper == null)
			{
				bnComboBoxDrives.Text = Properties.Resources.FailedToLoadRipperModule;
				return;
			}

			_workThread = new Thread(DrivesLookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(this);
		}

		bool outputFormatVisible = false;

		private void SetupControls ()
		{
			bool running = _workThread != null;

			bnComboBoxOutputFormat.Visible = outputFormatVisible;
			txtOutputPath.Visible = !outputFormatVisible;
			txtOutputPath.Enabled = !running && !outputFormatVisible;
			bnComboBoxRelease.Enabled = !running && releases.Count > 0;
			bnComboBoxDrives.Enabled = !running && drives.Count > 0;
			bnComboBoxOutputFormat.Enabled =
			listTracks.Enabled =
			groupBoxSettings.Enabled = !running;
			buttonPause.Visible = buttonPause.Enabled = buttonAbort.Visible = buttonAbort.Enabled = running;
			buttonGo.Visible = buttonGo.Enabled = !running;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			progressBarErrors.Value = 0;
			progressBarCD.Value = 0;
		}

		private void CheckStop()
		{
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
						toolStripStatusLabel1.Text = Properties.Resources.PausedMessage +  "...";
					});
					Monitor.Wait(_startStop);
				}
			}
		}

		private void UploadProgress(object sender, Krystalware.UploadHelper.UploadProgressEventArgs e)
		{
			CheckStop();
			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = e.uri;
				toolStripProgressBar1.Value = Math.Max(0, Math.Min(100, (int)(e.percent * 100)));
			});
		}

		private void CDReadProgress(object sender, ReadProgressArgs e)
		{		
			CheckStop();

			ICDRipper audioSource = sender as ICDRipper;
			int processed = e.Position - e.PassStart;
			TimeSpan elapsed = DateTime.Now - e.PassTime;
			double speed = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds / 75 : 1.0;

			double percentTrck = (double)(e.Position - e.PassStart) / (e.PassEnd - e.PassStart);
			string retry = e.Pass > 0 ? " (" + Properties.Resources.Retry + " " + e.Pass.ToString() + ")" : "";
			string status = (elapsed.TotalSeconds > 0 && e.Pass >= 0) ?
				string.Format("{0} @{1:00.00}x{2}...", e.Action, speed, retry) :
				string.Format("{0}{1}...", e.Action, retry);
			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = status;
				toolStripProgressBar1.Value = Math.Max(0, Math.Min(100, (int)(percentTrck * 100)));

				progressBarErrors.Maximum = (int)(Math.Log(e.PassEnd - e.PassStart + 1) * 10);
				progressBarErrors.Value = Math.Min(progressBarErrors.Maximum, (int)(Math.Log(e.ErrorsCount + 1) * 10));
				progressBarErrors.Enabled = e.Pass >= audioSource.CorrectionQuality;

				progressBarCD.Maximum = (int) audioSource.TOC.AudioLength;
				progressBarCD.Value = Math.Max(0, Math.Min(progressBarCD.Maximum, (int)e.PassStart + (e.PassEnd - e.PassStart) * (Math.Min(e.Pass, audioSource.CorrectionQuality) + 1) / (audioSource.CorrectionQuality + 1)));
			});
		}

		private void Rip(object o)
		{
			ICDRipper audioSource = o as ICDRipper;
			audioSource.ReadProgress += new EventHandler<ReadProgressArgs>(CDReadProgress);
			audioSource.DriveOffset = (int)numericWriteOffset.Value;

			try
			{
				cueSheet.Go();

				if ((cueSheet.CTDB.AccResult == HttpStatusCode.NotFound || cueSheet.CTDB.AccResult == HttpStatusCode.OK)
					&& audioSource.CorrectionQuality > 0 && audioSource.ErrorsCount == 0)
				{
					DBEntry confirm = null;
					foreach (DBEntry entry in cueSheet.CTDB.Entries)
						if (entry.toc.TrackOffsets == selectedDriveInfo.drive.TOC.TrackOffsets && !entry.hasErrors)
							confirm = entry;

					if (confirm != null)
						cueSheet.CTDB.Confirm(confirm);
					else
						cueSheet.CTDB.Submit(
							(int)cueSheet.ArVerify.WorstConfidence() + 1,
							(int)cueSheet.ArVerify.WorstTotal() + 1,
							cueSheet.Artist,
							cueSheet.Title);
				}

				bool canFix = false;
				if (cueSheet.CTDB.AccResult == HttpStatusCode.OK && audioSource.ErrorsCount != 0)
				{
					foreach (DBEntry entry in cueSheet.CTDB.Entries)
						if (entry.hasErrors && entry.canRecover)
							canFix = true;
				}
				this.Invoke((MethodInvoker)delegate()
				{					
					DialogResult dlgRes = audioSource.ErrorsCount != 0 ? 
						MessageBox.Show(this, cueSheet.GenerateAccurateRipStatus() + (canFix ? "\n" + Properties.Resources.DoneRippingRepair : "") + ".", Properties.Resources.DoneRippingErrors, MessageBoxButtons.OK, MessageBoxIcon.Error) :
						MessageBox.Show(this, cueSheet.GenerateAccurateRipStatus() + ".", Properties.Resources.DoneRipping, MessageBoxButtons.OK, MessageBoxIcon.Information);
				});
			}
			catch (StopException)
			{
			}
#if !DEBUG
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					string message = Properties.Resources.ExceptionMessage;
					for (Exception e = ex; e != null; e = e.InnerException)
						message += ": " + e.Message;
					DialogResult dlgRes = MessageBox.Show(this, message, Properties.Resources.ExceptionMessage, MessageBoxButtons.OK, MessageBoxIcon.Error);
				});
			}
#endif

			audioSource.ReadProgress -= new EventHandler<ReadProgressArgs>(CDReadProgress);
			try
			{
				audioSource.Close();
				audioSource.Open(audioSource.Path[0]);
			}
			catch (Exception ex)
			{
				this.BeginInvoke((MethodInvoker)delegate()
				{
					releases.Clear();
					selectedRelease = null;
					bnComboBoxRelease.Text = ex.Message;
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
			if (selectedDriveInfo == null)
				return;

			if (!bnComboBoxOutputFormat.Items.Contains(bnComboBoxOutputFormat.Text) && bnComboBoxOutputFormat.Text.Contains("%"))
			{
				bnComboBoxOutputFormat.Items.Insert(OutputPathUseTemplates.Length, bnComboBoxOutputFormat.Text);
				if (bnComboBoxOutputFormat.Items.Count > OutputPathUseTemplates.Length + 10)
					bnComboBoxOutputFormat.Items.RemoveAt(OutputPathUseTemplates.Length + 10);
			}

			cueSheet.CopyMetadata(selectedRelease.metadata);
			cueSheet.OutputStyle = bnComboBoxImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE :
				CUEStyle.GapsAppended;
			_pathOut = cueSheet.GenerateUniqueOutputPath(bnComboBoxOutputFormat.Text,
					cueSheet.OutputStyle == CUEStyle.SingleFileWithCUE ? "." + selectedFormat.ToString() : ".cue",
					CUEAction.Encode, null);
			if (_pathOut == "")
			{
				MessageBox.Show(this, "Output path generation failed", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			cueSheet.GenerateFilenames(SelectedOutputAudioType, selectedFormat.ToString(), _pathOut);
			selectedDriveInfo.drive.CorrectionQuality = trackBarSecureMode.Value;

			_workThread = new Thread(Rip);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(selectedDriveInfo.drive);			
		}

		private void buttonAbort_Click(object sender, EventArgs e)
		{
			_startStop.Stop();
		}

		private void buttonPause_Click(object sender, EventArgs e)
		{
			_startStop.Pause();
		}

		private void UpdateRelease()
		{
			listTracks.Items.Clear();
			selectedRelease = bnComboBoxRelease.SelectedItem as ReleaseInfo;
			comboBoxOutputFormat_TextUpdate(this, new EventArgs());
			if (selectedRelease == null)
				return;

			for (int i = 1; i <= selectedDriveInfo.drive.TOC.TrackCount; i++)
			{
				listTracks.Items.Add(new ListViewItem(new string[] { 
					selectedDriveInfo.drive.TOC[i].IsAudio ? selectedRelease.metadata.Tracks[i - selectedDriveInfo.drive.TOC.FirstAudio].Title : "Data track",
					selectedDriveInfo.drive.TOC[i].Number.ToString(), 
					selectedDriveInfo.drive.TOC[i].StartMSF, 
					selectedDriveInfo.drive.TOC[i].LengthMSF }));
			}
		}

		private void MusicBrainz_LookupProgress(object sender, XmlRequestEventArgs e)
		{
			CheckStop();
			//_progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
			//_progress.input = e.Uri.ToString();
			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " " + (e == null ? "FreeDB" : "MusicBrainz") + "...";
				toolStripProgressBar1.Value = (100 + 2 * toolStripProgressBar1.Value) / 3;
			});
		}

		private ReleaseInfo CreateCUESheet(ICDRipper audioSource, Release release, CDEntry cdEntry)
		{
			ReleaseInfo r = new ReleaseInfo(_config, audioSource.TOC);
			General.SetCUELine(r.metadata.Attributes, "REM", "GENRE", "", true);
			General.SetCUELine(r.metadata.Attributes, "REM", "DATE", "", false);
			if (release != null)
			{
				r.metadata.FillFromMusicBrainz(release);
				r.ImageKey = "musicbrainz";
			}
			else if (cdEntry != null)
			{
				r.metadata.FillFromFreedb(cdEntry);
				r.ImageKey = "freedb";
			}
			else
			{
				r.metadata.Artist = "Unknown Artist";
				r.metadata.Title = "Unknown Title";
				for (int i = 0; i < audioSource.TOC.AudioTracks; i++)
				{
					r.metadata.Tracks[i].Title = string.Format("Track {0:00}", i + 1);
					r.metadata.Tracks[i].Artist = r.metadata.Artist;
				}
			}
			if (r.metadata.Genre == "") r.metadata.Genre = "";
			if (r.metadata.Year == "") r.metadata.Year = "";
			return r;
		}

		private void Lookup(object o)
		{
			ICDRipper audioSource = o as ICDRipper;
			int mbresults_count = 0; // have to cache results.Count, because it sometimes hangs in it, and we don't want UI thread to hang.
			string musicbrainzError = "";

			releases.RaiseListChangedEvents = false;

			cueSheet = new CUESheet(_config);
			cueSheet.OpenCD(audioSource);
			cueSheet.Action = CUEAction.Encode;

			this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " CTDB..."; });
			cueSheet.UseCUEToolsDB(true, "CUERipper 2.0.8: " + selectedDriveInfo.drive.ARName);
			cueSheet.CTDB.UploadHelper.onProgress += new EventHandler<Krystalware.UploadHelper.UploadProgressEventArgs>(UploadProgress);
			this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " AccurateRip..."; });
			cueSheet.UseAccurateRip();
			this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " MusicBrainz..."; });

			General.SetCUELine(cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);
			General.SetCUELine(cueSheet.Attributes, "REM", "COMMENT", _config.createEACLOG ? "ExactAudioCopy v0.99pb4" : audioSource.RipperVersion, true);

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
					releases.Add(CreateCUESheet(audioSource, release, null));
				}
				mbresults_count = results.Count;
			}
			catch (Exception ex)
			{
				System.Diagnostics.Trace.WriteLine(ex.Message);
				if (!(ex is MusicBrainzNotFoundException))
					musicbrainzError = ex.Message;
			}
			MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);


			FreedbHelper m_freedb = new FreedbHelper();

			m_freedb.UserName = "gchudov";
			m_freedb.Hostname = "gmail.com";
			m_freedb.ClientName = "CUERipper";
			m_freedb.Version = "2.0.8";
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
						ReleaseInfo r = CreateCUESheet(audioSource, null, cdEntry);
						releases.Add(r);
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
							releases.Add(r);
						}
					}
				}
			}
			catch (Exception ex)
			{
				System.Diagnostics.Trace.WriteLine(ex.Message);
			}

			if (releases.Count == 0)
			{
				releases.Add(CreateCUESheet(audioSource, null, null));
			}
			_workThread = null;
			if (musicbrainzError != "")
				musicbrainzError = musicbrainzError + ": ";
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
				releases.RaiseListChangedEvents = true;
				releases.ResetBindings();
				//bnComboBoxRelease.SelectedIndex = 0;
				toolStripStatusAr.Enabled = cueSheet.ArVerify.ARStatus == null;
				toolStripStatusAr.Text = cueSheet.ArVerify.ARStatus == null ? cueSheet.ArVerify.WorstTotal().ToString() : "";
				toolStripStatusAr.ToolTipText = "AccurateRip: " + (cueSheet.ArVerify.ARStatus ?? "found") + ".";
				toolStripStatusCTDB.Enabled = cueSheet.CTDB.DBStatus == null;
				toolStripStatusCTDB.Text = cueSheet.CTDB.DBStatus == null ? cueSheet.CTDB.Total.ToString() : "";
				toolStripStatusCTDB.ToolTipText = "CUETools DB: " + (cueSheet.CTDB.DBStatus ?? "found") + ".";
				toolStripStatusLabelMusicBrainz.Enabled = true;
				toolStripStatusLabelMusicBrainz.BorderStyle = mbresults_count > 0 ? Border3DStyle.SunkenInner : Border3DStyle.RaisedInner;
				toolStripStatusLabelMusicBrainz.Text = mbresults_count > 0 ? mbresults_count.ToString() : "";
				toolStripStatusLabelMusicBrainz.ToolTipText = "Musicbrainz: " + (mbresults_count > 0 ? mbresults_count.ToString() + " entries found." : (musicbrainzError + "click to submit."));
			});
		}

		private void UpdateDrive()
		{
			selectedDriveInfo = bnComboBoxDrives.SelectedItem as DriveInfo;
			if (selectedDriveInfo == null)
				return;

			toolStripStatusAr.Enabled = false;
			toolStripStatusAr.Text = "";
			toolStripStatusAr.ToolTipText = "";
			toolStripStatusCTDB.Enabled = false;
			toolStripStatusCTDB.Text = "";
			toolStripStatusCTDB.ToolTipText = "";
			toolStripStatusLabelMusicBrainz.Enabled = false;
			toolStripStatusLabelMusicBrainz.Text = "";
			toolStripStatusLabelMusicBrainz.ToolTipText = "";
			buttonGo.Enabled = false;
			listTracks.Items.Clear();
			releases.Clear();
			selectedRelease = null;
			bnComboBoxRelease.Enabled = false;
			bnComboBoxRelease.Text = "";
			if (selectedDriveInfo == null || selectedDriveInfo.drive == null)
			{
				selectedDriveInfo = null;
				return;
			}
			if (cueSheet != null)
			{
				cueSheet.Close();
				cueSheet = null;
			}
			try
			{
				selectedDriveInfo.drive.Close();
				selectedDriveInfo.drive.Open(selectedDriveInfo.drive.Path[0]);
				numericWriteOffset.Value = selectedDriveInfo.drive.DriveOffset;
			}
			catch (Exception ex)
			{
				numericWriteOffset.Value = selectedDriveInfo.drive.DriveOffset;
				//selectedDriveInfo.drive.Close();
				bnComboBoxRelease.Text = ex.Message;
				return;
			}
			if (selectedDriveInfo.drive.TOC.AudioTracks == 0)
			{
				bnComboBoxRelease.Text = "No audio tracks";
				return;
			}
			UpdateRelease();
			_workThread = new Thread(Lookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(selectedDriveInfo.drive);
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
			if (selectedRelease == null) return;
			if (e.Label != null && selectedDriveInfo.drive.TOC[e.Item + 1].IsAudio)
				selectedRelease.metadata.Tracks[e.Item].Title = e.Label;
			else
				e.CancelEdit = true;
		}

		private void contextMenuStripRelease_Opening(object sender, CancelEventArgs e)
		{
			if (selectedRelease == null) return;
			bool isVarious = false;
			for (int i = 0; i < selectedRelease.metadata.TOC.AudioTracks; i++)
				if (selectedRelease.metadata.Tracks[i].Artist != selectedRelease.metadata.Artist)
					isVarious = true;
			variousToolStripMenuItem.Enabled = selectedRelease.ImageKey == "freedb" && !isVarious;
			ReleaseInfo r = new ReleaseInfo(_config, selectedRelease.metadata.TOC);
			r.metadata.CopyMetadata(selectedRelease.metadata);
			fixEncodingToolStripMenuItem.Enabled = selectedRelease.ImageKey == "freedb" && r.metadata.FreedbToEncoding();
		}

		private void editToolStripMenuItem_Click(object sender, EventArgs e)
		{
			if (selectedRelease == null) return;
			frmProperties frm = new frmProperties();
			frm.CUE = selectedRelease.metadata;
			frm.ShowDialog();
			releases.ResetItem(bnComboBoxRelease.SelectedIndex);
			comboBoxOutputFormat_TextUpdate(sender, e);
		}

		private void variousToolStripMenuItem_Click(object sender, EventArgs e)
		{
			if (selectedRelease == null) return;
			selectedRelease.metadata.FreedbToVarious();
			UpdateRelease();
			releases.ResetItem(bnComboBoxRelease.SelectedIndex);
		}

		private void fixEncodingToolStripMenuItem_Click(object sender, EventArgs e)
		{
			if (selectedRelease == null) return;
			selectedRelease.metadata.FreedbToEncoding();
			UpdateRelease();
			releases.ResetItem(bnComboBoxRelease.SelectedIndex);
			comboBoxOutputFormat_TextUpdate(sender, e);
		}

		private void frmCUERipper_FormClosed(object sender, FormClosedEventArgs e)
		{
			SettingsWriter sw = new SettingsWriter("CUERipper", "settings.txt", Application.ExecutablePath);
			_config.Save(sw);
			sw.Save("DefaultLosslessFormat", _defaultLosslessFormat);
			sw.Save("DefaultLossyFormat", _defaultLossyFormat);
			sw.Save("DefaultHybridFormat", _defaultHybridFormat);
			//sw.Save("CreateEACLOG", _config.createEACLOG);
			//sw.Save("PreserveHTOA", _config.preserveHTOA);
			//sw.Save("CreateM3U", _config.createM3U);
			sw.Save("OutputAudioType", (int)SelectedOutputAudioType);
			sw.Save("ComboImage", bnComboBoxImage.SelectedIndex);
			sw.Save("PathFormat", bnComboBoxOutputFormat.Text);
			sw.Save("SecureMode", trackBarSecureMode.Value);
			sw.Save("OutputPathUseTemplates", bnComboBoxOutputFormat.Items.Count - OutputPathUseTemplates.Length);
			for (int iFormat = bnComboBoxOutputFormat.Items.Count - 1; iFormat >= OutputPathUseTemplates.Length; iFormat--)
				sw.Save(string.Format("OutputPathUseTemplate{0}", iFormat - OutputPathUseTemplates.Length), bnComboBoxOutputFormat.Items[iFormat].ToString());

			sw.Close();
		}

		private void listTracks_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			if (!selectedDriveInfo.drive.TOC[e.Item + 1].IsAudio)
				e.CancelEdit = true;
		}

		private string SelectedOutputAudioFormat
		{
			get
			{
				return selectedFormat == null ? "dummy" : selectedFormat.ToString();
			}
			set
			{
				foreach (FormatInfo fmt in formats)
					if (fmt.ToString() == value)
						bnComboBoxFormat.SelectedItem = fmt;
			}
		}

		private CUEToolsFormat SelectedOutputAudioFmt
		{
			get
			{
				if (selectedFormat == null)
					return null;
				return selectedFormat.fmt;
			}
		}

		private AudioEncoderType SelectedOutputAudioType
		{
			get
			{
				return (bnComboBoxLosslessOrNot.SelectedItem as BNComboBoxItem<AudioEncoderType>).Value;
			}
			set
			{
				foreach (BNComboBoxItem<AudioEncoderType> item in losslessOrNot)
					if (value == item.Value)
					{
						bnComboBoxLosslessOrNot.SelectedItem = item;
						return;
					}
				throw new Exception("invalid value");
			}
		}

		private void checkBoxEACMode_CheckedChanged(object sender, EventArgs e)
		{
			_config.createEACLOG = checkBoxEACMode.Checked;
		}

		private void comboBoxEncoder_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (SelectedOutputAudioFormat == null)
				return;
			CUEToolsUDC encoder = bnComboBoxEncoder.SelectedItem as CUEToolsUDC;
			if (encoder == null)
				return;
			if (SelectedOutputAudioFormat.StartsWith("lossy."))
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else if (SelectedOutputAudioType == AudioEncoderType.Lossless)
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else
				SelectedOutputAudioFmt.encoderLossy = encoder;

			string[] modes = encoder.SupportedModes;
			if (modes == null || modes.Length < 2)
			{
				trackBarEncoderMode.Visible = false;
				labelEncoderMode.Visible = false;
				labelEncoderMinMode.Visible = false;
				labelEncoderMaxMode.Visible = false;
			}
			else
			{
				trackBarEncoderMode.Maximum = modes.Length - 1;
				trackBarEncoderMode.Value = encoder.DefaultModeIndex == -1 ? modes.Length - 1 : encoder.DefaultModeIndex;
				labelEncoderMode.Text = encoder.default_mode;
				labelEncoderMinMode.Text = modes[0];
				labelEncoderMaxMode.Text = modes[modes.Length - 1];
				trackBarEncoderMode.Visible = true;
				labelEncoderMode.Visible = true;
				labelEncoderMinMode.Visible = true;
				labelEncoderMaxMode.Visible = true;
			}
		}

		private void trackBarEncoderMode_Scroll(object sender, EventArgs e)
		{
			CUEToolsUDC encoder = bnComboBoxEncoder.SelectedItem as CUEToolsUDC;
			string[] modes = encoder.SupportedModes;
			encoder.default_mode = modes[trackBarEncoderMode.Value];
			labelEncoderMode.Text = encoder.default_mode;
		}

		private void trackBarSecureMode_Scroll(object sender, EventArgs e)
		{
			string[] modes = new string[] { "Burst", "Secure", "Paranoid" };
			labelSecureMode.Text = modes[trackBarSecureMode.Value];
		}

		private void toolStripStatusLabelMusicBrainz_Click(object sender, EventArgs e)
		{
			if (selectedDriveInfo == null)
				return;
			System.Diagnostics.Process.Start("http://musicbrainz.org/bare/cdlookup.html?toc=" + selectedDriveInfo.drive.TOC.MusicBrainzTOC);
		}

		private void frmCUERipper_KeyDown(object sender, KeyEventArgs e)
		{
			if (_workThread == null && e.KeyCode == Keys.F5)
				UpdateDrives();
		}

		private void comboBoxOutputFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			comboBoxOutputFormat_TextUpdate(sender, e);
		}

		private void comboBoxOutputFormat_TextUpdate(object sender, EventArgs e)
		{
			if (selectedFormat == null) return;
			CUEStyle style = bnComboBoxImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE : CUEStyle.GapsAppended;
			txtOutputPath.Text = selectedRelease == null ? "" : selectedRelease.metadata.GenerateUniqueOutputPath(bnComboBoxOutputFormat.Text,
					style == CUEStyle.SingleFileWithCUE ? "." + selectedFormat.ToString() : ".cue", CUEAction.Encode, null);
		}

		private void bnComboBoxRelease_SelectedIndexChanged(object sender, EventArgs e)
		{
			UpdateRelease();
		}

		private void bnComboBoxDrives_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (_workThread == null)
				UpdateDrive();
		}

		private FormatInfo selectedFormat;

		private void bnComboBoxFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			selectedFormat = bnComboBoxFormat.SelectedItem as FormatInfo;

			encoders.Clear();
			if (SelectedOutputAudioFmt == null)
				return;

			switch (SelectedOutputAudioType)
			{
				case AudioEncoderType.Lossless:
					_defaultLosslessFormat = SelectedOutputAudioFormat;
					break;
				case AudioEncoderType.Lossy:
					_defaultLossyFormat = SelectedOutputAudioFormat;
					break;
				case AudioEncoderType.Hybrid:
					_defaultHybridFormat = SelectedOutputAudioFormat;
					break;
			}

			encoders.RaiseListChangedEvents = false;

			foreach (CUEToolsUDC encoder in _config.encoders)
				if (encoder.extension == SelectedOutputAudioFmt.extension)
				{
					if (SelectedOutputAudioType == AudioEncoderType.Lossless && !encoder.lossless)
						continue;
					if (SelectedOutputAudioType == AudioEncoderType.Lossy && (encoder.lossless && !selectedFormat.lossyWAV))
						continue;
					encoders.Add(encoder);
				}

			CUEToolsUDC select = SelectedOutputAudioFormat.StartsWith("lossy.") ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioType == AudioEncoderType.Lossless ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioFmt.encoderLossy;
			encoders.RaiseListChangedEvents = true;
			encoders.ResetBindings();
			bnComboBoxEncoder.SelectedItem = select;

			comboBoxOutputFormat_TextUpdate(sender, e);
		}

		private void bnComboBoxLosslessOrNot_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (bnComboBoxLosslessOrNot.SelectedItem == null) return;
			formats.Clear();
			formats.RaiseListChangedEvents = false;
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (SelectedOutputAudioType == AudioEncoderType.Lossless && !format.Value.allowLossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Hybrid) // && format.Key != "wv") TODO!!!!!
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossy && !format.Value.allowLossy)
					continue;
				formats.Add(new FormatInfo(format.Value, false));
			}
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (!format.Value.allowLossyWAV)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
					continue;
				formats.Add(new FormatInfo(format.Value, true));
			}
			string select = null;
			switch (SelectedOutputAudioType)
			{
				case AudioEncoderType.Lossless:
					select = _defaultLosslessFormat;
					break;
				case AudioEncoderType.Lossy:
					select = _defaultLossyFormat;
					break;
				case AudioEncoderType.Hybrid:
					select = _defaultHybridFormat;
					break;
			}
			formats.RaiseListChangedEvents = true;
			formats.ResetBindings();
			SelectedOutputAudioFormat = select;
		}

		private void bnComboBoxOutputFormat_TextChanged(object sender, EventArgs e)
		{
			comboBoxOutputFormat_TextUpdate(sender, e);
		}

		private void txtOutputPath_Enter(object sender, EventArgs e)
		{
			if (outputFormatVisible)
				return;
			outputFormatVisible = true;
			bnComboBoxOutputFormat.Visible = true;
			bnComboBoxOutputFormat.Focus();
			txtOutputPath.Enabled = false;
			txtOutputPath.Visible = false;
		}

		private void bnComboBoxOutputFormat_MouseLeave(object sender, EventArgs e)
		{
			//if (!outputFormatVisible)
			//    return;
			//outputFormatVisible = false;
			//bnComboBoxOutputFormat.Visible = false;
			//txtOutputPath.Enabled = true;
			//txtOutputPath.Visible = true;
		}

		private void bnComboBoxOutputFormat_DroppedDown(object sender, EventArgs e)
		{
			if (!outputFormatVisible || bnComboBoxOutputFormat.IsDroppedDown || bnComboBoxOutputFormat.Focused)
				return;
			outputFormatVisible = false;
			bnComboBoxOutputFormat.Visible = false;
			txtOutputPath.Enabled = true;
			txtOutputPath.Visible = true;
		}

		private void bnComboBoxOutputFormat_Leave(object sender, EventArgs e)
		{
			bnComboBoxOutputFormat_DroppedDown(sender, e);
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

	public class FormatInfo
	{
		public CUEToolsFormat fmt;
		public bool lossyWAV;

		public FormatInfo(CUEToolsFormat fmt, bool lossyWAV)
		{
			this.fmt = fmt;
			this.lossyWAV = lossyWAV;
		}

		public override string ToString()
		{
			return lossyWAV ? "lossy." + fmt.extension : fmt.extension;
		}

		public string DotExtension
		{
			get
			{
				return fmt.DotExtension;
			}
		}
	}

	public class ReleaseInfo
	{
		public CUESheet metadata;
		private string imageKey;
		public string ImageKey
		{
			get
			{
				return imageKey;
			}
			set
			{
				imageKey = value;
			}
		}

		public ReleaseInfo(CUEConfig config, CDImageLayout TOC)
		{
			metadata = new CUESheet(config);
			metadata.TOC = TOC;
		}

		public override string ToString()
		{
			return string.Format("{0}{1} - {2}", metadata.Year != "" ? metadata.Year + ": " : "", metadata.Artist, metadata.Title);
		}
	}

	public class DriveInfo
	{
		public ICDRipper drive;
		public string error;
		DirectoryInfo di;
		CUEControls.IIconManager iconMgr;

		public int ImageKey
		{
			get
			{				
				return iconMgr.GetIconIndex(di, true);
			}
		}

		public DriveInfo(CUEControls.IIconManager iconMgr, string path, ICDRipper drive)
		{
			this.iconMgr = iconMgr;
			this.di = new DirectoryInfo(path);
			this.drive = drive;
		}

		public DriveInfo(CUEControls.IIconManager iconMgr, string path, string error)
		{
			this.iconMgr = iconMgr;
			this.di = new DirectoryInfo(path);
			this.error = error;
		}

		public override string ToString()
		{
			return drive != null ? drive.Path : this.di.FullName + ": " + error;
		}
	}
}
