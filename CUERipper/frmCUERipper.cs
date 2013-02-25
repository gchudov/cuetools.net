using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;
using CUEControls;
using CUETools.AccurateRip;
using CUETools.CTDB;
using CUETools.Processor;
using CUETools.Processor.Settings;
using CUETools.Ripper;
using Freedb;
using CUETools.Codecs;
using System.Xml;
using System.Xml.Serialization;

namespace CUERipper
{
	public partial class frmCUERipper : Form
	{
		private Thread _workThread = null;
		private StartStop _startStop;
		private CUEConfig _config;
        private CUERipperConfig cueRipperConfig;
		private CUESheet cueSheet;
		private DriveInfo selectedDriveInfo;
		private string _pathOut;
		private CUEControls.ShellIconMgr m_icon_mgr;
        private bool testAndCopy = false;
		internal CUERipperData data = new CUERipperData();
        public readonly static XmlSerializerNamespaces xmlEmptyNamespaces = new XmlSerializerNamespaces(new XmlQualifiedName[] { XmlQualifiedName.Empty });
        public readonly static XmlWriterSettings xmlEmptySettings = new XmlWriterSettings { Indent = true, OmitXmlDeclaration = true };

		public frmCUERipper()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_startStop = new StartStop();
            cueRipperConfig = new CUERipperConfig();
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
			"%music%\\%artist%\\[%year% - ]%album%\\%artist% - %album%[ '('disc %discnumberandname%')'].cue",
			"%music%\\%artist%\\[%year% - ]%album%[ '('disc %discnumberandname%')'][' ('%releasedateandlabel%')'][' ('%unique%')']\\%artist% - %album%.cue"
		};

		//// Calculate the graphics path that representing the figure in the bitmap 
		//// excluding the transparent color which is the top left pixel.
		//private static GraphicsPath CalculateControlGraphicsPath(Bitmap bitmap, Color colorTransparent)
		//{
		//    // Create GraphicsPath for our bitmap calculation
		//    GraphicsPath graphicsPath = new GraphicsPath();

		//    // Use the top left pixel as our transparent color
		//    colorTransparent = bitmap.GetPixel(0, 0); 

		//    // This is to store the column value where an opaque pixel is first found.
		//    // This value will determine where we start scanning for trailing 
		//    // opaque pixels.
		//    int colOpaquePixel = 0;

		//    // Go through all rows (Y axis)
		//    for (int row = 0; row < bitmap.Height; row++)
		//    {
		//        // Reset value
		//        colOpaquePixel = 0;

		//        // Go through all columns (X axis)
		//        for (int col = 0; col < bitmap.Width; col++)
		//        {
		//            // If this is an opaque pixel, mark it and search 
		//            // for anymore trailing behind
		//            if (bitmap.GetPixel(col, row) != colorTransparent)
		//            {
		//                // Opaque pixel found, mark current position
		//                colOpaquePixel = col;
		//                // Create another variable to set the current pixel position
		//                int colNext = col;
		//                // Starting from current found opaque pixel, search for 
		//                // anymore opaque pixels trailing behind, until a transparent
		//                // pixel is found or minimum width is reached
		//                for (colNext = colOpaquePixel; colNext < bitmap.Width; colNext++)
		//                    if (bitmap.GetPixel(colNext, row) == colorTransparent)
		//                        break;
		//                // Form a rectangle for line of opaque pixels found and 
		//                // add it to our graphics path
		//                graphicsPath.AddRectangle(new Rectangle(colOpaquePixel,
		//                                           row, colNext - colOpaquePixel, 1));
		//                // No need to scan the line of opaque pixels just found
		//                col = colNext;
		//            }
		//        }
		//    }

		//    // Return calculated graphics path
		//    return graphicsPath;
		//}

		//private static void CreateControlRegion(Button button, Bitmap bitmap, Color colorTransparent)
		//{
		//    // Return if control and bitmap are null
		//    if (button == null || bitmap == null)
		//        return;

		//    // Set our control's size to be the same as the bitmap
		//    button.Width = bitmap.Width;
		//    button.Height = bitmap.Height;

		//    // Do not show button text
		//    button.Text = "";

		//    // Change cursor to hand when over button
		//    button.Cursor = Cursors.Hand;

		//    // Set background image of button
		//    button.BackgroundImage = bitmap;

		//    // Calculate the graphics path based on the bitmap supplied
		//    GraphicsPath graphicsPath = CalculateControlGraphicsPath(bitmap, colorTransparent);

		//    // Apply new region

		//    button.Region = new Region(graphicsPath);
		//}

		private void frmCUERipper_Load(object sender, EventArgs e)
		{
			//buttonTrackMetadata.Parent = listTracks;
			//buttonTrackMetadata.ImageList = null;
			//CreateControlRegion(buttonTrackMetadata, new Bitmap(imageListChecked.Images[0]), imageListChecked.TransparentColor);
			//CreateControlRegion(buttonTrackMetadata, Properties.Resources.cdrepair, Color.White);

			SettingsReader sr = new SettingsReader("CUERipper", "settings.txt", Application.ExecutablePath);
			_config.Load(sr);
			//_config.createEACLOG = sr.LoadBoolean("CreateEACLOG") ?? true;
			//_config.preserveHTOA = sr.LoadBoolean("PreserveHTOA") ?? false;
			//_config.createM3U = sr.LoadBoolean("CreateM3U") ?? true;

			bindingSourceCR.DataSource = data;
			bnComboBoxDrives.ImageList = m_icon_mgr.ImageList;
			bnComboBoxFormat.ImageList = m_icon_mgr.ImageList;

            try
            {
                using (TextReader reader = new StringReader(sr.Load("CUERipper")))
                    cueRipperConfig = CUERipperConfig.serializer.Deserialize(reader) as CUERipperConfig;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(ex.Message);
            }


            SetupControls();

			int iFormat, nFormats = sr.LoadInt32("OutputPathUseTemplates", 0, 10) ?? 0;
			for (iFormat = 0; iFormat < OutputPathUseTemplates.Length; iFormat++)
				bnComboBoxOutputFormat.Items.Add(OutputPathUseTemplates[iFormat]);
			for (iFormat = nFormats - 1; iFormat >= 0; iFormat--)
				bnComboBoxOutputFormat.Items.Add(sr.Load(string.Format("OutputPathUseTemplate{0}", iFormat)) ?? "");

			bnComboBoxOutputFormat.Text = sr.Load("PathFormat") ?? "%music%\\%artist%\\[%year% - ]%album%\\%artist% - %album%.cue";
			SelectedOutputAudioType = (AudioEncoderType?)sr.LoadInt32("OutputAudioType", null, null) ?? AudioEncoderType.Lossless;
			bnComboBoxImage.SelectedIndex = sr.LoadInt32("ComboImage", 0, bnComboBoxImage.Items.Count - 1) ?? 0;
			trackBarSecureMode.Value = sr.LoadInt32("SecureMode", 0, trackBarSecureMode.Maximum - 1) ?? 1;
			trackBarSecureMode_Scroll(this, new EventArgs());
            this.checkBoxTestAndCopy.Checked = this.testAndCopy = sr.LoadBoolean("TestAndCopy") ?? this.testAndCopy;

			Size SizeIncrement = new Size(sr.LoadInt32("WidthIncrement", 0, null) ?? 0, sr.LoadInt32("HeightIncrement", 0, null) ?? 0);
			Size = MinimumSize + SizeIncrement;
			Left -= SizeIncrement.Width / 2;
			Top -= SizeIncrement.Height / 2;
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
				{
					if (_workThread == null)
						UpdateDrives();
				}
			}
			base.WndProc(ref m);
		}

		private void DrivesLookup(object o)
		{
			// Lookup
			data.Drives.RaiseListChangedEvents = false;
			foreach (char drive in CDDrivesList.DrivesAvailable())
			{
				this.BeginInvoke((MethodInvoker)(() =>
					toolStripStatusLabel1.Text = Properties.Resources.DetectingDrives + ": " + drive + ":\\..."));
				ICDRipper reader = Activator.CreateInstance(CUEProcessorPlugins.ripper) as ICDRipper;
				try
				{
					reader.Open(drive);
				}
				catch (Exception ex)
				{
					System.Diagnostics.Trace.WriteLine(ex.Message);
				}
				reader.Close();
				if (reader.ARName != null)
				{
                    int driveOffset;
                    if (cueRipperConfig.DriveOffsets.ContainsKey(reader.ARName))
                        reader.DriveOffset = cueRipperConfig.DriveOffsets[reader.ARName];
                    else if (AccurateRipVerify.FindDriveReadOffset(reader.ARName, out driveOffset))
						reader.DriveOffset = driveOffset;
					else
						reader.DriveOffset = 0;
				}
				data.Drives.Add(new DriveInfo(m_icon_mgr, drive + ":\\", reader));
			}
			this.BeginInvoke((MethodInvoker)delegate()
			{
				data.Drives.RaiseListChangedEvents = true;
				data.Drives.ResetBindings();
				for(int i = 0; i < bnComboBoxDrives.Items.Count; i++)
					if ((bnComboBoxDrives.Items[i] as DriveInfo).Path == cueRipperConfig.DefaultDrive)
						bnComboBoxDrives.SelectedIndex = i;
				_workThread = null;
				SetupControls();
				if (data.Drives.Count == 0)
					bnComboBoxDrives.Text = Properties.Resources.NoDrives;
				else
					UpdateDrive();
			});
		}

		private void UpdateDrives()
		{
			buttonGo.Enabled = false;
			foreach (DriveInfo driveInfo in data.Drives)
				if (driveInfo.drive != null)
					driveInfo.drive.Close();
			data.Drives.Clear();
			listTracks.Items.Clear();
			data.Releases.Clear();
			data.selectedRelease = null;
            ResetAlbumArt();
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

		private void SetupControls()
		{
			bool running = _workThread != null;

			bnComboBoxOutputFormat.Visible = outputFormatVisible;
			txtOutputPath.Visible = !outputFormatVisible;
			txtOutputPath.Enabled = !running && !outputFormatVisible;
			bnComboBoxRelease.Enabled = !running && data.Releases.Count > 0;
			bnComboBoxDrives.Enabled = !running && data.Drives.Count > 0;
			bnComboBoxOutputFormat.Enabled =			
			listTracks.Enabled =
			listMetadata.Enabled =
			groupBoxSettings.Enabled = !running;
			buttonGo.Enabled = !running && data.selectedRelease != null;
			buttonPause.Visible = buttonPause.Enabled = buttonAbort.Visible = buttonAbort.Enabled = running;
			buttonGo.Visible = !running;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			progressBarErrors.Value = 0;
			progressBarCD.Value = 0;

			buttonTracks.Enabled = data.selectedRelease != null && !running;
			buttonMetadata.Enabled = data.selectedRelease != null && !running;
			buttonFreedbSubmit.Enabled = data.selectedRelease != null && !running;
			buttonVA.Enabled = data.selectedRelease != null && !running &&
				data.selectedRelease.ImageKey == "freedb" && !data.selectedRelease.metadata.IsVarious() && (new CUEMetadata(data.selectedRelease.metadata)).FreedbToVarious();
			buttonEncoding.Enabled = data.selectedRelease != null && !running &&
				data.selectedRelease.ImageKey == "freedb" && (new CUEMetadata(data.selectedRelease.metadata)).FreedbToEncoding();
			buttonReload.Enabled = data.selectedRelease != null && !running;
            buttonSettings.Enabled = !running;
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
					this.BeginInvoke((MethodInvoker)(()=> toolStripStatusLabel1.Text = Properties.Resources.PausedMessage +  "..."));
					Monitor.Wait(_startStop);
				}
			}
            if (backgroundWorkerArtwork.IsBusy && backgroundWorkerArtwork.CancellationPending)
            {
                throw new StopException();
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
                if (this.testAndCopy)
                    cueSheet.TestBeforeCopy();
                else
                    cueSheet.ArTestVerify = null;

                cueSheet.Go();
                cueSheet.CTDB.Submit(
					(int)cueSheet.ArVerify.WorstConfidence() + 1,
					audioSource.CorrectionQuality == 0 ? 0 :
					100 - (int)(7 * Math.Log(audioSource.ErrorsCount + 1)), // ErrorsCount==1 ~= 95, ErrorsCount==max ~= 5;
					cueSheet.Metadata.Artist,
					cueSheet.Metadata.Title,
					cueSheet.TOC.Barcode);
				bool canFix = false;
				if (cueSheet.CTDB.QueryExceptionStatus == WebExceptionStatus.Success && audioSource.ErrorsCount != 0)
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
					data.Releases.Clear();
					data.selectedRelease = null;
					bnComboBoxRelease.Text = ex.Message;
				});
			}

			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate()
			{
				SetupControls();
				UpdateOutputPath();
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

            if (currentAlbumArt < albumArt.Count)
            {
                data.selectedRelease.metadata.AlbumArt.Clear();
                data.selectedRelease.metadata.AlbumArt.Add(albumArt[currentAlbumArt].meta);
                cueSheet.AddAlbumArt(albumArt[currentAlbumArt].contents);
            }            

			data.selectedRelease.metadata.Save();

			cueSheet.CopyMetadata(data.selectedRelease.metadata);
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
            if (cueSheet.Metadata.Comment == "")
                cueSheet.Metadata.Comment = selectedDriveInfo.drive.RipperVersion;
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

		private void ResizeList(ListView list, ColumnHeader title)
		{
			int colSum = 0;
			foreach (ColumnHeader col in list.Columns)
			{
				colSum += col.Width + SystemInformation.BorderSize.Width;
			}

			title.Width += list.Width - colSum - 2 * SystemInformation.BorderSize.Width - SystemInformation.VerticalScrollBarWidth;
		}

		private void UpdateRelease()
		{
			data.selectedRelease = bnComboBoxRelease.SelectedItem as CUEMetadataEntry;
			UpdateOutputPath();
			listTracks.BeginUpdate();
			listMetadata.BeginUpdate();
			listTracks.Items.Clear();
			listMetadata.Items.Clear();
			if (!data.metadataMode)
			{
				listTracks.Visible = true;
				listMetadata.Visible = false;
				if (data.selectedRelease != null)
				{
					columnHeaderArtist.Width = data.selectedRelease.metadata.IsVarious() ? 120 : 0;
					for (int i = 1; i <= selectedDriveInfo.drive.TOC.TrackCount; i++)
					{
						string title = "Data track";
						string artist = "";
						if (selectedDriveInfo.drive.TOC[i].IsAudio)
						{
							title = data.selectedRelease.metadata.Tracks[i - selectedDriveInfo.drive.TOC.FirstAudio].Title;
							artist = data.selectedRelease.metadata.Tracks[i - selectedDriveInfo.drive.TOC.FirstAudio].Artist;
						}
						listTracks.Items.Add(new ListViewItem(new string[] { 
							title,
							selectedDriveInfo.drive.TOC[i].Number.ToString(), 
							artist,
							selectedDriveInfo.drive.TOC[i].StartMSF, 
							selectedDriveInfo.drive.TOC[i].LengthMSF }));
					}
				}
			}
			else if (data.metadataTrack < 0)
			{
				listTracks.Visible = false;
				listMetadata.Visible = true;
				if (data.selectedRelease != null)
				{
					PropertyDescriptorCollection props = TypeDescriptor.GetProperties(data.selectedRelease.metadata);
                    PropertyDescriptorCollection sortedprops = props.Sort(new string[] { "Artist", "Title", "Genre", "Year", "DiscNumber", "TotalDiscs", "DiscName", "Label", "LabelNo", "Country", "ReleaseDate" });
					foreach (PropertyDescriptor p in sortedprops)
						if (p.Name != "Tracks" && p.Name != "AlbumArt" && p.Name != "Id" && !p.Attributes.Contains(new System.Xml.Serialization.XmlIgnoreAttribute()))
							listMetadata.Items.Add(new ListViewItem(new string[] { p.GetValue(data.selectedRelease.metadata).ToString(), p.Name }));
				}
			}
			else
			{
				listTracks.Visible = false;
				listMetadata.Visible = true;
				if (data.selectedRelease != null)
				{
					CUETrackMetadata track = data.selectedRelease.metadata.Tracks[data.metadataTrack];
					PropertyDescriptorCollection props = TypeDescriptor.GetProperties(track);
					props = props.Sort(new string[] { "ISRC", "Title", "Artist" });
					ListViewItem lvItem = new ListViewItem(new string[] { (data.metadataTrack + 1).ToString(), "Number" });
					lvItem.ForeColor = SystemColors.GrayText;
					listMetadata.Items.Add(lvItem);
					foreach (PropertyDescriptor p in props)
					{
						lvItem = new ListViewItem(new string[] { p.GetValue(track).ToString(), p.Name });
						if (p.Name == "ISRC")
							lvItem.ForeColor = SystemColors.GrayText;
						listMetadata.Items.Add(lvItem);
					}
				}
			}
			ResizeList(listTracks, Title);
			ResizeList(listMetadata, columnHeaderValue);
			listTracks.EndUpdate();
			listMetadata.EndUpdate();

            UpdateAlbumArt(true);
			SetupControls();
		}

		//private void MusicBrainz_LookupProgress(object sender, XmlRequestEventArgs e)
		//{
		//    CheckStop();
		//    //_progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
		//    //_progress.input = e.Uri.ToString();
		//    this.BeginInvoke((MethodInvoker)delegate()
		//    {
		//        toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " " + (e == null ? "FreeDB" : "MusicBrainz") + "...";
		//        toolStripProgressBar1.Value = (100 + 2 * toolStripProgressBar1.Value) / 3;
		//    });
		//}

		private void FreeDB_LookupProgress(object sender)
		{
			CheckStop();
			//_progress.percentDisk = (1.0 + _progress.percentDisk) / 2;
			//_progress.input = e.Uri.ToString();
            string text = Properties.Resources.LookingUpVia + " FreeDB..." + (sender is string ? " " + (sender as string) : "");
			this.BeginInvoke((MethodInvoker)delegate()
			{
				toolStripStatusLabel1.Text = text;
				toolStripProgressBar1.Value = (100 + 2 * toolStripProgressBar1.Value) / 3;
			});
		}

		private CUEMetadataEntry CreateCUESheet(ICDRipper audioSource, CTDBResponseMeta release)
		{
			CUEMetadataEntry entry = new CUEMetadataEntry(audioSource.TOC, release.source);
			entry.metadata.FillFromCtdb(release, entry.TOC.FirstAudio - 1);
			return entry;
		}

		//private CUEMetadataEntry CreateCUESheet(ICDRipper audioSource, Release release)
		//{
		//    CUEMetadataEntry entry = new CUEMetadataEntry(audioSource.TOC, "musicbrainz");
		//    entry.metadata.FillFromMusicBrainz(release, entry.TOC.FirstAudio - 1);
		//    return entry;
		//}

		private CUEMetadataEntry CreateCUESheet(ICDRipper audioSource, CDEntry cdEntry)
		{
			CUEMetadataEntry entry = new CUEMetadataEntry(audioSource.TOC, "freedb");
			entry.metadata.FillFromFreedb(cdEntry, entry.TOC.FirstAudio - 1);
			return entry;
		}

		private CUEMetadataEntry CreateCUESheet(ICDRipper audioSource)
		{
			CUEMetadataEntry entry = new CUEMetadataEntry(audioSource.TOC, "local");
			entry.metadata.Artist = "Unknown Artist";
			entry.metadata.Title = "Unknown Title";
			for (int i = 0; i < entry.TOC.AudioTracks; i++)
			{
				entry.metadata.Tracks[i].Title = string.Format("Track {0:00}", i + 1);
				entry.metadata.Tracks[i].Artist = entry.metadata.Artist;
			}
			return entry;
		}

		private bool loadAllMetadata = false;

        private void Lookup(object o)
        {
            ICDRipper audioSource = o as ICDRipper;
            int mbresults_count = 0; // have to cache results.Count, because it sometimes hangs in it, and we don't want UI thread to hang.
            string musicbrainzError = "";

            data.Releases.RaiseListChangedEvents = false;

            cueSheet = new CUESheet(_config);
            cueSheet.OpenCD(audioSource);
            cueSheet.Action = CUEAction.Encode;

            this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " CTDB..."; });
            cueSheet.UseCUEToolsDB("CUERipper " + CUESheet.CUEToolsVersion, selectedDriveInfo.drive.ARName, false, loadAllMetadata ? CTDBMetadataSearch.Extensive : _config.advanced.metadataSearch);
            cueSheet.CTDB.UploadHelper.onProgress += new EventHandler<Krystalware.UploadHelper.UploadProgressEventArgs>(UploadProgress);
            this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " AccurateRip..."; });
            cueSheet.UseAccurateRip();

            General.SetCUELine(cueSheet.Attributes, "REM", "DISCID", AccurateRipVerify.CalculateCDDBId(audioSource.TOC), false);

            try
            {
                CUEMetadata cache = CUEMetadata.Load(audioSource.TOC.TOCID);
                if (cache != null)
                    data.Releases.Add(new CUEMetadataEntry(cache, audioSource.TOC, "local"));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(ex.Message);
            }

            foreach (var ctdbMeta in cueSheet.CTDB.Metadata)
            {
                data.Releases.Add(CreateCUESheet(audioSource, ctdbMeta));
            }

            if (data.Releases.Count == 0 || loadAllMetadata)
            {
                loadAllMetadata = false;

                //this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " MusicBrainz..."; });

                //ReleaseQueryParameters p = new ReleaseQueryParameters();
                //p.DiscId = audioSource.TOC.MusicBrainzId;
                //Query<Release> results = Release.Query(p);
                //MusicBrainzService.Proxy = _config.GetProxy();
                //MusicBrainzService.XmlRequest += new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);

                //try
                //{
                //    foreach (Release release in results)
                //    {
                //        release.GetEvents();
                //        release.GetTracks();
                //        data.Releases.Add(CreateCUESheet(audioSource, release));
                //    }
                //    mbresults_count = results.Count;
                //}
                //catch (Exception ex)
                //{
                //    System.Diagnostics.Trace.WriteLine(ex.Message);
                //    if (!(ex is MusicBrainzNotFoundException))
                //        musicbrainzError = ex.Message;
                //}
                //MusicBrainzService.Proxy = null;
                //MusicBrainzService.XmlRequest -= new EventHandler<XmlRequestEventArgs>(MusicBrainz_LookupProgress);

                this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " Freedb..."; });

                FreedbHelper m_freedb = new FreedbHelper();
                m_freedb.Proxy = _config.GetProxy();
                m_freedb.UserName = _config.advanced.FreedbUser;
                m_freedb.Hostname = _config.advanced.FreedbDomain;
                m_freedb.ClientName = "CUERipper";
                m_freedb.Version = CUESheet.CUEToolsVersion;
                m_freedb.SetDefaultSiteAddress(Properties.Settings.Default.MAIN_FREEDB_SITEADDRESS);

                QueryResult queryResult;
                QueryResultCollection coll;
                string code = string.Empty;
                try
                {
                    FreeDB_LookupProgress(this);
                    code = m_freedb.Query(AccurateRipVerify.CalculateCDDBQuery(audioSource.TOC), out queryResult, out coll);
                    if (code == FreedbHelper.ResponseCodes.CODE_200)
                    {
                        bool duplicate = false;
                        foreach (var ctdbMeta in cueSheet.CTDB.Metadata)
                            if (ctdbMeta.source == "freedb" && ctdbMeta.id == queryResult.Category + "/" + queryResult.Discid)
                                duplicate = true;
                        if (!duplicate)
                        {
                            FreeDB_LookupProgress(queryResult.Category + "/" + queryResult.Discid);
                            CDEntry cdEntry;
                            code = m_freedb.Read(queryResult, out cdEntry);
                            if (code == FreedbHelper.ResponseCodes.CODE_210)
                            {
                                CUEMetadataEntry r = CreateCUESheet(audioSource, cdEntry);
                                data.Releases.Add(r);
                            }
                        }
                    }
                    else
                        if (code == FreedbHelper.ResponseCodes.CODE_210 ||
                            code == FreedbHelper.ResponseCodes.CODE_211)
                        {
                            foreach (QueryResult qr in coll)
                            {
                                bool duplicate = false;
                                foreach (var ctdbMeta in cueSheet.CTDB.Metadata)
                                    if (ctdbMeta.source == "freedb" && ctdbMeta.id == qr.Category + "/" + qr.Discid)
                                        duplicate = true;
                                if (!duplicate)
                                {
                                    CDEntry cdEntry;
                                    FreeDB_LookupProgress(qr.Category + "/" + qr.Discid);
                                    code = m_freedb.Read(qr, out cdEntry);
                                    if (code == FreedbHelper.ResponseCodes.CODE_210)
                                    {
                                        CUEMetadataEntry r = CreateCUESheet(audioSource, cdEntry);
                                        data.Releases.Add(r);
                                    }
                                }
                            }
                        }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(ex.Message);
                }
            }

            if (data.Releases.Count == 0)
            {
                data.Releases.Add(CreateCUESheet(audioSource));
            }
            _workThread = null;
            if (musicbrainzError != "")
                musicbrainzError = musicbrainzError + ": ";
            while (backgroundWorkerArtwork.IsBusy)
            {
                Thread.Sleep(100);
            }
            this.BeginInvoke((MethodInvoker)delegate()
            {
                SetupControls();
                data.Releases.RaiseListChangedEvents = true;
                data.Releases.ResetBindings();
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
                if (_config.advanced.coversSearch != CUEConfigAdvanced.CTDBCoversSearch.None)
                    backgroundWorkerArtwork.RunWorkerAsync(new BackgroundWorkerArtworkArgs() { cueSheet = cueSheet, meta = data.selectedRelease });
            });
        }

		private void UpdateDrive()
		{
			if (bnComboBoxDrives.SelectedItem as DriveInfo == null)
				return;

			if (selectedDriveInfo != null)
				selectedDriveInfo.drive.Close();

			selectedDriveInfo = bnComboBoxDrives.SelectedItem as DriveInfo;
            cueRipperConfig.DefaultDrive = selectedDriveInfo.Path;

			toolStripStatusAr.Enabled = false;
			toolStripStatusAr.Text = "";
			toolStripStatusAr.ToolTipText = "";
			toolStripStatusCTDB.Enabled = false;
			toolStripStatusCTDB.Text = "";
			toolStripStatusCTDB.ToolTipText = "";
			toolStripStatusLabelMusicBrainz.Enabled = false;
			toolStripStatusLabelMusicBrainz.Text = "";
			toolStripStatusLabelMusicBrainz.ToolTipText = "";
			listTracks.Items.Clear();
			data.Releases.Clear();
			data.selectedRelease = null;
            ResetAlbumArt();
            bnComboBoxRelease.Enabled = false;
			bnComboBoxRelease.Text = "";
			if (selectedDriveInfo == null)
			{
				SetupControls();
				return;
			}
			if (cueSheet != null)
			{
				cueSheet.Close();
				cueSheet = null;
			}

            numericWriteOffset.Value = selectedDriveInfo.drive.DriveOffset;
			try
			{
				selectedDriveInfo.drive.Open(selectedDriveInfo.drive.Path[0]);
			}
			catch (Exception ex)
			{
				selectedDriveInfo.drive.Close();
				bnComboBoxRelease.Text = ex.Message;
				SetupControls();
				return;
			}
			// cannot use data.Drives.ResetItem(bnComboBoxDrives.SelectedIndex); - causes recursion
			UpdateRelease();
			_workThread = new Thread(Lookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(selectedDriveInfo.drive);
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
                if (listTracks.FocusedItem != null && listTracks.FocusedItem.Index + 1 < listTracks.Items.Count)// && e.Label != null)
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
			if (data.selectedRelease == null) return;
			if (e.Label != null && selectedDriveInfo.drive.TOC[e.Item + 1].IsAudio)
				data.selectedRelease.metadata.Tracks[e.Item].Title = e.Label;
			else
				e.CancelEdit = true;
		}

		private void frmCUERipper_FormClosed(object sender, FormClosedEventArgs e)
		{
			SettingsWriter sw = new SettingsWriter("CUERipper", "settings.txt", Application.ExecutablePath);
			_config.Save(sw);
			//sw.Save("CreateEACLOG", _config.createEACLOG);
			//sw.Save("PreserveHTOA", _config.preserveHTOA);
			//sw.Save("CreateM3U", _config.createM3U);
			sw.Save("OutputAudioType", (int)SelectedOutputAudioType);
			sw.Save("ComboImage", bnComboBoxImage.SelectedIndex);
			sw.Save("PathFormat", bnComboBoxOutputFormat.Text);
			sw.Save("SecureMode", trackBarSecureMode.Value);
			sw.Save("OutputPathUseTemplates", bnComboBoxOutputFormat.Items.Count - OutputPathUseTemplates.Length);
            sw.Save("TestAndCopy", this.testAndCopy);
			var SizeIncrement = Size - MinimumSize;
			sw.Save("WidthIncrement", SizeIncrement.Width);
			sw.Save("HeightIncrement", SizeIncrement.Height);
			for (int iFormat = bnComboBoxOutputFormat.Items.Count - 1; iFormat >= OutputPathUseTemplates.Length; iFormat--)
				sw.Save(string.Format("OutputPathUseTemplate{0}", iFormat - OutputPathUseTemplates.Length), bnComboBoxOutputFormat.Items[iFormat].ToString());

            using (TextWriter tw = new StringWriter())
            using (XmlWriter xw = XmlTextWriter.Create(tw, xmlEmptySettings))
            {
                CUERipperConfig.serializer.Serialize(xw, cueRipperConfig, xmlEmptyNamespaces);
                sw.SaveText("CUERipper", tw.ToString());
            }

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
				foreach (FormatInfo fmt in data.Formats)
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
				return (bnComboBoxLosslessOrNot.SelectedItem as ImgComboBoxItem<AudioEncoderType>).Value;
			}
			set
			{
				foreach (ImgComboBoxItem<AudioEncoderType> item in data.LosslessOrNot)
					if (value == item.Value)
					{
						bnComboBoxLosslessOrNot.SelectedItem = item;
						return;
					}
				throw new Exception("invalid value");
			}
		}

		private void bnComboBoxEncoder_SelectedValueChanged(object sender, EventArgs e)
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
			{
				loadAllMetadata = true;
				UpdateDrives();
			}
		}

		private void UpdateOutputPath()
		{
			if (selectedFormat == null) return;
			if (data.selectedRelease == null)
			{
				txtOutputPath.Text = "";
				return;
			}
			CUEStyle style = bnComboBoxImage.SelectedIndex == 0 ? CUEStyle.SingleFileWithCUE : CUEStyle.GapsAppended;
			CUESheet sheet = new CUESheet(_config);
			sheet.TOC = selectedDriveInfo.drive.TOC;
			sheet.CopyMetadata(data.selectedRelease.metadata);
			txtOutputPath.Text = sheet.GenerateUniqueOutputPath(bnComboBoxOutputFormat.Text,
					style == CUEStyle.SingleFileWithCUE ? "." + selectedFormat.ToString() : ".cue", CUEAction.Encode, null);
		}

		private void bnComboBoxRelease_SelectedValueChanged(object sender, EventArgs e)
		{
			UpdateRelease();
		}

		private void bnComboBoxDrives_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (_workThread == null)
				UpdateDrive();
		}

		private FormatInfo selectedFormat;

		private void bnComboBoxFormat_SelectedValueChanged(object sender, EventArgs e)
		{
			selectedFormat = bnComboBoxFormat.SelectedItem as FormatInfo;

			data.Encoders.Clear();
			if (SelectedOutputAudioFmt == null)
				return;

			switch (SelectedOutputAudioType)
			{
				case AudioEncoderType.Lossless:
                    cueRipperConfig.DefaultLosslessFormat = SelectedOutputAudioFormat;
					break;
				case AudioEncoderType.Lossy:
                    cueRipperConfig.DefaultLossyFormat = SelectedOutputAudioFormat;
					break;
				case AudioEncoderType.Hybrid:
                    cueRipperConfig.DefaultHybridFormat = SelectedOutputAudioFormat;
					break;
			}

			data.Encoders.RaiseListChangedEvents = false;

			foreach (CUEToolsUDC encoder in _config.encoders)
				if (encoder.extension == SelectedOutputAudioFmt.extension)
				{
					if (SelectedOutputAudioType == AudioEncoderType.Lossless && !encoder.lossless)
						continue;
					if (SelectedOutputAudioType == AudioEncoderType.Lossy && (encoder.lossless && !selectedFormat.lossyWAV))
						continue;
					data.Encoders.Add(encoder);
				}

			CUEToolsUDC select = SelectedOutputAudioFormat.StartsWith("lossy.") ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioType == AudioEncoderType.Lossless ? SelectedOutputAudioFmt.encoderLossless
				: SelectedOutputAudioFmt.encoderLossy;
			data.Encoders.RaiseListChangedEvents = true;
			data.Encoders.ResetBindings();
			bnComboBoxEncoder.SelectedItem = select;

			UpdateOutputPath();
		}

		private void bnComboBoxImage_SelectedValueChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
		}

		private void bnComboBoxLosslessOrNot_SelectedValueChanged(object sender, EventArgs e)
		{
			if (bnComboBoxLosslessOrNot.SelectedItem == null) return;
			data.Formats.Clear();
			data.Formats.RaiseListChangedEvents = false;
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (SelectedOutputAudioType == AudioEncoderType.Lossless && !format.Value.allowLossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Hybrid) // && format.Key != "wv") TODO!!!!!
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossy && !format.Value.allowLossy)
					continue;
				data.Formats.Add(new FormatInfo(format.Value, false));
			}
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (!format.Value.allowLossyWAV)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
					continue;
				data.Formats.Add(new FormatInfo(format.Value, true));
			}
			string select = null;
			switch (SelectedOutputAudioType)
			{
				case AudioEncoderType.Lossless:
                    select = cueRipperConfig.DefaultLosslessFormat;
					break;
				case AudioEncoderType.Lossy:
                    select = cueRipperConfig.DefaultLossyFormat;
					break;
				case AudioEncoderType.Hybrid:
                    select = cueRipperConfig.DefaultHybridFormat;
					break;
			}
			data.Formats.RaiseListChangedEvents = true;
			data.Formats.ResetBindings();
			SelectedOutputAudioFormat = select;
		}

		private void bnComboBoxOutputFormat_TextChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
		}

		private void txtOutputPath_Enter(object sender, EventArgs e)
		{
			if (outputFormatVisible)
				return;
			txtOutputPath.Enabled = false;
			txtOutputPath.Visible = false;
			outputFormatVisible = true;
			bnComboBoxOutputFormat.Visible = true;
			bnComboBoxOutputFormat.Focus();
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
			if (!outputFormatVisible || bnComboBoxOutputFormat.DroppedDown || ActiveControl == bnComboBoxOutputFormat)
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

		private void listMetadata_Click(object sender, EventArgs e)
		{
			listMetadata.FocusedItem.BeginEdit();
		}

		private void listMetadata_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			if (data.selectedRelease == null || !data.metadataMode)
			{
				e.CancelEdit = true;
			}
			else if (data.metadataTrack < 0)
			{
			}
			else
			{
				if (listMetadata.Items[e.Item].SubItems[1].Text == "ISRC" ||
					listMetadata.Items[e.Item].SubItems[1].Text == "Number")
				{
					e.CancelEdit = true;
				}
			}
		}

		private void listMetadata_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			if (data.selectedRelease == null || e.Label == null || !data.metadataMode)
			{
				e.CancelEdit = true;
			}
			else if (data.metadataTrack < 0)
			{
				PropertyDescriptorCollection props = TypeDescriptor.GetProperties(data.selectedRelease.metadata);
				PropertyDescriptor prop = props[listMetadata.Items[e.Item].SubItems[1].Text];
				if (prop.Name == "Artist")
					data.selectedRelease.metadata.UpdateArtist(e.Label);
				else
					prop.SetValue(data.selectedRelease.metadata, e.Label);
				data.Releases.ResetItem(bnComboBoxRelease.SelectedIndex);
			}
			else
			{
				CUETrackMetadata track = data.selectedRelease.metadata.Tracks[data.metadataTrack];
				PropertyDescriptorCollection props = TypeDescriptor.GetProperties(track);
				props[listMetadata.Items[e.Item].SubItems[1].Text].SetValue(track, e.Label);
				//data.Tracks.ResetItem(data.metadataTrack);
			}
		}

		private void buttonMetadata_Click(object sender, EventArgs e)
		{
			buttonTracks.Visible = true;
			buttonTracks.Focus();
			buttonMetadata.Visible = false;
			data.metadataTrack = -1;
			data.metadataMode = true;
			UpdateRelease();
		}

		private void buttonTracks_Click(object sender, EventArgs e)
		{
			buttonMetadata.Visible = true;
			buttonMetadata.Focus();
			buttonTracks.Visible = false;
			data.metadataMode = false;
			UpdateRelease();
		}

		private void buttonReload_Click(object sender, EventArgs e)
		{
			loadAllMetadata = true;
			data.Releases.Clear();
			data.selectedRelease = null;
            ResetAlbumArt();
            UpdateRelease();
			_workThread = new Thread(Lookup);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start(selectedDriveInfo.drive);
		}

		private void buttonVA_Click(object sender, EventArgs e)
		{
			if (data.selectedRelease == null) return;
			data.selectedRelease.metadata.FreedbToVarious();
			UpdateRelease();
			data.Releases.ResetItem(bnComboBoxRelease.SelectedIndex);
			SetupControls();
		}

		private void buttonEncoding_Click(object sender, EventArgs e)
		{
			if (data.selectedRelease == null) return;
			data.selectedRelease.metadata.FreedbToEncoding();
			UpdateRelease();
			data.Releases.ResetItem(bnComboBoxRelease.SelectedIndex);
			UpdateOutputPath();
			SetupControls();
		}

		private void listTracks_Click(object sender, EventArgs e)
		{
			Point p = listTracks.PointToClient(MousePosition);
			ListViewItem lvItem = listTracks.GetItemAt(p.X, p.Y);
			if (lvItem != null)
			{
				ListViewItem.ListViewSubItem a = lvItem.GetSubItemAt(p.X, p.Y);
				if (a != null)
				{
					int track = lvItem.Index + 1 - selectedDriveInfo.drive.TOC.FirstAudio;
					if (a == lvItem.SubItems[0])
						lvItem.BeginEdit();
					else if (/*a == lvItem.SubItems[2] &&*/ track >= 0 && track < selectedDriveInfo.drive.TOC.AudioTracks)
					{
						buttonTracks.Visible = true;
						buttonTracks.Focus();
						buttonMetadata.Visible = false;
						data.metadataTrack = track;
						data.metadataMode = true;
						UpdateRelease();
					}
				}
			}
		}

		private void FreedbSubmit(object o)
		{
			StringCollection tmp = new StringCollection();
			tmp.Add("DTITLE=");
			CDEntry entry = new CDEntry(tmp);
			entry.Artist = data.selectedRelease.metadata.Artist;
			entry.Title = data.selectedRelease.metadata.Title;
			entry.Year = data.selectedRelease.metadata.Year;
			entry.Genre = data.selectedRelease.metadata.Genre;
			int i = 1;
            for (i = 1; i <= selectedDriveInfo.drive.TOC.TrackCount; i++)
            {
				Freedb.Track tt = new Freedb.Track();
                if (i >= selectedDriveInfo.drive.TOC.FirstAudio && i < selectedDriveInfo.drive.TOC.FirstAudio + selectedDriveInfo.drive.TOC.AudioTracks)
                {
                    CUETrackMetadata t = data.selectedRelease.metadata.Tracks[i - selectedDriveInfo.drive.TOC.FirstAudio];
				    if (t.Artist != "" && t.Artist != entry.Artist)
					    tt.Title = t.Artist + " / " + t.Title;
				    else
					    tt.Title = t.Title;
                } else
                    tt.Title = "Data track";
                tt.FrameOffset = 150 + (int)selectedDriveInfo.drive.TOC[i].Start;
                entry.Tracks.Add(tt);
            }
            /*
			foreach (CUETrackMetadata t in data.selectedRelease.metadata.Tracks)
			{
				Freedb.Track tt = new Freedb.Track();
				if (t.Artist != "" && t.Artist != entry.Artist)
					tt.Title = t.Artist + " / " + t.Title;
				else
					tt.Title = t.Title;
				tt.FrameOffset = 150 + (int)selectedDriveInfo.drive.TOC[i++].Start;
				entry.Tracks.Add(tt);
			}*/

			FreedbHelper m_freedb = new FreedbHelper();

			frmFreedbSubmit frm = new frmFreedbSubmit();
			foreach (string c in m_freedb.ValidCategories)
				frm.Data.Categories.Add(c);
			frm.Data.User = _config.advanced.FreedbUser;
			frm.Data.Domain = _config.advanced.FreedbDomain;
			frm.Data.Category = "misc";

			DialogResult dlgRes = DialogResult.Cancel;
			this.Invoke((MethodInvoker)delegate() { dlgRes = frm.ShowDialog(); });
			if (dlgRes == DialogResult.Cancel)
			{
				_workThread = null;
				this.BeginInvoke((MethodInvoker)delegate() { SetupControls(); });
				return;
			}

			data.selectedRelease.metadata.Save();

			_config.advanced.FreedbUser = frm.Data.User;
			_config.advanced.FreedbDomain = frm.Data.Domain;

			m_freedb.Proxy = _config.GetProxy();
			m_freedb.UserName = _config.advanced.FreedbUser;
			m_freedb.Hostname = _config.advanced.FreedbDomain;
			m_freedb.ClientName = "CUERipper";
			m_freedb.Version = CUESheet.CUEToolsVersion;
			//try
			//{
			//    string code = m_freedb.GetCategories(out tmp);
			//    if (code == FreedbHelper.ResponseCodes.CODE_210)
			//        m_freedb.ValidCategories = tmp;
			//}
			//catch
			//{
			//}
			uint length = selectedDriveInfo.drive.TOC.Length / 75 + 2;
			try
			{
				string res = m_freedb.Submit(entry, (int)length, AccurateRipVerify.CalculateCDDBId(selectedDriveInfo.drive.TOC), frm.Data.Category, false);
				this.BeginInvoke((MethodInvoker)delegate()
				{
					dlgRes = MessageBox.Show(this, res, "Submit result", MessageBoxButtons.OK, MessageBoxIcon.Information);
				});
			}
			catch (Exception ex)
			{
				this.BeginInvoke((MethodInvoker)delegate()
				{
					dlgRes = MessageBox.Show(this, ex.Message, "Submit result", MessageBoxButtons.OK, MessageBoxIcon.Error);
				});
			}
			_workThread = null;
			this.BeginInvoke((MethodInvoker)delegate() { SetupControls(); });
		}

		private void buttonFreedbSubmit_Click(object sender, EventArgs e)
		{
			_workThread = new Thread(FreedbSubmit);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			SetupControls();
			_workThread.Start();
		}

        List<AlbumArt> albumArt = new List<AlbumArt>();
        int currentAlbumArt = 0, frontAlbumArt = -1;

        private void ResetAlbumArt()
        {
            if (this.cueSheet != null)
            {
                this.cueSheet.CTDB.CancelRequest();
            }
            lock (albumArt)
            {
                if (backgroundWorkerArtwork.IsBusy)
                    backgroundWorkerArtwork.CancelAsync();
                albumArt.Clear();
            }
            UpdateAlbumArt(false);
        }

        private void UpdateAlbumArt(bool selectRelease)
        {
            if (albumArt.Count == 0)
            {
                pictureBox1.Image = null;
                return;
            }

            if (selectRelease && data.selectedRelease != null && data.selectedRelease.metadata.AlbumArt.Count > 0)
            {
                for (int i = 0; i < albumArt.Count; i++)
                {
                    foreach (var aa in data.selectedRelease.metadata.AlbumArt)
                    {
                        if (aa.uri == albumArt[i].meta.uri)
                        {
                            currentAlbumArt = i;
                            break;
                        }
                    }
                }
            }

            if (currentAlbumArt >= albumArt.Count)
                currentAlbumArt = 0;
            pictureBox1.Image = albumArt[currentAlbumArt].image;
        }

        private void backgroundWorkerArtwork_DoWork(object sender, DoWorkEventArgs e)
        {
            var args = e.Argument as BackgroundWorkerArtworkArgs;
            var cueSheet = args.cueSheet;
            albumArt.Clear();
            currentAlbumArt = 0;
            frontAlbumArt = -1;
            var knownUrls = new List<string>();
            var firstUrls = new List<string>();

            if (args.meta != null && args.meta.metadata.AlbumArt.Count > 0)
                foreach (var aa in args.meta.metadata.AlbumArt)
                    firstUrls.Add(aa.uri);

            for (int i = 0; i < 2; i++)
            {
                foreach (var metadata in cueSheet.CTDB.Metadata)
                {
                    if (metadata.coverart == null)
                        continue;
                    foreach (var coverart in metadata.coverart)
                    {
                        var uri = _config.advanced.coversSearch == CUEConfigAdvanced.CTDBCoversSearch.Large ?
                            coverart.uri : coverart.uri150 ?? coverart.uri;
                        if (knownUrls.Contains(uri) || !coverart.primary)
                            continue;
                        if (i == 0 && !firstUrls.Contains(coverart.uri))
                            continue;
                        var ms = new MemoryStream();
                        if (!cueSheet.CTDB.FetchFile(uri, ms))
                            continue;
                        lock (this.albumArt)
                        {
                            if (backgroundWorkerArtwork.CancellationPending)
                            {
                                e.Cancel = true;
                                return;
                            }
                            this.albumArt.Add(new AlbumArt(coverart, ms.ToArray()));
                        }
                        knownUrls.Add(uri);
                        backgroundWorkerArtwork.ReportProgress(0);
                    }
                }
            }
        }

        private void backgroundWorkerArtwork_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            UpdateAlbumArt(true);
        }

        private void backgroundWorkerArtwork_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            toolStripStatusLabel1.Text = "";
            toolStripProgressBar1.Value = 0;
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Left)
            {
                currentAlbumArt++;
                UpdateAlbumArt(false);
            }
        }

        private void buttonSettings_Click(object sender, EventArgs e)
        {
            var form = new Options(this._config);
            form.ShowDialog(this);
        }

        private void checkBoxTestAndCopy_Click(object sender, EventArgs e)
        {
            this.testAndCopy = checkBoxTestAndCopy.Checked;
        }

		private void frmCUERipper_ClientSizeChanged(object sender, EventArgs e)
		{
			ResizeList(listTracks, Title);
			ResizeList(listMetadata, columnHeaderValue);
		}

        private void numericWriteOffset_ValueChanged(object sender, EventArgs e)
        {
            if (selectedDriveInfo != null && selectedDriveInfo.drive.ARName != null)
            {
                cueRipperConfig.DriveOffsets[selectedDriveInfo.drive.ARName] = (int)numericWriteOffset.Value;
            }
        }
	}

    internal class BackgroundWorkerArtworkArgs
    {
        public CUESheet cueSheet;
        public CUEMetadataEntry meta;
    }

    internal class AlbumArt
    {
        public CTDBResponseMetaImage meta;

        public byte[] contents;

        public Image image;

        public AlbumArt(CTDBResponseMetaImage meta, byte[] contents)
        {
            this.meta = meta;
            this.contents = contents;
            using (MemoryStream imageStream = new MemoryStream(contents))
                try { this.image = Image.FromStream(imageStream); }
                catch { }
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

	public class DriveInfo
	{
		public ICDRipper drive;
		DirectoryInfo di;
		CUEControls.IIconManager iconMgr;

		public string Path
		{
			get
			{
				return di.FullName;
			}
		}

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

		public override string ToString()
		{
			return drive.Path;
		}
	}

	internal class CUERipperData
	{
		public CUERipperData()
		{
		}
		private BindingList<string> cueStyles = new BindingList<string> { "image", "tracks" };
		//private BindingList<string> losslessOrNot = new BindingList<string> { "lossless", "lossy" };
		private BindingList<ImgComboBoxItem<AudioEncoderType>> losslessOrNot = new BindingList<ImgComboBoxItem<AudioEncoderType>> { 
			new ImgComboBoxItem<AudioEncoderType>("lossless", "checked", AudioEncoderType.Lossless),
			new ImgComboBoxItem<AudioEncoderType>("lossy", "unchecked", AudioEncoderType.Lossy) 
		};
		private BindingList<CUEMetadataEntry> releases = new BindingList<CUEMetadataEntry>();
		private BindingList<DriveInfo> drives = new BindingList<DriveInfo>();
		private BindingList<FormatInfo> formats = new BindingList<FormatInfo>();
		private BindingList<CUEToolsUDC> encoders = new BindingList<CUEToolsUDC>();

		public CUEMetadataEntry selectedRelease { get; set; }
		public bool metadataMode { get; set; }
		public int metadataTrack { get; set; }

		public BindingList<string> CUEStyles
		{
			get
			{
				return cueStyles;
			}
		}

		public BindingList<ImgComboBoxItem<AudioEncoderType>> LosslessOrNot
		{
			get
			{
				return losslessOrNot;
			}
		}

		public BindingList<CUEMetadataEntry> Releases
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
	}
}
