// ****************************************************************************
// 
// CUE Tools
// Copyright (C) 2006-2007  Moitah (moitah@yahoo.com)
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// ****************************************************************************

// ****************************************************************************
// Access to AccurateRip is regulated, see
// http://www.accuraterip.com/3rdparty-access.htm for details.
// ****************************************************************************

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Globalization;
using System.Threading;
using System.Diagnostics;
using CUETools.Processor;

namespace JDP {
	public partial class frmCUETools : Form {
		public frmCUETools() {
			_config = new CUEConfig();
			InitializeComponent();
			m_icon_mgr = new CUEControls.ShellIconMgr();
			m_icon_mgr.SetExtensionIcon(".flac", global::JDP.Properties.Resources.flac);
			m_icon_mgr.SetExtensionIcon(".wv", global::JDP.Properties.Resources.wv);
			m_icon_mgr.SetExtensionIcon(".cue", global::JDP.Properties.Resources.cue);
		}

		private void btnBrowseOutput_Click(object sender, EventArgs e) {
			SaveFileDialog fileDlg = new SaveFileDialog();
			DialogResult dlgRes;

			fileDlg.Title = "Output CUE Sheet";
			fileDlg.Filter = "CUE Sheets (*.cue)|*.cue";

			dlgRes = fileDlg.ShowDialog();
			if (dlgRes == DialogResult.OK) {
				txtOutputPath.Text = fileDlg.FileName;
			}
		}

		private void AddNodesToBatch(TreeNodeCollection nodes)
		{
			foreach (TreeNode node in nodes)
			{
				if (node.Checked && node.Tag is FileSystemInfo)
				{
					_batchPaths.Add(((FileSystemInfo)node.Tag).FullName);
					if (!chkRecursive.Checked)
						AddNodesToBatch(node.Nodes);
				}
				else
					AddNodesToBatch(node.Nodes);
				node.Checked = false;
			}
		}

		private void btnConvert_Click(object sender, EventArgs e) {
			if ((_workThread != null) && (_workThread.IsAlive))
				return;
			if (!CheckWriteOffset()) return;
			_batchReport = new StringBuilder();
			_batchRoot = null;
			_batchProcessed = 0;
			if (!chkMulti.Checked && !chkRecursive.Checked)
			{
				StartConvert();
				return;
			}
			if (rbDontGenerate.Checked)
			{
				MessageBox.Show(this, "Batch mode cannot be used with the output path set manually.",
					"Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			if (chkMulti.Checked)
				AddNodesToBatch(fileSystemTreeView1.Nodes);
			else
			{
				_batchPaths.Add(txtInputPath.Text);
				_batchRoot = txtInputPath.Text;
			}
			if (_batchPaths.Count == 0)
			{
				MessageBox.Show(this, "Nothing selected!", "Done", MessageBoxButtons.OK,
					MessageBoxIcon.Information);
				return;
			}
			StartConvert();
		}

		private void ChangeCulture(Control control, ComponentResourceManager resources)
		{
			resources.ApplyResources(control, control.Name, Thread.CurrentThread.CurrentUICulture);
			foreach (Control c in control.Controls)
				ChangeCulture(c, resources);
		}

		private void btnSettings_Click(object sender, EventArgs e) {
			using (frmSettings settingsForm = new frmSettings()) {
				settingsForm.ReducePriority = _reducePriority;
				settingsForm.Config = _config;

				settingsForm.ShowDialog();

				if (Thread.CurrentThread.CurrentUICulture != CultureInfo.GetCultureInfo(_config.language))
				{
					Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(_config.language);
					ComponentResourceManager resources = new ComponentResourceManager(typeof(frmCUETools));
					ChangeCulture(this, resources);
				}

				_reducePriority = settingsForm.ReducePriority;
				_config = settingsForm.Config;
				updateOutputStyles();
				UpdateOutputPath();
				SaveSettings();
			}
		}

		private void btnAbout_Click(object sender, EventArgs e) {
			using (frmAbout aboutForm = new frmAbout())
			{
				aboutForm.ShowDialog(this);
			}
		}

		private void PathTextBox_DragEnter(object sender, DragEventArgs e) {
			if (e.Data.GetDataPresent(DataFormats.FileDrop) && !((TextBox)sender).ReadOnly) {
				e.Effect = DragDropEffects.Copy;
			}
		}

		private void PathTextBox_DragDrop(object sender, DragEventArgs e) {
			if (e.Data.GetDataPresent(DataFormats.FileDrop)) {
				string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
				if (files.Length == 1) {
					((TextBox)sender).Text = files[0];
				}
			}
		}

		public string InputPath {
			get {
				return txtInputPath.Text;
			}
			set {
				txtInputPath.Text = value;
			}
		}

		private void txtInputPath_TextChanged(object sender, EventArgs e) {
			UpdateOutputPath();
			UpdateActions();
		}

		private void rbCreateSubdirectory_CheckedChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void rbAppendFilename_CheckedChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void rbCustomFormat_CheckedChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void txtCreateSubdirectory_TextChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void txtAppendFilename_TextChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void txtCustomFormat_TextChanged(object sender, EventArgs e) {
			UpdateOutputPath();
		}

		private void frmCUETools_Load(object sender, EventArgs e) {
			_batchPaths = new List<string>();
			LoadSettings();
			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;

			fileSystemTreeView1.CheckBoxes = chkMulti.Checked;
			fileSystemTreeView1.IconManager = m_icon_mgr;
			if (InputPath != "")
			{
				TreeNode node = null;
				try
				{
					node = fileSystemTreeView1.LookupNode(InputPath) ??
						fileSystemTreeView1.LookupNode(Path.GetDirectoryName(InputPath));
				}
				catch
				{
				}
				if (node != null)
				{
					fileSystemTreeView1.SelectedNode = node;
					node.Expand();
				}
			}

			SetupControls(false);
			UpdateOutputPath();
			updateOutputStyles();
		}

		private void frmCUETools_FormClosed(object sender, FormClosedEventArgs e) {
			SaveSettings();
		}


		// ********************************************************************************

		private CUEControls.ShellIconMgr m_icon_mgr;
		List<string> _batchPaths;
		StringBuilder _batchReport;
		string _batchRoot;
		int _batchProcessed;
		bool _usePregapForFirstTrackInSingleFile;
		bool _reducePriority;
		Thread _workThread;
		CUESheet _workClass;
		CUEConfig _config;

		private bool IsCDROM(string pathIn)
		{
			return pathIn.Length == 3 && pathIn.Substring(1) == ":\\" && new DriveInfo(pathIn).DriveType == DriveType.CDRom;
		}

		private void StartConvert() {
			try
			{
				_workThread = null;
				if (_batchPaths.Count != 0)
				{
					txtInputPath.Text = _batchPaths[0];
					txtInputPath.SelectAll();
				}

				string pathIn = txtInputPath.Text;
				if (!File.Exists(pathIn))
				{
					if (!Directory.Exists(pathIn) && !IsCDROM(pathIn))
						throw new Exception("Input CUE Sheet not found.");
					if (!pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
					{
						pathIn = pathIn + Path.DirectorySeparatorChar;
						txtInputPath.Text = pathIn;
					}
				}

				CUESheet cueSheet = new CUESheet(_config);
				cueSheet.PasswordRequired += new ArchivePasswordRequiredHandler(PasswordRequired);
				cueSheet.CUEToolsProgress += new CUEToolsProgressHandler(SetStatus);
				cueSheet.CUEToolsSelection += new CUEToolsSelectionHandler(MakeSelection);
				cueSheet.WriteOffset = (int)numericWriteOffset.Value;

				object[] p = new object[7];

				_workThread = new Thread(WriteAudioFilesThread);
				_workClass = cueSheet;

				p[0] = cueSheet;
				p[1] = pathIn;
				p[2] = SelectedCUEStyle;
				p[3] = SelectedAction;
				p[4] = SelectedOutputAudioFormat;
				p[5] = chkLossyWAV.Checked;
				p[6] = chkRecursive.Checked;

				SetupControls(true);
				_workThread.Priority = ThreadPriority.BelowNormal;
				_workThread.IsBackground = true;
				_workThread.Start(p);
			}
			catch (Exception ex)
			{
				if (!ShowErrorMessage(ex))
					_batchPaths.Clear();
				if ((_workThread == null) && (_batchPaths.Count != 0))
				{
					_batchPaths.RemoveAt(0);
					if (_batchPaths.Count == 0)
					{
						frmReport reportForm = new frmReport();
						reportForm.Message = _batchReport.ToString();
						reportForm.ShowDialog(this);
						//ShowBatchDoneMessage();
					}
					else
						StartConvert();
				}
			}
		}

		private void PasswordRequired(object sender, ArchivePasswordRequiredEventArgs e)
		{
			this.Invoke((MethodInvoker)delegate()
			{
				frmPassword dlg = new frmPassword();
				if (dlg.ShowDialog() == DialogResult.OK)
				{
					e.Password = dlg.txtPassword.Text;
					e.ContinueOperation = true;
				} else
					e.ContinueOperation = false;
			});
		}

		private void MakeSelection(object sender, CUEToolsSelectionEventArgs e)
		{
			if (_batchPaths.Count != 0)
				return;
			this.Invoke((MethodInvoker)delegate()
			{
				frmChoice dlg = new frmChoice();
				dlg.Choices = e.choices;
				if (dlg.ShowDialog(this) == DialogResult.OK)
					e.selection = dlg.ChosenIndex;
			});
		}

		private bool TryDummyCUE(string pathIn, out string cueSheetContents, out string ext)
		{
			string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a", "*.tta", "*.tak" };
			for (int i = 0; i < audioExts.Length; i++)
			{
				cueSheetContents = CUESheet.CreateDummyCUESheet(pathIn, audioExts[i]);
				if (cueSheetContents != null)
				{
					ext = audioExts[i].Substring(1);
					return true;
				}
			}
			cueSheetContents = null;
			ext = null;
			return false;
		}

		private void BatchLog(string format, string pathIn, params object[] args)
		{
			if (_batchRoot == null || !pathIn.StartsWith(_batchRoot))
				_batchReport.Append(pathIn);
			else
			{
				_batchReport.Append(".");
				_batchReport.Append(pathIn, _batchRoot.Length, pathIn.Length - _batchRoot.Length);
			}
			_batchReport.Append(": ");
			_batchReport.AppendFormat(format, args);
			_batchReport.Append("\r\n");
		}

		private void WriteAudioFilesThread(object o) {
			object[] p = (object[])o;

			CUESheet cueSheet = (CUESheet)p[0];
			string pathIn = (string)p[1];
			CUEStyle cueStyle = (CUEStyle)p[2];
			CUEAction action = (CUEAction)p[3];
			OutputAudioFormat outputFormat = (OutputAudioFormat)p[4];
			bool lossyWAV = (bool)p[5];
			bool recursive = (bool)p[6];
			DialogResult dlgRes = DialogResult.OK;

			try
			{
				if (action == CUEAction.CreateDummyCUE)
				{
					if (_batchPaths.Count > 0 && Directory.Exists(pathIn))
					{
						if (recursive)
							_batchPaths.InsertRange(1, Directory.GetDirectories(pathIn));
					}
					if (!Directory.Exists(pathIn))
						BatchLog("no such directory.", pathIn);
					else
					{
						if (Directory.GetFiles(pathIn, "*.cue").Length != 0)
							BatchLog("already contains a cue sheet.", pathIn);
						else
						{
							string cueSheetContents, ext;
							if (TryDummyCUE(pathIn, out cueSheetContents, out ext))
							{
								string cueName = Path.GetFileName(Path.GetDirectoryName(pathIn)) + ".cuetools" + ext + ".cue";
								string fullCueName = Path.Combine(pathIn, cueName);
								bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(cueSheetContents)) != cueSheetContents;
								StreamWriter sw1 = new StreamWriter(fullCueName, false, utf8Required ? Encoding.UTF8 : CUESheet.Encoding);
								sw1.Write(cueSheetContents);
								sw1.Close();
								BatchLog("created ok.", fullCueName);
							} else
								BatchLog("no audio files.", pathIn);
						}
					}
				}
				else if (action == CUEAction.CorrectFilenames)
				{
					if (_batchPaths.Count > 0 && Directory.Exists(pathIn))
					{
						string [] cues = Directory.GetFiles(pathIn, "*.cue", recursive ? 
							SearchOption.AllDirectories : SearchOption.TopDirectoryOnly);
						if (cues.Length == 0)
							BatchLog("no cue files.", pathIn);
						else
							_batchPaths.InsertRange(1, cues);
					}
					try
					{
						if (Directory.Exists(pathIn))
						{
							if (_batchPaths.Count == 0)
								throw new Exception ("is a directory");
						}
						else
						{
							if (Path.GetExtension(pathIn).ToLower() != ".cue")
								throw new Exception("is not a .cue file");
							string cue = null;
							using (StreamReader sr = new StreamReader(pathIn, CUESheet.Encoding))
								cue = sr.ReadToEnd();
							string fixedCue = CUESheet.CorrectAudioFilenames(Path.GetDirectoryName(pathIn), cue, true, null);
							if (fixedCue != cue)
							{
								using (StreamWriter sw = new StreamWriter(pathIn, false, CUESheet.Encoding))
									sw.Write(fixedCue);
								BatchLog("corrected.", pathIn);
							}
							else
								BatchLog("no changes.", pathIn);
						}
					}
					catch (Exception ex)
					{
						BatchLog("{0}.", pathIn, ex.Message);
					}
				}
				else
				{
					bool foundImages = false;
					bool foundAudio = false;
					bool processThis = true;

					if (_batchPaths.Count > 0 && Directory.Exists(pathIn))
					{
						if (recursive)
							_batchPaths.InsertRange(1, Directory.GetDirectories(pathIn));
						string[] cueFiles = Directory.GetFiles(pathIn, "*.cue");
						if (cueFiles.Length > 0)
						{
							_batchPaths.InsertRange(1, cueFiles);
							foundImages = true;
						}
						else
						{
							string cueSheetContents, ext1;
							foundAudio = TryDummyCUE(pathIn, out cueSheetContents, out ext1);
							string[] audioExts = new string[] { "*.flac", "*.wv", "*.ape" };
							foreach (string ext in audioExts)
								foreach (string audioFile in Directory.GetFiles(pathIn, ext))
								{
									TagLib.UserDefined.AdditionalFileTypes.Config = _config;
									TagLib.File.IFileAbstraction file = new TagLib.File.LocalFileAbstraction(audioFile);
									try
									{
										TagLib.File fileInfo = TagLib.File.Create(file);
										NameValueCollection tags = Tagging.Analyze(fileInfo);
										if (tags.Get("CUESHEET") != null)
										{
											_batchPaths.Insert(1, audioFile);
											foundImages = true;
										}
									}
									catch
									{
									}
								}
						}
						processThis = !foundImages && foundAudio;
					}

					if (processThis)
					{
						bool convertAction = action == CUEAction.Convert || action == CUEAction.VerifyAndConvert || action == CUEAction.VerifyThenConvert;
						string pathOut = null;
						List<object> releases = null;

						cueSheet.Action = action;
						cueSheet.Open(pathIn);
						if (action != CUEAction.Convert)
							cueSheet.DataTrackLengthMSF = txtDataTrackLength.Text;
						cueSheet.PreGapLengthMSF = txtPreGapLength.Text;
						cueSheet.Lookup();

						if (_batchPaths.Count == 0 && convertAction)
						{
							if (rbFreedbAlways.Checked || (rbFreedbIf.Checked &&
								(cueSheet.Artist == "" || cueSheet.Title == "" || cueSheet.Year == "")))
								releases = cueSheet.LookupAlbumInfo();
						}

						this.Invoke((MethodInvoker)delegate()
						{
							toolStripStatusLabelAR.Visible = action != CUEAction.Convert;// && cueSheet.ArVerify.ARStatus == null;
							toolStripStatusLabelAR.Text = cueSheet.ArVerify.ARStatus == null ? cueSheet.ArVerify.Total(0).ToString() : "?";
							toolStripStatusLabelAR.ToolTipText = "AccurateRip: " + (cueSheet.ArVerify.ARStatus ?? "found") + ".";
							if (releases != null)
							{
								frmChoice dlg = new frmChoice();
								dlg.CUE = cueSheet;
								dlg.Choices = releases;
								dlgRes = dlg.ShowDialog(this);
								if (dlgRes == DialogResult.Cancel)
								{
									cueSheet.Close();
									SetupControls(false);
								}
							}
							UpdateOutputPath(
								cueSheet.Year != "" ? cueSheet.Year : "YYYY",
								cueSheet.Artist != "" ? cueSheet.Artist : "Unknown Artist",
								cueSheet.Title != "" ? cueSheet.Title : "Unknown Title");
							pathOut = txtOutputPath.Text;
						});

						if (dlgRes == DialogResult.Cancel)
							return;

						bool outputAudio = convertAction && outputFormat != OutputAudioFormat.NoAudio;
						bool outputCUE = convertAction && (cueStyle == CUEStyle.SingleFile || (cueStyle == CUEStyle.SingleFileWithCUE && _config.createCUEFileWhenEmbedded));

						cueSheet.GenerateFilenames(outputFormat, lossyWAV, pathOut);
						string outDir = Path.GetDirectoryName(pathOut);
						if (cueStyle == CUEStyle.SingleFileWithCUE)
							cueSheet.SingleFilename = Path.GetFileName(pathOut);
						if (outDir == "")
							outDir = ".";

						bool outputExists = false;
						if (outputCUE)
							outputExists = File.Exists(pathOut);
						if (outputAudio)
						{
							if (cueStyle == CUEStyle.SingleFile || cueStyle == CUEStyle.SingleFileWithCUE)
								outputExists |= File.Exists(Path.Combine(outDir, cueSheet.SingleFilename));
							else
							{
								if (cueStyle == CUEStyle.GapsAppended && _config.preserveHTOA)
									outputExists |= File.Exists(Path.Combine(outDir, cueSheet.HTOAFilename));
								for (int i = 0; i < cueSheet.TrackCount; i++)
									outputExists |= File.Exists(Path.Combine(outDir, cueSheet.TrackFilenames[i]));
							}
						}
						dlgRes = DialogResult.Cancel;
						if (outputExists)
						{
							this.Invoke((MethodInvoker)delegate()
							{
								dlgRes = MessageBox.Show(this, "One or more output file already exists, " +
									"do you want to overwrite?", "Overwrite?", (_batchPaths.Count == 0) ?
									MessageBoxButtons.YesNo : MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);
								if (dlgRes == DialogResult.Yes)
									outputExists = false;
								else if (_batchPaths.Count == 0)
									SetupControls(false);
							});
							if (outputExists && _batchPaths.Count == 0)
							{
								cueSheet.Close();
								return;
							}
						}
						if (!outputExists)
						{
							cueSheet.UsePregapForFirstTrackInSingleFile = _usePregapForFirstTrackInSingleFile && !outputAudio;
							string status = cueSheet.WriteAudioFiles(outDir, cueStyle);
							if (_batchPaths.Count > 0)
							{
								_batchProcessed++;
								BatchLog("{0}.", pathIn, status);
							}
							cueSheet.CheckStop();
						}
					}
				}
				this.Invoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0)
					{
						if (cueSheet.IsCD)
						{
							frmReport reportForm = new frmReport();
							reportForm.Message = cueSheet.LOGContents;
							reportForm.ShowDialog(this);
						}
						else if (action == CUEAction.CreateDummyCUE || action == CUEAction.CorrectFilenames)
						{
							frmReport reportForm = new frmReport();
							reportForm.Message = _batchReport.ToString();
							reportForm.ShowDialog(this);
						}
						else if (cueSheet.Action == CUEAction.Verify ||
							cueSheet.Action == CUEAction.VerifyPlusCRCs ||
						(cueSheet.Action != CUEAction.Convert && outputFormat != OutputAudioFormat.NoAudio))
						{
							frmReport reportForm = new frmReport();
							StringWriter sw = new StringWriter();
							cueSheet.GenerateAccurateRipLog(sw);
							reportForm.Message = sw.ToString();
							sw.Close();
							reportForm.ShowDialog(this);
						}
						else
							ShowFinishedMessage(cueSheet.PaddedToFrame);
						SetupControls(false);
					}
				});
			}
			catch (StopException)
			{
			}
#if !DEBUG
			catch (Exception ex)
			{
				if (_batchPaths.Count == 0)
				{
					this.Invoke((MethodInvoker)delegate()
					{
						SetupControls(false);
						ShowErrorMessage(ex);
					});
				}
				else
				{
					_batchProcessed++;
					BatchLog("{0}.", pathIn, ex.Message);
				}
			}
#endif
			try
			{
				cueSheet.CheckStop();
			}
			catch (StopException)
			{
				_batchPaths.Clear();
				this.Invoke((MethodInvoker)delegate()
				{
					SetupControls(false);
					MessageBox.Show(this, "Conversion was stopped.", "Stopped", MessageBoxButtons.OK,
						MessageBoxIcon.Exclamation);
				});
			}
			cueSheet.Close();

			if (_batchPaths.Count != 0) {
				_batchPaths.RemoveAt(0);
				this.BeginInvoke((MethodInvoker)delegate() {
					if (_batchPaths.Count == 0) {
						SetupControls(false);
						frmReport reportForm = new frmReport();
						reportForm.Message = _batchReport.ToString();
						reportForm.ShowDialog(this);
						//ShowBatchDoneMessage();
					}
					else {
						StartConvert();
					}
				});
			}
		}

		public void SetStatus(object sender, CUEToolsProgressEventArgs e)
		{
			this.BeginInvoke((MethodInvoker)delegate() {
				toolStripStatusLabel1.Text = e.status;
				toolStripProgressBar1.Value = Math.Max(0,Math.Min(100,(int)(e.percentTrck*100)));
				toolStripProgressBar2.Value = Math.Max(0,Math.Min(100,(int)(e.percentDisk*100)));
			});
		}

		private void SetupControls(bool running) {
			bool converting = (SelectedAction == CUEAction.Convert || SelectedAction == CUEAction.VerifyAndConvert || SelectedAction == CUEAction.VerifyThenConvert);
			bool verifying = (SelectedAction == CUEAction.Verify || SelectedAction == CUEAction.VerifyPlusCRCs || SelectedAction == CUEAction.VerifyAndConvert || SelectedAction == CUEAction.VerifyThenConvert);
			//grpInput.Enabled = !running;
			txtInputPath.Enabled = !running;
			grpExtra.Enabled = !running;
			grpOutputPathGeneration.Enabled = !running;
			grpAudioOutput.Enabled = !running && converting;
			grpAction.Enabled = !running;
			grpOutputStyle.Enabled = !running && converting;
			grpFreedb.Enabled = !running && converting;
			txtDataTrackLength.Enabled = !running && verifying;
			txtPreGapLength.Enabled = !running;
			btnAbout.Enabled = !running;
			btnSettings.Enabled = !running;
			btnConvert.Visible = !running;
			btnStop.Enabled = btnPause.Enabled = btnResume.Enabled = running;
			btnStop.Visible = btnPause.Visible = running;
			btnResume.Visible = false;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			toolStripProgressBar2.Value = 0;
			toolStripStatusLabelAR.Visible = false;			
			if (_batchPaths.Count > 0)
			{
				fileSystemTreeView1.Visible = false;
				textBatchReport.Visible = true;
				textBatchReport.Text = _batchReport.ToString();
				textBatchReport.SelectAll();
				textBatchReport.ScrollToCaret();
				//toolStripStatusLabelProcessed.Visible = true;
				//toolStripStatusLabelProcessed.Text = "Processed: " + _batchProcessed.ToString();
				//toolStripStatusLabelProcessed.ToolTipText = _batchReport.ToString();
			}
			else
			{
				bool wasHidden = !fileSystemTreeView1.Visible;
				fileSystemTreeView1.Visible = true;
				toolStripStatusLabelProcessed.Visible = false;
				textBatchReport.Visible = false;
				if (wasHidden && fileSystemTreeView1.SelectedPath != null)
				{
					txtInputPath.Text = fileSystemTreeView1.SelectedPath;
					txtInputPath.SelectAll();
				}
			}
			UpdateActions();
		}

		private bool ShowErrorMessage(Exception ex) {
			string message = "Exception";
			for (Exception e = ex; e != null; e = e.InnerException)
				message += ": " + e.Message;
			DialogResult dlgRes = MessageBox.Show(this, message, "Error", (_batchPaths.Count == 0) ? 
				MessageBoxButtons.OK : MessageBoxButtons.OKCancel, MessageBoxIcon.Error);
			return (dlgRes == DialogResult.OK);
		}

		private void ShowFinishedMessage(bool warnAboutPadding) {
			if (_batchPaths.Count != 0) {
				return;
			}
			if (warnAboutPadding) {
				MessageBox.Show(this, "One or more input file doesn't end on a CD frame boundary.  " +
					"The output has been padded where necessary to fix this.  If your input " +
					"files are from a CD source, this may indicate a problem with your files.",
					"Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
			}
			MessageBox.Show(this, "Conversion was successful!", "Done", MessageBoxButtons.OK,
				MessageBoxIcon.Information);
		}

		//private void ShowBatchDoneMessage() {
		//    MessageBox.Show(this, "Batch conversion is complete!", "Done", MessageBoxButtons.OK,
		//        MessageBoxIcon.Information);
		//}

		private bool CheckWriteOffset() {
			if (numericWriteOffset.Value == 0 || rbNoAudio.Checked || rbActionVerify.Checked || rbActionVerifyAndCRCs.Checked)
			{
				return true;
			}

			DialogResult dlgRes = MessageBox.Show(this, "Write offset setting is non-zero which " +
				"will cause some samples to be discarded.  You should only use this setting " +
				"to make temporary files for burning.  Are you sure you want to continue?",
				"Write offset is enabled", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
			return (dlgRes == DialogResult.Yes);
		}

		private void LoadSettings() {
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt");
			txtCreateSubdirectory.Text = sr.Load("OutputSubdirectory") ?? "New";
			txtAppendFilename.Text = sr.Load("OutputFilenameSuffix") ?? "-New";
			txtCustomFormat.Text = sr.Load("OutputCustomFormat") ?? "%music%\\Converted\\%artist%\\%year% - %album%\\%artist% - %album%.cue";
			SelectedOutputPathGeneration = (OutputPathGeneration?)sr.LoadInt32("OutputPathGeneration", null, null) ?? OutputPathGeneration.CreateSubdirectory;
			SelectedOutputAudioFormat = (OutputAudioFormat?)sr.LoadInt32("OutputAudioFormat", null, null) ?? OutputAudioFormat.WAV;
			SelectedAction = (CUEAction?)sr.LoadInt32("AccurateRipMode", null, null) ?? CUEAction.Convert;
			SelectedCUEStyle = (CUEStyle?)sr.LoadInt32("CUEStyle", null, null) ?? CUEStyle.SingleFileWithCUE;
			numericWriteOffset.Value = sr.LoadInt32("WriteOffset", null, null) ?? 0;
			_usePregapForFirstTrackInSingleFile = sr.LoadBoolean("UsePregapForFirstTrackInSingleFile") ?? false;
			_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
			chkMulti.Checked = sr.LoadBoolean("BatchProcessing") ?? false;
			chkRecursive.Checked = sr.LoadBoolean("RecursiveProcessing") ?? true;
			chkLossyWAV.Checked = sr.LoadBoolean("LossyWav") ?? false;
			switch (sr.LoadInt32("FreedbLookup", null, null) ?? 2)
			{
				case 0: rbFreedbNever.Checked = true; break;
				case 1: rbFreedbIf.Checked = true; break;
				case 2: rbFreedbAlways.Checked = true; break;
			}
			_config.Load(sr);
		}

		private void SaveSettings() {
			SettingsWriter sw = new SettingsWriter("CUE Tools", "settings.txt");
			sw.Save("OutputPathGeneration", (int)SelectedOutputPathGeneration);
			sw.Save("OutputSubdirectory", txtCreateSubdirectory.Text);
			sw.Save("OutputFilenameSuffix", txtAppendFilename.Text);
			sw.Save("OutputCustomFormat", txtCustomFormat.Text);
			sw.Save("OutputAudioFormat", (int)SelectedOutputAudioFormat);
			sw.Save("AccurateRipMode", (int)SelectedAction);
			sw.Save("CUEStyle", (int)SelectedCUEStyle);
			sw.Save("WriteOffset", (int)numericWriteOffset.Value);
			sw.Save("UsePregapForFirstTrackInSingleFile", _usePregapForFirstTrackInSingleFile);
			sw.Save("ReducePriority", _reducePriority);
			sw.Save("BatchProcessing", chkMulti.Checked);
			sw.Save("RecursiveProcessing", chkRecursive.Checked);
			sw.Save("LossyWav", chkLossyWAV.Checked);
			sw.Save("FreedbLookup", rbFreedbNever.Checked ? 0 : rbFreedbIf.Checked ? 1 : 2);
			_config.Save(sw);
			sw.Close();
		}

		private void BuildOutputPathFindReplace(string inputPath, string format, List<string> find, List<string> replace) {
			int i, j, first, last, maxFindLen;
			string range;
			string[] rangeSplit;
			List<string> tmpFind = new List<string>();
			List<string> tmpReplace = new List<string>();

			i = 0;
			last = 0;
			while (i < format.Length) {
				if (format[i++] == '%') {
					j = i;
					while (j < format.Length) {
						char c = format[j];
						if (((c < '0') || (c > '9')) && (c != '-') && (c != ':')) {
							break;
						}
						j++;
					}
					range = format.Substring(i, j - i);
					if (range.Length != 0) {
						rangeSplit = range.Split(new char[] { ':' }, 2);
						if (Int32.TryParse(rangeSplit[0], out first)) {
							if (rangeSplit.Length == 1) {
								last = first;
							}
							if ((rangeSplit.Length == 1) || Int32.TryParse(rangeSplit[1], out last)) {
								tmpFind.Add("%" + range);
								tmpReplace.Add(General.EmptyStringToNull(GetDirectoryElements(Path.GetDirectoryName(inputPath), first, last)));
							}
						}
					}
					i = j;
				}
			}

			// Sort so that longest find strings are first, so when the replacing is done the
			// longer strings are checked first.  This avoids problems with overlapping find
			// strings, for example if one of the strings is "%1" and another is "%1:3".
			maxFindLen = 0;
			for (i = 0; i < tmpFind.Count; i++) {
				if (tmpFind[i].Length > maxFindLen) {
					maxFindLen = tmpFind[i].Length;
				}
			}
			for (j = maxFindLen; j >= 1; j--) {
				for (i = 0; i < tmpFind.Count; i++) {
					if (tmpFind[i].Length == j) {
						find.Add(tmpFind[i]);
						replace.Add(tmpReplace[i]);
					}
				}
			}

			find.Add("%F");
			replace.Add(Path.GetFileNameWithoutExtension(inputPath));
		}

		private string GetDirectoryElements(string dir, int first, int last) {
			if (dir == null)
				return "";
			string[] dirSplit = dir.Split(Path.DirectorySeparatorChar,
				Path.AltDirectorySeparatorChar);
			int count = dirSplit.Length;

			if ((first == 0) && (last == 0)) {
				first = 1;
				last = count;
			}

			if (first < 0) first = (count + 1) + first;
			if (last < 0) last = (count + 1) + last;

			if ((first < 1) && (last < 1)) {
				return String.Empty;
			}
			else if ((first > count) && (last > count)) {
				return String.Empty;
			}
			else {
				int i;
				StringBuilder sb = new StringBuilder();

				if (first < 1) first = 1;
				if (first > count) first = count;
				if (last < 1) last = 1;
				if (last > count) last = count;

				if (last >= first) {
					for (i = first; i <= last; i++) {
						sb.Append(dirSplit[i - 1]);
						sb.Append(Path.DirectorySeparatorChar);
					}
				}
				else {
					for (i = first; i >= last; i--) {
						sb.Append(dirSplit[i - 1]);
						sb.Append(Path.DirectorySeparatorChar);
					}
				}

				return sb.ToString(0, sb.Length - 1);
			}
		}

		private CUEStyle SelectedCUEStyle {
			get {
				if (rbGapsAppended.Checked)	 return CUEStyle.GapsAppended;
				if (rbGapsPrepended.Checked) return CUEStyle.GapsPrepended;
				if (rbGapsLeftOut.Checked)	 return CUEStyle.GapsLeftOut;
				if (rbEmbedCUE.Checked)		return CUEStyle.SingleFileWithCUE;
											 return CUEStyle.SingleFile;
			}
			set {
				switch (value) {
					case CUEStyle.SingleFileWithCUE: rbEmbedCUE.Checked = true; break;
					case CUEStyle.SingleFile:	 rbSingleFile.Checked = true; break;
					case CUEStyle.GapsAppended:	 rbGapsAppended.Checked = true; break;
					case CUEStyle.GapsPrepended: rbGapsPrepended.Checked = true; break;
					case CUEStyle.GapsLeftOut:	 rbGapsLeftOut.Checked = true; break;
				}
			}
		}

		private OutputPathGeneration SelectedOutputPathGeneration {
			get {
				if (rbCreateSubdirectory.Checked) return OutputPathGeneration.CreateSubdirectory;
				if (rbAppendFilename.Checked)	  return OutputPathGeneration.AppendFilename;
				if (rbCustomFormat.Checked)		  return OutputPathGeneration.CustomFormat;
												  return OutputPathGeneration.Disabled;
			}
			set {
				switch (value) {
					case OutputPathGeneration.CreateSubdirectory: rbCreateSubdirectory.Checked = true; break;
					case OutputPathGeneration.AppendFilename:	  rbAppendFilename.Checked = true; break;
					case OutputPathGeneration.CustomFormat:		  rbCustomFormat.Checked = true; break;
					case OutputPathGeneration.Disabled:			  rbDontGenerate.Checked = true; break;
				}
			}
		}

		private OutputAudioFormat SelectedOutputAudioFormat {
			get {
				if (rbFLAC.Checked)    return OutputAudioFormat.FLAC;
				if (rbWavPack.Checked) return OutputAudioFormat.WavPack;
				if (rbAPE.Checked)	   return OutputAudioFormat.APE;
				if (rbTTA.Checked)	   return OutputAudioFormat.TTA;
				if (rbNoAudio.Checked) return OutputAudioFormat.NoAudio;
				if (rbWAV.Checked)     return OutputAudioFormat.WAV;
				if (rbUDC1.Checked)	   return OutputAudioFormat.UDC1;
				return OutputAudioFormat.NoAudio;
				//throw new Exception("output format invalid");
			}
			set {
				switch (value) {
					case OutputAudioFormat.FLAC:    rbFLAC.Checked = true; break;
					case OutputAudioFormat.WavPack: rbWavPack.Checked = true; break;
					case OutputAudioFormat.APE: rbAPE.Checked = true; break;
					case OutputAudioFormat.TTA: rbTTA.Checked = true; break;
					case OutputAudioFormat.WAV: rbWAV.Checked = true; break;
					case OutputAudioFormat.NoAudio: rbNoAudio.Checked = true; break;
					case OutputAudioFormat.UDC1: rbUDC1.Checked = true; break;
				}
			}
		}

		private CUEAction SelectedAction
		{
			get
			{
				return
					rbActionVerifyAndCRCs.Checked ? CUEAction.VerifyPlusCRCs :
					rbActionVerify.Checked ? CUEAction.Verify :
					rbActionVerifyThenEncode.Checked ? CUEAction.VerifyThenConvert :
					rbActionVerifyAndEncode.Checked ? CUEAction.VerifyAndConvert :
					rbActionCorrectFilenames.Checked ? CUEAction.CorrectFilenames :
					rbActionCreateCUESheet.Checked ? CUEAction.CreateDummyCUE :
					CUEAction.Convert;
			}
			set
			{
				switch (value)
				{
					case CUEAction.VerifyPlusCRCs:
						rbActionVerifyAndCRCs.Checked = true;
						break;
					case CUEAction.Verify:
						rbActionVerify.Checked = true;
						break;
					case CUEAction.VerifyThenConvert:
						rbActionVerifyThenEncode.Checked = true;
						break;
					case CUEAction.VerifyAndConvert:
						rbActionVerifyAndEncode.Checked = true;
						break;
					case CUEAction.CorrectFilenames:
						rbActionCorrectFilenames.Checked = true;
						break;
					case CUEAction.CreateDummyCUE:
						rbActionCreateCUESheet.Checked = true;
						break;
					default:
						rbActionEncode.Checked = true;
						break;
				}
			}
		}

		private void UpdateOutputPath() {
			UpdateOutputPath("YYYY", "Artist", "Album");
		}

		private void UpdateOutputPath(string year, string artist, string album) {
			/* if (rbArVerify.Checked)
			{
				txtOutputPath.Text = txtInputPath.Text;
				txtOutputPath.ReadOnly = true;
				btnBrowseOutput.Enabled = false;
			}
			else */ if (rbDontGenerate.Checked)
			{
				txtOutputPath.ReadOnly = false;
				btnBrowseOutput.Enabled = true;
			}
			else
			{
				txtOutputPath.ReadOnly = true;
				btnBrowseOutput.Enabled = false;
				txtOutputPath.Text = GenerateOutputPath(year, artist, album);
			}
		}

		private string GenerateOutputPath(string year, string artist, string album) {
			string pathIn, pathOut, dir, file, ext;

			pathIn = txtInputPath.Text;
			pathOut = String.Empty;

			if ((pathIn.Length != 0) && (File.Exists(pathIn) || Directory.Exists(pathIn)))
			{
				if (Directory.Exists(pathIn))
				{
					if (!pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
						pathIn = pathIn + Path.DirectorySeparatorChar;
					dir = Path.GetDirectoryName(pathIn) ?? pathIn;
					file = Path.GetFileNameWithoutExtension(dir);
				}
				else
				{
					dir = Path.GetDirectoryName(pathIn);
					file = Path.GetFileNameWithoutExtension(pathIn);
				}
				ext = ".cue";
				if (rbEmbedCUE.Checked)
					ext = General.FormatExtension (SelectedOutputAudioFormat, _config);
				if (chkLossyWAV.Checked)
					ext = ".lossy" + ext;
				if (_config.detectHDCD && _config.decodeHDCD && (!chkLossyWAV.Checked || !_config.decodeHDCDtoLW16))
				{
					if (_config.decodeHDCDto24bit)
						ext = ".24bit" + ext;
					else
						ext = ".20bit" + ext;
				}
				
				if (rbCreateSubdirectory.Checked) {
					pathOut = Path.Combine(Path.Combine(dir, txtCreateSubdirectory.Text), file + ext);
				}
				else if (rbAppendFilename.Checked) {
					pathOut = Path.Combine(dir, file + txtAppendFilename.Text + ext);
				}
				else if (rbCustomFormat.Checked) {
					string format = txtCustomFormat.Text;
					List<string> find = new List<string>();
					List<string> replace = new List<string>();
					bool rs = _config.replaceSpaces;

					find.Add("%music%");
					find.Add("%artist%");
					find.Add("%D");
					find.Add("%album%");
					find.Add("%C");
					find.Add("%year%");
					find.Add("%Y");
					replace.Add(m_icon_mgr.GetFolderPath(CUEControls.ExtraSpecialFolder.MyMusic));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? artist.Replace(' ', '_') : artist)));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? artist.Replace(' ', '_') : artist)));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? album.Replace(' ', '_') : album)));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? album.Replace(' ', '_') : album)));
					replace.Add(year);
					replace.Add(year);
					BuildOutputPathFindReplace(pathIn, format, find, replace);

					pathOut = General.ReplaceMultiple(format, find, replace);
					if (pathOut == null) pathOut = String.Empty;
					pathOut = Path.ChangeExtension(pathOut, ext);
				}
			}

			return pathOut;
		}

		private void updateOutputStyles()
		{
			rbEmbedCUE.Enabled = rbFLAC.Checked || rbWavPack.Checked || rbAPE.Checked || (rbUDC1.Checked && _config.udc1APEv2);
			chkLossyWAV.Enabled = rbFLAC.Checked || rbWavPack.Checked || rbWAV.Checked;
			rbNoAudio.Enabled = !rbEmbedCUE.Checked && !chkLossyWAV.Checked;
			rbWAV.Enabled = !rbEmbedCUE.Checked;
			rbTTA.Enabled = rbAPE.Enabled = !chkLossyWAV.Checked;
			rbUDC1.Enabled = _config.udc1Extension != "" && _config.udc1Encoder != "" && (_config.udc1APEv2 || !rbEmbedCUE.Checked) && !chkLossyWAV.Checked;
			rbUDC1.Text = _config.udc1Extension == "" ? "User" : _config.udc1Extension.ToUpper(); 
			// _config.udc1Extension.Substring(0, 1).ToUpper() + _config.udc1Extension.Substring(1);
		}

		private void rbWAV_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
		}

		private void rbFLAC_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbWavPack_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbEmbedCUE_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbNoAudio_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbAPE_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void btnStop_Click(object sender, EventArgs e)
		{
			if ((_workThread != null) && (_workThread.IsAlive))
				_workClass.Stop();
		}

		private void btnPause_Click(object sender, EventArgs e)
		{
			if ((_workThread != null) && (_workThread.IsAlive))
			{
				_workClass.Pause();
				btnPause.Visible = !btnPause.Visible;
				btnResume.Visible = !btnResume.Visible;
			}
		}

		private void chkLossyWAV_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbTTA_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void rbUDC1_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void btnCodec_Click(object sender, EventArgs e)
		{
			contextMenuStripUDC.Show(btnCodec, btnCodec.Width, btnCodec.Height);
			return;
		}

		private void contextMenuStripUDC_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			contextMenuStripUDC.Hide();
			string executable = null, extension = null, decParams = null, encParams = null;
			bool apev2 = false, id3v2 = false;
			switch (e.ClickedItem.Text)
			{
				case "TAK":
					extension = "tak";
					executable = "takc.exe";
					decParams = "-d %I -";
					encParams = "-e -p4m -overwrite - %O";
					apev2 = true;
					id3v2 = false;
					break;
				case "ALAC":
					extension = "m4a";
					executable = "ffmpeg.exe";
					decParams = "%I -f wav -";
					encParams = "-i - -f ipod -acodec alac -y %O";
					apev2 = false;
					id3v2 = false;
					break;
				case "MP3":
					extension = "mp3";
					executable = "lame.exe";
					decParams = "--decode %I -";
					encParams = "--vbr-new -V2 - %O";
					apev2 = false;
					id3v2 = true;
					break;
				case "OGG":
					extension = "ogg";
					executable = "oggenc.exe";
					encParams = "- -o %O";
					decParams = "";
					apev2 = false;
					id3v2 = false;
					break;
				default:
					return;
			}

			string path = Path.Combine(Application.StartupPath, executable);
			if (!File.Exists(path))
			{
				OpenFileDialog fileDlg = new OpenFileDialog();
				DialogResult dlgRes;
				fileDlg.Title = "Select the path to encoder";
				fileDlg.Filter = executable + "|" + executable;
				if (Directory.Exists(Application.StartupPath))
					fileDlg.InitialDirectory = Application.StartupPath;
				dlgRes = fileDlg.ShowDialog();
				if (dlgRes != DialogResult.OK)
					return;
				path = fileDlg.FileName;
			}
			_config.udc1Extension = extension;
			_config.udc1Decoder = path;
			_config.udc1Params = decParams;
			_config.udc1Encoder = path;
			_config.udc1EncParams = encParams;
			_config.udc1APEv2 = apev2;
			_config.udc1ID3v2 = id3v2;
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void fileSystemTreeView1_NodeAttributes(object sender, CUEControls.FileSystemTreeViewNodeAttributesEventArgs e)
		{
			if ((e.file.Attributes & FileAttributes.Hidden) != 0)
			{
				e.isVisible = false;
				return;
			}
			if ((e.file.Attributes & FileAttributes.Directory) != 0)
			{
				e.isVisible = true;
				e.isExpandable = true;
				//		  e.isExpandable = false;
				//        foreach (FileSystemInfo subfile in ((DirectoryInfo)e.file).GetFileSystemInfos())
				//            if (IsVisible(subfile))
				//            {
				//                e.isExpandable = true;
				//                break;
				//            }
				return;
			}
			string ext = e.file.Extension.ToLower();
			if (ext == ".cue")
			{
				e.isVisible = true;
				e.isExpandable = false;
				return;
			}
			if (ext == ".zip")
			{
				e.isVisible = false;
				e.isExpandable = false;
				try
				{
					using (ICSharpCode.SharpZipLib.Zip.ZipFile unzip = new ICSharpCode.SharpZipLib.Zip.ZipFile(e.file.FullName))
					{
						foreach (ICSharpCode.SharpZipLib.Zip.ZipEntry entry in unzip)
						{
							if (entry.IsFile && Path.GetExtension(entry.Name).ToLower() == ".cue")
							{
								e.isVisible = true;
								break;
							}

						}
						unzip.Close();
					}
				}
				catch
				{
				}
				return;
			}
			if (ext == ".rar")
			{
				e.isVisible = true;
				e.isExpandable = false;
				return;
			}
			if (ext != "" && ".flac;.ape;.wv;".Contains(ext))
			{
				TagLib.UserDefined.AdditionalFileTypes.Config = _config;
				TagLib.File.IFileAbstraction file = new TagLib.File.LocalFileAbstraction(e.file.FullName);
				try
				{
					TagLib.File fileInfo = TagLib.File.Create(file);
					NameValueCollection tags = Tagging.Analyze(fileInfo);
					e.isVisible = tags.Get("CUESHEET") != null;
				}
				catch
				{
					e.isVisible = false;
				}
				e.isExpandable = false;
				return;
			}
			return;
		}

		private void UpdateActions()
		{
			if (chkMulti.Checked)
			{
				rbActionCorrectFilenames.Enabled = true;
				rbActionCreateCUESheet.Enabled = true;
				rbActionEncode.Enabled = true;
				rbActionVerifyAndCRCs.Enabled = true;
				rbActionVerify.Enabled = true;
				rbActionVerifyThenEncode.Enabled = true;
				rbActionVerifyAndEncode.Enabled = true;
			}
			else if (chkRecursive.Checked)
			{
				string pathIn = txtInputPath.Text;
				rbActionCorrectFilenames.Enabled = 
					rbActionCreateCUESheet.Enabled =
					rbActionVerifyAndEncode.Enabled =
					rbActionVerifyThenEncode.Enabled =
					rbActionVerify.Enabled =
					rbActionVerifyAndCRCs.Enabled =
					rbActionEncode.Enabled = pathIn.Length != 0 && Directory.Exists(pathIn);
			}
			else
			{
				string pathIn = txtInputPath.Text;
				string cueSheetContents, ext;
				rbActionCorrectFilenames.Enabled = pathIn.Length != 0
					&& File.Exists(pathIn)
					&& Path.GetExtension(pathIn).ToLower() == ".cue";
				rbActionCreateCUESheet.Enabled = pathIn.Length != 0
					&& Directory.Exists(pathIn)
					&& Directory.GetFiles(pathIn, "*.cue").Length == 0
					&& TryDummyCUE(pathIn, out cueSheetContents, out ext);
				rbActionVerifyAndEncode.Enabled =
					rbActionVerifyThenEncode.Enabled =
					rbActionVerify.Enabled =
					rbActionVerifyAndCRCs.Enabled =
					rbActionEncode.Enabled = pathIn.Length != 0
						&& (File.Exists(pathIn) || IsCDROM(pathIn) || rbActionCreateCUESheet.Enabled);
			}
			btnConvert.Enabled = btnConvert.Visible &&
				 ((rbActionCorrectFilenames.Enabled && rbActionCorrectFilenames.Checked)
				|| (rbActionCreateCUESheet.Enabled && rbActionCreateCUESheet.Checked)
				|| (rbActionEncode.Enabled && rbActionEncode.Checked)
				|| (rbActionVerifyAndCRCs.Enabled && rbActionVerifyAndCRCs.Checked)
				|| (rbActionVerify.Enabled && rbActionVerify.Checked)
				|| (rbActionVerifyThenEncode.Enabled && rbActionVerifyThenEncode.Checked)
				|| (rbActionVerifyAndEncode.Enabled && rbActionVerifyAndEncode.Checked));
		}

		private void fileSystemTreeView1_AfterSelect(object sender, TreeViewEventArgs e)
		{
			if (fileSystemTreeView1.SelectedPath != null)
			{
				txtInputPath.Text = fileSystemTreeView1.SelectedPath;
				txtInputPath.SelectAll();
			}
		}

		private void chkMulti_CheckedChanged(object sender, EventArgs e)
		{
			fileSystemTreeView1.CheckBoxes = chkMulti.Checked;
			if (fileSystemTreeView1.SelectedNode != null)
			{
				if (chkMulti.Checked)
					fileSystemTreeView1.SelectedNode.Checked = true;
				fileSystemTreeView1.SelectedNode.Expand();
			}
			UpdateActions();
		}

		private void chkRecursive_CheckedChanged(object sender, EventArgs e)
		{
			UpdateActions();
		}

		private void fileSystemTreeView1_AfterExpand(object sender, TreeViewEventArgs e)
		{
			fileSystemTreeView1_AfterCheck(sender, e);
		}

		private void fileSystemTreeView1_AfterCheck(object sender, TreeViewEventArgs e)
		{
			if (chkMulti.Checked && chkRecursive.Checked)
				foreach (TreeNode node in e.Node.Nodes)
					node.Checked = e.Node.Checked;
		}

		private void fileSystemTreeView1_DragEnter(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				e.Effect = DragDropEffects.Copy;
			}
		}

		private void fileSystemTreeView1_DragDrop(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				string[] folders = e.Data.GetData(DataFormats.FileDrop) as string[];
				if (folders != null)
				{
					if (folders.Length > 1 && !chkMulti.Checked)
					{
						chkMulti.Checked = true;
						if (fileSystemTreeView1.SelectedNode != null && fileSystemTreeView1.SelectedNode.Checked)
							fileSystemTreeView1.SelectedNode.Checked = false;
					}
					if (chkMulti.Checked)
						foreach (string folder in folders)
						{
							TreeNode node = fileSystemTreeView1.LookupNode(folder);
							if (node != null) node.Checked = true;
						}
					else
						fileSystemTreeView1.SelectedPath = folders[0];
					fileSystemTreeView1.Focus();
				}
			}
		}

		private void rbAction_CheckedChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
			SetupControls(false);
		}

		public void OnSecondCall(string[] args)
		{
			this.Invoke((MethodInvoker)delegate()
			{
				if (args.Length == 1)
				{
					TreeNode node = null;
					try
					{
						node = fileSystemTreeView1.LookupNode(args[0]) ??
							fileSystemTreeView1.LookupNode(Path.GetDirectoryName(args[0]));
					}
					catch
					{
					}
					if (node != null)
					{
						fileSystemTreeView1.SelectedNode = node;
						node.Expand();
					}
				}
				if (WindowState == FormWindowState.Minimized)
					WindowState = FormWindowState.Normal;
				fileSystemTreeView1.Select();
				Activate();
			});
		}

		private void setAsMyMusicFolderToolStripMenuItem_Click(object sender, EventArgs e)
		{
			DirectoryInfo dir = (DirectoryInfo)contextMenuStripFileTree.Tag;
			try
			{
				fileSystemTreeView1.IconManager.SetFolderPath(CUEControls.ExtraSpecialFolder.MyMusic, dir.FullName);
			}
			catch (Exception ex)
			{
				MessageBox.Show(this, ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			fileSystemTreeView1.Nodes[0].Collapse();
			fileSystemTreeView1.SelectedFolder = CUEControls.ExtraSpecialFolder.MyMusic;
			fileSystemTreeView1.SelectedNode.Expand();
		}

		private void resetToOriginalLocationToolStripMenuItem_Click(object sender, EventArgs e)
		{
			CUEControls.ExtraSpecialFolder dir = (CUEControls.ExtraSpecialFolder)contextMenuStripFileTree.Tag;
			try
			{
				fileSystemTreeView1.IconManager.SetFolderPath(dir, null);
			}
			catch (Exception ex)
			{
				MessageBox.Show(this, ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			fileSystemTreeView1.Nodes[0].Collapse();
			fileSystemTreeView1.SelectedFolder = CUEControls.ExtraSpecialFolder.MyMusic;
			fileSystemTreeView1.SelectedNode.Expand();
		}

		private void fileSystemTreeView1_MouseDown(object sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Right)
			{				
				TreeViewHitTestInfo info = fileSystemTreeView1.HitTest(e.Location);
				if (info.Node != null)
				{
					contextMenuStripFileTree.Tag = info.Node.Tag;
					SelectedNodeName.Text = info.Node.Text;
					SelectedNodeName.Image = m_icon_mgr.ImageList.Images[info.Node.ImageIndex];
					if (info.Node.Tag is DirectoryInfo)
					{
						resetToOriginalLocationToolStripMenuItem.Visible = false;
						setAsMyMusicFolderToolStripMenuItem.Visible = true;
						setAsMyMusicFolderToolStripMenuItem.Image = m_icon_mgr.ImageList.Images[m_icon_mgr.GetIconIndex(CUEControls.ExtraSpecialFolder.MyMusic, true)];
					}
					else if (info.Node.Tag is CUEControls.ExtraSpecialFolder && ((CUEControls.ExtraSpecialFolder)info.Node.Tag) == CUEControls.ExtraSpecialFolder.MyMusic)
					{
						resetToOriginalLocationToolStripMenuItem.Visible = true;
						setAsMyMusicFolderToolStripMenuItem.Visible = false;
					}
					else
						return;
					fileSystemTreeView1.SelectedNode = info.Node;
					contextMenuStripFileTree.Show(fileSystemTreeView1, e.Location);
				}					
			}
		}

	}

	enum OutputPathGeneration {
		CreateSubdirectory,
		AppendFilename,
		CustomFormat,
		Disabled
	}
}
