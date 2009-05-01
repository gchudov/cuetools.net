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
using System.Net;
using System.Globalization;
using System.Threading;
using System.Diagnostics;
using CUETools.Processor;
using CUETools.CDImage;

namespace JDP {
	public partial class frmCUETools : Form {
		public frmCUETools() {
			_config = new CUEConfig();
			InitializeComponent();
			m_icon_mgr = new CUEControls.ShellIconMgr();
			m_icon_mgr.SetExtensionIcon(".flac", global::JDP.Properties.Resources.flac1);
			m_icon_mgr.SetExtensionIcon(".wv", global::JDP.Properties.Resources.wv1);
			m_icon_mgr.SetExtensionIcon(".ape", global::JDP.Properties.Resources.ape);
			m_icon_mgr.SetExtensionIcon(".tta", global::JDP.Properties.Resources.tta);
			m_icon_mgr.SetExtensionIcon(".wav", global::JDP.Properties.Resources.wave);
			m_icon_mgr.SetExtensionIcon(".mp3", global::JDP.Properties.Resources.mp3);
			m_icon_mgr.SetExtensionIcon(".m4a", global::JDP.Properties.Resources.ipod_sound);
			m_icon_mgr.SetExtensionIcon(".ogg", global::JDP.Properties.Resources.ogg);
			m_icon_mgr.SetExtensionIcon(".cue", global::JDP.Properties.Resources.cue3);
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
				if (node.IsExpanded)
					AddNodesToBatch(node.Nodes);
				else if ((chkMulti.CheckState == CheckState.Indeterminate || node.Checked) && node.Tag is FileSystemInfo)
					_batchPaths.Add(((FileSystemInfo)node.Tag).FullName);
			}
		}

		private void btnConvert_Click(object sender, EventArgs e) {
			if ((_workThread != null) && (_workThread.IsAlive))
				return;
			if (!CheckWriteOffset()) return;
			_batchReport = new StringBuilder();
			_batchRoot = null;
			_batchProcessed = 0;

			// TODO!!!
			//if (SelectedOutputAudioFmt != null)
			//{
			//    CUEToolsUDC encoder = _config.encoders[SelectedOutputAudioFmt.encoder];
			//    if (encoder.path != null)
			//    {
			//        if (Path.GetDirectoryName(encoder.path) == "" && Directory.Exists(Application.StartupPath))
			//            encoder.path = Path.Combine(Application.StartupPath, encoder.path);
			//        if (!File.Exists(encoder.path))
			//        {
			//            string executable = Path.GetFileName(encoder.path);
			//            OpenFileDialog fileDlg = new OpenFileDialog();
			//            DialogResult dlgRes;
			//            fileDlg.Title = "Select the path to encoder";
			//            fileDlg.Filter = executable + "|" + executable;
			//            if (Directory.Exists(Application.StartupPath))
			//                fileDlg.InitialDirectory = Application.StartupPath;
			//            dlgRes = fileDlg.ShowDialog();
			//            if (dlgRes == DialogResult.OK)
			//                encoder.path = fileDlg.FileName;
			//        }
			//    }
			//}

			if (chkMulti.CheckState == CheckState.Unchecked && !Directory.Exists(txtInputPath.Text))
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
				_batchRoot = txtInputPath.Text;
				if (Directory.Exists(_batchRoot) && !_batchRoot.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
					_batchRoot = _batchRoot + Path.DirectorySeparatorChar;
				_batchPaths.Add(_batchRoot);
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
				settingsForm.IconMgr = m_icon_mgr;
				settingsForm.ReducePriority = _reducePriority;
				settingsForm.Config = _config;

				settingsForm.ShowDialog();

				if (Thread.CurrentThread.CurrentUICulture != CultureInfo.GetCultureInfo(_config.language))
				{
					Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(_config.language);
					ComponentResourceManager resources = new ComponentResourceManager(typeof(frmCUETools));
					int savedWidth = Width;
					Width = MinimumSize.Width;
					ChangeCulture(this, resources);
					Width = savedWidth;
					PerformLayout();
				}

				_reducePriority = settingsForm.ReducePriority;
				_config = settingsForm.Config;
				updateOutputStyles();
				UpdateOutputPath();
				SetupScripts();
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
					if (chkMulti.CheckState == CheckState.Unchecked)
						fileSystemTreeView1.SelectedPath = files[0];
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
			labelFormat.ImageList = m_icon_mgr.ImageList;
			labelCorrectorFormat.ImageList = m_icon_mgr.ImageList;
			LoadSettings();

			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;

			fileSystemTreeView1.CheckBoxes = chkMulti.CheckState == CheckState.Checked;
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
			SetupScripts();
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
		string _defaultLosslessFormat, _defaultLossyFormat, _defaultHybridFormat, _defaultNoAudioFormat;
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
				//if (!File.Exists(pathIn) && !Directory.Exists(pathIn) && !IsCDROM(pathIn))
				//    throw new Exception("Invalid input path.");
				//if (Directory.Exists(pathIn) && !pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
				//{
				//    pathIn = pathIn + Path.DirectorySeparatorChar;
				//    txtInputPath.Text = pathIn;
				//}

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
				p[5] = SelectedOutputAudioType;
				p[6] = comboBoxScript.SelectedItem;

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

		private void BatchLog(string format, string pathIn, params object[] args)
		{
			if (_batchRoot == null || !pathIn.StartsWith(_batchRoot) || pathIn == _batchRoot)
				_batchReport.Append(pathIn);
			else
			{
				_batchReport.Append(".\\");
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
			string outputFormat = (string)p[4];
			AudioEncoderType audioEncoderType = (AudioEncoderType)p[5];
			CUEToolsScript script = (CUEToolsScript)p[6];
			DialogResult dlgRes = DialogResult.OK;
			string status = null;

			try
			{
				if (action == CUEAction.CreateDummyCUE)
				{
					if (Directory.Exists(pathIn))
					{
						if (_batchPaths.Count == 0)
							throw new Exception("is a directory");
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, pathIn);
						int directoriesFound = 0, cueSheetsFound = 0;
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.Folder)
								_batchPaths.Insert(++directoriesFound, fileGroup.main.FullName);
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.CUESheetFile)
								throw new Exception("already contains a cue sheet");
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.TrackFiles || fileGroup.type == FileGroupInfoType.FileWithCUE)
								_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
					}
					else if (File.Exists(pathIn))
					{
						pathIn = Path.GetFullPath(pathIn);
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, Path.GetDirectoryName(pathIn));
						FileGroupInfo fileGroup = FileGroupInfo.WhichContains(fileGroups, pathIn, FileGroupInfoType.TrackFiles)
							?? FileGroupInfo.WhichContains(fileGroups, pathIn, FileGroupInfoType.FileWithCUE);
						if (fileGroup == null)
							throw new Exception("doesn't seem to be part of an album");
						string cueSheetContents;
						if (_batchPaths.Count == 0)
						{
							cueSheet.Open(fileGroup.main.FullName);
							cueSheetContents = cueSheet.CUESheetContents();
							cueSheet.Close();
						} else
							cueSheetContents = CUESheet.CreateDummyCUESheet(_config, fileGroup);
						string fullCueName;
						if (fileGroup.type == FileGroupInfoType.FileWithCUE)
							fullCueName = Path.ChangeExtension(fileGroup.main.FullName, ".cue");
						else
						{
							string cueName = Path.GetFileName(Path.GetDirectoryName(pathIn)) + (fileGroup.discNo != 1 ? ".cd" + fileGroup.discNo.ToString() : "") + ".cuetools" + Path.GetExtension(pathIn) + ".cue";
							fullCueName = Path.Combine(Path.GetDirectoryName(pathIn), cueName);
						}
						if (File.Exists(fullCueName))
							throw new Exception("file already exists");
						bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(cueSheetContents)) != cueSheetContents;
						StreamWriter sw1 = new StreamWriter(fullCueName, false, utf8Required ? Encoding.UTF8 : CUESheet.Encoding);
						sw1.Write(cueSheetContents);
						sw1.Close();
						BatchLog("created ok.", fullCueName);
					}
					else
					{
						//if (_batchPaths.Count > 0)
						//BatchLog("invalid path", pathIn);
						throw new Exception("invalid path");
					}
				}
				else if (action == CUEAction.CorrectFilenames)
				{
					if (Directory.Exists(pathIn))
					{
						if (_batchPaths.Count == 0)
							throw new Exception("is a directory");
						string[] cues = Directory.GetFiles(pathIn, "*.cue", SearchOption.AllDirectories);
						if (cues.Length == 0)
							BatchLog("no cue files.", pathIn);
						else
							_batchPaths.InsertRange(1, cues);
					}
					else if (File.Exists(pathIn))
					{
						if (Path.GetExtension(pathIn).ToLower() != ".cue")
							throw new Exception("is not a .cue file");
						string cue = null;
						using (StreamReader sr = new StreamReader(pathIn, CUESheet.Encoding))
							cue = sr.ReadToEnd();
						string extension;
						string fixedCue;
						if (rbCorrectorLocateFiles.Checked)
							fixedCue = CUESheet.CorrectAudioFilenames(_config, Path.GetDirectoryName(pathIn), cue, true, null, out extension);
						else
						{
							extension = (string)comboBoxCorrectorFormat.SelectedItem;
							using (StringReader sr = new StringReader(cue))
							{
								using (StringWriter sw = new StringWriter())
								{
									string lineStr;
									while ((lineStr = sr.ReadLine()) != null)
									{
										CUELine line = new CUELine(lineStr);
										if (line.Params.Count == 3 && line.Params[0].ToUpper() == "FILE" 
											&& (line.Params[2].ToUpper() != "BINARY" && line.Params[2].ToUpper() != "MOTOROLA")
											)
											sw.WriteLine("FILE \"" + Path.ChangeExtension(line.Params[1], "." + extension) + "\" WAVE");
										else
											sw.WriteLine(lineStr);
									}
									fixedCue = sw.ToString();
								}
							}
						}
						if (fixedCue != cue)
						{
							if (checkBoxCorrectorOverwrite.Checked)
							{
								using (StreamWriter sw = new StreamWriter(pathIn, false, CUESheet.Encoding))
									sw.Write(fixedCue);
								BatchLog("corrected ({0}).", pathIn, extension);
							}
							else
							{
								string pathFixed = Path.ChangeExtension(pathIn, extension + ".cue");
								if (File.Exists(pathFixed))
									BatchLog("corrected cue already exists.", pathIn);
								else
								{
									using (StreamWriter sw = new StreamWriter(pathFixed, false, CUESheet.Encoding))
										sw.Write(fixedCue);
									BatchLog("corrected ({0}).", pathIn, extension);
								}
							}
						}
						else
							BatchLog("no changes.", pathIn);
					} else
						throw new Exception("invalid path");
				}
				else
				{
					if (Directory.Exists(pathIn))
					{
						if (_batchPaths.Count == 0)
							throw new Exception("is a directory");
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, pathIn);
						int directoriesFound = 0, cueSheetsFound = 0;
						foreach(FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.Folder)
								_batchPaths.Insert(++directoriesFound, fileGroup.main.FullName);
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.CUESheetFile)
								_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
						if (cueSheetsFound == 0)
							foreach (FileGroupInfo fileGroup in fileGroups)
								if (fileGroup.type == FileGroupInfoType.FileWithCUE)
									_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
						if (cueSheetsFound == 0)
							foreach (FileGroupInfo fileGroup in fileGroups)
								if (fileGroup.type == FileGroupInfoType.TrackFiles)
									_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
					}
					else if (File.Exists(pathIn) || IsCDROM(pathIn))
					{
						bool convertAction = action == CUEAction.Convert || action == CUEAction.VerifyAndConvert;
						string pathOut = null;
						List<object> releases = null;

						if (Directory.Exists(pathIn) && !pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
							pathIn = pathIn + Path.DirectorySeparatorChar;

						cueSheet.Action = action;
						cueSheet.OutputStyle = cueStyle;
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
							toolStripStatusLabelAR.Text = cueSheet.ArVerify.ARStatus == null ? cueSheet.ArVerify.WorstTotal().ToString() : "?";
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
								pathIn,
								cueSheet.Year != "" ? cueSheet.Year : "YYYY",
								cueSheet.Artist != "" ? cueSheet.Artist : "Unknown Artist",
								cueSheet.Title != "" ? cueSheet.Title : "Unknown Title");
							pathOut = txtOutputPath.Text;
						});

						if (dlgRes == DialogResult.Cancel)
							return;

						bool outputAudio = convertAction && audioEncoderType != AudioEncoderType.NoAudio;
						bool outputCUE = convertAction && (cueStyle != CUEStyle.SingleFileWithCUE || _config.createCUEFileWhenEmbedded);

						cueSheet.GenerateFilenames(audioEncoderType, outputFormat, pathOut);
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
							if (script == null)
								status = cueSheet.Go();
							else if (script.builtin)
							{
								CUESheet processor = cueSheet;
								if (script.name == "default")
									status = processor.Go();
								if (script.name == "only if found")
									status = processor.ArVerify.AccResult != HttpStatusCode.OK ?
										processor.WriteReport() :
										processor.Go();
								if (script.name == "fix offset")
								{
									if (processor.ArVerify.AccResult != HttpStatusCode.OK)
										status = processor.WriteReport();
									else
									{
										processor.WriteOffset = 0;
										processor.Action = CUEAction.Verify;
										status = processor.Go();

										uint tracksMatch;
										int bestOffset;
										processor.FindBestOffset(processor.Config.fixOffsetMinimumConfidence, !processor.Config.fixOffsetToNearest, out tracksMatch, out bestOffset);
										if (tracksMatch * 100 >= processor.Config.fixOffsetMinimumTracksPercent * processor.TrackCount)
										{
											processor.WriteOffset = bestOffset;
											processor.Action = CUEAction.VerifyAndConvert;
											status = processor.Go();
										}
									}
								}
								if (script.name == "encode if verified")
								{
									if (processor.ArVerify.AccResult != HttpStatusCode.OK)
										status = processor.WriteReport();
									else
									{
										processor.Action = CUEAction.Verify;
										status = processor.Go();

										uint tracksMatch;
										int bestOffset;
										processor.FindBestOffset(processor.Config.encodeWhenConfidence, false, out tracksMatch, out bestOffset);
										if (tracksMatch * 100 >= processor.Config.encodeWhenPercent * processor.TrackCount &&  (!_config.encodeWhenZeroOffset || bestOffset == 0))
										{
											processor.Action = CUEAction.VerifyAndConvert;
											status = processor.Go();
										}
									}
								}
							}
							else
								status = cueSheet.ExecuteScript(script.code);
								
							//if (_batchPaths.Count > 0)
							{
								_batchProcessed++;
								BatchLog("{0}.", pathIn, status);
							}
							cueSheet.CheckStop();
						}
					} else
						throw new Exception("invalid path");
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
							(cueSheet.Action == CUEAction.VerifyAndConvert && audioEncoderType != AudioEncoderType.NoAudio))
						{
							frmReport reportForm = new frmReport();
							StringWriter sw = new StringWriter();
							cueSheet.GenerateAccurateRipLog(sw);
							if (status != null)
								reportForm.Text += ": " + status;
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
			bool converting = (SelectedAction == CUEAction.Convert || SelectedAction == CUEAction.VerifyAndConvert);
			bool verifying = (SelectedAction == CUEAction.Verify || SelectedAction == CUEAction.VerifyAndConvert);
			//grpInput.Enabled = !running;
			fileSystemTreeView1.Enabled = !running;
			txtInputPath.Enabled = !running;
			txtInputPath.ReadOnly = chkMulti.Checked;
			chkMulti.Enabled = !running;
			grpExtra.Enabled = !running && (converting || verifying);
			groupBoxCorrector.Enabled = !running && SelectedAction == CUEAction.CorrectFilenames;
			grpOutputPathGeneration.Enabled = !running;
			grpAudioOutput.Enabled = !running && converting;
			grpAction.Enabled = !running;
			grpOutputStyle.Enabled = !running && converting;
			grpFreedb.Enabled = !running && !chkMulti.Checked && converting;
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
				textBatchReport.ReadOnly = true;
				textBatchReport.Text = _batchReport.ToString();
				textBatchReport.SelectAll();
				textBatchReport.ScrollToCaret();
				//toolStripStatusLabelProcessed.Visible = true;
				//toolStripStatusLabelProcessed.Text = "Processed: " + _batchProcessed.ToString();
				//toolStripStatusLabelProcessed.ToolTipText = _batchReport.ToString();
			}
			//else if (chkMulti.CheckState == CheckState.Indeterminate)
			//{
			//    fileSystemTreeView1.Visible = false;
			//    textBatchReport.Visible = true;
			//    textBatchReport.ReadOnly = false;
			//}
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

			if (!running)
				UpdateActions();

			//rbGapsLeftOut.Visible = 
			//    rbGapsPrepended.Visible = 
			//    rbCorrectorLocateFiles.Visible =
			//    rbCorrectorChangeExtension.Visible =
			//    comboBoxCorrectorFormat.Visible =
			//    radioButtonAudioHybrid.Visible = 
			//    radioButtonAudioNone.Visible =
			//    grpExtra.Visible = 
			//    comboBoxScript.Visible =
			//    checkBoxAdvancedMode.Checked;
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
			if (numericWriteOffset.Value == 0 || SelectedOutputAudioType == AudioEncoderType.NoAudio || rbActionVerify.Checked)
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
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt", Application.ExecutablePath);
			_config.Load(sr);
			_defaultLosslessFormat = sr.Load("DefaultLosslessFormat") ?? "flac";
			_defaultLossyFormat = sr.Load("DefaultLossyFormat") ?? "mp3";
			_defaultHybridFormat = sr.Load("DefaultHybridFormat") ?? "lossy.flac";
			_defaultNoAudioFormat = sr.Load("DefaultNoAudioFormat") ?? "wav";
			txtCreateSubdirectory.Text = sr.Load("OutputSubdirectory") ?? "New";
			txtAppendFilename.Text = sr.Load("OutputFilenameSuffix") ?? "-New";
			txtCustomFormat.Text = sr.Load("OutputCustomFormat") ?? "%music%\\Converted\\%artist%\\%year% - %album%\\%artist% - %album%.cue";
			SelectedOutputPathGeneration = (OutputPathGeneration?)sr.LoadInt32("OutputPathGeneration", null, null) ?? OutputPathGeneration.CustomFormat;
			SelectedOutputAudioType = (AudioEncoderType?)sr.LoadInt32("OutputAudioType", null, null) ?? AudioEncoderType.Lossless;
			SelectedOutputAudioFormat = sr.Load("OutputAudioFmt") ?? "flac";
			SelectedAction = (CUEAction?)sr.LoadInt32("AccurateRipMode", null, null) ?? CUEAction.VerifyAndConvert;
			SelectedCUEStyle = (CUEStyle?)sr.LoadInt32("CUEStyle", null, null) ?? CUEStyle.SingleFileWithCUE;
			numericWriteOffset.Value = sr.LoadInt32("WriteOffset", null, null) ?? 0;
			_usePregapForFirstTrackInSingleFile = sr.LoadBoolean("UsePregapForFirstTrackInSingleFile") ?? false;
			_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
			chkMulti.CheckState = (CheckState) (sr.LoadInt32("BatchProcessing", (int) CheckState.Unchecked, (int) CheckState.Indeterminate) ?? (int) CheckState.Unchecked);
			switch (sr.LoadInt32("FreedbLookup", null, null) ?? 2)
			{
				case 0: rbFreedbNever.Checked = true; break;
				case 1: rbFreedbIf.Checked = true; break;
				case 2: rbFreedbAlways.Checked = true; break;
			}
			rbCorrectorChangeExtension.Checked = true;
			switch (sr.LoadInt32("CorrectorLookup", null, null) ?? 0)
			{
				case 0: rbCorrectorLocateFiles.Checked = true; break;
				case 1: rbCorrectorChangeExtension.Checked = true; break;
			}
			checkBoxCorrectorOverwrite.Checked = sr.LoadBoolean("CorrectorOverwrite") ?? true;
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
				comboBoxCorrectorFormat.Items.Add(format.Key);
			comboBoxCorrectorFormat.SelectedItem = sr.Load("CorrectorFormat") ?? "flac";
			Width = sr.LoadInt32("Width", Width, null) ?? Width;
			Top = sr.LoadInt32("Top", 0, null) ?? Top;
			Left = sr.LoadInt32("Left", 0, null) ?? Left;
			PerformLayout();
		}

		private void SaveSettings() {
			SettingsWriter sw = new SettingsWriter("CUE Tools", "settings.txt", Application.ExecutablePath);
			SaveScripts(SelectedAction);
			sw.Save("DefaultLosslessFormat", _defaultLosslessFormat);
			sw.Save("DefaultLossyFormat", _defaultLossyFormat);
			sw.Save("DefaultHybridFormat", _defaultHybridFormat);
			sw.Save("DefaultNoAudioFormat", _defaultNoAudioFormat);
			sw.Save("OutputPathGeneration", (int)SelectedOutputPathGeneration);
			sw.Save("OutputSubdirectory", txtCreateSubdirectory.Text);
			sw.Save("OutputFilenameSuffix", txtAppendFilename.Text);
			sw.Save("OutputCustomFormat", txtCustomFormat.Text);
			sw.Save("OutputAudioFmt", SelectedOutputAudioFormat);
			sw.Save("OutputAudioType", (int)SelectedOutputAudioType);
			sw.Save("AccurateRipMode", (int)SelectedAction);
			sw.Save("CUEStyle", (int)SelectedCUEStyle);
			sw.Save("WriteOffset", (int)numericWriteOffset.Value);
			sw.Save("UsePregapForFirstTrackInSingleFile", _usePregapForFirstTrackInSingleFile);
			sw.Save("ReducePriority", _reducePriority);
			sw.Save("BatchProcessing", (int)chkMulti.CheckState);
			sw.Save("FreedbLookup", rbFreedbNever.Checked ? 0 : rbFreedbIf.Checked ? 1 : 2);
			sw.Save("CorrectorLookup", rbCorrectorLocateFiles.Checked ? 0 : 1);
			sw.Save("CorrectorOverwrite", checkBoxCorrectorOverwrite.Checked);
			sw.Save("CorrectorFormat", (string) (comboBoxCorrectorFormat.SelectedItem ?? "flac"));
			sw.Save("Width", Width);
			sw.Save("Top", Top);
			sw.Save("Left", Left);
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
					radioButtonAudioNone.Checked ? AudioEncoderType.NoAudio :
					radioButtonAudioHybrid.Checked ? AudioEncoderType.Hybrid :
					radioButtonAudioLossy.Checked ? AudioEncoderType.Lossy :
					AudioEncoderType.Lossless;
			}
			set
			{
				switch (value)
				{
					case AudioEncoderType.NoAudio:
						radioButtonAudioNone.Checked = true;
						break;
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

		private string SelectedOutputAudioFormat {
			get 
			{
				return (string) (comboBoxAudioFormat.SelectedItem ?? "dummy");
			}
			set
			{
				comboBoxAudioFormat.SelectedItem = value;
			}
		}

		private CUEAction SelectedAction
		{
			get
			{
				return
					rbActionVerify.Checked ? CUEAction.Verify :
					rbActionVerifyAndEncode.Checked ? CUEAction.VerifyAndConvert :
					rbActionCorrectFilenames.Checked ? CUEAction.CorrectFilenames :
					rbActionCreateCUESheet.Checked ? CUEAction.CreateDummyCUE :
					CUEAction.Convert;
			}
			set
			{
				switch (value)
				{
					case CUEAction.Verify:
						rbActionVerify.Checked = true;
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
			UpdateOutputPath(txtInputPath.Text, "YYYY", "Artist", "Album");
		}

		private void UpdateOutputPath(string pathIn, string year, string artist, string album) {
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
				txtOutputPath.Text = GenerateOutputPath(pathIn, year, artist, album);
			}
		}

		private string GenerateOutputPath(string pathIn, string year, string artist, string album) 
		{
			if (!IsCDROM(pathIn) && !File.Exists(pathIn) && !Directory.Exists(pathIn))
				return "";
			if (SelectedAction == CUEAction.Verify && _config.arLogToSourceFolder)
				return Path.ChangeExtension(pathIn, ".cue");
			if (SelectedAction == CUEAction.CreateDummyCUE)
				return Path.ChangeExtension(pathIn, ".cue");
			if (SelectedAction == CUEAction.CorrectFilenames)
				return pathIn;

			string pathOut, dir, file, ext;

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
					ext = "." + SelectedOutputAudioFormat;
				if (_config.detectHDCD && _config.decodeHDCD && (!ext.StartsWith(".lossy.") || !_config.decodeHDCDtoLW16))
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
			rbEmbedCUE.Enabled = SelectedOutputAudioType != AudioEncoderType.NoAudio && SelectedOutputAudioFmt != null && SelectedOutputAudioFmt.allowEmbed;
			//checkBoxNoAudio.Enabled = !rbEmbedCUE.Checked;
			//comboBoxAudioFormat.Enabled = ;
		}

		private void rbEmbedCUE_CheckedChanged(object sender, EventArgs e)
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

		private void btnCodec_Click(object sender, EventArgs e)
		{
		}

		private void fileSystemTreeView1_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode != Keys.F5) return;
			if (fileSystemTreeView1.Nodes.Count == 0) return;
			string was = fileSystemTreeView1.SelectedPath;
			fileSystemTreeView1.Nodes[0].Collapse();
			fileSystemTreeView1.Nodes[0].Expand();
			if (was != null)
				fileSystemTreeView1.SelectedPath = was;
		}

		private void fileSystemTreeView1_NodeExpand(object sender, CUEControls.FileSystemTreeViewNodeExpandEventArgs e)
		{
			List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_config, e.files);
			foreach (FileGroupInfo fileGroup in fileGroups)
			{
				TreeNode node = fileSystemTreeView1.NewNode(fileGroup.main, fileGroup.type == FileGroupInfoType.Folder);
				if (fileGroup.type == FileGroupInfoType.TrackFiles)
					node.Text = node.Text + ": " + fileGroup.files.Count.ToString() + " files";
				e.node.Nodes.Add(node);
			}
			//toolTip1.Show
		}

		private void UpdateActions()
		{
			if (chkMulti.Checked)
			{
				rbActionCorrectFilenames.Enabled = true;
				rbActionCreateCUESheet.Enabled = true;
				rbActionEncode.Enabled = true;
				rbActionVerify.Enabled = true;
				rbActionVerifyAndEncode.Enabled = true;
				rbDontGenerate.Enabled = false;
			}
			else
			{
				string pathIn = txtInputPath.Text;
				rbActionCorrectFilenames.Enabled = pathIn.Length != 0
					&& ((File.Exists(pathIn) && Path.GetExtension(pathIn).ToLower() == ".cue")
					 || Directory.Exists(pathIn));
				rbActionCreateCUESheet.Enabled = pathIn.Length != 0
					&& ((File.Exists(pathIn) && CUESheet.CreateDummyCUESheet(_config, pathIn) != null)
					 || Directory.Exists(pathIn));
				rbActionVerifyAndEncode.Enabled =
					rbActionVerify.Enabled =
					rbActionEncode.Enabled = pathIn.Length != 0 
					    && (File.Exists(pathIn) || Directory.Exists(pathIn) || IsCDROM(pathIn));
				rbDontGenerate.Enabled = pathIn.Length != 0
				    && (IsCDROM(pathIn) || File.Exists(pathIn));
			}

			btnConvert.Enabled = btnConvert.Visible &&
				 ((rbActionCorrectFilenames.Enabled && rbActionCorrectFilenames.Checked)
				|| (rbActionCreateCUESheet.Enabled && rbActionCreateCUESheet.Checked)
				|| (rbActionEncode.Enabled && rbActionEncode.Checked)
				|| (rbActionVerify.Enabled && rbActionVerify.Checked)
				|| (rbActionVerifyAndEncode.Enabled && rbActionVerifyAndEncode.Checked));

			comboBoxScript.Enabled = btnConvert.Enabled && comboBoxScript.Items.Count > 1;
		}

		private void fileSystemTreeView1_AfterSelect(object sender, TreeViewEventArgs e)
		{
			if (fileSystemTreeView1.SelectedPath != null)
			{
				txtInputPath.Text = fileSystemTreeView1.SelectedPath;
				txtInputPath.SelectAll();
			}
		}

		private void chkRecursive_CheckedChanged(object sender, EventArgs e)
		{
			SetupControls(false);
		}

		private void fileSystemTreeView1_AfterExpand(object sender, TreeViewEventArgs e)
		{
			fileSystemTreeView1_AfterCheck(sender, e);
		}

		private void fileSystemTreeView1_AfterCheck(object sender, TreeViewEventArgs e)
		{
			if (chkMulti.Checked)
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
					//if (folders.Length > 1 && !chkMulti.Checked)
					//{
					//    chkMulti.CheckState = CheckState.Checked;
					//    if (fileSystemTreeView1.SelectedNode != null && fileSystemTreeView1.SelectedNode.Checked)
					//        fileSystemTreeView1.SelectedNode.Checked = false;
					//}
					if (folders.Length > 1 && chkMulti.CheckState == CheckState.Unchecked)
						chkMulti.CheckState = CheckState.Indeterminate;
					switch (chkMulti.CheckState)
					{
						case CheckState.Unchecked:
							fileSystemTreeView1.SelectedPath = folders[0];
							break;
						case CheckState.Checked:
							foreach (string folder in folders)
							{
								TreeNode node = fileSystemTreeView1.LookupNode(folder);
								if (node != null) node.Checked = true;
							}
							break;
						case CheckState.Indeterminate:
							fileSystemTreeView1.Nodes.Clear();
							foreach (string folder in folders)
							{
								TreeNode node = Directory.Exists(folder)
									? fileSystemTreeView1.NewNode(new DirectoryInfo(folder), true)
									: fileSystemTreeView1.NewNode(new FileInfo(folder), false);
								fileSystemTreeView1.Nodes.Add(node);
							}
							break;
					}
					fileSystemTreeView1.Focus();
				}
			}
		}

		private void SetupScripts()
		{
			comboBoxScript.Items.Clear();
			foreach (KeyValuePair<string, CUEToolsScript> script in _config.scripts)
				if (script.Value.conditions.Contains(SelectedAction))
					comboBoxScript.Items.Add(script.Value);
			comboBoxScript.Enabled = btnConvert.Enabled && comboBoxScript.Items.Count > 1;
			comboBoxScript.SelectedItem = comboBoxScript.Items.Count > 0 ? comboBoxScript.Items[0] : null;
			try
			{
				switch (SelectedAction)
				{
					case CUEAction.Verify:
						comboBoxScript.SelectedItem = _config.scripts[_config.defaultVerifyScript];
						break;
					case CUEAction.Convert:
						comboBoxScript.SelectedItem = _config.scripts[_config.defaultConvertScript];
						break;
					case CUEAction.VerifyAndConvert:
						comboBoxScript.SelectedItem = _config.scripts[_config.defaultVerifyAndConvertScript];
						break;
				}
			}
			catch
			{
			}
		}

		private void SaveScripts(CUEAction action)
		{
			switch (action)
			{
				case CUEAction.Verify:
					_config.defaultVerifyScript = ((CUEToolsScript)comboBoxScript.SelectedItem).name;
					break;
				case CUEAction.Convert:
					_config.defaultConvertScript = ((CUEToolsScript)comboBoxScript.SelectedItem).name;
					break;
				case CUEAction.VerifyAndConvert:
					_config.defaultVerifyAndConvertScript = ((CUEToolsScript)comboBoxScript.SelectedItem).name;
					break;
			}
		}

		private void rbAction_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton && !((RadioButton)sender).Checked)
			{
				if (sender == rbActionVerify && comboBoxScript.SelectedItem != null)
					SaveScripts(CUEAction.Verify);
				if (sender == rbActionEncode && comboBoxScript.SelectedItem != null)
					SaveScripts(CUEAction.Convert);
				if (sender == rbActionVerifyAndEncode && comboBoxScript.SelectedItem != null)
					SaveScripts(CUEAction.VerifyAndConvert);
				return;
			}
			UpdateOutputPath();
			SetupScripts();
			SetupControls(false);
		}

		public bool OnSecondCall(string[] args)
		{
			if ((_workThread != null) && (_workThread.IsAlive))
				return false;
			this.Invoke((MethodInvoker)delegate()
			{
				if (args.Length == 1)
				{
					if (chkMulti.CheckState == CheckState.Indeterminate)
					{
						fileSystemTreeView1.Nodes.Clear();
						string folder = args[0];
						{
							TreeNode node = Directory.Exists(folder)
								? fileSystemTreeView1.NewNode(new DirectoryInfo(folder), true)
								: fileSystemTreeView1.NewNode(new FileInfo(folder), false);
							fileSystemTreeView1.Nodes.Add(node);
						}
					}
					else
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
							if (chkMulti.CheckState == CheckState.Checked)
								node.Checked = true;
						}
					}
				}
				if (WindowState == FormWindowState.Minimized)
					WindowState = FormWindowState.Normal;
				fileSystemTreeView1.Select();
				Activate();
			});
			return true;
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

		private void chkMulti_CheckStateChanged(object sender, EventArgs e)
		{
			switch (chkMulti.CheckState)
			{
				case CheckState.Unchecked:
					fileSystemTreeView1.CheckBoxes = false;
					fileSystemTreeView1.Nodes.Clear();
					fileSystemTreeView1.IconManager = m_icon_mgr;
					if (fileSystemTreeView1.Nodes.Count > 0)
						fileSystemTreeView1.Nodes[0].Expand();
					try { fileSystemTreeView1.SelectedPath = txtInputPath.Text; }
					catch { }
					break;
				case CheckState.Checked:
					fileSystemTreeView1.CheckBoxes = true;
					if (fileSystemTreeView1.SelectedNode == null && fileSystemTreeView1.Nodes.Count > 0)
						fileSystemTreeView1.SelectedNode = fileSystemTreeView1.Nodes[0];
					if (fileSystemTreeView1.SelectedNode != null && fileSystemTreeView1.SelectedNode.Tag is FileSystemInfo)
					{
						fileSystemTreeView1.SelectedNode.Checked = true;
						fileSystemTreeView1.SelectedNode.Expand();
					}
					break;
				case CheckState.Indeterminate:
					fileSystemTreeView1.CheckBoxes = false;
					fileSystemTreeView1.Nodes.Clear();
					int icon = m_icon_mgr.GetIconIndex(CUEControls.ExtraSpecialFolder.Desktop, true);
					fileSystemTreeView1.Nodes.Add(null, "Drag the files here", icon, icon);
					break;
			}
			SetupControls(false);
		}

		private void comboBoxAudioFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
			UpdateOutputPath();
			labelFormat.ImageKey = SelectedOutputAudioFmt == null ? null : "." + SelectedOutputAudioFmt.extension;
			comboBoxEncoder.Items.Clear();
			if (SelectedOutputAudioFmt == null)
				return;

			if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
			{
				comboBoxEncoder.Enabled = false;
			}
			else
			{
				foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
					if (encoder.Value.extension == SelectedOutputAudioFmt.extension)
					{
						if (SelectedOutputAudioFormat.StartsWith("lossy.") && !encoder.Value.lossless)
							continue;
						else if (SelectedOutputAudioType == AudioEncoderType.Lossless && !encoder.Value.lossless)
							continue;
						else if (SelectedOutputAudioType == AudioEncoderType.Lossy && encoder.Value.lossless)
							continue;
						comboBoxEncoder.Items.Add(encoder.Key);
					}
				comboBoxEncoder.SelectedItem = SelectedOutputAudioFormat.StartsWith("lossy.") ? SelectedOutputAudioFmt.encoderLossless
					: SelectedOutputAudioType == AudioEncoderType.Lossless ? SelectedOutputAudioFmt.encoderLossless
					: SelectedOutputAudioFmt.encoderLossy;
				comboBoxEncoder.Enabled = true;
			}

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
				case AudioEncoderType.NoAudio:
					_defaultNoAudioFormat = SelectedOutputAudioFormat;
					break;
			}
		}

		private void comboBoxCorrectorFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			CUEToolsFormat fmt;
			if (comboBoxCorrectorFormat.SelectedItem == null || !_config.formats.TryGetValue((string)comboBoxCorrectorFormat.SelectedItem, out fmt))
				return;
			labelCorrectorFormat.ImageKey = "." + fmt.extension;
		}

		private void rbCorrectorChangeExtension_CheckedChanged(object sender, EventArgs e)
		{
			labelCorrectorFormat.Visible = comboBoxCorrectorFormat.Enabled = rbCorrectorChangeExtension.Checked;
		}

		private void radioButtonAudioLossless_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton && !((RadioButton)sender).Checked)
				return;
			labelFormat.ImageKey = null;
			comboBoxAudioFormat.Items.Clear();
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				if (SelectedOutputAudioType == AudioEncoderType.Lossless && !format.Value.allowLossless)
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Hybrid) // && format.Key != "wv") TODO!!!!!
					continue;
				if (SelectedOutputAudioType == AudioEncoderType.Lossy && !format.Value.allowLossy)
					continue;
				//if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
				//continue;
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
				case AudioEncoderType.NoAudio:
					SelectedOutputAudioFormat = _defaultNoAudioFormat;
					break;
			}
			//if (comboBoxAudioFormat.Items.Count > 0)
			//    comboBoxAudioFormat.SelectedIndex = 0;
			//comboBoxAudioFormat.Enabled = comboBoxAudioFormat.Items.Count > 0;
			updateOutputStyles();
			UpdateOutputPath();
		}

		private void comboBoxEncoder_SelectedIndexChanged(object sender, EventArgs e)
		{
			if (SelectedOutputAudioType == AudioEncoderType.NoAudio)
				return;
			if (SelectedOutputAudioFormat == null)
				return;
			if (SelectedOutputAudioFormat.StartsWith("lossy."))
				SelectedOutputAudioFmt.encoderLossless = (string) comboBoxEncoder.SelectedItem;
			else if (SelectedOutputAudioType == AudioEncoderType.Lossless)
				SelectedOutputAudioFmt.encoderLossless = (string) comboBoxEncoder.SelectedItem;
			else
				SelectedOutputAudioFmt.encoderLossy = (string) comboBoxEncoder.SelectedItem;
		}

		private void checkBoxAdvancedMode_CheckedChanged(object sender, EventArgs e)
		{
			SetupControls(false);
		}
	}

	enum OutputPathGeneration {
		CreateSubdirectory,
		AppendFilename,
		CustomFormat,
		Disabled
	}
}
