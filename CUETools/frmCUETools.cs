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
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using CUEToolsLib;

namespace JDP {
	public partial class frmCUETools : Form {
		public frmCUETools() {
			_config = new CUEConfig();
			InitializeComponent();
		}

		private void btnBrowseInput_Click(object sender, EventArgs e) {
			OpenFileDialog fileDlg = new OpenFileDialog();
			DialogResult dlgRes;

			fileDlg.Title = "Input CUE Sheet or album image";
			fileDlg.Filter = "CUE Sheets (*.cue)|*.cue|FLAC images (*.flac)|*.flac|WavPack images (*.wv)|*.wv|APE images (*.ape)|*.ape";

			try
			{
				if (Directory.Exists (Path.GetDirectoryName (txtInputPath.Text)))
					fileDlg.InitialDirectory = Path.GetDirectoryName (txtInputPath.Text);
			}
			catch { }
			dlgRes = fileDlg.ShowDialog();
			if (dlgRes == DialogResult.OK) {
				txtInputPath.Text = fileDlg.FileName;
			}
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

		private void btnConvert_Click(object sender, EventArgs e) {
			if ((_workThread != null) && (_workThread.IsAlive))
				return;
			if (!CheckWriteOffset()) return;
			StartConvert();
		}

		private void btnBatch_Click(object sender, EventArgs e) {
			if (rbDontGenerate.Checked) {
				MessageBox.Show("Batch mode cannot be used with the output path set manually.",
					"Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			FolderBrowserDialog folderDialog = new FolderBrowserDialog();
			folderDialog.Description = "Select the folder containing the CUE sheets you want to convert.  Subfolders will be included automatically.";
			folderDialog.ShowNewFolderButton = false;
			try
			{
				if (Directory.Exists(Path.GetDirectoryName(txtInputPath.Text)))
					folderDialog.SelectedPath = Path.GetDirectoryName(txtInputPath.Text);
			}
			catch { }
			if (folderDialog.ShowDialog() == DialogResult.OK) {
				if (!CheckWriteOffset()) return;
				AddDirToBatch(folderDialog.SelectedPath);
				StartConvert();
			}
		}

		private void btnFilenameCorrector_Click(object sender, EventArgs e) {
			if ((_fcForm == null) || _fcForm.IsDisposed) {
				_fcForm = new frmFilenameCorrector();
				CenterSubForm(_fcForm);
				_fcForm.Show();
			}
			else {
				_fcForm.Activate();
			}
		}

		private void btnSettings_Click(object sender, EventArgs e) {
			using (frmSettings settingsForm = new frmSettings()) {
				settingsForm.WriteOffset = _writeOffset;
				settingsForm.Config = _config;

				CenterSubForm(settingsForm);
				settingsForm.ShowDialog();

				_writeOffset = settingsForm.WriteOffset;
				_config = settingsForm.Config;
				UpdateOutputPath();
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
			SetupControls(false);
			UpdateOutputPath();
			updateOutputStyles();
		}

		private void frmCUETools_FormClosed(object sender, FormClosedEventArgs e) {
			SaveSettings();
		}


		// ********************************************************************************

		frmFilenameCorrector _fcForm;
		List<string> _batchPaths;
		bool _usePregapForFirstTrackInSingleFile;
		int _writeOffset;
		Thread _workThread;
		CUESheet _workClass;
		CUEConfig _config;

		private void StartConvert() {
			string pathIn, pathOut, outDir;
			CUESheet cueSheet;
			CUEStyle cueStyle;

			try {
				_workThread = null;
				if (_batchPaths.Count != 0) {
					txtInputPath.Text = _batchPaths[0];
				}
				pathIn = txtInputPath.Text;
				cueStyle = SelectedCUEStyle;

				bool outputAudio = !rbNoAudio.Checked && !rbArVerify.Checked;
				bool outputCUE = (cueStyle != CUEStyle.SingleFileWithCUE) && !rbArVerify.Checked;
				bool accurateRip = !rbArNone.Checked;

				if (!File.Exists(pathIn)) {
					throw new Exception("Input CUE Sheet not found.");
				}

				cueSheet = new CUESheet(pathIn, _config);

				cueSheet.WriteOffset = _writeOffset;

				UpdateOutputPath(cueSheet.Artist != "" ? cueSheet.Artist : "Unknown Artist", cueSheet.Title != "" ? cueSheet.Title : "Unknown Title");
				pathOut = txtOutputPath.Text;
				cueSheet.GenerateFilenames(SelectedOutputAudioFormat, pathOut);
				outDir = Path.GetDirectoryName(pathOut);
				if (cueStyle == CUEStyle.SingleFileWithCUE)
					cueSheet.SingleFilename = Path.GetFileName (pathOut);

				bool outputExists = false;
				if (outputCUE)
					outputExists = File.Exists(pathOut);
				if (outputAudio) {
					if (cueStyle == CUEStyle.SingleFile || cueStyle == CUEStyle.SingleFileWithCUE) {
						outputExists |= File.Exists(Path.Combine(outDir, cueSheet.SingleFilename));
					}
					else {
						if ((cueStyle == CUEStyle.GapsAppended) && _config.preserveHTOA) {
							outputExists |= File.Exists(Path.Combine(outDir, cueSheet.HTOAFilename));
						}
						for (int i = 0; i < cueSheet.TrackCount; i++) {
							outputExists |= File.Exists(Path.Combine(outDir, cueSheet.TrackFilenames[i]));
						}
					}
				}
				if (outputExists) {
					DialogResult dlgRes = MessageBox.Show("One or more output file already exists, " +
						"do you want to overwrite?", "Overwrite?", (_batchPaths.Count == 0) ?
						MessageBoxButtons.YesNo : MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);

					if (dlgRes == DialogResult.Cancel) {
						_batchPaths.Clear();
					}
					if (dlgRes != DialogResult.Yes) {
						goto SkipConversion;
					}
				}

				cueSheet.UsePregapForFirstTrackInSingleFile = _usePregapForFirstTrackInSingleFile && !outputAudio;
				cueSheet.AccurateRip = accurateRip;
				cueSheet.AccurateOffset = rbArApplyOffset.Checked;
				cueSheet.DataTrackLength = txtDataTrackLength.Text;

				if (outputAudio || accurateRip) {
					object[] p = new object[3];

					_workThread = new Thread(WriteAudioFilesThread);
					_workClass = cueSheet;

					p[0] = cueSheet;
					p[1] = outDir;
					p[2] = cueStyle;

					SetupControls(true);
					//System.Diagnostics; Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.High;
					_workThread.Priority = ThreadPriority.BelowNormal;
					_workThread.IsBackground = true;
					_workThread.Start(p);
				}
				else {
					if (!Directory.Exists(outDir))
						Directory.CreateDirectory(outDir);
					if (outputCUE)
						cueSheet.Write(pathOut, cueStyle);
					ShowFinishedMessage(cueSheet.PaddedToFrame);
				}
			}
			catch (Exception ex) {
				if (!ShowErrorMessage(ex)) {
					_batchPaths.Clear();
				}
			}

		SkipConversion:
			if ((_workThread == null) && (_batchPaths.Count != 0)) {
				_batchPaths.RemoveAt(0);
				if (_batchPaths.Count == 0) {
					ShowBatchDoneMessage();
				}
				else {
					StartConvert();
				}
			}
		}

		private void WriteAudioFilesThread(object o) {
			object[] p = (object[])o;

			CUESheet cueSheet = (CUESheet)p[0];
			string outDir = (string)p[1];
			CUEStyle cueStyle = (CUEStyle)p[2];

			try {
				cueSheet.WriteAudioFiles(outDir, cueStyle, new SetStatus(this.SetStatus));
				this.Invoke((MethodInvoker)delegate() {
					if (_batchPaths.Count == 0)
					{
						if (cueSheet.AccurateRip)
						{
							using (frmReport reportForm = new frmReport())
							{
								StringWriter sw = new StringWriter();
								cueSheet.GenerateAccurateRipLog(sw);
								reportForm.Message = sw.ToString();
								sw.Close();
								CenterSubForm(reportForm);
								reportForm.ShowDialog(this);
							}
						}
						else
							ShowFinishedMessage(cueSheet.PaddedToFrame);
						SetupControls(false);
					}
				});
			}
			catch (StopException) {
				_batchPaths.Clear();
				this.Invoke((MethodInvoker)delegate() {
					SetupControls(false);
					MessageBox.Show("Conversion was stopped.", "Stopped", MessageBoxButtons.OK,
						MessageBoxIcon.Exclamation);
				});
			}
			catch (Exception ex) {
				this.Invoke((MethodInvoker)delegate() {
					if (_batchPaths.Count == 0) SetupControls(false);
					if (!ShowErrorMessage(ex)) {
						_batchPaths.Clear();
						SetupControls(false);
					}
				});
			}

			if (_batchPaths.Count != 0) {
				_batchPaths.RemoveAt(0);
				this.BeginInvoke((MethodInvoker)delegate() {
					if (_batchPaths.Count == 0) {
						SetupControls(false);
						ShowBatchDoneMessage();
					}
					else {
						StartConvert();
					}
				});
			}
		}

		public void SetStatus(string status, uint percentTrack, double percentDisk, string input, string output) {
			this.BeginInvoke((MethodInvoker)delegate() {
				toolStripStatusLabel1.Text = status;
				toolStripProgressBar1.Value = (int)percentTrack;
				toolStripProgressBar2.Value = (int)(percentDisk*100);
			});
		}

		private void SetupControls(bool running) {
			grpCUEPaths.Enabled = !running;
			grpOutputPathGeneration.Enabled = !running;
			grpAudioOutput.Enabled = !running && !rbArVerify.Checked;
			grpAccurateRip.Enabled = !running;
			grpOutputStyle.Enabled = !running && !rbArVerify.Checked;
			txtDataTrackLength.Enabled = !running && !rbArNone.Checked;
			btnAbout.Enabled = !running;
			btnSettings.Enabled = !running;
			btnFilenameCorrector.Enabled = !running;
			btnCUECreator.Enabled = !running;
			btnBatch.Enabled = !running;
			btnConvert.Enabled = !running;
			btnConvert.Visible = !running;
			btnStop.Enabled = btnPause.Enabled = running;
			btnStop.Visible = btnPause.Visible = running;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			toolStripProgressBar2.Value = 0;
		}

		private bool ShowErrorMessage(Exception ex) {
			DialogResult dlgRes = MessageBox.Show(ex.Message, "Error", (_batchPaths.Count == 0) ? 
				MessageBoxButtons.OK : MessageBoxButtons.OKCancel, MessageBoxIcon.Error);
			return (dlgRes == DialogResult.OK);
		}

		private void ShowFinishedMessage(bool warnAboutPadding) {
			if (_batchPaths.Count != 0) {
				return;
			}
			if (warnAboutPadding) {
				MessageBox.Show("One or more input file doesn't end on a CD frame boundary.  " +
					"The output has been padded where necessary to fix this.  If your input " +
					"files are from a CD source, this may indicate a problem with your files.",
					"Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
			}
			MessageBox.Show("Conversion was successful!", "Done", MessageBoxButtons.OK,
				MessageBoxIcon.Information);
		}

		private void ShowBatchDoneMessage() {
			MessageBox.Show("Batch conversion is complete!", "Done", MessageBoxButtons.OK,
				MessageBoxIcon.Information);
		}

		private bool CheckWriteOffset() {
			if ((_writeOffset == 0) || rbNoAudio.Checked || rbArVerify.Checked) {
				return true;
			}

			DialogResult dlgRes = MessageBox.Show("Write offset setting is non-zero which " +
				"will cause some samples to be discarded.  You should only use this setting " +
				"to make temporary files for burning.  Are you sure you want to continue?",
				"Write offset is enabled", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
			return (dlgRes == DialogResult.Yes);
		}

		private void AddDirToBatch(string dir) {
			string[] files = Directory.GetFiles(dir, "*.cue");
			string[] subDirs = Directory.GetDirectories(dir);
			_batchPaths.AddRange(files);
			foreach (string subDir in subDirs) {
				AddDirToBatch(subDir);
			}
		}

		private void LoadSettings() {
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt");
			string val;

			val = sr.Load("OutputPathGeneration");
			if (val != null) {
				try {
					SelectedOutputPathGeneration = (OutputPathGeneration)Int32.Parse(val);
				}
				catch { }
			}

			val = sr.Load("OutputSubdirectory");
			if (val != null) {
				txtCreateSubdirectory.Text = val;
			}

			val = sr.Load("OutputFilenameSuffix");
			if (val != null) {
				txtAppendFilename.Text = val;
			}

			val = sr.Load("OutputCustomFormat");
			if (val != null) {
				txtCustomFormat.Text = val;
			}

			val = sr.Load("OutputAudioFormat");
			if (val != null) {
				try {
					SelectedOutputAudioFormat = (OutputAudioFormat)Int32.Parse(val);
				}
				catch { }
			}

			val = sr.Load("AccurateRipMode");
			if (val != null) {
				try {
					SelectedAccurateRipMode = (AccurateRipMode)Int32.Parse(val);
				}
				catch { }
			}

			val = sr.Load("CUEStyle");
			if (val != null) {
				try {
					SelectedCUEStyle = (CUEStyle)Int32.Parse(val);
				}
				catch { }
			}

			val = sr.Load("WriteOffset");
			if (val != null) {
				if (!Int32.TryParse(val, out _writeOffset)) _writeOffset = 0;
			}

			val = sr.Load("UsePregapForFirstTrackInSingleFile");
			_usePregapForFirstTrackInSingleFile = (val != null) ? (val != "0") : false;

			_config.Load(sr);
		}

		private void SaveSettings() {
			SettingsWriter sw = new SettingsWriter("CUE Tools", "settings.txt");

			sw.Save("OutputPathGeneration", ((int)SelectedOutputPathGeneration).ToString());

			sw.Save("OutputSubdirectory", txtCreateSubdirectory.Text);

			sw.Save("OutputFilenameSuffix", txtAppendFilename.Text);

			sw.Save("OutputCustomFormat", txtCustomFormat.Text);

			sw.Save("OutputAudioFormat", ((int)SelectedOutputAudioFormat).ToString());

			sw.Save("AccurateRipMode", ((int)SelectedAccurateRipMode).ToString());

			sw.Save("CUEStyle", ((int)SelectedCUEStyle).ToString());

			sw.Save("WriteOffset", _writeOffset.ToString());

			sw.Save("UsePregapForFirstTrackInSingleFile", _usePregapForFirstTrackInSingleFile ? "1" : "0");

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
				if (rbNoAudio.Checked) return OutputAudioFormat.NoAudio;
									   return OutputAudioFormat.WAV;
			}
			set {
				switch (value) {
					case OutputAudioFormat.FLAC:    rbFLAC.Checked = true; break;
					case OutputAudioFormat.WavPack: rbWavPack.Checked = true; break;
					case OutputAudioFormat.APE: rbAPE.Checked = true; break;
					case OutputAudioFormat.WAV: rbWAV.Checked = true; break;
					case OutputAudioFormat.NoAudio: rbNoAudio.Checked = true; break;
				}
			}
		}

		private AccurateRipMode SelectedAccurateRipMode {
			get {
				if (rbArVerify.Checked)			return AccurateRipMode.Verify;
				if (rbArApplyOffset.Checked)			return AccurateRipMode.Offset;
				return AccurateRipMode.None;
			}
			set {
				switch (value) {
					case AccurateRipMode.Verify:		rbArVerify.Checked = true; break;
					case AccurateRipMode.Offset:		rbArApplyOffset.Checked = true; break;
					default:			rbArNone.Checked = true; break;
				}
			}
		}

		private void CenterSubForm(Form form) {
			int centerX, centerY, formX, formY;
			Rectangle formRect, maxRect;

			centerX = ((Left * 2) + Width ) / 2;
			centerY = ((Top  * 2) + Height) / 2;
			formX   = ((Left * 2) + Width  - form.Width ) / 2;
			formY   = ((Top  * 2) + Height - form.Height) / 2;

			formRect = new Rectangle(formX, formY, form.Width, form.Height);
			maxRect = Screen.GetWorkingArea(new Point(centerX, centerY));

			if (formRect.Right > maxRect.Right) {
				formRect.X -= formRect.Right - maxRect.Right;
			}
			if (formRect.Bottom > maxRect.Bottom) {
				formRect.Y -= formRect.Bottom - maxRect.Bottom;
			}
			if (formRect.X < maxRect.X) {
				formRect.X = maxRect.X;
			}
			if (formRect.Y < maxRect.Y) {
				formRect.Y = maxRect.Y;
			}

			form.Location = formRect.Location;
		}

		private void UpdateOutputPath() {
			UpdateOutputPath("Artist", "Album");
		}

		private void UpdateOutputPath(string artist, string album) {
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
				txtOutputPath.Text = GenerateOutputPath(artist, album);
			}
		}

		private string GenerateOutputPath(string artist, string album) {
			string pathIn, pathOut, dir, file, ext;

			pathIn = txtInputPath.Text;
			pathOut = String.Empty;

			if ((pathIn.Length != 0) && File.Exists(pathIn)) {
				dir = Path.GetDirectoryName(pathIn);
				file = Path.GetFileNameWithoutExtension(pathIn);
				ext = ".cue";

				if (rbEmbedCUE.Checked)
					ext = General.FormatExtension (SelectedOutputAudioFormat);
				
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

					find.Add("%D");
					find.Add("%C");
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? artist.Replace(' ', '_') : artist)));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? album.Replace(' ', '_') : album)));
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
			rbEmbedCUE.Enabled = rbFLAC.Checked || rbWavPack.Checked || rbAPE.Checked;
			rbNoAudio.Enabled = rbWAV.Enabled = !rbEmbedCUE.Checked;
		}

		private void rbWAV_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
		}

		private void rbFLAC_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
		}

		private void rbWavPack_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
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

		private void rbArVerify_CheckedChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
			SetupControls (false);
		}

		private void CUECreator (string dir)
		{
			string[] cueFiles = Directory.GetFiles(dir, "*.cue");
			if (cueFiles.Length == 0)
			{
				string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape" };
				for (int i = 0; i < audioExts.Length; i++)
				{
					string [] audioFiles = Directory.GetFiles(dir, audioExts[i]);
					if (audioFiles.Length < 2)
						continue;
					Array.Sort (audioFiles);
					string cueName = Path.GetFileName(dir) + ".cuetools" + audioExts[i].Substring(1) + ".cue";
					cueName = Path.Combine(dir, cueName);
					StreamWriter sw = new StreamWriter(cueName, false, CUESheet.Encoding);
					sw.WriteLine(String.Format("REM COMMENT \"CUETools generated dummy CUE sheet\""));
					for (int iFile = 0; iFile < audioFiles.Length; iFile++)
					{
						sw.WriteLine(String.Format("FILE \"{0}\" WAVE", Path.GetFileName (audioFiles[iFile])));
						sw.WriteLine(String.Format("  TRACK {0:00} AUDIO", iFile+1));
						sw.WriteLine(String.Format("    INDEX 01 00:00:00"));
					}
					sw.Close();
					break;
				}
			}
			string[] subDirs = Directory.GetDirectories(dir);
			foreach (string subDir in subDirs)
			{
				CUECreator (subDir);
			}
		}

        private void btnCUECreator_Click(object sender, EventArgs e)
        {
			FolderBrowserDialog folderDialog = new FolderBrowserDialog();
			folderDialog.Description = "Select the folder containing the audio files without CUE sheets. Subfolders will be included automatically.";
			folderDialog.ShowNewFolderButton = false;
			if (folderDialog.ShowDialog() == DialogResult.OK)
			{
				try
				{
					CUECreator(folderDialog.SelectedPath);
				}
				catch (Exception ex)
				{
					ShowErrorMessage(ex);
				}
			}
		}

		private void rbAPE_CheckedChanged(object sender, EventArgs e)
		{
			updateOutputStyles();
		}

		private void btnStop_Click(object sender, EventArgs e)
		{
			if ((_workThread != null) && (_workThread.IsAlive))
				_workClass.Stop();
		}

		private void btnPause_Click(object sender, EventArgs e)
		{
			if ((_workThread != null) && (_workThread.IsAlive))
				_workClass.Pause();
		}
	}

	enum OutputPathGeneration {
		CreateSubdirectory,
		AppendFilename,
		CustomFormat,
		Disabled
	}

	enum AccurateRipMode {
		None,
		Verify,
		Offset
	}
}
