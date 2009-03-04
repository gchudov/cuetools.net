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
using System.Diagnostics;
using CUETools.Processor;

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
			fileDlg.Filter = "CUE Sheets (*.cue)|*.cue|FLAC images (*.flac)|*.flac|WavPack images (*.wv)|*.wv|APE images (*.ape)|*.ape|RAR archives (*.rar)|*.rar";

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
				settingsForm.ReducePriority = _reducePriority;
				settingsForm.Config = _config;

				CenterSubForm(settingsForm);
				settingsForm.ShowDialog();

				_writeOffset = settingsForm.WriteOffset;
				_reducePriority = settingsForm.ReducePriority;
				_config = settingsForm.Config;
				updateOutputStyles();
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
			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;
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
		bool _reducePriority;
		Thread _workThread;
		CUESheet _workClass;
		CUEConfig _config;

		private void StartConvert() {
			try
			{
				_workThread = null;
				if (_batchPaths.Count != 0)
				{
					txtInputPath.Text = _batchPaths[0];
				}

				string pathIn = txtInputPath.Text;
				if (!File.Exists(pathIn))
				{
					if (!Directory.Exists(pathIn))
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
				cueSheet.WriteOffset = _writeOffset;

				object[] p = new object[6];

				_workThread = new Thread(WriteAudioFilesThread);
				_workClass = cueSheet;

				p[0] = cueSheet;
				p[1] = pathIn;
				p[2] = SelectedCUEStyle;
				p[3] = SelectedAccurateRipMode;
				p[4] = SelectedOutputAudioFormat;
				p[5] = chkLossyWAV.Checked;

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
						ShowBatchDoneMessage();
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

		private void WriteAudioFilesThread(object o) {
			object[] p = (object[])o;

			CUESheet cueSheet = (CUESheet)p[0];
			string pathIn = (string)p[1];
			CUEStyle cueStyle = (CUEStyle)p[2];
			AccurateRipMode accurateRip = (AccurateRipMode)p[3];
			OutputAudioFormat outputFormat = (OutputAudioFormat)p[4];
			bool lossyWAV = (bool)p[5];
			DialogResult dlgRes = DialogResult.OK;

			try
			{

				bool outputAudio = outputFormat != OutputAudioFormat.NoAudio && accurateRip != AccurateRipMode.Verify && accurateRip != AccurateRipMode.VerifyPlusCRCs;
				bool outputCUE = cueStyle != CUEStyle.SingleFileWithCUE && accurateRip != AccurateRipMode.Verify && accurateRip != AccurateRipMode.VerifyPlusCRCs;
				string pathOut = null;
				List<object> releases = null;

				cueSheet.Open(pathIn);

				if (_batchPaths.Count == 0 && accurateRip != AccurateRipMode.Verify && accurateRip != AccurateRipMode.VerifyPlusCRCs)
				{
					if (rbFreedbAlways.Checked || (rbFreedbIf.Checked && 
						(cueSheet.Artist == "" || cueSheet.Title == "" || cueSheet.Year == "")))
						releases = cueSheet.LookupAlbumInfo();
				}

				this.Invoke((MethodInvoker)delegate()
				{
					if (releases != null && releases.Count > 0)
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
					cueSheet.AccurateRip = accurateRip;
					if (accurateRip != AccurateRipMode.None)
						cueSheet.DataTrackLength = txtDataTrackLength.Text;
					cueSheet.PreGapLengthMSF = txtPreGapLength.Text;

					cueSheet.WriteAudioFiles(outDir, cueStyle);
				}
				this.Invoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0)
					{
						if (cueSheet.IsCD)
						{
							frmReport reportForm = new frmReport();
							reportForm.Message = cueSheet.LOGContents;
							CenterSubForm(reportForm);
							reportForm.ShowDialog(this);
						}
						else if (cueSheet.AccurateRip == AccurateRipMode.Verify ||
							cueSheet.AccurateRip == AccurateRipMode.VerifyPlusCRCs ||
						(cueSheet.AccurateRip != AccurateRipMode.None && outputFormat != OutputAudioFormat.NoAudio))
						{
							frmReport reportForm = new frmReport();
							StringWriter sw = new StringWriter();
							cueSheet.GenerateAccurateRipLog(sw);
							reportForm.Message = sw.ToString();
							sw.Close();
							CenterSubForm(reportForm);
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
				_batchPaths.Clear();
				this.Invoke((MethodInvoker)delegate()
				{
					SetupControls(false);
					MessageBox.Show(this, "Conversion was stopped.", "Stopped", MessageBoxButtons.OK,
						MessageBoxIcon.Exclamation);
				});
			}
#if !DEBUG
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0) SetupControls(false);
					if (!ShowErrorMessage(ex))
					{
						_batchPaths.Clear();
						SetupControls(false);
					}
				});
			}
#endif
			cueSheet.Close();

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

		public void SetStatus(object sender, CUEToolsProgressEventArgs e)
		{
			this.BeginInvoke((MethodInvoker)delegate() {
				toolStripStatusLabel1.Text = e.status;
				toolStripProgressBar1.Value = Math.Max(0,Math.Min(100,(int)(e.percentTrck*100)));
				toolStripProgressBar2.Value = Math.Max(0,Math.Min(100,(int)(e.percentDisk*100)));
			});
		}

		private void SetupControls(bool running) {
			grpCUEPaths.Enabled = !running;
			grpOutputPathGeneration.Enabled = !running;
			grpAudioOutput.Enabled = !running && !rbArVerify.Checked && !rbArPlusCRC.Checked;
			grpAccurateRip.Enabled = !running;
			grpOutputStyle.Enabled = !running && !rbArVerify.Checked && !rbArPlusCRC.Checked;
			groupBox1.Enabled = !running && !rbArVerify.Checked && !rbArPlusCRC.Checked;
			txtDataTrackLength.Enabled = !running && !rbArNone.Checked;
			txtPreGapLength.Enabled = !running;
			btnAbout.Enabled = !running;
			btnSettings.Enabled = !running;
			btnFilenameCorrector.Enabled = !running;
			btnCUECreator.Enabled = !running;
			btnBatch.Enabled = !running;
			btnConvert.Enabled = !running;
			btnConvert.Visible = !running;
			btnStop.Enabled = btnPause.Enabled = btnResume.Enabled = running;
			btnStop.Visible = btnPause.Visible = running;
			btnResume.Visible = false;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar1.Value = 0;
			toolStripProgressBar2.Value = 0;
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

		private void ShowBatchDoneMessage() {
			MessageBox.Show(this, "Batch conversion is complete!", "Done", MessageBoxButtons.OK,
				MessageBoxIcon.Information);
		}

		private bool CheckWriteOffset() {
			if ((_writeOffset == 0) || rbNoAudio.Checked || rbArVerify.Checked || rbArPlusCRC.Checked)
			{
				return true;
			}

			DialogResult dlgRes = MessageBox.Show(this, "Write offset setting is non-zero which " +
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
			txtCreateSubdirectory.Text = sr.Load("OutputSubdirectory") ?? "New";
			txtAppendFilename.Text = sr.Load("OutputFilenameSuffix") ?? "-New";
			txtCustomFormat.Text = sr.Load("OutputCustomFormat") ?? "%1:-2\\New\\%-1\\%F.cue";
			SelectedOutputPathGeneration = (OutputPathGeneration?)sr.LoadInt32("OutputPathGeneration", null, null) ?? OutputPathGeneration.CreateSubdirectory;
			SelectedOutputAudioFormat = (OutputAudioFormat?)sr.LoadInt32("OutputAudioFormat", null, null) ?? OutputAudioFormat.WAV;
			SelectedAccurateRipMode = (AccurateRipMode?)sr.LoadInt32("AccurateRipMode", null, null) ?? AccurateRipMode.None;
			SelectedCUEStyle = (CUEStyle?)sr.LoadInt32("CUEStyle", null, null) ?? CUEStyle.SingleFileWithCUE;
			_writeOffset = sr.LoadInt32("WriteOffset", null, null) ?? 0;
			_usePregapForFirstTrackInSingleFile = sr.LoadBoolean("UsePregapForFirstTrackInSingleFile") ?? false;
			_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
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
			sw.Save("AccurateRipMode", (int)SelectedAccurateRipMode);
			sw.Save("CUEStyle", (int)SelectedCUEStyle);
			sw.Save("WriteOffset", _writeOffset);
			sw.Save("UsePregapForFirstTrackInSingleFile", _usePregapForFirstTrackInSingleFile);
			sw.Save("ReducePriority", _reducePriority);
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

		private AccurateRipMode SelectedAccurateRipMode
		{
			get
			{
				return
					rbArPlusCRC.Checked ? AccurateRipMode.VerifyPlusCRCs :
					rbArVerify.Checked ? AccurateRipMode.Verify :
					rbArApplyOffset.Checked ? AccurateRipMode.VerifyThenConvert :
					rbArAndEncode.Checked ? AccurateRipMode.VerifyAndConvert :
					AccurateRipMode.None;
			}
			set
			{
				switch (value)
				{
					case AccurateRipMode.VerifyPlusCRCs:
						rbArPlusCRC.Checked = true;
						break;
					case AccurateRipMode.Verify:
						rbArVerify.Checked = true;
						break;
					case AccurateRipMode.VerifyThenConvert:
						rbArApplyOffset.Checked = true;
						break;
					case AccurateRipMode.VerifyAndConvert:
						rbArAndEncode.Checked = true;
						break;
					default:
						rbArNone.Checked = true;
						break;
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

					find.Add("%D");
					find.Add("%C");
					find.Add("%Y");
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? artist.Replace(' ', '_') : artist)));
					replace.Add(General.EmptyStringToNull(_config.CleanseString(rs ? album.Replace(' ', '_') : album)));
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
				string[] audioExts = new string[] { "*.wav", "*.flac", "*.wv", "*.ape", "*.m4a", "*.tta" };
				for (int i = 0; i < audioExts.Length; i++)
				{
					string cueSheet = CUESheet.CreateDummyCUESheet(dir, audioExts[i]);
					if (cueSheet == null)
						continue;
					string cueName = Path.GetFileName(dir) + ".cuetools" + audioExts[i].Substring(1) + ".cue";
					cueName = Path.Combine(dir, cueName);
					bool utf8Required = CUESheet.Encoding.GetString(CUESheet.Encoding.GetBytes(cueSheet)) != cueSheet;
					StreamWriter sw1 = new StreamWriter(cueName, false, utf8Required ? Encoding.UTF8 : CUESheet.Encoding);
					sw1.Write(cueSheet);
					sw1.Close();
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

		private void rbArApplyOffset_CheckedChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
			SetupControls(false);
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

		private void rbArPlusCRC_CheckedChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
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
