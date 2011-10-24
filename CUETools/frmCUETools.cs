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
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.CTDB;
using CUETools.Compression;
using CUETools.Processor;
using CUETools.Processor.Settings;

namespace JDP
{
	public partial class frmCUETools : Form {
		public frmCUETools() {
			_profile = _defaultProfile = new CUEToolsProfile("default");
			InitializeComponent();
			//splitContainer1.AutoScaleMode = AutoScaleMode.Font;
			if (Type.GetType("Mono.Runtime", false) == null)
				m_icon_mgr = new CUEControls.ShellIconMgr();
			else
				m_icon_mgr = new CUEControls.MonoIconMgr();
			m_icon_mgr.SetExtensionIcon(".flac", global::JDP.Properties.Resources.flac1);
			m_icon_mgr.SetExtensionIcon(".wv", global::JDP.Properties.Resources.wv1);
			m_icon_mgr.SetExtensionIcon(".ape", global::JDP.Properties.Resources.ape);
			m_icon_mgr.SetExtensionIcon(".tta", global::JDP.Properties.Resources.tta);
			m_icon_mgr.SetExtensionIcon(".wav", global::JDP.Properties.Resources.wave);
			m_icon_mgr.SetExtensionIcon(".mp3", global::JDP.Properties.Resources.mp3);
			m_icon_mgr.SetExtensionIcon(".m4a", global::JDP.Properties.Resources.ipod_sound);
			m_icon_mgr.SetExtensionIcon(".ogg", global::JDP.Properties.Resources.ogg);
			m_icon_mgr.SetExtensionIcon(".cue", global::JDP.Properties.Resources.cue3);
			m_icon_mgr.SetExtensionIcon(".#puzzle", global::JDP.Properties.Resources.puzzle);
			m_icon_mgr.SetExtensionIcon(".#users", global::JDP.Properties.Resources.users);
			m_icon_mgr.SetExtensionIcon(".#alarm_clock", global::JDP.Properties.Resources.alarm_clock);
			m_icon_mgr.SetExtensionIcon(".#calendar", global::JDP.Properties.Resources.calendar);
			m_icon_mgr.SetExtensionIcon(".#ar", global::JDP.Properties.Resources.AR);
			m_icon_mgr.SetExtensionIcon(".#images", global::JDP.Properties.Resources.images);
			m_icon_mgr.SetExtensionIcon(".#images_question", global::JDP.Properties.Resources.images_question);
			m_icon_mgr.SetExtensionIcon(".#pictures", global::JDP.Properties.Resources.pictures);
			m_icon_mgr.SetExtensionIcon(".#picture", global::JDP.Properties.Resources.picture);

			//m_state_image_list = new ImageList();
			//m_state_image_list.ImageSize = new Size(16, 16);
			//m_state_image_list.ColorDepth = ColorDepth.Depth32Bit;
			//m_state_image_list.Images.Add("blank", new Bitmap(16, 16));
			//m_state_image_list.Images.Add("cue", Properties.Resources.cue3);
		}

		private void AddCheckedNodesToBatch(TreeNodeCollection nodes)
		{
			foreach (TreeNode node in nodes)
			{
				if (node.IsExpanded)
					AddCheckedNodesToBatch(node.Nodes);
				else if (node.Checked)
				{
					if (node is CUEControls.FileSystemTreeNodeFileSystemInfo || node is FileSystemTreeNodeLocalDBEntry)
						if ((node as CUEControls.FileSystemTreeNode).Path != null)
							_batchPaths.Add((node as CUEControls.FileSystemTreeNode).Path);
					if (node is FileSystemTreeNodeLocalDBFolder)
						foreach (var entry in (node as FileSystemTreeNodeLocalDBFolder).Group)
							if (entry.Path != null)
								_batchPaths.Add(entry.Path);
				}
			}
		}

		private void AddAllNodesToBatch(TreeNode node)
		{
			if (node is CUEControls.FileSystemTreeNodeFileSystemInfo)
			{
				_batchPaths.Add((node as CUEControls.FileSystemTreeNodeFileSystemInfo).Path);
				return;
			}
			if (node is FileSystemTreeNodeLocalDBFolder)
			{
				foreach (var entry in (node as FileSystemTreeNodeLocalDBFolder).Group)
					if (entry.Path != null)
						_batchPaths.Add(entry.Path);
				return;
			}
			if (node is FileSystemTreeNodeLocalDBEntry)
			{
				if ((node as FileSystemTreeNodeLocalDBEntry).Path != null)
					_batchPaths.Add((node as FileSystemTreeNodeLocalDBEntry).Path);
				return;
			}
			if (node.IsExpanded || !(node is CUEControls.FileSystemTreeNode))
			{
				AddAllNodesToBatch(node.Nodes);
				return;
			}
		}

		private void AddAllNodesToBatch(TreeNodeCollection nodes)
		{
			foreach (TreeNode node in nodes)
			{
				AddAllNodesToBatch(node);
			}
		}

		DialogResult overwriteResult = DialogResult.None;

		private void btnConvert_Click(object sender, EventArgs e) {
			if ((_workThread != null) && (_workThread.IsAlive))
				return;

			if (!comboBoxOutputFormat.Items.Contains(comboBoxOutputFormat.Text) && comboBoxOutputFormat.Text.Contains("%"))
			{
				comboBoxOutputFormat.Items.Insert(OutputPathUseTemplates.Length, comboBoxOutputFormat.Text);
				if (comboBoxOutputFormat.Items.Count > OutputPathUseTemplates.Length + 10)
					comboBoxOutputFormat.Items.RemoveAt(OutputPathUseTemplates.Length + 10);
			}

			if (!CheckWriteOffset()) return;
			_batchReport = new StringBuilder();
			_batchRoot = null;
			_batchProcessed = 0;
			overwriteResult = DialogResult.None;

			// TODO!!!
			//if (SelectedOutputAudioFmt != null)
			//{
			//    CUEToolsUDC encoder = _profile._config.encoders[SelectedOutputAudioFmt.encoder];
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

			if ((FileBrowserState == FileBrowserStateEnum.Hidden
				|| (FileBrowserState == FileBrowserStateEnum.Tree
				  && !(fileSystemTreeView1.SelectedNode is FileSystemTreeNodeLocalDBFolder)))
				&& !Directory.Exists(InputPath))
			{
				StartConvert();
				return;
			}
			if (!OutputPathUseTemplate)
			{
				MessageBox.Show(this, "Batch mode cannot be used with the output path set manually.",
					"Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			if (FileBrowserState == FileBrowserStateEnum.Checkboxes)
				AddCheckedNodesToBatch(fileSystemTreeView1.Nodes);
			else if (FileBrowserState == FileBrowserStateEnum.DragDrop)
				AddAllNodesToBatch(fileSystemTreeView1.Nodes);
			else if (FileBrowserState == FileBrowserStateEnum.Tree && fileSystemTreeView1.SelectedNode is FileSystemTreeNodeLocalDBFolder)
			{
				foreach (var entry in (fileSystemTreeView1.SelectedNode as FileSystemTreeNodeLocalDBFolder).Group)
					if (entry.Path != null)
						_batchPaths.Add(entry.Path);
			}
			else
			{
				_batchRoot = InputPath;
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

		private void toolStripButtonSettings_Click(object sender, EventArgs e)
		{
			using (frmSettings settingsForm = new frmSettings())
			{
				settingsForm.IconMgr = m_icon_mgr;
				settingsForm.ReducePriority = _reducePriority;
				settingsForm.Config = _profile._config;

				DialogResult res = settingsForm.ShowDialog(this);

				if (res == DialogResult.Cancel)
					return;

				if (Thread.CurrentThread.CurrentUICulture != CultureInfo.GetCultureInfo(_profile._config.language))
				{
					Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(_profile._config.language);
					ComponentResourceManager resources = new ComponentResourceManager(typeof(frmCUETools));
					int savedWidth = Width;
					Width = MinimumSize.Width;
					ChangeCulture(this, resources);
					Width = savedWidth;
					PerformLayout();
				}

				_reducePriority = settingsForm.ReducePriority;
				SelectedOutputAudioType = SelectedOutputAudioType;
				SetupScripts();
				SaveSettings();
			}
		}

		private void toolStripButtonAbout_Click(object sender, EventArgs e)
		{
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
					if (FileBrowserState == FileBrowserStateEnum.Tree)
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

		private void txtInputPath_TextChanged(object sender, EventArgs e) 
		{
			UpdateOutputPath();
			UpdateActions();
		}

		private void comboBoxOutputFormat_TextUpdate(object sender, EventArgs e)
		{
			UpdateOutputPath();
		}

		private void comboBoxOutputFormat_SelectedIndexChanged(object sender, EventArgs e)
		{
			UpdateOutputPath();
		}

		private void frmCUETools_Load(object sender, EventArgs e) {
			_batchPaths = new List<string>();
			try
			{
				_localDB = CUEToolsLocalDB.Load();
			}
			catch (Exception ex)
			{
				MessageBox.Show(this, ex.Message, "Error loading local database",
					MessageBoxButtons.OK, MessageBoxIcon.Error);
				string tempPath = CUEToolsLocalDB.LocalDBPath + "." + DateTime.Now.Ticks.ToString() + ".tmp";
				File.Move(CUEToolsLocalDB.LocalDBPath, tempPath);
				_localDB = new CUEToolsLocalDB();
			}
			//foreach (var entry in _localDB)
			//{
			//    try
			//    {
			//        CUEMetadata cache = CUEMetadata.Load(entry.DiscID);
			//        foreach (var entry2 in _localDB)
			//            if (entry2.DiscID == entry.DiscID)
			//                entry2.Metadata = new CUEMetadata(cache);
			//    }
			//    catch
			//    {
			//    }
			//}
			//_localDB.Dirty = true;
			//foreach (var entries in _localDB)
			//    if (entries.InputPaths != null)
			//        entries.InputPaths = entries.InputPaths.ConvertAll(i => CUEToolsLocalDBEntry.NormalizePath(i));
			//_localDB.Dirty = true;
			//foreach (var entries in _localDB)
			//    entry.AudioPaths = entry.AudioPaths.ConvertAll(p => p.ToLower());
			//_localDB.Dirty = true;
			//var t = CUEToolsLocalDB.Group(_localDB, i => string.Join("$", i.AudioPaths.ToArray()) + "$" + i.DiscID.ToString() + "$" + i.TrackOffsets.ToString() + "$" + i.FirstAudio.ToString() + "$" + i.TrackCount.ToString(), null);
			//foreach (var entries in t)
			//{
			//    if (entries.Count > 1)
			//    {
			//        for (int i = 1; i < entries.Count; i++)
			//        {
			//            entries[0].InputPaths.AddRange(entries[i].InputPaths);
			//            _localDB.Remove(entries[i]);
			//            _localDB.Dirty = true;
			//        }
			//    }
			//}
			labelFormat.ImageList = m_icon_mgr.ImageList;
			toolStripCorrectorFormat.ImageList = m_icon_mgr.ImageList;
			toolStripDropDownButtonCorrectorFormat.DropDown.ImageList = m_icon_mgr.ImageList;
			OpenMinimumSize = MinimumSize;
			ClosedMinimumSize = new Size(Width - grpInput.Width, Height - textBatchReport.Height);
			SizeIncrement = Size - OpenMinimumSize;
			LoadSettings();

			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;

			motdImage = null;
			if (File.Exists(MOTDImagePath))
				using (FileStream imageStream = new FileStream(MOTDImagePath, FileMode.Open, FileAccess.Read))
					try { motdImage = Image.FromStream(imageStream); }
					catch { }

			if (File.Exists(MOTDTextPath))
				try
				{
					using (StreamReader sr = new StreamReader(MOTDTextPath, Encoding.UTF8))
					{
						string version = sr.ReadLine();
						if (version != MOTDVersion)
						{
							string motd = sr.ReadToEnd();
							_batchReport = new StringBuilder();
							_batchReport.Append(motd);
							ReportState = true;
						}
					}
				}
				catch { }

			SetupControls(false);
			UpdateOutputPath();
			updateOutputStyles();
		}

		private void frmCUETools_FormClosed(object sender, FormClosedEventArgs e) {
			SaveDatabase();
			SaveSettings();
		}


		public enum FileBrowserStateEnum
		{
			Tree = 0,
			Checkboxes = 1,
			DragDrop = 2,
			Hidden = 4
		}

		private enum CorrectorModeEnum {
			Locate = 0,
			Extension = 1
		}

		// ********************************************************************************

		private CUEControls.IIconManager m_icon_mgr;
		//private ImageList m_state_image_list;
		List<string> _batchPaths;
		StringBuilder _batchReport;
		string _batchRoot;
		int _batchProcessed;
		bool _usePregapForFirstTrackInSingleFile;
		bool _reducePriority;
		string _defaultLosslessFormat, _defaultLossyFormat, _defaultHybridFormat, _defaultNoAudioFormat;
		int _choiceWidth, _choiceHeight;
		bool _choiceMaxed;
		Thread _workThread;
		CUESheet _workClass;
		CUEToolsProfile _profile, _defaultProfile;
		Size OpenMinimumSize, ClosedMinimumSize, SizeIncrement;
		FileBrowserStateEnum _fileBrowserState = FileBrowserStateEnum.DragDrop;
		FileBrowserStateEnum _fileBrowserControlState = FileBrowserStateEnum.Hidden;
		bool _outputPathUseTemplate = true;
		bool _reportState = true;
		CorrectorModeEnum _correctorMode;
		DateTime _startedAt;
		DateTime lastMOTD;
		Image motdImage = null;
		string profilePath;
		string [] OutputPathUseTemplates = {
			"%music%\\Converted\\%artist%\\[%year% - ]%album%[ '('disc %discnumberandname%')'][' ('%releasedateandlabel%')'][' ('%unique%')']\\%artist% - %album%.cue",
			"%music%\\Converted\\%artist%\\[%year% - ]%album%[' ('%releasedateandlabel%')'][' ('%unique%')']\\%artist% - %album%[ '('disc %discnumberandname%')'].cue",
			"[%directoryname%\\]%filename%-new[%unique%].cue",
			"[%directoryname%\\]new[%unique%]\\%filename%.cue"
		};
		CUEToolsLocalDB _localDB;

		private bool IsCDROM(string pathIn)
		{
			return pathIn.Length == 3 && pathIn.Substring(1) == ":\\" && new DriveInfo(pathIn).DriveType == DriveType.CDRom;
		}

		private void StartConvert() {
			try
			{
				_workThread = null;
				_workClass = null;
				if (_batchPaths.Count != 0)
				{
					InputPath = _batchPaths[0];
					txtInputPath.SelectAll();
				}

				string pathIn = InputPath;
				//if (!File.Exists(pathIn) && !Directory.Exists(pathIn) && !IsCDROM(pathIn))
				//    throw new Exception("Invalid input path.");
				//if (Directory.Exists(pathIn) && !pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
				//{
				//    pathIn = pathIn + Path.DirectorySeparatorChar;
				//    InputPath = pathIn;
				//}

				CUESheet cueSheet = new CUESheet(_profile._config);
				cueSheet.PasswordRequired += new EventHandler<CompressionPasswordRequiredEventArgs>(PasswordRequired);
				cueSheet.CUEToolsProgress += new EventHandler<CUEToolsProgressEventArgs>(SetStatus);
				cueSheet.CUEToolsSelection += new EventHandler<CUEToolsSelectionEventArgs>(MakeSelection);
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

		private void PasswordRequired(object sender, CompressionPasswordRequiredEventArgs e)
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
				if (_choiceWidth != 0 && _choiceHeight != 0)
					dlg.Size = new Size(_choiceWidth, _choiceHeight);
				if (_choiceMaxed)
					dlg.WindowState = FormWindowState.Maximized;
				dlg.Choices = e.choices;
				if (dlg.ShowDialog(this) == DialogResult.OK)
					e.selection = dlg.ChosenIndex;
				_choiceMaxed = dlg.WindowState == FormWindowState.Maximized;
				if (!_choiceMaxed)
				{
					_choiceHeight = dlg.Height;
					_choiceWidth = dlg.Width;
				}
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

		string MOTDImagePath
		{
			get
			{
				return System.IO.Path.Combine(profilePath, "motd.jpg");
			}
		}

		string MOTDTextPath
		{
			get
			{
				return System.IO.Path.Combine(profilePath, "motd.txt");
			}
		}

		string MOTDVersion
		{
			get
			{
				return "CUETools 2.1.2";
			}
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
			bool outputAudio = action == CUEAction.Encode && audioEncoderType != AudioEncoderType.NoAudio;
			bool useAR = action == CUEAction.Verify || (outputAudio && checkBoxUseAccurateRip.Checked);
			bool useCUEToolsDB = action == CUEAction.Verify && checkBoxVerifyUseCDRepair.Checked;
			bool useLocalDB = action != CUEAction.Verify || checkBoxVerifyUseLocal.Checked;
			bool skipRecent = action == CUEAction.Verify && checkBoxSkipRecent.Checked;

			try
			{
				if (_profile._config.checkForUpdates && DateTime.UtcNow - lastMOTD > TimeSpan.FromDays(1) && _batchReport.Length == 0)
				{
					this.Invoke((MethodInvoker)(() => toolStripStatusLabel1.Text = "Checking for updates..."));
					IWebProxy proxy = _profile._config.GetProxy();
					HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://cuetools.net/motd/motd.jpg");
					req.Proxy = proxy;
					req.Method = "GET";
					try
					{
						using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
							if (resp.StatusCode == HttpStatusCode.OK)
							{
								using (Stream respStream = resp.GetResponseStream())
								using (FileStream motd = new FileStream(MOTDImagePath, FileMode.Create, FileAccess.Write))
								{
									byte[] buff = new byte[0x8000];
									do
									{
										int count = respStream.Read(buff, 0, buff.Length);
										if (count == 0) break;
										motd.Write(buff, 0, count);
									} while (true);
								}
							}
							else
							{
								File.Delete(MOTDImagePath);
							}
						lastMOTD = DateTime.UtcNow;
					}
					catch (WebException ex)
					{
						if (ex.Status == WebExceptionStatus.ProtocolError && ex.Response != null && ex.Response is HttpWebResponse)
						{
							HttpWebResponse resp = (HttpWebResponse)ex.Response;
							if (resp.StatusCode == HttpStatusCode.NotFound)
							{
								File.Delete(MOTDImagePath);
								lastMOTD = DateTime.UtcNow;
							}
						}
					}

					motdImage = null;
					if (File.Exists(MOTDImagePath))
						using (FileStream imageStream = new FileStream(MOTDImagePath, FileMode.Open, FileAccess.Read))
							try { motdImage = Image.FromStream(imageStream); }
							catch { }

					req = (HttpWebRequest)WebRequest.Create("http://cuetools.net/motd/motd.txt");
					req.Proxy = proxy;
					req.Method = "GET";
					try
					{
						using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
						{
							if (resp.StatusCode == HttpStatusCode.OK)
							{
								using (Stream respStream = resp.GetResponseStream())
								using (FileStream motd = new FileStream(MOTDTextPath, FileMode.Create, FileAccess.Write))
								using (StreamReader sr = new StreamReader(respStream, Encoding.UTF8))
								using (StreamWriter sw = new StreamWriter(motd, Encoding.UTF8))
								{
									string motdText = sr.ReadToEnd();
									sw.Write(motdText);
								}
							}
							else
							{
								File.Delete(MOTDTextPath);
							}
						}
						lastMOTD = DateTime.UtcNow;
					}
					catch { }
					if (File.Exists(MOTDTextPath))
						try
						{
							using (StreamReader sr = new StreamReader(MOTDTextPath, Encoding.UTF8))
							{
								string version = sr.ReadLine();
								if (version != MOTDVersion)
								{
									string motd = sr.ReadToEnd();
									_batchReport.Append(motd);
								}
							}
						}
						catch { }
				}
				if (action == CUEAction.CreateDummyCUE)
				{
					if (Directory.Exists(pathIn))
					{
						if (_batchPaths.Count == 0)
							throw new Exception("is a directory");
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_profile._config, pathIn);
						int directoriesFound = 0, cueSheetsFound = 0;
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.Folder)
								_batchPaths.Insert(++directoriesFound, fileGroup.main.FullName);
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.CUESheetFile)
								throw new Exception("already contains a cue sheet");
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.TrackFiles || fileGroup.type == FileGroupInfoType.FileWithCUE || fileGroup.type == FileGroupInfoType.M3UFile)
								_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
					}
					else if (File.Exists(pathIn))
					{
						pathIn = Path.GetFullPath(pathIn);
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_profile._config, Path.GetDirectoryName(pathIn));
						FileGroupInfo fileGroup = fileGroups.Find(f => f.type == FileGroupInfoType.TrackFiles && f.Contains(pathIn)) ??
							fileGroups.Find(f => f.type == FileGroupInfoType.FileWithCUE && f.Contains(pathIn)) ??
							fileGroups.Find(f => f.type == FileGroupInfoType.M3UFile && f.Contains(pathIn));
						if (fileGroup == null)
							throw new Exception("doesn't seem to be part of an album");
						string cueSheetContents;
						if (_batchPaths.Count == 0)
						{
							cueSheet.Open(fileGroup.main.FullName);
							cueSheetContents = cueSheet.CUESheetContents();
							cueSheet.Close();
						}
						else
							cueSheetContents = CUESheet.CreateDummyCUESheet(_profile._config, fileGroup);
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
						if (CorrectorMode == CorrectorModeEnum.Locate)
							fixedCue = CUESheet.CorrectAudioFilenames(_profile._config, Path.GetDirectoryName(pathIn), cue, true, null, out extension);
						else
						{
							extension = toolStripDropDownButtonCorrectorFormat.Text;
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
							if (toolStripButtonCorrectorOverwrite.Checked)
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
					}
					else
						throw new Exception("invalid path");
				}
				else
				{
					if (Directory.Exists(pathIn) && !IsCDROM(pathIn))
					{
						if (_batchPaths.Count == 0)
							throw new Exception("is a directory");
						List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_profile._config, pathIn);
						int directoriesFound = 0, cueSheetsFound = 0;
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.Folder)
								_batchPaths.Insert(++directoriesFound, fileGroup.main.FullName);
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.FileWithCUE)
								_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
						foreach (FileGroupInfo fileGroup in fileGroups)
							if (fileGroup.type == FileGroupInfoType.CUESheetFile)
							{
								string cue;
								using (TextReader tr = new StreamReader(fileGroup.main.FullName))
									cue = tr.ReadToEnd();
								foreach (FileGroupInfo fileGroup2 in fileGroups)
									if (fileGroup2.type == FileGroupInfoType.FileWithCUE && fileGroup2.TOC != null)
									{
										CDImageLayout toc = CUESheet.CUE2TOC(cue, (int)fileGroup2.TOC.AudioLength);
										if (toc != null && toc.TrackOffsets == fileGroup2.TOC.TrackOffsets)
										{
											cue = null;
											break;
										}
									}
								if (cue != null)
									_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
							}
						if (cueSheetsFound == 0)
							foreach (FileGroupInfo fileGroup in fileGroups)
								if (fileGroup.type == FileGroupInfoType.TrackFiles)
									_batchPaths.Insert(directoriesFound + (++cueSheetsFound), fileGroup.main.FullName);
					}
					else if (File.Exists(pathIn) || IsCDROM(pathIn))
					{
						string pathOut = null;

						if (Directory.Exists(pathIn) && !pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
							pathIn = pathIn + Path.DirectorySeparatorChar;

						var fullInputPath = CUEToolsLocalDBEntry.NormalizePath(pathIn);
						var recentEntry = skipRecent ? _localDB.Find(item =>
							item.HasPath(fullInputPath) && item.VerificationDate != DateTime.MinValue && item.Status != null && item.VerificationDate.AddDays(30) > DateTime.Now) : null;
						if (recentEntry != null)
							throw new Exception("recently verified: " + recentEntry.Status);

						cueSheet.Action = action;
						cueSheet.OutputStyle = cueStyle;
						cueSheet.Open(pathIn);
						cueSheet.PreGapLengthMSF = txtPreGapLength.Text;
						if (useAR || useCUEToolsDB)
							cueSheet.DataTrackLengthMSF = txtDataTrackLength.Text;
						if (useLocalDB)
							cueSheet.UseLocalDB(_localDB);
						if (useCUEToolsDB)
							cueSheet.UseCUEToolsDB("CUETools " + CUESheet.CUEToolsVersion, null, true, CTDBMetadataSearch.None);
						if (useAR)
							cueSheet.UseAccurateRip();

						List<string> fullAudioPaths = cueSheet.SourcePaths.ConvertAll(sp => CUEToolsLocalDBEntry.NormalizePath(sp));
						recentEntry = skipRecent ? _localDB.Find(item =>
							item.Equals(cueSheet.TOC, fullAudioPaths) && item.VerificationDate != null && item.Status != null && item.VerificationDate.AddDays(30) > DateTime.Now) : null;
						if (recentEntry != null)
						{
							if (useLocalDB)
							{
								_localDB.Dirty = true;
								if (recentEntry.InputPaths == null)
									recentEntry.InputPaths = new List<string>();
								if (!recentEntry.InputPaths.Contains(fullInputPath))
									recentEntry.InputPaths.Add(fullInputPath);
							}
							throw new Exception("recently verified: " + recentEntry.Status);
						}

						this.Invoke((MethodInvoker)delegate()
						{
							toolStripStatusLabelAR.Visible = useAR;
							toolStripStatusLabelCTDB.Visible = useCUEToolsDB;

							if (_batchPaths.Count == 0 && action == CUEAction.Encode && (checkBoxUseFreeDb.Checked || checkBoxUseMusicBrainz.Checked))
							{
								frmChoice dlg = new frmChoice();
								if (_choiceWidth != 0 && _choiceHeight != 0)
									dlg.Size = new Size(_choiceWidth, _choiceHeight);
								if (_choiceMaxed)
									dlg.WindowState = FormWindowState.Maximized;
								dlg.CUE = cueSheet;
								dlg.LookupAlbumInfo(_profile._config.advanced.CacheMetadata,
									true,
									true,
									CTDBMetadataSearch.Default);
								dlgRes = dlg.ShowDialog(this);
								_choiceMaxed = dlg.WindowState == FormWindowState.Maximized;
								if (!_choiceMaxed)
								{
									_choiceHeight = dlg.Height;
									_choiceWidth = dlg.Width;
								}
								if (dlgRes == DialogResult.Cancel)
								{
									cueSheet.Close();
									SetupControls(false);
								}
								else if (_profile._config.advanced.CacheMetadata && dlg.ChosenRelease != null)
								{
									var entry = cueSheet.OpenLocalDBEntry();
									if (entry != null)
									{
										_localDB.Dirty = true;
										entry.Metadata.CopyMetadata(dlg.ChosenRelease.metadata);
									}
								}
								dlg.Close();
							}
							else if (_profile._config.advanced.CacheMetadata)
							{
								recentEntry = _localDB.Find(item => item.Equals(cueSheet.TOC, fullAudioPaths));
								if (recentEntry != null)
									cueSheet.CopyMetadata(recentEntry.Metadata);
							}

							UpdateOutputPath(pathIn, cueSheet);
							pathOut = txtOutputPath.Text;
							if (dlgRes != DialogResult.Cancel && cueSheet.AlbumArt.Count != 0)
								pictureBoxMotd.Image = cueSheet.Cover;
							else
								pictureBoxMotd.Image = motdImage;
						});

						if (dlgRes == DialogResult.Cancel)
							return;

						cueSheet.GenerateFilenames(audioEncoderType, outputFormat, pathOut);

						List<string> outputExists = cueSheet.OutputExists();

						dlgRes = DialogResult.Cancel;
						if (outputExists.Count > 0)
						{
							this.Invoke((MethodInvoker)delegate()
							{
								if (overwriteResult == DialogResult.None)
								{
									using (frmOverwrite frm = new frmOverwrite())
									{
										outputExists.ForEach(path => frm.textFiles.AppendText(path + "\n"));
										dlgRes = frm.ShowDialog(this);
										if (frm.checkBoxRemember.Checked)
											overwriteResult = dlgRes;
									}
								}
								else
									dlgRes = overwriteResult;
								if (dlgRes == DialogResult.Yes)
									outputExists.Clear();
								else if (_batchPaths.Count == 0)
									SetupControls(false);
							});
							if (outputExists.Count > 0 && _batchPaths.Count == 0)
							{
								cueSheet.Close();
								return;
							}
						}
						if (outputExists.Count == 0)
						{
							cueSheet.UsePregapForFirstTrackInSingleFile = _usePregapForFirstTrackInSingleFile && !outputAudio;
							if (script == null || (script.builtin && script.name == "default"))
							{
								status = cueSheet.Go();
								if (cueSheet.Config.advanced.CTDBSubmit
									&& useAR 
									&& useCUEToolsDB
									&& cueSheet.ArVerify.ARStatus == null
									&& cueSheet.ArVerify.WorstConfidence() >= 2
									&& (cueSheet.AccurateRipId == null || AccurateRipVerify.CalculateAccurateRipId(cueSheet.TOC) == cueSheet.AccurateRipId)
									&& cueSheet.CTDB.MatchingEntry == null
									&& (cueSheet.CTDB.QueryExceptionStatus == WebExceptionStatus.Success
									 || (cueSheet.CTDB.QueryExceptionStatus == WebExceptionStatus.ProtocolError && cueSheet.CTDB.QueryResponseStatus == HttpStatusCode.NotFound)
									 )
									)
								{
									DialogResult res = DialogResult.OK;
									if (cueSheet.Config.advanced.CTDBAsk)
									{
										bool remember = true;
										this.Invoke((MethodInvoker)delegate()
										{
											var confirm = new frmSubmit();
											res = confirm.ShowDialog(this);
											remember = confirm.checkBoxRemember.Checked;
										});
										if (remember)
										{
											cueSheet.Config.advanced.CTDBSubmit = res == DialogResult.OK;
											cueSheet.Config.advanced.CTDBAsk = false;
										}
									}
									if (res == DialogResult.OK)
									{
										cueSheet.CTDB.Submit(
											(int)cueSheet.ArVerify.WorstConfidence(),
											100,
											cueSheet.Metadata.Artist,
											cueSheet.Metadata.Title + (cueSheet.Metadata.Title != "" && cueSheet.Metadata.DiscNumberAndName != "" ? " (disc " + cueSheet.Metadata.DiscNumberAndName + ")" : ""),
											cueSheet.Metadata.Barcode);
										if (cueSheet.CTDB.SubStatus != null)
											status += ", submit: " + cueSheet.CTDB.SubStatus;
									}
								}
							}
							else
								status = cueSheet.ExecuteScript(script);

							if (_batchPaths.Count > 0)
							{
								_batchProcessed++;
								BatchLog("{0}.", pathIn, status);
							}
							cueSheet.CheckStop();
						}
					}
					else
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
							ReportState = true;
							//frmReport reportForm = new frmReport();
							//reportForm.Message = _batchReport.ToString();
							//reportForm.ShowDialog(this);
						}
						else if (useAR && cueSheet.Processed)
						{
							using (StringWriter sw = new StringWriter())
							{
								cueSheet.GenerateAccurateRipLog(sw);
								_batchReport.Append(sw.ToString());
							}
							ReportState = true;

							//frmReport reportForm = new frmReport();
							//StringWriter sw = new StringWriter();
							//cueSheet.GenerateAccurateRipLog(sw);
							//if (status != null)
							//    reportForm.Text += ": " + status;
							//reportForm.Message = sw.ToString();
							//_batchReport.Append(sw.ToString());
							//sw.Close();
							//reportForm.ShowDialog(this);
						}
						else
							ShowFinishedMessage(cueSheet.PaddedToFrame, status);
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
					String msg = "";
					for (Exception e = ex; e != null; e = e.InnerException)
						msg += ": " + e.Message;
					BatchLog("{0}.", pathIn, msg.Substring(2));

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
				if (_batchPaths.Count == 0)
				{
					SaveDatabase();
				}
				this.BeginInvoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0) {
						SetupControls(false);
						ReportState = true;
						//frmReport reportForm = new frmReport();
						//reportForm.Message = _batchReport.ToString();
						//reportForm.ShowDialog(this);
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
			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (e.percent == 0)
				{
					_startedAt = DateTime.Now;
					toolStripStatusLabelProcessed.Visible = false;
					toolStripProgressBar2.ToolTipText = "";
				}
				else if (e.percent > 0.02)
				{
					TimeSpan span = DateTime.Now - _startedAt;
					TimeSpan eta = new TimeSpan((long)(span.Ticks / e.percent));
					string speedStr = "";
					if (span.TotalSeconds > 0 && e.offset > 0)
					{
						double speed = e.offset / span.TotalSeconds / 44100;
						speedStr = String.Format("{0:00.00}x", speed);
					}
					toolStripProgressBar2.ToolTipText = String.Format("{0}:{1:00}/{2}:{3:00}", (int)span.TotalMinutes, span.Seconds, (int)eta.TotalMinutes, eta.Seconds);
					toolStripStatusLabelProcessed.Text = String.Format("{0}@{1}", toolStripProgressBar2.ToolTipText, speedStr);
					toolStripStatusLabelProcessed.Visible = true;
				}
				toolStripStatusLabel1.Text = e.status.Replace("&", "&&");
				toolStripProgressBar2.Value = Math.Max(0, Math.Min(100, (int)(e.percent * 100)));

				toolStripStatusLabelAR.Enabled = e.cueSheet != null && e.cueSheet.ArVerify != null && e.cueSheet.ArVerify.ARStatus == null;
				toolStripStatusLabelAR.Text = e.cueSheet != null && e.cueSheet.ArVerify != null && e.cueSheet.ArVerify.ExceptionStatus == WebExceptionStatus.Success ? e.cueSheet.ArVerify.WorstTotal().ToString() : "";
				toolStripStatusLabelAR.ToolTipText = e.cueSheet != null && e.cueSheet.ArVerify != null ? "AccurateRip: " + (e.cueSheet.ArVerify.ARStatus ?? "found") + "." : "";
				toolStripStatusLabelCTDB.Enabled = e.cueSheet != null && e.cueSheet.CTDB != null && e.cueSheet.CTDB.DBStatus == null;
				toolStripStatusLabelCTDB.Text = e.cueSheet != null && e.cueSheet.CTDB != null && e.cueSheet.CTDB.DBStatus == null ? e.cueSheet.CTDB.Total.ToString() : "";
				toolStripStatusLabelCTDB.ToolTipText = e.cueSheet != null && e.cueSheet.CTDB != null ? "CUETools DB: " + (e.cueSheet.CTDB.DBStatus ?? "found") + "." : "";
			});
		}

		private void SetupControls(bool running) {
			bool converting = (SelectedAction == CUEAction.Encode);
			bool verifying = (SelectedAction == CUEAction.Verify || (SelectedAction == CUEAction.Encode && SelectedOutputAudioType != AudioEncoderType.NoAudio && checkBoxUseAccurateRip.Checked));
			//grpInput.Enabled = !running;
			toolStripMenu.Enabled = !running;
			fileSystemTreeView1.Enabled = !running;
			txtInputPath.Enabled = !running;
			grpExtra.Enabled = !running && (converting || verifying);
			//groupBoxCorrector.Enabled = !running && SelectedAction == CUEAction.CorrectFilenames;
			//grpOutputStyle.Enabled = !running && converting;
			groupBoxMode.Enabled = !running;
			toolStripCorrectorFormat.Visible = SelectedAction == CUEAction.CorrectFilenames;
			tableLayoutPanelCUEStyle.Visible = converting;
			tableLayoutPanelVerifyMode.Visible = SelectedAction == CUEAction.Verify;
			grpOutputPathGeneration.Enabled = !running;
			grpAudioOutput.Enabled = !running && converting;
			grpAction.Enabled = !running;
			//checkBoxUseFreeDb.Enabled = 
			//    checkBoxUseMusicBrainz.Enabled =
			//    checkBoxUseAccurateRip.Enabled =
			//    !(FileBrowserState == FileBrowserStateEnum.DragDrop || FileBrowserState == FileBrowserStateEnum.Checkboxes) && converting;
			txtDataTrackLength.Enabled = !running && verifying;
			txtPreGapLength.Enabled = !running;
			btnConvert.Visible = !running;
			btnStop.Enabled = btnPause.Enabled = btnResume.Enabled = running;
			btnStop.Visible = btnPause.Visible = running;
			btnResume.Visible = false;
			toolStripStatusLabel1.Text = String.Empty;
			toolStripProgressBar2.Value = 0;
			toolStripStatusLabelAR.Visible = false;
			toolStripStatusLabelCTDB.Visible = false;
			if (ReportState)
			{
				string newText = _batchReport != null ? _batchReport.ToString() : "";
				string oldText = textBatchReport.Text;
				if (oldText != "" && newText.StartsWith(oldText))
					textBatchReport.AppendText(newText.Substring(oldText.Length));
				else
					textBatchReport.Text = newText;
			}

			if (!running)
			{
				UpdateActions();
				pictureBoxMotd.Image = motdImage;
				toolStripStatusLabelProcessed.Visible = false;
			}

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

		private void ShowFinishedMessage(bool warnAboutPadding, string status) {
			if (_batchPaths.Count != 0) {
				return;
			}
			if (warnAboutPadding) {
				MessageBox.Show(this, "One or more input file doesn't end on a CD frame boundary.  " +
					"The output has been padded where necessary to fix this.  If your input " +
					"files are from a CD source, this may indicate a problem with your files.",
					"Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
			}
			MessageBox.Show(this, status, "Done", MessageBoxButtons.OK,
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

		private void ActivateProfile()
		{
			SelectedOutputAudioType = _profile._outputAudioType;
			SelectedOutputAudioFormat = _profile._outputAudioFormat;
			SelectedAction = _profile._action;
			SelectedCUEStyle = _profile._CUEStyle;
			numericWriteOffset.Value = _profile._writeOffset;
			comboBoxOutputFormat.Text = _profile._outputTemplate ?? comboBoxOutputFormat.Items[0].ToString();
			toolStripDropDownButtonProfile.Text = _profile._name;
			SelectedScript = _profile._script;
			checkBoxUseFreeDb.Checked = _profile._useFreeDb;
			checkBoxUseMusicBrainz.Checked = _profile._useMusicBrainz;
			checkBoxUseAccurateRip.Checked = _profile._useAccurateRip;
			checkBoxVerifyUseCDRepair.Checked = _profile._useCUEToolsDB;
			checkBoxVerifyUseLocal.Checked = _profile._useLocalDB;
			checkBoxSkipRecent.Checked = _profile._skipRecent;
		}

		private void ActivateProfile(string profileName)
		{
			if (profileName == _defaultProfile._name)
				return;
			_profile = new CUEToolsProfile(profileName);
			SettingsReader sr = new SettingsReader("CUE Tools", string.Format("profile-{0}.txt", _profile._name), Application.ExecutablePath);
			_profile.Load(sr);
			_profile._config.encoders = _defaultProfile._config.encoders;
			_profile._config.decoders = _defaultProfile._config.decoders;
			_profile._config.formats = _defaultProfile._config.formats;
			_profile._config.scripts = _defaultProfile._config.scripts;
			ActivateProfile();
		}

		private void DeactivateProfile()
		{
			SaveProfile();

			if (_profile != _defaultProfile)
			{
				_defaultProfile._config.encoders = _profile._config.encoders;
				_defaultProfile._config.decoders = _profile._config.decoders;
				_defaultProfile._config.formats = _profile._config.formats;
				_defaultProfile._config.scripts = _profile._config.scripts;
				_profile = _defaultProfile;
				ActivateProfile();
			}
		}

		private void SaveProfile()
		{
			_profile._outputAudioType = SelectedOutputAudioType;
			_profile._outputAudioFormat = SelectedOutputAudioFormat;
			_profile._action = SelectedAction;
			_profile._CUEStyle = SelectedCUEStyle;
			_profile._writeOffset = (int) numericWriteOffset.Value;
			_profile._outputTemplate = comboBoxOutputFormat.Text;
			_profile._script = SelectedScript;
			_profile._useFreeDb = checkBoxUseFreeDb.Checked;
			_profile._useMusicBrainz = checkBoxUseMusicBrainz.Checked;
			_profile._useAccurateRip = checkBoxUseAccurateRip.Checked;
			_profile._useCUEToolsDB = checkBoxVerifyUseCDRepair.Checked;
			_profile._useLocalDB = checkBoxVerifyUseLocal.Checked;
			_profile._skipRecent = checkBoxSkipRecent.Checked;
			
			if (_profile != _defaultProfile)
			{
				SettingsWriter sw = new SettingsWriter("CUE Tools", string.Format("profile-{0}.txt", _profile._name), Application.ExecutablePath);
				_profile.Save(sw);
				sw.Close();
			}
		}

		private void LoadSettings() {
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt", Application.ExecutablePath);
			profilePath = sr.ProfilePath;
			_profile.Load(sr);
			lastMOTD = sr.LoadDate("LastMOTD") ?? DateTime.FromBinary(0);
			_choiceWidth = sr.LoadInt32("ChoiceWidth", null, null) ?? 0;
			_choiceHeight = sr.LoadInt32("ChoiceHeight", null, null) ?? 0;
			_choiceMaxed = sr.LoadBoolean("ChoiceMaxed") ?? false;
			_defaultLosslessFormat = sr.Load("DefaultLosslessFormat") ?? "flac";
			_defaultLossyFormat = sr.Load("DefaultLossyFormat") ?? "mp3";
			_defaultHybridFormat = sr.Load("DefaultHybridFormat") ?? "lossy.flac";
			_defaultNoAudioFormat = sr.Load("DefaultNoAudioFormat") ?? "wav";
			int iFormat, nFormats = sr.LoadInt32("OutputPathUseTemplates", 0, 10) ?? 0;
			for (iFormat = 0; iFormat < OutputPathUseTemplates.Length; iFormat++)
				comboBoxOutputFormat.Items.Add(OutputPathUseTemplates[iFormat]);
			for (iFormat = nFormats - 1; iFormat >= 0; iFormat --)
				comboBoxOutputFormat.Items.Add(sr.Load(string.Format("OutputPathUseTemplate{0}", iFormat)) ?? "");
			OutputPathUseTemplate = !(sr.LoadBoolean("DontGenerate") ?? false);

			ActivateProfile();

			_usePregapForFirstTrackInSingleFile = sr.LoadBoolean("UsePregapForFirstTrackInSingleFile") ?? false;
			_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
			CorrectorMode = (CorrectorModeEnum)(sr.LoadInt32("CorrectorLookup", null, null) ?? (int) CorrectorModeEnum.Locate);
			toolStripButtonCorrectorOverwrite.Checked = sr.LoadBoolean("CorrectorOverwrite") ?? true;
			string correctorFormat = sr.Load("CorrectorFormat") ?? "flac";
			foreach (KeyValuePair<string, CUEToolsFormat> format in _profile._config.formats)
			{
				ToolStripItem item = new ToolStripMenuItem(format.Key);
				item.ImageKey = "." + format.Value.extension;
				toolStripDropDownButtonCorrectorFormat.DropDownItems.Add(item);
				if (correctorFormat == format.Key)
				{
					toolStripDropDownButtonCorrectorFormat.Text = item.Text;
					toolStripDropDownButtonCorrectorFormat.ImageKey = item.ImageKey;
				}
			}
			SizeIncrement.Width = sr.LoadInt32("WidthIncrement", 0, null) ?? 0;
			SizeIncrement.Height = sr.LoadInt32("HeightIncrement", 0, null) ?? 0;
			Size = OpenMinimumSize + SizeIncrement;
			Top = sr.LoadInt32("Top", 0, null) ?? Top;
			Left = sr.LoadInt32("Left", 0, null) ?? Left;
			if (InputPath == "")
			{
				InputPath = sr.Load("InputPath") ?? "";
				FileBrowserState = (FileBrowserStateEnum)(sr.LoadInt32("FileBrowserState", (int)FileBrowserStateEnum.Tree, (int)FileBrowserStateEnum.Hidden) ?? (int)FileBrowserStateEnum.Hidden);
			}
			else
				FileBrowserState = FileBrowserStateEnum.Hidden;
			ReportState = sr.LoadBoolean("ReportState") ?? false;
			PerformLayout();
			string profiles = sr.Load("Profiles") ?? "verify convert fix";
			foreach (string prof in profiles.Split(' '))
				toolStripDropDownButtonProfile.DropDownItems.Add(prof);
		}

		private void SaveSettings() 
		{
			SaveProfile();

			SettingsWriter sw = new SettingsWriter("CUE Tools", "settings.txt", Application.ExecutablePath);
			SaveScripts(SelectedAction);
			sw.Save("LastMOTD", lastMOTD);
			sw.Save("ChoiceWidth", _choiceWidth);
			sw.Save("ChoiceHeight", _choiceHeight);
			sw.Save("ChoiceMaxed", _choiceMaxed);
			sw.Save("InputPath", InputPath);
			sw.Save("DefaultLosslessFormat", _defaultLosslessFormat);
			sw.Save("DefaultLossyFormat", _defaultLossyFormat);
			sw.Save("DefaultHybridFormat", _defaultHybridFormat);
			sw.Save("DefaultNoAudioFormat", _defaultNoAudioFormat);
			sw.Save("DontGenerate", !_outputPathUseTemplate);
			sw.Save("OutputPathUseTemplates", comboBoxOutputFormat.Items.Count - OutputPathUseTemplates.Length);
			for (int iFormat = comboBoxOutputFormat.Items.Count - 1; iFormat >= OutputPathUseTemplates.Length; iFormat--)
				sw.Save(string.Format("OutputPathUseTemplate{0}", iFormat - OutputPathUseTemplates.Length), comboBoxOutputFormat.Items[iFormat].ToString());

			sw.Save("UsePregapForFirstTrackInSingleFile", _usePregapForFirstTrackInSingleFile);
			sw.Save("ReducePriority", _reducePriority);
			sw.Save("FileBrowserState", (int)FileBrowserState);
			sw.Save("ReportState", ReportState);
			sw.Save("CorrectorLookup", (int) CorrectorMode);
			sw.Save("CorrectorOverwrite", toolStripButtonCorrectorOverwrite.Checked);
			sw.Save("CorrectorFormat", toolStripDropDownButtonCorrectorFormat.Text);
			sw.Save("WidthIncrement", FileBrowserState == FileBrowserStateEnum.Hidden ? SizeIncrement.Width : Width - OpenMinimumSize.Width);
			sw.Save("HeightIncrement", !ReportState ? SizeIncrement.Height : Height - OpenMinimumSize.Height);
			sw.Save("Top", Top);
			sw.Save("Left", Left);

			StringBuilder profiles = new StringBuilder();
			foreach(ToolStripItem item in toolStripDropDownButtonProfile.DropDownItems)
				if (item != toolStripTextBoxAddProfile
					&& item != toolStripMenuItemDeleteProfile
					&& item != defaultToolStripMenuItem
					&& item != toolStripSeparator5
					)
				{
					if (profiles.Length > 0)
						profiles.Append(' ');
					profiles.Append(item.Text);
				}
			sw.Save("Profiles", profiles.ToString());

			_defaultProfile.Save(sw);
			sw.Close();
		}

		private CUEStyle SelectedCUEStyle {
			get {
				if (rbEmbedCUE.Checked)		return CUEStyle.SingleFileWithCUE;
				if (rbSingleFile.Checked)   return CUEStyle.SingleFile;
				return _profile._config.gapsHandling;
			}
			set {
				switch (value) {
					case CUEStyle.SingleFileWithCUE: rbEmbedCUE.Checked = true; break;
					case CUEStyle.SingleFile:	 rbSingleFile.Checked = true; break;
					default: rbTracks.Checked = true; break;
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
				return _profile._config.formats.TryGetValue(formatName, out fmt) ? fmt : null;
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
						radioButtonAudioNone.Checked = false;
						radioButtonAudioNone.Checked = true;
						break;
					case AudioEncoderType.Hybrid:
						radioButtonAudioHybrid.Checked = false;
						radioButtonAudioHybrid.Checked = true;
						break;
					case AudioEncoderType.Lossy:
						radioButtonAudioLossy.Checked = false;
						radioButtonAudioLossy.Checked = true;
						break;
					default:
						radioButtonAudioLossless.Checked = false;
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

		private void ResetSize()
		{
			MinimumSize = new Size(
				_fileBrowserState == FileBrowserStateEnum.Hidden ? ClosedMinimumSize.Width : OpenMinimumSize.Width,
				!_reportState ? ClosedMinimumSize.Height : OpenMinimumSize.Height);
			MaximumSize = new Size(
				_fileBrowserState == FileBrowserStateEnum.Hidden ? ClosedMinimumSize.Width : 4 * OpenMinimumSize.Width,
				!_reportState ? ClosedMinimumSize.Height : 4 * OpenMinimumSize.Height);
			Width = _fileBrowserState == FileBrowserStateEnum.Hidden ? ClosedMinimumSize.Width : OpenMinimumSize.Width + SizeIncrement.Width;
			Height = !_reportState ? ClosedMinimumSize.Height : OpenMinimumSize.Height + SizeIncrement.Height;
			PerformLayout();
		}

		private void SaveSize()
		{
			if (_fileBrowserState != FileBrowserStateEnum.Hidden)
				SizeIncrement.Width = Width - OpenMinimumSize.Width;
			if (_reportState)
				SizeIncrement.Height = Height - OpenMinimumSize.Height;
		}

		private bool ReportState
		{
			get
			{
				return _reportState;
			}
			set
			{
				toolStripButtonShowLog.Checked = value;
				SaveSize();
				_reportState = value;
				ResetSize();
			}
		}

		bool OutputPathUseTemplate
		{
			get
			{
				return _outputPathUseTemplate;
			}
			set
			{
				_outputPathUseTemplate = value;
				toolStripSplitButtonOutputBrowser.DefaultItem = _outputPathUseTemplate
					? toolStripMenuItemOutputManual : toolStripMenuItemOutputBrowse;
				toolStripSplitButtonOutputBrowser.Text = toolStripSplitButtonOutputBrowser.DefaultItem.Text;
				toolStripSplitButtonOutputBrowser.Image = toolStripSplitButtonOutputBrowser.DefaultItem.Image;
				toolStripSplitButtonOutputBrowser.Enabled = true;// toolStripSplitButtonOutputBrowser.DefaultItem.Enabled;
				UpdateOutputPath();
			}
		}

		private ToolStripMenuItem FileBrowserStateButton(FileBrowserStateEnum state)
		{
			switch (state)
			{
				case FileBrowserStateEnum.Tree: return toolStripMenuItemInputBrowserFiles;
				case FileBrowserStateEnum.Checkboxes: return toolStripMenuItemInputBrowserMulti;
				case FileBrowserStateEnum.DragDrop: return toolStripMenuItemInputBrowserDrag;
				case FileBrowserStateEnum.Hidden: return toolStripMenuItemInputBrowserHide;
			}
			return null;
		}

		private FileBrowserStateEnum FileBrowserState
		{
			get
			{
				return _fileBrowserState;
			}
			set
			{
				ToolStripMenuItem inputBtn = FileBrowserStateButton(value);
				if (inputBtn == null) { value = FileBrowserStateEnum.Hidden; inputBtn = toolStripMenuItemInputBrowserHide; }
				ToolStripMenuItem defaultBtn = FileBrowserStateButton(value != FileBrowserStateEnum.Hidden
					? FileBrowserStateEnum.Hidden : _fileBrowserControlState == FileBrowserStateEnum.Hidden
					? FileBrowserStateEnum.Tree : _fileBrowserControlState);
				toolStripSplitButtonInputBrowser.Text = defaultBtn.Text;
				toolStripSplitButtonInputBrowser.Image = defaultBtn.Image;
				toolStripSplitButtonInputBrowser.DefaultItem = defaultBtn;
				grpInput.Text = inputBtn.Text;

				if (value == _fileBrowserState && _fileBrowserControlState != FileBrowserStateEnum.Hidden)
					return;

				UseWaitCursor = true;

				SaveSize();
				_fileBrowserState = value;
				ResetSize();

				switch (value)
				{
					case FileBrowserStateEnum.Tree:
					case FileBrowserStateEnum.Checkboxes:
						if (value == FileBrowserStateEnum.Checkboxes)
							OutputPathUseTemplate = true;
						if (_fileBrowserControlState != value)
						{
							fileSystemTreeView1.CheckBoxes = value == FileBrowserStateEnum.Checkboxes;
							if (_fileBrowserControlState != FileBrowserStateEnum.Tree &&
								_fileBrowserControlState != FileBrowserStateEnum.Checkboxes)
							{
								fileSystemTreeView1.Nodes.Clear();
								fileSystemTreeView1.IconManager = m_icon_mgr;
								fileSystemTreeView1.Nodes.Add(new FileSystemTreeNodeLocalDB(m_icon_mgr, _localDB));
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
							}
							foreach (TreeNode node in fileSystemTreeView1.Nodes)
								node.Expand();
							
							if (value == FileBrowserStateEnum.Checkboxes
								&& fileSystemTreeView1.SelectedNode != null
								&& fileSystemTreeView1.SelectedNode is CUEControls.FileSystemTreeNodeFileSystemInfo)
							{
								fileSystemTreeView1.SelectedNode.Checked = true;
								fileSystemTreeView1.SelectedNode.Expand();
							}
							_fileBrowserControlState = value;
						}
						fileSystemTreeView1.Select();
						fileSystemTreeView1.ShowRootLines = false;
						break;
					case FileBrowserStateEnum.DragDrop:
						OutputPathUseTemplate = true;
						if (_fileBrowserControlState != value)
						{
							fileSystemTreeView1.CheckBoxes = false;
							fileSystemTreeView1.Nodes.Clear();
							int icon = m_icon_mgr.GetIconIndex(CUEControls.ExtraSpecialFolder.Desktop, true);
							fileSystemTreeView1.Nodes.Add(null, "Drag the files here", icon, icon);
							fileSystemTreeView1.IconManager = m_icon_mgr;
							_fileBrowserControlState = value;
						}
						fileSystemTreeView1.Select();
						fileSystemTreeView1.ShowRootLines = false;
						break;
					case FileBrowserStateEnum.Hidden:
						break;
				}
				UseWaitCursor = false;
			}
		}

		private CUEAction SelectedAction
		{
			get
			{
				return
					rbActionVerify.Checked ? CUEAction.Verify :
					rbActionCorrectFilenames.Checked ? CUEAction.CorrectFilenames :
					rbActionCreateCUESheet.Checked ? CUEAction.CreateDummyCUE :
					CUEAction.Encode;
			}
			set
			{
				switch (value)
				{
					case CUEAction.Verify:
						rbActionVerify.Checked = true;
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

		private void UpdateOutputPath() 
		{
			UpdateOutputPath(InputPath, null);
		}

		private void UpdateOutputPath(string pathIn, CUESheet cueSheet)
		{
			if (!OutputPathUseTemplate)
			{
				txtOutputPath.ReadOnly = false;
				comboBoxOutputFormat.Enabled = false;
			}
			else
			{
				txtOutputPath.ReadOnly = true;
				comboBoxOutputFormat.Enabled =
					SelectedAction != CUEAction.CorrectFilenames &&
					SelectedAction != CUEAction.CreateDummyCUE &&
					(SelectedAction != CUEAction.Verify || !_profile._config.arLogToSourceFolder);

				txtOutputPath.Text = CUESheet.GenerateUniqueOutputPath(
					_profile._config,
					comboBoxOutputFormat.Text,
					SelectedCUEStyle == CUEStyle.SingleFileWithCUE ? "." + SelectedOutputAudioFormat : ".cue",
					SelectedAction,
					new NameValueCollection(),
					pathIn,
					cueSheet);
			}
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
			if ((_workThread == null || _workThread.IsAlive) && _workClass != null)
				_workClass.Stop();
		}

		private void btnPause_Click(object sender, EventArgs e)
		{
			if ((_workThread == null || _workThread.IsAlive) && _workClass != null)
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
			switch (e.KeyCode)
			{
				case Keys.F5:
					string was = fileSystemTreeView1.SelectedPath;
					foreach (TreeNode node in fileSystemTreeView1.Nodes)
						node.Collapse();
					if (was != null)
						fileSystemTreeView1.SelectedPath = was;
					break;
				case Keys.Delete:
					if (FileBrowserState == FileBrowserStateEnum.DragDrop && fileSystemTreeView1.SelectedNode != null)
						fileSystemTreeView1.Nodes.Remove(fileSystemTreeView1.SelectedNode);
					break;
			}
		}

		private void fileSystemTreeView1_NodeExpand(object sender, CUEControls.FileSystemTreeViewNodeExpandEventArgs e)
		{
			List<FileGroupInfo> fileGroups = CUESheet.ScanFolder(_profile._config, e.files);
			foreach (FileGroupInfo fileGroup in fileGroups)
			{
				TreeNode node = fileSystemTreeView1.NewNode(fileGroup.main);
				node.Text = fileGroup.ToString();
				e.node.Nodes.Add(node);
			}
			//toolTip1.Show
		}

		private void UpdateActions()
		{
			if (FileBrowserState == FileBrowserStateEnum.DragDrop || FileBrowserState == FileBrowserStateEnum.Checkboxes)
			{
				rbActionCorrectFilenames.Enabled = true;
				rbActionCreateCUESheet.Enabled = true;
				rbActionEncode.Enabled = true;
				rbActionVerify.Enabled = true;
				//toolStripSplitButtonOutputBrowser.Enabled = false;
				toolStripMenuItemOutputManual.Enabled = 
					toolStripMenuItemOutputBrowse.Enabled = false;
				txtInputPath.ReadOnly = true;
			}
			else
			{
				string pathIn = InputPath;
				bool is_file = !string.IsNullOrEmpty(pathIn) && File.Exists(pathIn);
				bool is_cue = is_file && Path.GetExtension(pathIn).ToLower() == ".cue";
				bool is_directory = !string.IsNullOrEmpty(pathIn) && Directory.Exists(pathIn);
				bool is_folder = FileBrowserState == FileBrowserStateEnum.Tree && fileSystemTreeView1.SelectedNode is FileSystemTreeNodeLocalDBFolder;
				bool is_cdrom = !string.IsNullOrEmpty(pathIn) && IsCDROM(pathIn);
				rbActionCorrectFilenames.Enabled = is_cue || is_directory;
				rbActionCreateCUESheet.Enabled = (is_file && !is_cue) || is_directory;
				rbActionVerify.Enabled = rbActionEncode.Enabled = is_file || is_directory || is_folder || is_cdrom;
				toolStripMenuItemOutputManual.Enabled = toolStripMenuItemOutputBrowse.Enabled = is_file || is_cdrom;
				txtInputPath.ReadOnly = is_folder;
			}

			btnConvert.Enabled = btnConvert.Visible &&
				 ((rbActionCorrectFilenames.Enabled && rbActionCorrectFilenames.Checked)
				|| (rbActionCreateCUESheet.Enabled && rbActionCreateCUESheet.Checked)
				|| (rbActionEncode.Enabled && rbActionEncode.Checked)
				|| (rbActionVerify.Enabled && rbActionVerify.Checked)
				);

			comboBoxScript.Enabled = btnConvert.Enabled && comboBoxScript.Items.Count > 1;
		}

		private void fileSystemTreeView1_AfterSelect(object sender, TreeViewEventArgs e)
		{
			InputPath = fileSystemTreeView1.SelectedPath ?? "";
			txtInputPath.SelectAll();
			UpdateActions();
			if (fileSystemTreeView1.SelectedNode is FileSystemTreeNodeLocalDBEntry)
			{
				var entry = (fileSystemTreeView1.SelectedNode as FileSystemTreeNodeLocalDBEntry).Item;
				textBatchReport.Text = (entry.Log ?? "").Replace("\r", "").Replace("\n", "\r\n");
			}
			else if (fileSystemTreeView1.SelectedNode is FileSystemTreeNodeLocalDBFolder)
			{
				var group = (fileSystemTreeView1.SelectedNode as FileSystemTreeNodeLocalDBFolder).Group;
				StringBuilder report = new StringBuilder();
				foreach (var entry in group)
					if (entry.Path != null)
						if (entry.Status == null || entry.OffsetSafeCRC == null)
							report.AppendFormat("{0}: never verified\r\n", entry.Path);
						else
							report.AppendFormat("{0}: CRC {1,8:X}, {2}\r\n", entry.Path, entry.OffsetSafeCRC.Value[0], entry.Status);
				textBatchReport.Text = report.ToString();
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
			if (FileBrowserState == FileBrowserStateEnum.Checkboxes)
				foreach (TreeNode node in e.Node.Nodes)
					node.Checked = e.Node.Checked;
		}

		private void fileSystemTreeView1_DragOver(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				if (((e.KeyState & 8) != 0 && FileBrowserState == FileBrowserStateEnum.DragDrop) || FileBrowserState == FileBrowserStateEnum.Checkboxes)
					e.Effect = DragDropEffects.Copy;
				else
					e.Effect = DragDropEffects.Move;
			}
		}

		private void fileSystemTreeView1_DragDrop(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				string[] folders = e.Data.GetData(DataFormats.FileDrop) as string[];
				if (folders != null)
				{
					//if (folders.Length > 1 && !(FileBrowserState == FileBrowserStateEnum.DragDrop || FileBrowserState == FileBrowserStateEnum.Checkboxes))
					//{
					//    FileBrowserState = FileBrowserStateEnum.Checked;
					//    if (fileSystemTreeView1.SelectedNode != null && fileSystemTreeView1.SelectedNode.Checked)
					//        fileSystemTreeView1.SelectedNode.Checked = false;
					//}
					if (folders.Length > 1 && FileBrowserState == FileBrowserStateEnum.Tree)
						FileBrowserState = FileBrowserStateEnum.DragDrop;
					switch (FileBrowserState)
					{
						case FileBrowserStateEnum.Tree:
							fileSystemTreeView1.SelectedPath = folders[0];
							break;
						case FileBrowserStateEnum.Checkboxes:
							foreach (string folder in folders)
							{
								TreeNode node = fileSystemTreeView1.LookupNode(folder);
								if (node != null) node.Checked = true;
							}
							break;
						case FileBrowserStateEnum.DragDrop:
							if (e.Effect == DragDropEffects.Move)
								fileSystemTreeView1.Nodes.Clear();
							foreach (string folder in folders)
							{
								TreeNode node = Directory.Exists(folder)
									? fileSystemTreeView1.NewNode(new DirectoryInfo(folder))
									: fileSystemTreeView1.NewNode(new FileInfo(folder));
								fileSystemTreeView1.Nodes.Add(node);
							}
							break;
					}
					fileSystemTreeView1.Focus();
				}
			}
		}

		private string SelectedScript
		{
			get
			{
				return comboBoxScript.SelectedItem != null ? ((CUEToolsScript)comboBoxScript.SelectedItem).name : "default";
			}
			set
			{
				CUEAction action = SelectedAction;
				comboBoxScript.Items.Clear();
				foreach (KeyValuePair<string, CUEToolsScript> script in _profile._config.scripts)
					if (script.Value.conditions.Contains(action))
						comboBoxScript.Items.Add(script.Value);
				comboBoxScript.Enabled = btnConvert.Enabled && comboBoxScript.Items.Count > 1;
				comboBoxScript.SelectedItem =
					(value != null && _profile._config.scripts.ContainsKey(value)) ? _profile._config.scripts[value] :
					(comboBoxScript.Items.Count > 0 ? comboBoxScript.Items[0] : null);
			}
		}

		private void SetupScripts()
		{
			switch (SelectedAction)
			{
				case CUEAction.Verify:
					SelectedScript = _profile._config.defaultVerifyScript;
					break;
				case CUEAction.Encode:
					SelectedScript = _profile._config.defaultEncodeScript;
					break;
				default:
					SelectedScript = null;
					break;
			}
		}

		private void SaveScripts(CUEAction action)
		{
			switch (action)
			{
				case CUEAction.Verify:
					_profile._config.defaultVerifyScript = SelectedScript;
					break;
				case CUEAction.Encode:
					_profile._config.defaultEncodeScript = SelectedScript;
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
					SaveScripts(CUEAction.Encode);
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
					InputPath = args[0];
					TreeNode node = null;
					switch (FileBrowserState)
					{
						case FileBrowserStateEnum.DragDrop:
							fileSystemTreeView1.Nodes.Clear();
							node = Directory.Exists(InputPath)
								? fileSystemTreeView1.NewNode(new DirectoryInfo(InputPath))
								: fileSystemTreeView1.NewNode(new FileInfo(InputPath));
							fileSystemTreeView1.Nodes.Add(node);
							break;
						case FileBrowserStateEnum.Tree:
						case FileBrowserStateEnum.Checkboxes:
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
								if (FileBrowserState == FileBrowserStateEnum.Checkboxes)
									node.Checked = true;
							}
							fileSystemTreeView1.Select();
							break;
					}
				}
				if (WindowState == FormWindowState.Minimized)
					WindowState = FormWindowState.Normal;
				Activate();
			});
			return true;
		}

		private void setAsMyMusicFolderToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var node = (contextMenuStripFileTree.Tag as CUEControls.FileSystemTreeNode);
			try
			{
				fileSystemTreeView1.IconManager.SetFolderPath(CUEControls.ExtraSpecialFolder.MyMusic, node.Path);
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
			var node = (contextMenuStripFileTree.Tag as CUEControls.FileSystemTreeNode);
			CUEControls.ExtraSpecialFolder dir = (node as CUEControls.FileSystemTreeNodeSpecialFolder).Folder;
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

		private void addFolderToLocalDatabaseToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var node = contextMenuStripFileTree.Tag as CUEControls.FileSystemTreeNode;
			if (node == null || node.Path == null)
				return;
			SetupControls(true);
			backgroundWorkerAddToLocalDB.RunWorkerAsync(node.Path);
		}

		private void SaveDatabase()
		{
			SetStatus(this, new CUEToolsProgressEventArgs() { status = "Saving local database..." });
			try
			{
				_localDB.Save();
			}
			catch (Exception ex)
			{
				if (this.InvokeRequired)
					this.Invoke((MethodInvoker)(() => ShowErrorMessage(ex)));
				else
					this.ShowErrorMessage(ex);
			}
			SetStatus(this, new CUEToolsProgressEventArgs());
		}

		private void editMetadataToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var node = contextMenuStripFileTree.Tag as CUEControls.FileSystemTreeNode;
			string path = null;
			if (node != null && node is FileSystemTreeNodeLocalDBEntry)
				path = node.Path;
			if (node != null && node is FileSystemTreeNodeLocalDBCollision)
				path = (node as FileSystemTreeNodeLocalDBCollision).Group[0].Path;
			if (path == null)
				return;
			var CueSheet = new CUESheet(_profile._config);
			CueSheet.PasswordRequired += new EventHandler<CompressionPasswordRequiredEventArgs>(PasswordRequired);
			CueSheet.CUEToolsProgress += new EventHandler<CUEToolsProgressEventArgs>(SetStatus);
			//cueSheet.CUEToolsSelection += new EventHandler<CUEToolsSelectionEventArgs>(MakeSelection);
			try
			{
				CueSheet.Open(path);
			}
			catch (Exception ex)
			{
				ShowErrorMessage(ex);
				return;
			}
			CueSheet.UseLocalDB(_localDB);
			frmChoice dlg = new frmChoice();
			if (_choiceWidth != 0 && _choiceHeight != 0)
				dlg.Size = new Size(_choiceWidth, _choiceHeight);
			if (_choiceMaxed)
				dlg.WindowState = FormWindowState.Maximized;
			dlg.CUE = CueSheet;
			dlg.LookupAlbumInfo(true, node is FileSystemTreeNodeLocalDBEntry, true, CTDBMetadataSearch.Default);
			var dlgRes = dlg.ShowDialog(this);
			_choiceMaxed = dlg.WindowState == FormWindowState.Maximized;
			if (!_choiceMaxed)
			{
				_choiceHeight = dlg.Height;
				_choiceWidth = dlg.Width;
			}
			if (dlgRes == DialogResult.OK && dlg.ChosenRelease != null)
			{
				if (node is FileSystemTreeNodeLocalDBCollision)
				{
					var group = (node as FileSystemTreeNodeLocalDBCollision).Group;
					foreach (var item in group)
						item.Metadata.CopyMetadata(dlg.ChosenRelease.metadata);
				}
				else if (node is FileSystemTreeNodeLocalDBEntry)
				{
					var item = (node as FileSystemTreeNodeLocalDBEntry).Item;
					item.Metadata.CopyMetadata(dlg.ChosenRelease.metadata);
				}
				node.Text = node.DisplayName;
				_localDB.Dirty = true;
				SaveDatabase();
			}
			CueSheet.Close();
		}

		private void fileSystemTreeView1_MouseDown(object sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Right)
			{				
				TreeViewHitTestInfo info = fileSystemTreeView1.HitTest(e.Location);
				if (info.Node as CUEControls.FileSystemTreeNode != null)
				{
					contextMenuStripFileTree.Tag = info.Node as CUEControls.FileSystemTreeNode;
					SelectedNodeName.Text = info.Node.Text;
					SelectedNodeName.Image = m_icon_mgr.ImageList.Images[info.Node.ImageIndex];
					resetToOriginalLocationToolStripMenuItem.Visible = false;
					setAsMyMusicFolderToolStripMenuItem.Visible = false;
					editMetadataToolStripMenuItem.Visible = false;
					addFolderToLocalDatabaseToolStripMenuItem.Visible = false;
					removeItemFromDatabaseToolStripMenuItem.Visible = false;
					if (info.Node is CUEControls.FileSystemTreeNodeFileSystemInfo && (info.Node as CUEControls.FileSystemTreeNodeFileSystemInfo).File is DirectoryInfo)
					{
						setAsMyMusicFolderToolStripMenuItem.Visible = true;
						setAsMyMusicFolderToolStripMenuItem.Image = m_icon_mgr.ImageList.Images[m_icon_mgr.GetIconIndex(CUEControls.ExtraSpecialFolder.MyMusic, true)];
						addFolderToLocalDatabaseToolStripMenuItem.Visible = true;
					}
					else if (info.Node is CUEControls.FileSystemTreeNodeSpecialFolder && (info.Node as CUEControls.FileSystemTreeNodeSpecialFolder).Folder == CUEControls.ExtraSpecialFolder.MyMusic)
					{
						resetToOriginalLocationToolStripMenuItem.Visible = true;
					}
					else if (info.Node is FileSystemTreeNodeLocalDBCollision)
					{
						editMetadataToolStripMenuItem.Visible = true;
					}
					else if (info.Node as FileSystemTreeNodeLocalDBEntry != null)
					{
						editMetadataToolStripMenuItem.Visible = true;
					}

					if (info.Node is FileSystemTreeNodeLocalDBGroup || info.Node is FileSystemTreeNodeLocalDBEntry)
					{
						removeItemFromDatabaseToolStripMenuItem.Visible = true;
					}

					fileSystemTreeView1.SelectedNode = info.Node;
					contextMenuStripFileTree.Show(fileSystemTreeView1, e.Location);
				}
			}
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
				foreach (CUEToolsUDC encoder in _profile._config.encoders)
					if (encoder.extension == SelectedOutputAudioFmt.extension)
					{
						if (SelectedOutputAudioFormat.StartsWith("lossy."))
						{
							if (!encoder.lossless)
								continue;
						} else if (SelectedOutputAudioType == AudioEncoderType.Lossless && !encoder.lossless)
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

		private void radioButtonAudioLossless_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton && !((RadioButton)sender).Checked)
				return;
			labelFormat.ImageKey = null;
			comboBoxAudioFormat.Items.Clear();
			foreach (KeyValuePair<string, CUEToolsFormat> format in _profile._config.formats)
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
			foreach (KeyValuePair<string, CUEToolsFormat> format in _profile._config.formats)
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
			CUEToolsUDC encoder = comboBoxEncoder.SelectedItem as CUEToolsUDC;
			if (SelectedOutputAudioFormat.StartsWith("lossy."))
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else if (SelectedOutputAudioType == AudioEncoderType.Lossless)
				SelectedOutputAudioFmt.encoderLossless = encoder;
			else
				SelectedOutputAudioFmt.encoderLossy = encoder;
			string [] modes = encoder.SupportedModes;
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
			CUEToolsUDC encoder = comboBoxEncoder.SelectedItem as CUEToolsUDC;
			string[] modes = encoder.SupportedModes;
			encoder.default_mode = modes[trackBarEncoderMode.Value];
			labelEncoderMode.Text = encoder.default_mode;
		}

		//private void toolStripButton1_Click(object sender, EventArgs e)
		//{
		//    FileBrowserState = FileBrowserStateEnum.Tree;
		//    SetupControls(false);
		//}

		//private void toolStripButton2_Click(object sender, EventArgs e)
		//{
		//    FileBrowserState = FileBrowserStateEnum.Checkboxes;
		//    SetupControls(false);
		//}

		//private void toolStripButton3_Click(object sender, EventArgs e)
		//{
		//    FileBrowserState = FileBrowserStateEnum.DragDrop;
		//    SetupControls(false);
		//}

		//private void toolStripButton5_Click(object sender, EventArgs e)
		//{
		//    FileBrowserState = FileBrowserStateEnum.Hidden;
		//    SetupControls(false);
		//}

		private void toolStripButton4_Click(object sender, EventArgs e)
		{
			ReportState = !ReportState;
			SetupControls(false);
		}

		private void toolStripButtonHelp_Click(object sender, EventArgs e)
		{
			System.Diagnostics.Process.Start("http://www.cuetools.net");
		}

		private void toolStripDropDownButtonProfile_DropDownItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			if (e.ClickedItem == toolStripTextBoxAddProfile
			  || e.ClickedItem == toolStripSeparator5)
				return;
			if (e.ClickedItem == toolStripMenuItemDeleteProfile)
			{
				foreach(ToolStripItem item in toolStripDropDownButtonProfile.DropDownItems)
					if (item.Text == toolStripDropDownButtonProfile.Text
						&& item != toolStripTextBoxAddProfile
						&& item != toolStripMenuItemDeleteProfile
						&& item != defaultToolStripMenuItem
						)
					{
						toolStripDropDownButtonProfile.DropDownItems.Remove(item);
						_profile = _defaultProfile;
						ActivateProfile();
						return;
					}
				return;
			}
			string profileName = e.ClickedItem.Text;
			if (profileName == _profile._name)
				return;
			DeactivateProfile();
			ActivateProfile(profileName);
		}

		private void toolStripTextBoxAddProfile_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode != Keys.Enter) return;
			if (toolStripTextBoxAddProfile.Text.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0
				|| toolStripTextBoxAddProfile.Text.IndexOf(' ') >= 0
				|| toolStripTextBoxAddProfile.Text.Length <= 0)
			{
				//System.Media.SystemSounds.Beep.Play();
				return;
			}			
			ToolStripItem item = toolStripDropDownButtonProfile.DropDownItems.Add(toolStripTextBoxAddProfile.Text);
			toolStripDropDownButtonProfile.DropDown.Close();
			e.Handled = true;
			e.SuppressKeyPress = true;

			string profileName = item.Text;
			DeactivateProfile();
			_profile = new CUEToolsProfile(profileName);
			ActivateProfile();
		}

		private void toolStripSplitButtonInputBrowser_ButtonClick(object sender, EventArgs e)
		{
			//FileBrowserState = _fileBrowserState != FileBrowserStateEnum.Hidden
			//    ? FileBrowserStateEnum.Hidden
			//    : _fileBrowserControlState == FileBrowserStateEnum.Hidden
			//    ? FileBrowserStateEnum.Tree
			//    : _fileBrowserControlState;
			//SetupControls(false);
		}

		private void toolStripSplitButtonInputBrowser_DropDownItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			toolStripSplitButtonInputBrowser.DropDown.Close(ToolStripDropDownCloseReason.ItemClicked);
			if (e.ClickedItem == toolStripMenuItemInputBrowserFiles)
				FileBrowserState = FileBrowserStateEnum.Tree;
			if (e.ClickedItem == toolStripMenuItemInputBrowserMulti)
				FileBrowserState = FileBrowserStateEnum.Checkboxes;
			if (e.ClickedItem == toolStripMenuItemInputBrowserDrag)
				FileBrowserState = FileBrowserStateEnum.DragDrop;
			if (e.ClickedItem == toolStripMenuItemInputBrowserHide)
				FileBrowserState = FileBrowserStateEnum.Hidden;
			SetupControls(false);
		}

		private void toolStripSplitButtonOutputBrowser_DropDownItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			toolStripSplitButtonOutputBrowser.DropDown.Close(ToolStripDropDownCloseReason.ItemClicked);
			if (e.ClickedItem == toolStripMenuItemOutputBrowse)
			{
				OutputPathUseTemplate = false;
				SaveFileDialog fileDlg = new SaveFileDialog();
				DialogResult dlgRes;

				fileDlg.Title = "Output CUE Sheet";
				fileDlg.Filter = "CUE Sheets (*.cue)|*.cue";

				dlgRes = fileDlg.ShowDialog();
				if (dlgRes == DialogResult.OK)
					txtOutputPath.Text = fileDlg.FileName;
				UpdateOutputPath();
				return;
			}
			if (e.ClickedItem == toolStripMenuItemOutputManual)
			{
				OutputPathUseTemplate = false;
				UpdateOutputPath();
				return;
			}
			if (e.ClickedItem == toolStripMenuItemOutputTemplate)
			{
				OutputPathUseTemplate = true;
				UpdateOutputPath();
				return;
			}
		}

		private void toolStripDropDownButtonCorrectorFormat_DropDownItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			CUEToolsFormat fmt;
			if (!_profile._config.formats.TryGetValue(e.ClickedItem.Text, out fmt))
				return;
			toolStripDropDownButtonCorrectorFormat.DropDown.Close(ToolStripDropDownCloseReason.ItemClicked);
			toolStripDropDownButtonCorrectorFormat.ImageKey = e.ClickedItem.ImageKey;
			toolStripDropDownButtonCorrectorFormat.Text = e.ClickedItem.Text;
		}

		private CorrectorModeEnum CorrectorMode
		{
			get
			{
				return _correctorMode;
			}
			set
			{
				ToolStripMenuItem item = null;
				switch (value)
				{
					case CorrectorModeEnum.Extension:
						item = toolStripMenuItemCorrectorModeChangeExtension;
						break;
					case CorrectorModeEnum.Locate:
						item = toolStripMenuItemCorrectorModeLocateFiles;
						break;
				}
				toolStripDropDownButtonCorrectorMode.Text = item.Text;
				toolStripDropDownButtonCorrectorMode.Image = item.Image;
				toolStripDropDownButtonCorrectorMode.ToolTipText = item.ToolTipText;
				toolStripDropDownButtonCorrectorFormat.Visible = value == CorrectorModeEnum.Extension;
				_correctorMode = value;
			}
		}

		private void toolStripDropDownButtonCorrectorMode_DropDownItemClicked(object sender, ToolStripItemClickedEventArgs e)
		{
			CorrectorMode = e.ClickedItem == toolStripMenuItemCorrectorModeChangeExtension ?
				CorrectorModeEnum.Extension : CorrectorModeEnum.Locate;
		}

		private void pictureBoxMotd_Click(object sender, EventArgs e)
		{
			if (motdImage != null && pictureBoxMotd.Image == motdImage)
				System.Diagnostics.Process.Start("http://www.cuetools.net/doku.php/cuetools:download");
		}

		private void checkBoxUseAccurateRip_CheckedChanged(object sender, EventArgs e)
		{
			SetupControls(false);
		}

		private void backgroundWorkerAddToLocalDB_DoWork(object sender, DoWorkEventArgs e)
		{
			var folder = e.Argument as string;
			CUESheet cueSheet = new CUESheet(_profile._config);
			cueSheet.CUEToolsProgress += new EventHandler<CUEToolsProgressEventArgs>(SetStatus);
			cueSheet.UseLocalDB(_localDB);
			_workThread = null;
			_workClass = cueSheet;
			cueSheet.ScanLocalDB(folder);
		}

		private void backgroundWorkerAddToLocalDB_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			SetStatus(sender, new CUEToolsProgressEventArgs());
			SetupControls(false);
		}

		private void removeItemFromDatabaseToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var items = (contextMenuStripFileTree.Tag is FileSystemTreeNodeLocalDBGroup)
				? new List<CUEToolsLocalDBEntry>((contextMenuStripFileTree.Tag as FileSystemTreeNodeLocalDBGroup).Group)
				: (contextMenuStripFileTree.Tag is FileSystemTreeNodeLocalDBEntry)
				? new List<CUEToolsLocalDBEntry>(new CUEToolsLocalDBEntry[] { (contextMenuStripFileTree.Tag as FileSystemTreeNodeLocalDBEntry).Item })
				: null;
			if (items == null)
				return;
			foreach (var node in fileSystemTreeView1.Nodes)
			{
				if (node is FileSystemTreeNodeLocalDB)
				{
					(node as FileSystemTreeNodeLocalDB).Purge(items);
					//_localDB.RemoveAll(i => items.Contains(i));
					_localDB.Dirty = true;
					SaveDatabase();
					return;
				}
			}
		}
	}
}
