namespace JDP {
	partial class frmCUETools {
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing) {
			if (disposing && (components != null)) {
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent() {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmCUETools));
            this.toolStripContainer1 = new System.Windows.Forms.ToolStripContainer();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabelProcessed = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabelCTDB = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabelAR = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.textBatchReport = new System.Windows.Forms.TextBox();
            this.grpInput = new System.Windows.Forms.GroupBox();
            this.fileSystemTreeView1 = new CUEControls.FileSystemTreeView();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.groupBoxMode = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanelVerifyMode = new System.Windows.Forms.TableLayoutPanel();
            this.checkBoxSkipRecent = new System.Windows.Forms.CheckBox();
            this.checkBoxVerifyUseLocal = new System.Windows.Forms.CheckBox();
            this.checkBoxVerifyUseCDRepair = new System.Windows.Forms.CheckBox();
            this.tableLayoutPanelCUEStyle = new System.Windows.Forms.TableLayoutPanel();
            this.checkBoxUseAccurateRip = new System.Windows.Forms.CheckBox();
            this.checkBoxUseFreeDb = new System.Windows.Forms.CheckBox();
            this.rbTracks = new System.Windows.Forms.RadioButton();
            this.rbEmbedCUE = new System.Windows.Forms.RadioButton();
            this.rbSingleFile = new System.Windows.Forms.RadioButton();
            this.checkBoxUseMusicBrainz = new System.Windows.Forms.CheckBox();
            this.toolStripCorrectorFormat = new System.Windows.Forms.ToolStrip();
            this.toolStripButtonCorrectorOverwrite = new System.Windows.Forms.ToolStripButton();
            this.toolStripDropDownButtonCorrectorMode = new System.Windows.Forms.ToolStripDropDownButton();
            this.toolStripMenuItemCorrectorModeLocateFiles = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemCorrectorModeChangeExtension = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripDropDownButtonCorrectorFormat = new System.Windows.Forms.ToolStripDropDownButton();
            this.grpAudioOutput = new System.Windows.Forms.GroupBox();
            this.labelEncoderMaxMode = new System.Windows.Forms.Label();
            this.labelEncoderMinMode = new System.Windows.Forms.Label();
            this.labelEncoderMode = new System.Windows.Forms.Label();
            this.trackBarEncoderMode = new System.Windows.Forms.TrackBar();
            this.comboBoxEncoder = new System.Windows.Forms.ComboBox();
            this.radioButtonAudioNone = new System.Windows.Forms.RadioButton();
            this.radioButtonAudioLossy = new System.Windows.Forms.RadioButton();
            this.radioButtonAudioHybrid = new System.Windows.Forms.RadioButton();
            this.radioButtonAudioLossless = new System.Windows.Forms.RadioButton();
            this.labelFormat = new System.Windows.Forms.Label();
            this.comboBoxAudioFormat = new System.Windows.Forms.ComboBox();
            this.pictureBoxMotd = new System.Windows.Forms.PictureBox();
            this.grpOutputPathGeneration = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanelPaths = new System.Windows.Forms.TableLayoutPanel();
            this.labelOutputTemplate = new System.Windows.Forms.Label();
            this.comboBoxOutputFormat = new System.Windows.Forms.ComboBox();
            this.txtInputPath = new System.Windows.Forms.TextBox();
            this.txtOutputPath = new System.Windows.Forms.TextBox();
            this.toolStripInput = new System.Windows.Forms.ToolStrip();
            this.toolStripLabelInput = new System.Windows.Forms.ToolStripLabel();
            this.toolStripSplitButtonInputBrowser = new System.Windows.Forms.ToolStripSplitButton();
            this.toolStripMenuItemInputBrowserFiles = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemInputBrowserMulti = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemInputBrowserDrag = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemInputBrowserHide = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripOutput = new System.Windows.Forms.ToolStrip();
            this.toolStripLabelOutput = new System.Windows.Forms.ToolStripLabel();
            this.toolStripSplitButtonOutputBrowser = new System.Windows.Forms.ToolStripSplitButton();
            this.toolStripMenuItemOutputBrowse = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemOutputManual = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItemOutputTemplate = new System.Windows.Forms.ToolStripMenuItem();
            this.grpAction = new System.Windows.Forms.GroupBox();
            this.comboBoxScript = new System.Windows.Forms.ComboBox();
            this.rbActionCorrectFilenames = new System.Windows.Forms.RadioButton();
            this.rbActionCreateCUESheet = new System.Windows.Forms.RadioButton();
            this.rbActionVerify = new System.Windows.Forms.RadioButton();
            this.rbActionEncode = new System.Windows.Forms.RadioButton();
            this.grpExtra = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel4 = new System.Windows.Forms.TableLayoutPanel();
            this.labelPregap = new System.Windows.Forms.Label();
            this.lblWriteOffset = new System.Windows.Forms.Label();
            this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
            this.txtPreGapLength = new System.Windows.Forms.MaskedTextBox();
            this.labelDataTrack = new System.Windows.Forms.Label();
            this.txtDataTrackLength = new System.Windows.Forms.MaskedTextBox();
            this.panelGo = new System.Windows.Forms.Panel();
            this.btnConvert = new System.Windows.Forms.Button();
            this.btnStop = new System.Windows.Forms.Button();
            this.btnResume = new System.Windows.Forms.Button();
            this.btnPause = new System.Windows.Forms.Button();
            this.toolStripMenu = new System.Windows.Forms.ToolStrip();
            this.toolStripDropDownButtonProfile = new System.Windows.Forms.ToolStripDropDownButton();
            this.toolStripTextBoxAddProfile = new System.Windows.Forms.ToolStripTextBox();
            this.toolStripMenuItemDeleteProfile = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator5 = new System.Windows.Forms.ToolStripSeparator();
            this.defaultToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripButtonAbout = new System.Windows.Forms.ToolStripButton();
            this.toolStripButtonHelp = new System.Windows.Forms.ToolStripButton();
            this.toolStripButtonSettings = new System.Windows.Forms.ToolStripButton();
            this.toolStripButtonShowLog = new System.Windows.Forms.ToolStripButton();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.contextMenuStripFileTree = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.SelectedNodeName = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.setAsMyMusicFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.resetToOriginalLocationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.editMetadataToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addFolderToLocalDatabaseToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.removeItemFromDatabaseToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.backgroundWorkerAddToLocalDB = new System.ComponentModel.BackgroundWorker();
            this.toolStripContainer1.BottomToolStripPanel.SuspendLayout();
            this.toolStripContainer1.ContentPanel.SuspendLayout();
            this.toolStripContainer1.TopToolStripPanel.SuspendLayout();
            this.toolStripContainer1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.grpInput.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.groupBoxMode.SuspendLayout();
            this.tableLayoutPanelVerifyMode.SuspendLayout();
            this.tableLayoutPanelCUEStyle.SuspendLayout();
            this.toolStripCorrectorFormat.SuspendLayout();
            this.grpAudioOutput.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMotd)).BeginInit();
            this.grpOutputPathGeneration.SuspendLayout();
            this.tableLayoutPanelPaths.SuspendLayout();
            this.toolStripInput.SuspendLayout();
            this.toolStripOutput.SuspendLayout();
            this.grpAction.SuspendLayout();
            this.grpExtra.SuspendLayout();
            this.tableLayoutPanel4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
            this.panelGo.SuspendLayout();
            this.toolStripMenu.SuspendLayout();
            this.contextMenuStripFileTree.SuspendLayout();
            this.SuspendLayout();
            // 
            // toolStripContainer1
            // 
            // 
            // toolStripContainer1.BottomToolStripPanel
            // 
            this.toolStripContainer1.BottomToolStripPanel.Controls.Add(this.statusStrip1);
            // 
            // toolStripContainer1.ContentPanel
            // 
            resources.ApplyResources(this.toolStripContainer1.ContentPanel, "toolStripContainer1.ContentPanel");
            this.toolStripContainer1.ContentPanel.Controls.Add(this.tableLayoutPanel1);
            resources.ApplyResources(this.toolStripContainer1, "toolStripContainer1");
            // 
            // toolStripContainer1.LeftToolStripPanel
            // 
            this.toolStripContainer1.LeftToolStripPanel.MaximumSize = new System.Drawing.Size(32, 0);
            this.toolStripContainer1.LeftToolStripPanelVisible = false;
            this.toolStripContainer1.Name = "toolStripContainer1";
            this.toolStripContainer1.RightToolStripPanelVisible = false;
            // 
            // toolStripContainer1.TopToolStripPanel
            // 
            this.toolStripContainer1.TopToolStripPanel.BackColor = System.Drawing.SystemColors.GradientInactiveCaption;
            this.toolStripContainer1.TopToolStripPanel.Controls.Add(this.toolStripMenu);
            this.toolStripContainer1.TopToolStripPanel.RenderMode = System.Windows.Forms.ToolStripRenderMode.System;
            // 
            // statusStrip1
            // 
            resources.ApplyResources(this.statusStrip1, "statusStrip1");
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLabelProcessed,
            this.toolStripStatusLabelCTDB,
            this.toolStripStatusLabelAR,
            this.toolStripProgressBar2});
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.ShowItemToolTips = true;
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
            this.toolStripStatusLabel1.Spring = true;
            // 
            // toolStripStatusLabelProcessed
            // 
            this.toolStripStatusLabelProcessed.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.toolStripStatusLabelProcessed.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            this.toolStripStatusLabelProcessed.Name = "toolStripStatusLabelProcessed";
            resources.ApplyResources(this.toolStripStatusLabelProcessed, "toolStripStatusLabelProcessed");
            // 
            // toolStripStatusLabelCTDB
            // 
            this.toolStripStatusLabelCTDB.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.toolStripStatusLabelCTDB.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            resources.ApplyResources(this.toolStripStatusLabelCTDB, "toolStripStatusLabelCTDB");
            this.toolStripStatusLabelCTDB.Image = global::JDP.Properties.Resources.cdrepair1;
            this.toolStripStatusLabelCTDB.Name = "toolStripStatusLabelCTDB";
            // 
            // toolStripStatusLabelAR
            // 
            this.toolStripStatusLabelAR.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.toolStripStatusLabelAR.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            resources.ApplyResources(this.toolStripStatusLabelAR, "toolStripStatusLabelAR");
            this.toolStripStatusLabelAR.Image = global::JDP.Properties.Resources.AR;
            this.toolStripStatusLabelAR.Name = "toolStripStatusLabelAR";
            this.toolStripStatusLabelAR.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
            // 
            // toolStripProgressBar2
            // 
            this.toolStripProgressBar2.AutoToolTip = true;
            this.toolStripProgressBar2.Name = "toolStripProgressBar2";
            resources.ApplyResources(this.toolStripProgressBar2, "toolStripProgressBar2");
            this.toolStripProgressBar2.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            // 
            // tableLayoutPanel1
            // 
            resources.ApplyResources(this.tableLayoutPanel1, "tableLayoutPanel1");
            this.tableLayoutPanel1.Controls.Add(this.textBatchReport, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.grpInput, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.tableLayoutPanel2, 1, 0);
            this.tableLayoutPanel1.GrowStyle = System.Windows.Forms.TableLayoutPanelGrowStyle.AddColumns;
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            // 
            // textBatchReport
            // 
            this.tableLayoutPanel1.SetColumnSpan(this.textBatchReport, 2);
            resources.ApplyResources(this.textBatchReport, "textBatchReport");
            this.textBatchReport.Name = "textBatchReport";
            this.textBatchReport.ReadOnly = true;
            this.textBatchReport.TabStop = false;
            // 
            // grpInput
            // 
            this.grpInput.Controls.Add(this.fileSystemTreeView1);
            resources.ApplyResources(this.grpInput, "grpInput");
            this.grpInput.Name = "grpInput";
            this.grpInput.TabStop = false;
            // 
            // fileSystemTreeView1
            // 
            this.fileSystemTreeView1.AllowDrop = true;
            resources.ApplyResources(this.fileSystemTreeView1, "fileSystemTreeView1");
            this.fileSystemTreeView1.BackColor = System.Drawing.SystemColors.Control;
            this.fileSystemTreeView1.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.fileSystemTreeView1.CheckBoxes = true;
            this.fileSystemTreeView1.FullRowSelect = true;
            this.fileSystemTreeView1.HideSelection = false;
            this.fileSystemTreeView1.ItemHeight = 16;
            this.fileSystemTreeView1.Name = "fileSystemTreeView1";
            this.fileSystemTreeView1.ShowLines = false;
            this.fileSystemTreeView1.ShowRootLines = false;
            this.fileSystemTreeView1.SpecialFolders = new CUEControls.ExtraSpecialFolder[] {
        CUEControls.ExtraSpecialFolder.MyComputer,
        CUEControls.ExtraSpecialFolder.Profile,
        CUEControls.ExtraSpecialFolder.MyMusic,
        CUEControls.ExtraSpecialFolder.CommonMusic};
            this.fileSystemTreeView1.NodeExpand += new CUEControls.FileSystemTreeViewNodeExpandHandler(this.fileSystemTreeView1_NodeExpand);
            this.fileSystemTreeView1.AfterCheck += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterCheck);
            this.fileSystemTreeView1.AfterExpand += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterExpand);
            this.fileSystemTreeView1.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterSelect);
            this.fileSystemTreeView1.DragDrop += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragDrop);
            this.fileSystemTreeView1.DragOver += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragOver);
            this.fileSystemTreeView1.KeyDown += new System.Windows.Forms.KeyEventHandler(this.fileSystemTreeView1_KeyDown);
            this.fileSystemTreeView1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.fileSystemTreeView1_MouseDown);
            // 
            // tableLayoutPanel2
            // 
            resources.ApplyResources(this.tableLayoutPanel2, "tableLayoutPanel2");
            this.tableLayoutPanel2.Controls.Add(this.groupBoxMode, 1, 1);
            this.tableLayoutPanel2.Controls.Add(this.grpAudioOutput, 2, 1);
            this.tableLayoutPanel2.Controls.Add(this.pictureBoxMotd, 0, 2);
            this.tableLayoutPanel2.Controls.Add(this.grpOutputPathGeneration, 0, 0);
            this.tableLayoutPanel2.Controls.Add(this.grpAction, 0, 1);
            this.tableLayoutPanel2.Controls.Add(this.grpExtra, 1, 2);
            this.tableLayoutPanel2.Controls.Add(this.panelGo, 2, 3);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            // 
            // groupBoxMode
            // 
            this.groupBoxMode.Controls.Add(this.tableLayoutPanelVerifyMode);
            this.groupBoxMode.Controls.Add(this.tableLayoutPanelCUEStyle);
            this.groupBoxMode.Controls.Add(this.toolStripCorrectorFormat);
            resources.ApplyResources(this.groupBoxMode, "groupBoxMode");
            this.groupBoxMode.Name = "groupBoxMode";
            this.groupBoxMode.TabStop = false;
            // 
            // tableLayoutPanelVerifyMode
            // 
            resources.ApplyResources(this.tableLayoutPanelVerifyMode, "tableLayoutPanelVerifyMode");
            this.tableLayoutPanelVerifyMode.Controls.Add(this.checkBoxSkipRecent, 2, 0);
            this.tableLayoutPanelVerifyMode.Controls.Add(this.checkBoxVerifyUseLocal, 1, 0);
            this.tableLayoutPanelVerifyMode.Controls.Add(this.checkBoxVerifyUseCDRepair, 0, 0);
            this.tableLayoutPanelVerifyMode.Name = "tableLayoutPanelVerifyMode";
            // 
            // checkBoxSkipRecent
            // 
            resources.ApplyResources(this.checkBoxSkipRecent, "checkBoxSkipRecent");
            this.checkBoxSkipRecent.Image = global::JDP.Properties.Resources.alarm_clock__minus;
            this.checkBoxSkipRecent.Name = "checkBoxSkipRecent";
            this.toolTip1.SetToolTip(this.checkBoxSkipRecent, resources.GetString("checkBoxSkipRecent.ToolTip"));
            this.checkBoxSkipRecent.UseVisualStyleBackColor = true;
            // 
            // checkBoxVerifyUseLocal
            // 
            resources.ApplyResources(this.checkBoxVerifyUseLocal, "checkBoxVerifyUseLocal");
            this.checkBoxVerifyUseLocal.Image = global::JDP.Properties.Resources.puzzle__arrow;
            this.checkBoxVerifyUseLocal.Name = "checkBoxVerifyUseLocal";
            this.toolTip1.SetToolTip(this.checkBoxVerifyUseLocal, resources.GetString("checkBoxVerifyUseLocal.ToolTip"));
            this.checkBoxVerifyUseLocal.UseVisualStyleBackColor = true;
            // 
            // checkBoxVerifyUseCDRepair
            // 
            resources.ApplyResources(this.checkBoxVerifyUseCDRepair, "checkBoxVerifyUseCDRepair");
            this.checkBoxVerifyUseCDRepair.Image = global::JDP.Properties.Resources.cdrepair1;
            this.checkBoxVerifyUseCDRepair.Name = "checkBoxVerifyUseCDRepair";
            this.toolTip1.SetToolTip(this.checkBoxVerifyUseCDRepair, resources.GetString("checkBoxVerifyUseCDRepair.ToolTip"));
            this.checkBoxVerifyUseCDRepair.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanelCUEStyle
            // 
            resources.ApplyResources(this.tableLayoutPanelCUEStyle, "tableLayoutPanelCUEStyle");
            this.tableLayoutPanelCUEStyle.Controls.Add(this.checkBoxUseAccurateRip, 2, 4);
            this.tableLayoutPanelCUEStyle.Controls.Add(this.checkBoxUseFreeDb, 1, 4);
            this.tableLayoutPanelCUEStyle.Controls.Add(this.rbTracks, 0, 2);
            this.tableLayoutPanelCUEStyle.Controls.Add(this.rbEmbedCUE, 0, 0);
            this.tableLayoutPanelCUEStyle.Controls.Add(this.rbSingleFile, 0, 1);
            this.tableLayoutPanelCUEStyle.Controls.Add(this.checkBoxUseMusicBrainz, 0, 4);
            this.tableLayoutPanelCUEStyle.Name = "tableLayoutPanelCUEStyle";
            // 
            // checkBoxUseAccurateRip
            // 
            resources.ApplyResources(this.checkBoxUseAccurateRip, "checkBoxUseAccurateRip");
            this.checkBoxUseAccurateRip.Image = global::JDP.Properties.Resources.AR;
            this.checkBoxUseAccurateRip.MinimumSize = new System.Drawing.Size(0, 16);
            this.checkBoxUseAccurateRip.Name = "checkBoxUseAccurateRip";
            this.toolTip1.SetToolTip(this.checkBoxUseAccurateRip, resources.GetString("checkBoxUseAccurateRip.ToolTip"));
            this.checkBoxUseAccurateRip.UseVisualStyleBackColor = true;
            this.checkBoxUseAccurateRip.CheckedChanged += new System.EventHandler(this.checkBoxUseAccurateRip_CheckedChanged);
            // 
            // checkBoxUseFreeDb
            // 
            resources.ApplyResources(this.checkBoxUseFreeDb, "checkBoxUseFreeDb");
            this.checkBoxUseFreeDb.Image = global::JDP.Properties.Resources.freedb16;
            this.checkBoxUseFreeDb.MinimumSize = new System.Drawing.Size(0, 16);
            this.checkBoxUseFreeDb.Name = "checkBoxUseFreeDb";
            this.toolTip1.SetToolTip(this.checkBoxUseFreeDb, resources.GetString("checkBoxUseFreeDb.ToolTip"));
            this.checkBoxUseFreeDb.UseVisualStyleBackColor = true;
            // 
            // rbTracks
            // 
            resources.ApplyResources(this.rbTracks, "rbTracks");
            this.tableLayoutPanelCUEStyle.SetColumnSpan(this.rbTracks, 3);
            this.rbTracks.Name = "rbTracks";
            this.rbTracks.TabStop = true;
            this.toolTip1.SetToolTip(this.rbTracks, resources.GetString("rbTracks.ToolTip"));
            this.rbTracks.UseVisualStyleBackColor = true;
            // 
            // rbEmbedCUE
            // 
            resources.ApplyResources(this.rbEmbedCUE, "rbEmbedCUE");
            this.tableLayoutPanelCUEStyle.SetColumnSpan(this.rbEmbedCUE, 3);
            this.rbEmbedCUE.Name = "rbEmbedCUE";
            this.rbEmbedCUE.TabStop = true;
            this.toolTip1.SetToolTip(this.rbEmbedCUE, resources.GetString("rbEmbedCUE.ToolTip"));
            this.rbEmbedCUE.UseVisualStyleBackColor = true;
            this.rbEmbedCUE.CheckedChanged += new System.EventHandler(this.rbEmbedCUE_CheckedChanged);
            // 
            // rbSingleFile
            // 
            resources.ApplyResources(this.rbSingleFile, "rbSingleFile");
            this.rbSingleFile.Checked = true;
            this.tableLayoutPanelCUEStyle.SetColumnSpan(this.rbSingleFile, 3);
            this.rbSingleFile.Name = "rbSingleFile";
            this.rbSingleFile.TabStop = true;
            this.toolTip1.SetToolTip(this.rbSingleFile, resources.GetString("rbSingleFile.ToolTip"));
            this.rbSingleFile.UseVisualStyleBackColor = true;
            // 
            // checkBoxUseMusicBrainz
            // 
            resources.ApplyResources(this.checkBoxUseMusicBrainz, "checkBoxUseMusicBrainz");
            this.checkBoxUseMusicBrainz.Image = global::JDP.Properties.Resources.musicbrainz;
            this.checkBoxUseMusicBrainz.MinimumSize = new System.Drawing.Size(0, 16);
            this.checkBoxUseMusicBrainz.Name = "checkBoxUseMusicBrainz";
            this.toolTip1.SetToolTip(this.checkBoxUseMusicBrainz, resources.GetString("checkBoxUseMusicBrainz.ToolTip"));
            this.checkBoxUseMusicBrainz.UseVisualStyleBackColor = true;
            // 
            // toolStripCorrectorFormat
            // 
            resources.ApplyResources(this.toolStripCorrectorFormat, "toolStripCorrectorFormat");
            this.toolStripCorrectorFormat.BackColor = System.Drawing.Color.Transparent;
            this.toolStripCorrectorFormat.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripButtonCorrectorOverwrite,
            this.toolStripDropDownButtonCorrectorMode,
            this.toolStripDropDownButtonCorrectorFormat});
            this.toolStripCorrectorFormat.LayoutStyle = System.Windows.Forms.ToolStripLayoutStyle.Flow;
            this.toolStripCorrectorFormat.Name = "toolStripCorrectorFormat";
            this.toolStripCorrectorFormat.RenderMode = System.Windows.Forms.ToolStripRenderMode.System;
            this.toolStripCorrectorFormat.TabStop = true;
            // 
            // toolStripButtonCorrectorOverwrite
            // 
            this.toolStripButtonCorrectorOverwrite.AutoToolTip = false;
            this.toolStripButtonCorrectorOverwrite.CheckOnClick = true;
            this.toolStripButtonCorrectorOverwrite.Image = global::JDP.Properties.Resources.disk;
            resources.ApplyResources(this.toolStripButtonCorrectorOverwrite, "toolStripButtonCorrectorOverwrite");
            this.toolStripButtonCorrectorOverwrite.Name = "toolStripButtonCorrectorOverwrite";
            // 
            // toolStripDropDownButtonCorrectorMode
            // 
            this.toolStripDropDownButtonCorrectorMode.AutoToolTip = false;
            this.toolStripDropDownButtonCorrectorMode.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripMenuItemCorrectorModeLocateFiles,
            this.toolStripMenuItemCorrectorModeChangeExtension});
            resources.ApplyResources(this.toolStripDropDownButtonCorrectorMode, "toolStripDropDownButtonCorrectorMode");
            this.toolStripDropDownButtonCorrectorMode.Name = "toolStripDropDownButtonCorrectorMode";
            this.toolStripDropDownButtonCorrectorMode.DropDownItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStripDropDownButtonCorrectorMode_DropDownItemClicked);
            // 
            // toolStripMenuItemCorrectorModeLocateFiles
            // 
            this.toolStripMenuItemCorrectorModeLocateFiles.Image = global::JDP.Properties.Resources.find;
            this.toolStripMenuItemCorrectorModeLocateFiles.Name = "toolStripMenuItemCorrectorModeLocateFiles";
            resources.ApplyResources(this.toolStripMenuItemCorrectorModeLocateFiles, "toolStripMenuItemCorrectorModeLocateFiles");
            // 
            // toolStripMenuItemCorrectorModeChangeExtension
            // 
            this.toolStripMenuItemCorrectorModeChangeExtension.Image = global::JDP.Properties.Resources.link_go;
            this.toolStripMenuItemCorrectorModeChangeExtension.Name = "toolStripMenuItemCorrectorModeChangeExtension";
            resources.ApplyResources(this.toolStripMenuItemCorrectorModeChangeExtension, "toolStripMenuItemCorrectorModeChangeExtension");
            // 
            // toolStripDropDownButtonCorrectorFormat
            // 
            this.toolStripDropDownButtonCorrectorFormat.AutoToolTip = false;
            resources.ApplyResources(this.toolStripDropDownButtonCorrectorFormat, "toolStripDropDownButtonCorrectorFormat");
            this.toolStripDropDownButtonCorrectorFormat.Name = "toolStripDropDownButtonCorrectorFormat";
            this.toolStripDropDownButtonCorrectorFormat.DropDownItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStripDropDownButtonCorrectorFormat_DropDownItemClicked);
            // 
            // grpAudioOutput
            // 
            this.grpAudioOutput.Controls.Add(this.labelEncoderMaxMode);
            this.grpAudioOutput.Controls.Add(this.labelEncoderMinMode);
            this.grpAudioOutput.Controls.Add(this.labelEncoderMode);
            this.grpAudioOutput.Controls.Add(this.trackBarEncoderMode);
            this.grpAudioOutput.Controls.Add(this.comboBoxEncoder);
            this.grpAudioOutput.Controls.Add(this.radioButtonAudioNone);
            this.grpAudioOutput.Controls.Add(this.radioButtonAudioLossy);
            this.grpAudioOutput.Controls.Add(this.radioButtonAudioHybrid);
            this.grpAudioOutput.Controls.Add(this.radioButtonAudioLossless);
            this.grpAudioOutput.Controls.Add(this.labelFormat);
            this.grpAudioOutput.Controls.Add(this.comboBoxAudioFormat);
            resources.ApplyResources(this.grpAudioOutput, "grpAudioOutput");
            this.grpAudioOutput.Name = "grpAudioOutput";
            this.tableLayoutPanel2.SetRowSpan(this.grpAudioOutput, 2);
            this.grpAudioOutput.TabStop = false;
            // 
            // labelEncoderMaxMode
            // 
            resources.ApplyResources(this.labelEncoderMaxMode, "labelEncoderMaxMode");
            this.labelEncoderMaxMode.Name = "labelEncoderMaxMode";
            // 
            // labelEncoderMinMode
            // 
            resources.ApplyResources(this.labelEncoderMinMode, "labelEncoderMinMode");
            this.labelEncoderMinMode.Name = "labelEncoderMinMode";
            // 
            // labelEncoderMode
            // 
            resources.ApplyResources(this.labelEncoderMode, "labelEncoderMode");
            this.labelEncoderMode.Name = "labelEncoderMode";
            this.toolTip1.SetToolTip(this.labelEncoderMode, resources.GetString("labelEncoderMode.ToolTip"));
            // 
            // trackBarEncoderMode
            // 
            resources.ApplyResources(this.trackBarEncoderMode, "trackBarEncoderMode");
            this.trackBarEncoderMode.LargeChange = 1;
            this.trackBarEncoderMode.Name = "trackBarEncoderMode";
            this.trackBarEncoderMode.Scroll += new System.EventHandler(this.trackBarEncoderMode_Scroll);
            // 
            // comboBoxEncoder
            // 
            resources.ApplyResources(this.comboBoxEncoder, "comboBoxEncoder");
            this.comboBoxEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxEncoder.FormattingEnabled = true;
            this.comboBoxEncoder.Name = "comboBoxEncoder";
            this.toolTip1.SetToolTip(this.comboBoxEncoder, resources.GetString("comboBoxEncoder.ToolTip"));
            this.comboBoxEncoder.SelectedIndexChanged += new System.EventHandler(this.comboBoxEncoder_SelectedIndexChanged);
            // 
            // radioButtonAudioNone
            // 
            resources.ApplyResources(this.radioButtonAudioNone, "radioButtonAudioNone");
            this.radioButtonAudioNone.Name = "radioButtonAudioNone";
            this.radioButtonAudioNone.TabStop = true;
            this.toolTip1.SetToolTip(this.radioButtonAudioNone, resources.GetString("radioButtonAudioNone.ToolTip"));
            this.radioButtonAudioNone.UseVisualStyleBackColor = true;
            this.radioButtonAudioNone.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
            // 
            // radioButtonAudioLossy
            // 
            resources.ApplyResources(this.radioButtonAudioLossy, "radioButtonAudioLossy");
            this.radioButtonAudioLossy.Name = "radioButtonAudioLossy";
            this.radioButtonAudioLossy.TabStop = true;
            this.toolTip1.SetToolTip(this.radioButtonAudioLossy, resources.GetString("radioButtonAudioLossy.ToolTip"));
            this.radioButtonAudioLossy.UseVisualStyleBackColor = true;
            this.radioButtonAudioLossy.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
            // 
            // radioButtonAudioHybrid
            // 
            resources.ApplyResources(this.radioButtonAudioHybrid, "radioButtonAudioHybrid");
            this.radioButtonAudioHybrid.Name = "radioButtonAudioHybrid";
            this.radioButtonAudioHybrid.TabStop = true;
            this.toolTip1.SetToolTip(this.radioButtonAudioHybrid, resources.GetString("radioButtonAudioHybrid.ToolTip"));
            this.radioButtonAudioHybrid.UseVisualStyleBackColor = true;
            this.radioButtonAudioHybrid.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
            // 
            // radioButtonAudioLossless
            // 
            resources.ApplyResources(this.radioButtonAudioLossless, "radioButtonAudioLossless");
            this.radioButtonAudioLossless.Name = "radioButtonAudioLossless";
            this.radioButtonAudioLossless.TabStop = true;
            this.toolTip1.SetToolTip(this.radioButtonAudioLossless, resources.GetString("radioButtonAudioLossless.ToolTip"));
            this.radioButtonAudioLossless.UseVisualStyleBackColor = true;
            this.radioButtonAudioLossless.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
            // 
            // labelFormat
            // 
            resources.ApplyResources(this.labelFormat, "labelFormat");
            this.labelFormat.MinimumSize = new System.Drawing.Size(16, 16);
            this.labelFormat.Name = "labelFormat";
            // 
            // comboBoxAudioFormat
            // 
            resources.ApplyResources(this.comboBoxAudioFormat, "comboBoxAudioFormat");
            this.comboBoxAudioFormat.BackColor = System.Drawing.SystemColors.Window;
            this.comboBoxAudioFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxAudioFormat.FormattingEnabled = true;
            this.comboBoxAudioFormat.Name = "comboBoxAudioFormat";
            this.toolTip1.SetToolTip(this.comboBoxAudioFormat, resources.GetString("comboBoxAudioFormat.ToolTip"));
            this.comboBoxAudioFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxAudioFormat_SelectedIndexChanged);
            // 
            // pictureBoxMotd
            // 
            resources.ApplyResources(this.pictureBoxMotd, "pictureBoxMotd");
            this.pictureBoxMotd.Name = "pictureBoxMotd";
            this.tableLayoutPanel2.SetRowSpan(this.pictureBoxMotd, 2);
            this.pictureBoxMotd.TabStop = false;
            this.pictureBoxMotd.Click += new System.EventHandler(this.pictureBoxMotd_Click);
            // 
            // grpOutputPathGeneration
            // 
            this.tableLayoutPanel2.SetColumnSpan(this.grpOutputPathGeneration, 3);
            this.grpOutputPathGeneration.Controls.Add(this.tableLayoutPanelPaths);
            resources.ApplyResources(this.grpOutputPathGeneration, "grpOutputPathGeneration");
            this.grpOutputPathGeneration.Name = "grpOutputPathGeneration";
            this.grpOutputPathGeneration.TabStop = false;
            // 
            // tableLayoutPanelPaths
            // 
            resources.ApplyResources(this.tableLayoutPanelPaths, "tableLayoutPanelPaths");
            this.tableLayoutPanelPaths.Controls.Add(this.labelOutputTemplate, 0, 2);
            this.tableLayoutPanelPaths.Controls.Add(this.comboBoxOutputFormat, 1, 2);
            this.tableLayoutPanelPaths.Controls.Add(this.txtInputPath, 1, 0);
            this.tableLayoutPanelPaths.Controls.Add(this.txtOutputPath, 1, 1);
            this.tableLayoutPanelPaths.Controls.Add(this.toolStripInput, 0, 0);
            this.tableLayoutPanelPaths.Controls.Add(this.toolStripOutput, 0, 1);
            this.tableLayoutPanelPaths.Name = "tableLayoutPanelPaths";
            // 
            // labelOutputTemplate
            // 
            resources.ApplyResources(this.labelOutputTemplate, "labelOutputTemplate");
            this.labelOutputTemplate.Name = "labelOutputTemplate";
            // 
            // comboBoxOutputFormat
            // 
            resources.ApplyResources(this.comboBoxOutputFormat, "comboBoxOutputFormat");
            this.comboBoxOutputFormat.FormattingEnabled = true;
            this.comboBoxOutputFormat.Name = "comboBoxOutputFormat";
            this.toolTip1.SetToolTip(this.comboBoxOutputFormat, resources.GetString("comboBoxOutputFormat.ToolTip"));
            this.comboBoxOutputFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxOutputFormat_SelectedIndexChanged);
            this.comboBoxOutputFormat.TextUpdate += new System.EventHandler(this.comboBoxOutputFormat_TextUpdate);
            // 
            // txtInputPath
            // 
            this.txtInputPath.AllowDrop = true;
            this.txtInputPath.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtInputPath.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
            this.txtInputPath.BackColor = System.Drawing.SystemColors.Window;
            resources.ApplyResources(this.txtInputPath, "txtInputPath");
            this.txtInputPath.Name = "txtInputPath";
            this.toolTip1.SetToolTip(this.txtInputPath, resources.GetString("txtInputPath.ToolTip"));
            this.txtInputPath.TextChanged += new System.EventHandler(this.txtInputPath_TextChanged);
            this.txtInputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
            this.txtInputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
            // 
            // txtOutputPath
            // 
            this.txtOutputPath.AllowDrop = true;
            this.txtOutputPath.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtOutputPath.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
            resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
            this.txtOutputPath.Name = "txtOutputPath";
            this.toolTip1.SetToolTip(this.txtOutputPath, resources.GetString("txtOutputPath.ToolTip"));
            this.txtOutputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
            this.txtOutputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
            // 
            // toolStripInput
            // 
            resources.ApplyResources(this.toolStripInput, "toolStripInput");
            this.toolStripInput.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStripInput.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripLabelInput,
            this.toolStripSplitButtonInputBrowser});
            this.toolStripInput.Name = "toolStripInput";
            this.toolStripInput.RenderMode = System.Windows.Forms.ToolStripRenderMode.System;
            // 
            // toolStripLabelInput
            // 
            this.toolStripLabelInput.Name = "toolStripLabelInput";
            resources.ApplyResources(this.toolStripLabelInput, "toolStripLabelInput");
            // 
            // toolStripSplitButtonInputBrowser
            // 
            this.toolStripSplitButtonInputBrowser.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripSplitButtonInputBrowser.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripSplitButtonInputBrowser.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripMenuItemInputBrowserFiles,
            this.toolStripMenuItemInputBrowserMulti,
            this.toolStripMenuItemInputBrowserDrag,
            this.toolStripMenuItemInputBrowserHide});
            this.toolStripSplitButtonInputBrowser.Image = global::JDP.Properties.Resources.folder;
            resources.ApplyResources(this.toolStripSplitButtonInputBrowser, "toolStripSplitButtonInputBrowser");
            this.toolStripSplitButtonInputBrowser.Name = "toolStripSplitButtonInputBrowser";
            this.toolStripSplitButtonInputBrowser.ButtonClick += new System.EventHandler(this.toolStripSplitButtonInputBrowser_ButtonClick);
            this.toolStripSplitButtonInputBrowser.DropDownItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStripSplitButtonInputBrowser_DropDownItemClicked);
            // 
            // toolStripMenuItemInputBrowserFiles
            // 
            this.toolStripMenuItemInputBrowserFiles.Image = global::JDP.Properties.Resources.folder;
            this.toolStripMenuItemInputBrowserFiles.Name = "toolStripMenuItemInputBrowserFiles";
            resources.ApplyResources(this.toolStripMenuItemInputBrowserFiles, "toolStripMenuItemInputBrowserFiles");
            // 
            // toolStripMenuItemInputBrowserMulti
            // 
            this.toolStripMenuItemInputBrowserMulti.Image = global::JDP.Properties.Resources.folder_add;
            this.toolStripMenuItemInputBrowserMulti.Name = "toolStripMenuItemInputBrowserMulti";
            resources.ApplyResources(this.toolStripMenuItemInputBrowserMulti, "toolStripMenuItemInputBrowserMulti");
            // 
            // toolStripMenuItemInputBrowserDrag
            // 
            this.toolStripMenuItemInputBrowserDrag.Image = global::JDP.Properties.Resources.folder_feed;
            this.toolStripMenuItemInputBrowserDrag.Name = "toolStripMenuItemInputBrowserDrag";
            resources.ApplyResources(this.toolStripMenuItemInputBrowserDrag, "toolStripMenuItemInputBrowserDrag");
            // 
            // toolStripMenuItemInputBrowserHide
            // 
            this.toolStripMenuItemInputBrowserHide.Image = global::JDP.Properties.Resources.folder_delete;
            this.toolStripMenuItemInputBrowserHide.Name = "toolStripMenuItemInputBrowserHide";
            resources.ApplyResources(this.toolStripMenuItemInputBrowserHide, "toolStripMenuItemInputBrowserHide");
            // 
            // toolStripOutput
            // 
            resources.ApplyResources(this.toolStripOutput, "toolStripOutput");
            this.toolStripOutput.GripMargin = new System.Windows.Forms.Padding(0);
            this.toolStripOutput.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStripOutput.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripLabelOutput,
            this.toolStripSplitButtonOutputBrowser});
            this.toolStripOutput.LayoutStyle = System.Windows.Forms.ToolStripLayoutStyle.HorizontalStackWithOverflow;
            this.toolStripOutput.Name = "toolStripOutput";
            this.toolStripOutput.RenderMode = System.Windows.Forms.ToolStripRenderMode.System;
            // 
            // toolStripLabelOutput
            // 
            this.toolStripLabelOutput.Margin = new System.Windows.Forms.Padding(0);
            this.toolStripLabelOutput.Name = "toolStripLabelOutput";
            resources.ApplyResources(this.toolStripLabelOutput, "toolStripLabelOutput");
            // 
            // toolStripSplitButtonOutputBrowser
            // 
            this.toolStripSplitButtonOutputBrowser.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripSplitButtonOutputBrowser.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripSplitButtonOutputBrowser.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripMenuItemOutputBrowse,
            this.toolStripMenuItemOutputManual,
            this.toolStripMenuItemOutputTemplate});
            this.toolStripSplitButtonOutputBrowser.Image = global::JDP.Properties.Resources.folder;
            resources.ApplyResources(this.toolStripSplitButtonOutputBrowser, "toolStripSplitButtonOutputBrowser");
            this.toolStripSplitButtonOutputBrowser.Name = "toolStripSplitButtonOutputBrowser";
            this.toolStripSplitButtonOutputBrowser.DropDownItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStripSplitButtonOutputBrowser_DropDownItemClicked);
            // 
            // toolStripMenuItemOutputBrowse
            // 
            this.toolStripMenuItemOutputBrowse.Image = global::JDP.Properties.Resources.folder;
            this.toolStripMenuItemOutputBrowse.Name = "toolStripMenuItemOutputBrowse";
            resources.ApplyResources(this.toolStripMenuItemOutputBrowse, "toolStripMenuItemOutputBrowse");
            // 
            // toolStripMenuItemOutputManual
            // 
            this.toolStripMenuItemOutputManual.Image = global::JDP.Properties.Resources.cog;
            this.toolStripMenuItemOutputManual.Name = "toolStripMenuItemOutputManual";
            resources.ApplyResources(this.toolStripMenuItemOutputManual, "toolStripMenuItemOutputManual");
            // 
            // toolStripMenuItemOutputTemplate
            // 
            this.toolStripMenuItemOutputTemplate.Image = global::JDP.Properties.Resources.folder_page;
            this.toolStripMenuItemOutputTemplate.Name = "toolStripMenuItemOutputTemplate";
            resources.ApplyResources(this.toolStripMenuItemOutputTemplate, "toolStripMenuItemOutputTemplate");
            // 
            // grpAction
            // 
            this.grpAction.Controls.Add(this.comboBoxScript);
            this.grpAction.Controls.Add(this.rbActionCorrectFilenames);
            this.grpAction.Controls.Add(this.rbActionCreateCUESheet);
            this.grpAction.Controls.Add(this.rbActionVerify);
            this.grpAction.Controls.Add(this.rbActionEncode);
            resources.ApplyResources(this.grpAction, "grpAction");
            this.grpAction.Name = "grpAction";
            this.grpAction.TabStop = false;
            // 
            // comboBoxScript
            // 
            resources.ApplyResources(this.comboBoxScript, "comboBoxScript");
            this.comboBoxScript.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxScript.FormattingEnabled = true;
            this.comboBoxScript.Name = "comboBoxScript";
            this.toolTip1.SetToolTip(this.comboBoxScript, resources.GetString("comboBoxScript.ToolTip"));
            // 
            // rbActionCorrectFilenames
            // 
            resources.ApplyResources(this.rbActionCorrectFilenames, "rbActionCorrectFilenames");
            this.rbActionCorrectFilenames.Name = "rbActionCorrectFilenames";
            this.rbActionCorrectFilenames.TabStop = true;
            this.toolTip1.SetToolTip(this.rbActionCorrectFilenames, resources.GetString("rbActionCorrectFilenames.ToolTip"));
            this.rbActionCorrectFilenames.UseVisualStyleBackColor = true;
            this.rbActionCorrectFilenames.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
            // 
            // rbActionCreateCUESheet
            // 
            resources.ApplyResources(this.rbActionCreateCUESheet, "rbActionCreateCUESheet");
            this.rbActionCreateCUESheet.Name = "rbActionCreateCUESheet";
            this.rbActionCreateCUESheet.TabStop = true;
            this.toolTip1.SetToolTip(this.rbActionCreateCUESheet, resources.GetString("rbActionCreateCUESheet.ToolTip"));
            this.rbActionCreateCUESheet.UseVisualStyleBackColor = true;
            this.rbActionCreateCUESheet.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
            // 
            // rbActionVerify
            // 
            resources.ApplyResources(this.rbActionVerify, "rbActionVerify");
            this.rbActionVerify.Name = "rbActionVerify";
            this.toolTip1.SetToolTip(this.rbActionVerify, resources.GetString("rbActionVerify.ToolTip"));
            this.rbActionVerify.UseVisualStyleBackColor = true;
            this.rbActionVerify.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
            // 
            // rbActionEncode
            // 
            resources.ApplyResources(this.rbActionEncode, "rbActionEncode");
            this.rbActionEncode.Name = "rbActionEncode";
            this.toolTip1.SetToolTip(this.rbActionEncode, resources.GetString("rbActionEncode.ToolTip"));
            this.rbActionEncode.UseVisualStyleBackColor = true;
            this.rbActionEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
            // 
            // grpExtra
            // 
            this.grpExtra.Controls.Add(this.tableLayoutPanel4);
            resources.ApplyResources(this.grpExtra, "grpExtra");
            this.grpExtra.Name = "grpExtra";
            this.tableLayoutPanel2.SetRowSpan(this.grpExtra, 2);
            this.grpExtra.TabStop = false;
            // 
            // tableLayoutPanel4
            // 
            resources.ApplyResources(this.tableLayoutPanel4, "tableLayoutPanel4");
            this.tableLayoutPanel4.Controls.Add(this.labelPregap, 0, 0);
            this.tableLayoutPanel4.Controls.Add(this.lblWriteOffset, 0, 2);
            this.tableLayoutPanel4.Controls.Add(this.numericWriteOffset, 1, 2);
            this.tableLayoutPanel4.Controls.Add(this.txtPreGapLength, 1, 0);
            this.tableLayoutPanel4.Controls.Add(this.labelDataTrack, 0, 1);
            this.tableLayoutPanel4.Controls.Add(this.txtDataTrackLength, 1, 1);
            this.tableLayoutPanel4.Name = "tableLayoutPanel4";
            // 
            // labelPregap
            // 
            resources.ApplyResources(this.labelPregap, "labelPregap");
            this.labelPregap.Name = "labelPregap";
            // 
            // lblWriteOffset
            // 
            resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
            this.lblWriteOffset.Name = "lblWriteOffset";
            // 
            // numericWriteOffset
            // 
            resources.ApplyResources(this.numericWriteOffset, "numericWriteOffset");
            this.numericWriteOffset.Maximum = new decimal(new int[] {
            99999,
            0,
            0,
            0});
            this.numericWriteOffset.Minimum = new decimal(new int[] {
            99999,
            0,
            0,
            -2147483648});
            this.numericWriteOffset.Name = "numericWriteOffset";
            this.toolTip1.SetToolTip(this.numericWriteOffset, resources.GetString("numericWriteOffset.ToolTip"));
            // 
            // txtPreGapLength
            // 
            this.txtPreGapLength.Culture = new System.Globalization.CultureInfo("");
            this.txtPreGapLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
            resources.ApplyResources(this.txtPreGapLength, "txtPreGapLength");
            this.txtPreGapLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
            this.txtPreGapLength.Name = "txtPreGapLength";
            this.txtPreGapLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
            this.toolTip1.SetToolTip(this.txtPreGapLength, resources.GetString("txtPreGapLength.ToolTip"));
            // 
            // labelDataTrack
            // 
            resources.ApplyResources(this.labelDataTrack, "labelDataTrack");
            this.labelDataTrack.Name = "labelDataTrack";
            // 
            // txtDataTrackLength
            // 
            this.txtDataTrackLength.Culture = new System.Globalization.CultureInfo("");
            this.txtDataTrackLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
            resources.ApplyResources(this.txtDataTrackLength, "txtDataTrackLength");
            this.txtDataTrackLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
            this.txtDataTrackLength.Name = "txtDataTrackLength";
            this.txtDataTrackLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
            this.toolTip1.SetToolTip(this.txtDataTrackLength, resources.GetString("txtDataTrackLength.ToolTip"));
            // 
            // panelGo
            // 
            this.panelGo.Controls.Add(this.btnConvert);
            this.panelGo.Controls.Add(this.btnStop);
            this.panelGo.Controls.Add(this.btnResume);
            this.panelGo.Controls.Add(this.btnPause);
            resources.ApplyResources(this.panelGo, "panelGo");
            this.panelGo.Name = "panelGo";
            // 
            // btnConvert
            // 
            resources.ApplyResources(this.btnConvert, "btnConvert");
            this.btnConvert.Name = "btnConvert";
            this.btnConvert.UseVisualStyleBackColor = true;
            this.btnConvert.Click += new System.EventHandler(this.btnConvert_Click);
            // 
            // btnStop
            // 
            resources.ApplyResources(this.btnStop, "btnStop");
            this.btnStop.Name = "btnStop";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // btnResume
            // 
            resources.ApplyResources(this.btnResume, "btnResume");
            this.btnResume.Name = "btnResume";
            this.btnResume.UseVisualStyleBackColor = true;
            this.btnResume.Click += new System.EventHandler(this.btnPause_Click);
            // 
            // btnPause
            // 
            resources.ApplyResources(this.btnPause, "btnPause");
            this.btnPause.Name = "btnPause";
            this.btnPause.UseVisualStyleBackColor = true;
            this.btnPause.Click += new System.EventHandler(this.btnPause_Click);
            // 
            // toolStripMenu
            // 
            this.toolStripMenu.BackColor = System.Drawing.SystemColors.GradientInactiveCaption;
            resources.ApplyResources(this.toolStripMenu, "toolStripMenu");
            this.toolStripMenu.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStripMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripDropDownButtonProfile,
            this.toolStripSeparator3,
            this.toolStripButtonAbout,
            this.toolStripButtonHelp,
            this.toolStripButtonSettings,
            this.toolStripButtonShowLog});
            this.toolStripMenu.Name = "toolStripMenu";
            this.toolStripMenu.RenderMode = System.Windows.Forms.ToolStripRenderMode.System;
            this.toolStripMenu.Stretch = true;
            // 
            // toolStripDropDownButtonProfile
            // 
            this.toolStripDropDownButtonProfile.AutoToolTip = false;
            this.toolStripDropDownButtonProfile.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripTextBoxAddProfile,
            this.toolStripMenuItemDeleteProfile,
            this.toolStripSeparator5,
            this.defaultToolStripMenuItem});
            this.toolStripDropDownButtonProfile.Image = global::JDP.Properties.Resources.basket;
            resources.ApplyResources(this.toolStripDropDownButtonProfile, "toolStripDropDownButtonProfile");
            this.toolStripDropDownButtonProfile.Name = "toolStripDropDownButtonProfile";
            this.toolStripDropDownButtonProfile.DropDownItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStripDropDownButtonProfile_DropDownItemClicked);
            // 
            // toolStripTextBoxAddProfile
            // 
            this.toolStripTextBoxAddProfile.Name = "toolStripTextBoxAddProfile";
            resources.ApplyResources(this.toolStripTextBoxAddProfile, "toolStripTextBoxAddProfile");
            this.toolStripTextBoxAddProfile.KeyDown += new System.Windows.Forms.KeyEventHandler(this.toolStripTextBoxAddProfile_KeyDown);
            // 
            // toolStripMenuItemDeleteProfile
            // 
            this.toolStripMenuItemDeleteProfile.Image = global::JDP.Properties.Resources.delete;
            this.toolStripMenuItemDeleteProfile.Name = "toolStripMenuItemDeleteProfile";
            resources.ApplyResources(this.toolStripMenuItemDeleteProfile, "toolStripMenuItemDeleteProfile");
            // 
            // toolStripSeparator5
            // 
            this.toolStripSeparator5.Name = "toolStripSeparator5";
            resources.ApplyResources(this.toolStripSeparator5, "toolStripSeparator5");
            // 
            // defaultToolStripMenuItem
            // 
            this.defaultToolStripMenuItem.Image = global::JDP.Properties.Resources.basket;
            this.defaultToolStripMenuItem.Name = "defaultToolStripMenuItem";
            resources.ApplyResources(this.defaultToolStripMenuItem, "defaultToolStripMenuItem");
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            resources.ApplyResources(this.toolStripSeparator3, "toolStripSeparator3");
            // 
            // toolStripButtonAbout
            // 
            this.toolStripButtonAbout.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripButtonAbout.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButtonAbout.Image = global::JDP.Properties.Resources.information;
            resources.ApplyResources(this.toolStripButtonAbout, "toolStripButtonAbout");
            this.toolStripButtonAbout.Name = "toolStripButtonAbout";
            this.toolStripButtonAbout.Click += new System.EventHandler(this.toolStripButtonAbout_Click);
            // 
            // toolStripButtonHelp
            // 
            this.toolStripButtonHelp.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripButtonHelp.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButtonHelp.Image = global::JDP.Properties.Resources.world_go;
            resources.ApplyResources(this.toolStripButtonHelp, "toolStripButtonHelp");
            this.toolStripButtonHelp.Name = "toolStripButtonHelp";
            this.toolStripButtonHelp.Click += new System.EventHandler(this.toolStripButtonHelp_Click);
            // 
            // toolStripButtonSettings
            // 
            this.toolStripButtonSettings.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripButtonSettings.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButtonSettings.Image = global::JDP.Properties.Resources.cog;
            resources.ApplyResources(this.toolStripButtonSettings, "toolStripButtonSettings");
            this.toolStripButtonSettings.Name = "toolStripButtonSettings";
            this.toolStripButtonSettings.Click += new System.EventHandler(this.toolStripButtonSettings_Click);
            // 
            // toolStripButtonShowLog
            // 
            this.toolStripButtonShowLog.Alignment = System.Windows.Forms.ToolStripItemAlignment.Right;
            this.toolStripButtonShowLog.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.toolStripButtonShowLog.Image = global::JDP.Properties.Resources.report;
            resources.ApplyResources(this.toolStripButtonShowLog, "toolStripButtonShowLog");
            this.toolStripButtonShowLog.Name = "toolStripButtonShowLog";
            this.toolStripButtonShowLog.Click += new System.EventHandler(this.toolStripButton4_Click);
            // 
            // toolTip1
            // 
            this.toolTip1.AutoPopDelay = 15000;
            this.toolTip1.InitialDelay = 500;
            this.toolTip1.IsBalloon = true;
            this.toolTip1.ReshowDelay = 100;
            // 
            // contextMenuStripFileTree
            // 
            this.contextMenuStripFileTree.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.SelectedNodeName,
            this.toolStripSeparator2,
            this.setAsMyMusicFolderToolStripMenuItem,
            this.resetToOriginalLocationToolStripMenuItem,
            this.editMetadataToolStripMenuItem,
            this.addFolderToLocalDatabaseToolStripMenuItem,
            this.removeItemFromDatabaseToolStripMenuItem});
            this.contextMenuStripFileTree.Name = "contextMenuStripFileTree";
            resources.ApplyResources(this.contextMenuStripFileTree, "contextMenuStripFileTree");
            // 
            // SelectedNodeName
            // 
            resources.ApplyResources(this.SelectedNodeName, "SelectedNodeName");
            this.SelectedNodeName.Name = "SelectedNodeName";
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            resources.ApplyResources(this.toolStripSeparator2, "toolStripSeparator2");
            // 
            // setAsMyMusicFolderToolStripMenuItem
            // 
            this.setAsMyMusicFolderToolStripMenuItem.Name = "setAsMyMusicFolderToolStripMenuItem";
            resources.ApplyResources(this.setAsMyMusicFolderToolStripMenuItem, "setAsMyMusicFolderToolStripMenuItem");
            this.setAsMyMusicFolderToolStripMenuItem.Click += new System.EventHandler(this.setAsMyMusicFolderToolStripMenuItem_Click);
            // 
            // resetToOriginalLocationToolStripMenuItem
            // 
            this.resetToOriginalLocationToolStripMenuItem.Name = "resetToOriginalLocationToolStripMenuItem";
            resources.ApplyResources(this.resetToOriginalLocationToolStripMenuItem, "resetToOriginalLocationToolStripMenuItem");
            this.resetToOriginalLocationToolStripMenuItem.Click += new System.EventHandler(this.resetToOriginalLocationToolStripMenuItem_Click);
            // 
            // editMetadataToolStripMenuItem
            // 
            this.editMetadataToolStripMenuItem.Name = "editMetadataToolStripMenuItem";
            resources.ApplyResources(this.editMetadataToolStripMenuItem, "editMetadataToolStripMenuItem");
            this.editMetadataToolStripMenuItem.Click += new System.EventHandler(this.editMetadataToolStripMenuItem_Click);
            // 
            // addFolderToLocalDatabaseToolStripMenuItem
            // 
            this.addFolderToLocalDatabaseToolStripMenuItem.Name = "addFolderToLocalDatabaseToolStripMenuItem";
            resources.ApplyResources(this.addFolderToLocalDatabaseToolStripMenuItem, "addFolderToLocalDatabaseToolStripMenuItem");
            this.addFolderToLocalDatabaseToolStripMenuItem.Click += new System.EventHandler(this.addFolderToLocalDatabaseToolStripMenuItem_Click);
            // 
            // removeItemFromDatabaseToolStripMenuItem
            // 
            this.removeItemFromDatabaseToolStripMenuItem.Name = "removeItemFromDatabaseToolStripMenuItem";
            resources.ApplyResources(this.removeItemFromDatabaseToolStripMenuItem, "removeItemFromDatabaseToolStripMenuItem");
            this.removeItemFromDatabaseToolStripMenuItem.Click += new System.EventHandler(this.removeItemFromDatabaseToolStripMenuItem_Click);
            // 
            // backgroundWorkerAddToLocalDB
            // 
            this.backgroundWorkerAddToLocalDB.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorkerAddToLocalDB_DoWork);
            this.backgroundWorkerAddToLocalDB.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorkerAddToLocalDB_RunWorkerCompleted);
            // 
            // frmCUETools
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.toolStripContainer1);
            this.Name = "frmCUETools";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUETools_FormClosed);
            this.Load += new System.EventHandler(this.frmCUETools_Load);
            this.toolStripContainer1.BottomToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.BottomToolStripPanel.PerformLayout();
            this.toolStripContainer1.ContentPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.PerformLayout();
            this.toolStripContainer1.ResumeLayout(false);
            this.toolStripContainer1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.grpInput.ResumeLayout(false);
            this.tableLayoutPanel2.ResumeLayout(false);
            this.groupBoxMode.ResumeLayout(false);
            this.tableLayoutPanelVerifyMode.ResumeLayout(false);
            this.tableLayoutPanelVerifyMode.PerformLayout();
            this.tableLayoutPanelCUEStyle.ResumeLayout(false);
            this.tableLayoutPanelCUEStyle.PerformLayout();
            this.toolStripCorrectorFormat.ResumeLayout(false);
            this.toolStripCorrectorFormat.PerformLayout();
            this.grpAudioOutput.ResumeLayout(false);
            this.grpAudioOutput.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMotd)).EndInit();
            this.grpOutputPathGeneration.ResumeLayout(false);
            this.tableLayoutPanelPaths.ResumeLayout(false);
            this.tableLayoutPanelPaths.PerformLayout();
            this.toolStripInput.ResumeLayout(false);
            this.toolStripInput.PerformLayout();
            this.toolStripOutput.ResumeLayout(false);
            this.toolStripOutput.PerformLayout();
            this.grpAction.ResumeLayout(false);
            this.grpAction.PerformLayout();
            this.grpExtra.ResumeLayout(false);
            this.tableLayoutPanel4.ResumeLayout(false);
            this.tableLayoutPanel4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
            this.panelGo.ResumeLayout(false);
            this.toolStripMenu.ResumeLayout(false);
            this.toolStripMenu.PerformLayout();
            this.contextMenuStripFileTree.ResumeLayout(false);
            this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Button btnConvert;
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.RadioButton rbSingleFile;
		private System.Windows.Forms.GroupBox grpOutputPathGeneration;
		private System.Windows.Forms.GroupBox grpAudioOutput;
		private System.Windows.Forms.GroupBox grpAction;
		private System.Windows.Forms.RadioButton rbActionVerify;
		private System.Windows.Forms.RadioButton rbActionEncode;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.RadioButton rbEmbedCUE;
		private System.Windows.Forms.MaskedTextBox txtDataTrackLength;
		private System.Windows.Forms.Label labelDataTrack;
		private System.Windows.Forms.Button btnStop;
		private System.Windows.Forms.Button btnPause;
		private System.Windows.Forms.Button btnResume;
		private System.Windows.Forms.MaskedTextBox txtPreGapLength;
		private System.Windows.Forms.Label labelPregap;
		private CUEControls.FileSystemTreeView fileSystemTreeView1;
		private System.Windows.Forms.TextBox txtInputPath;
		private System.Windows.Forms.GroupBox grpInput;
		private System.Windows.Forms.GroupBox grpExtra;
		private System.Windows.Forms.RadioButton rbActionCorrectFilenames;
		private System.Windows.Forms.RadioButton rbActionCreateCUESheet;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripFileTree;
		private System.Windows.Forms.ToolStripMenuItem setAsMyMusicFolderToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem SelectedNodeName;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
		private System.Windows.Forms.ToolStripMenuItem resetToOriginalLocationToolStripMenuItem;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.Label lblWriteOffset;
		private System.Windows.Forms.TextBox textBatchReport;
		private System.Windows.Forms.ComboBox comboBoxAudioFormat;
		private System.Windows.Forms.Label labelFormat;
		private System.Windows.Forms.GroupBox groupBoxMode;
		private System.Windows.Forms.ComboBox comboBoxScript;
		private System.Windows.Forms.RadioButton radioButtonAudioNone;
		private System.Windows.Forms.RadioButton radioButtonAudioLossy;
		private System.Windows.Forms.RadioButton radioButtonAudioHybrid;
		private System.Windows.Forms.RadioButton radioButtonAudioLossless;
		private System.Windows.Forms.ComboBox comboBoxEncoder;
		private System.Windows.Forms.ToolStripContainer toolStripContainer1;
		private System.Windows.Forms.ToolStrip toolStripMenu;
		private System.Windows.Forms.ToolStripButton toolStripButtonShowLog;
		private System.Windows.Forms.ComboBox comboBoxOutputFormat;
		private System.Windows.Forms.Label labelOutputTemplate;
		private System.Windows.Forms.TrackBar trackBarEncoderMode;
		private System.Windows.Forms.Label labelEncoderMode;
		private System.Windows.Forms.Label labelEncoderMaxMode;
		private System.Windows.Forms.Label labelEncoderMinMode;
		private System.Windows.Forms.RadioButton rbTracks;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel2;
		private System.Windows.Forms.Panel panelGo;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelProcessed;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelAR;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar2;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel4;
		private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButtonProfile;
		private System.Windows.Forms.ToolStripMenuItem defaultToolStripMenuItem;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
		private System.Windows.Forms.ToolStripButton toolStripButtonAbout;
		private System.Windows.Forms.ToolStripButton toolStripButtonSettings;
		private System.Windows.Forms.ToolStripTextBox toolStripTextBoxAddProfile;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemDeleteProfile;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator5;
		private System.Windows.Forms.ToolStripButton toolStripButtonHelp;
		private System.Windows.Forms.PictureBox pictureBoxMotd;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanelPaths;
		private System.Windows.Forms.ToolStrip toolStripInput;
		private System.Windows.Forms.ToolStripSplitButton toolStripSplitButtonInputBrowser;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemInputBrowserFiles;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemInputBrowserMulti;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemInputBrowserDrag;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemInputBrowserHide;
		private System.Windows.Forms.ToolStrip toolStripOutput;
		private System.Windows.Forms.ToolStripSplitButton toolStripSplitButtonOutputBrowser;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemOutputBrowse;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemOutputTemplate;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemOutputManual;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanelCUEStyle;
		private System.Windows.Forms.ToolStripLabel toolStripLabelInput;
		private System.Windows.Forms.ToolStripLabel toolStripLabelOutput;
		private System.Windows.Forms.ToolStrip toolStripCorrectorFormat;
		private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButtonCorrectorFormat;
		private System.Windows.Forms.ToolStripDropDownButton toolStripDropDownButtonCorrectorMode;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemCorrectorModeLocateFiles;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItemCorrectorModeChangeExtension;
		private System.Windows.Forms.ToolStripButton toolStripButtonCorrectorOverwrite;
		private System.Windows.Forms.CheckBox checkBoxUseMusicBrainz;
		private System.Windows.Forms.CheckBox checkBoxUseAccurateRip;
		private System.Windows.Forms.CheckBox checkBoxUseFreeDb;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanelVerifyMode;
		private System.Windows.Forms.CheckBox checkBoxVerifyUseCDRepair;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelCTDB;
		private System.Windows.Forms.CheckBox checkBoxVerifyUseLocal;
		private System.Windows.Forms.CheckBox checkBoxSkipRecent;
		private System.Windows.Forms.ToolStripMenuItem editMetadataToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem addFolderToLocalDatabaseToolStripMenuItem;
		private System.ComponentModel.BackgroundWorker backgroundWorkerAddToLocalDB;
		private System.Windows.Forms.ToolStripMenuItem removeItemFromDatabaseToolStripMenuItem;
	}
}

