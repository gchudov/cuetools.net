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
			this.statusStrip2 = new System.Windows.Forms.StatusStrip();
			this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelProcessed = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelAR = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
			this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
			this.grpInput = new System.Windows.Forms.GroupBox();
			this.textBatchReport = new System.Windows.Forms.TextBox();
			this.fileSystemTreeView1 = new CUEControls.FileSystemTreeView();
			this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
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
			this.grpOutputPathGeneration = new System.Windows.Forms.GroupBox();
			this.labelOutputTemplate = new System.Windows.Forms.Label();
			this.labelOutput = new System.Windows.Forms.Label();
			this.labelInput = new System.Windows.Forms.Label();
			this.checkBoxDontGenerate = new System.Windows.Forms.CheckBox();
			this.comboBoxOutputFormat = new System.Windows.Forms.ComboBox();
			this.txtInputPath = new System.Windows.Forms.TextBox();
			this.btnBrowseOutput = new System.Windows.Forms.Button();
			this.txtOutputPath = new System.Windows.Forms.TextBox();
			this.grpAction = new System.Windows.Forms.GroupBox();
			this.checkBoxAdvancedMode = new System.Windows.Forms.CheckBox();
			this.comboBoxScript = new System.Windows.Forms.ComboBox();
			this.rbActionCorrectFilenames = new System.Windows.Forms.RadioButton();
			this.rbActionCreateCUESheet = new System.Windows.Forms.RadioButton();
			this.rbActionVerifyAndEncode = new System.Windows.Forms.RadioButton();
			this.rbActionVerify = new System.Windows.Forms.RadioButton();
			this.rbActionEncode = new System.Windows.Forms.RadioButton();
			this.groupBoxCorrector = new System.Windows.Forms.GroupBox();
			this.rbCorrectorLocateFiles = new System.Windows.Forms.RadioButton();
			this.rbCorrectorChangeExtension = new System.Windows.Forms.RadioButton();
			this.checkBoxCorrectorOverwrite = new System.Windows.Forms.CheckBox();
			this.labelCorrectorFormat = new System.Windows.Forms.Label();
			this.comboBoxCorrectorFormat = new System.Windows.Forms.ComboBox();
			this.grpExtra = new System.Windows.Forms.GroupBox();
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.txtPreGapLength = new System.Windows.Forms.MaskedTextBox();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.labelPregap = new System.Windows.Forms.Label();
			this.txtDataTrackLength = new System.Windows.Forms.MaskedTextBox();
			this.labelDataTrack = new System.Windows.Forms.Label();
			this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
			this.labelMotd = new System.Windows.Forms.Label();
			this.grpOutputStyle = new System.Windows.Forms.GroupBox();
			this.rbTracks = new System.Windows.Forms.RadioButton();
			this.rbEmbedCUE = new System.Windows.Forms.RadioButton();
			this.rbSingleFile = new System.Windows.Forms.RadioButton();
			this.grpFreedb = new System.Windows.Forms.GroupBox();
			this.rbFreedbAlways = new System.Windows.Forms.RadioButton();
			this.rbFreedbIf = new System.Windows.Forms.RadioButton();
			this.rbFreedbNever = new System.Windows.Forms.RadioButton();
			this.panel1 = new System.Windows.Forms.Panel();
			this.btnAbout = new System.Windows.Forms.Button();
			this.btnSettings = new System.Windows.Forms.Button();
			this.btnConvert = new System.Windows.Forms.Button();
			this.btnStop = new System.Windows.Forms.Button();
			this.btnResume = new System.Windows.Forms.Button();
			this.btnPause = new System.Windows.Forms.Button();
			this.toolStrip1 = new System.Windows.Forms.ToolStrip();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.toolStripButton1 = new System.Windows.Forms.ToolStripButton();
			this.toolStripButton2 = new System.Windows.Forms.ToolStripButton();
			this.toolStripButton3 = new System.Windows.Forms.ToolStripButton();
			this.toolStripButton4 = new System.Windows.Forms.ToolStripButton();
			this.toolStripButton5 = new System.Windows.Forms.ToolStripButton();
			this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.contextMenuStripFileTree = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.SelectedNodeName = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.setAsMyMusicFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.resetToOriginalLocationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripContainer1.BottomToolStripPanel.SuspendLayout();
			this.toolStripContainer1.ContentPanel.SuspendLayout();
			this.toolStripContainer1.LeftToolStripPanel.SuspendLayout();
			this.toolStripContainer1.SuspendLayout();
			this.statusStrip2.SuspendLayout();
			this.tableLayoutPanel1.SuspendLayout();
			this.grpInput.SuspendLayout();
			this.tableLayoutPanel2.SuspendLayout();
			this.grpAudioOutput.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).BeginInit();
			this.grpOutputPathGeneration.SuspendLayout();
			this.grpAction.SuspendLayout();
			this.groupBoxCorrector.SuspendLayout();
			this.grpExtra.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.tableLayoutPanel3.SuspendLayout();
			this.grpOutputStyle.SuspendLayout();
			this.grpFreedb.SuspendLayout();
			this.panel1.SuspendLayout();
			this.toolStrip1.SuspendLayout();
			this.contextMenuStripFileTree.SuspendLayout();
			this.SuspendLayout();
			// 
			// toolStripContainer1
			// 
			// 
			// toolStripContainer1.BottomToolStripPanel
			// 
			this.toolStripContainer1.BottomToolStripPanel.Controls.Add(this.statusStrip2);
			// 
			// toolStripContainer1.ContentPanel
			// 
			resources.ApplyResources(this.toolStripContainer1.ContentPanel, "toolStripContainer1.ContentPanel");
			this.toolStripContainer1.ContentPanel.Controls.Add(this.tableLayoutPanel1);
			resources.ApplyResources(this.toolStripContainer1, "toolStripContainer1");
			// 
			// toolStripContainer1.LeftToolStripPanel
			// 
			this.toolStripContainer1.LeftToolStripPanel.Controls.Add(this.toolStrip1);
			this.toolStripContainer1.LeftToolStripPanel.MaximumSize = new System.Drawing.Size(32, 0);
			this.toolStripContainer1.Name = "toolStripContainer1";
			this.toolStripContainer1.RightToolStripPanelVisible = false;
			this.toolStripContainer1.TopToolStripPanelVisible = false;
			// 
			// statusStrip2
			// 
			resources.ApplyResources(this.statusStrip2, "statusStrip2");
			this.statusStrip2.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLabelProcessed,
            this.toolStripStatusLabelAR,
            this.toolStripProgressBar1,
            this.toolStripProgressBar2});
			this.statusStrip2.Name = "statusStrip2";
			// 
			// toolStripStatusLabel1
			// 
			this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
			resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
			this.toolStripStatusLabel1.Spring = true;
			// 
			// toolStripStatusLabelProcessed
			// 
			this.toolStripStatusLabelProcessed.Name = "toolStripStatusLabelProcessed";
			resources.ApplyResources(this.toolStripStatusLabelProcessed, "toolStripStatusLabelProcessed");
			// 
			// toolStripStatusLabelAR
			// 
			resources.ApplyResources(this.toolStripStatusLabelAR, "toolStripStatusLabelAR");
			this.toolStripStatusLabelAR.Image = global::JDP.Properties.Resources.AR;
			this.toolStripStatusLabelAR.Name = "toolStripStatusLabelAR";
			this.toolStripStatusLabelAR.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
			// 
			// toolStripProgressBar1
			// 
			this.toolStripProgressBar1.AutoToolTip = true;
			this.toolStripProgressBar1.Name = "toolStripProgressBar1";
			resources.ApplyResources(this.toolStripProgressBar1, "toolStripProgressBar1");
			this.toolStripProgressBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
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
			this.tableLayoutPanel1.Controls.Add(this.grpInput, 0, 0);
			this.tableLayoutPanel1.Controls.Add(this.tableLayoutPanel2, 1, 0);
			this.tableLayoutPanel1.GrowStyle = System.Windows.Forms.TableLayoutPanelGrowStyle.AddColumns;
			this.tableLayoutPanel1.Name = "tableLayoutPanel1";
			// 
			// grpInput
			// 
			this.grpInput.Controls.Add(this.textBatchReport);
			this.grpInput.Controls.Add(this.fileSystemTreeView1);
			resources.ApplyResources(this.grpInput, "grpInput");
			this.grpInput.Name = "grpInput";
			this.grpInput.TabStop = false;
			// 
			// textBatchReport
			// 
			resources.ApplyResources(this.textBatchReport, "textBatchReport");
			this.textBatchReport.Name = "textBatchReport";
			this.textBatchReport.ReadOnly = true;
			this.textBatchReport.TabStop = false;
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
			this.fileSystemTreeView1.AfterCheck += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterCheck);
			this.fileSystemTreeView1.NodeExpand += new CUEControls.FileSystemTreeViewNodeExpandHandler(this.fileSystemTreeView1_NodeExpand);
			this.fileSystemTreeView1.DragDrop += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragDrop);
			this.fileSystemTreeView1.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterSelect);
			this.fileSystemTreeView1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.fileSystemTreeView1_MouseDown);
			this.fileSystemTreeView1.DragEnter += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragEnter);
			this.fileSystemTreeView1.KeyDown += new System.Windows.Forms.KeyEventHandler(this.fileSystemTreeView1_KeyDown);
			this.fileSystemTreeView1.AfterExpand += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterExpand);
			// 
			// tableLayoutPanel2
			// 
			resources.ApplyResources(this.tableLayoutPanel2, "tableLayoutPanel2");
			this.tableLayoutPanel2.Controls.Add(this.grpAudioOutput, 1, 1);
			this.tableLayoutPanel2.Controls.Add(this.grpOutputPathGeneration, 0, 0);
			this.tableLayoutPanel2.Controls.Add(this.grpAction, 0, 1);
			this.tableLayoutPanel2.Controls.Add(this.groupBoxCorrector, 1, 2);
			this.tableLayoutPanel2.Controls.Add(this.grpExtra, 0, 2);
			this.tableLayoutPanel2.Controls.Add(this.tableLayoutPanel3, 2, 1);
			this.tableLayoutPanel2.Controls.Add(this.panel1, 2, 2);
			this.tableLayoutPanel2.Name = "tableLayoutPanel2";
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
			this.comboBoxAudioFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboBoxAudioFormat.FormattingEnabled = true;
			this.comboBoxAudioFormat.Name = "comboBoxAudioFormat";
			this.toolTip1.SetToolTip(this.comboBoxAudioFormat, resources.GetString("comboBoxAudioFormat.ToolTip"));
			this.comboBoxAudioFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxAudioFormat_SelectedIndexChanged);
			// 
			// grpOutputPathGeneration
			// 
			this.tableLayoutPanel2.SetColumnSpan(this.grpOutputPathGeneration, 3);
			this.grpOutputPathGeneration.Controls.Add(this.labelOutputTemplate);
			this.grpOutputPathGeneration.Controls.Add(this.labelOutput);
			this.grpOutputPathGeneration.Controls.Add(this.labelInput);
			this.grpOutputPathGeneration.Controls.Add(this.checkBoxDontGenerate);
			this.grpOutputPathGeneration.Controls.Add(this.comboBoxOutputFormat);
			this.grpOutputPathGeneration.Controls.Add(this.txtInputPath);
			this.grpOutputPathGeneration.Controls.Add(this.btnBrowseOutput);
			this.grpOutputPathGeneration.Controls.Add(this.txtOutputPath);
			resources.ApplyResources(this.grpOutputPathGeneration, "grpOutputPathGeneration");
			this.grpOutputPathGeneration.Name = "grpOutputPathGeneration";
			this.grpOutputPathGeneration.TabStop = false;
			// 
			// labelOutputTemplate
			// 
			resources.ApplyResources(this.labelOutputTemplate, "labelOutputTemplate");
			this.labelOutputTemplate.Name = "labelOutputTemplate";
			// 
			// labelOutput
			// 
			resources.ApplyResources(this.labelOutput, "labelOutput");
			this.labelOutput.Name = "labelOutput";
			// 
			// labelInput
			// 
			resources.ApplyResources(this.labelInput, "labelInput");
			this.labelInput.Name = "labelInput";
			// 
			// checkBoxDontGenerate
			// 
			resources.ApplyResources(this.checkBoxDontGenerate, "checkBoxDontGenerate");
			this.checkBoxDontGenerate.Name = "checkBoxDontGenerate";
			this.toolTip1.SetToolTip(this.checkBoxDontGenerate, resources.GetString("checkBoxDontGenerate.ToolTip"));
			this.checkBoxDontGenerate.UseVisualStyleBackColor = true;
			this.checkBoxDontGenerate.CheckedChanged += new System.EventHandler(this.checkBoxDontGenerate_CheckedChanged);
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
			resources.ApplyResources(this.txtInputPath, "txtInputPath");
			this.txtInputPath.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
			this.txtInputPath.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
			this.txtInputPath.BackColor = System.Drawing.SystemColors.Window;
			this.txtInputPath.Name = "txtInputPath";
			this.toolTip1.SetToolTip(this.txtInputPath, resources.GetString("txtInputPath.ToolTip"));
			this.txtInputPath.TextChanged += new System.EventHandler(this.txtInputPath_TextChanged);
			this.txtInputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtInputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// btnBrowseOutput
			// 
			resources.ApplyResources(this.btnBrowseOutput, "btnBrowseOutput");
			this.btnBrowseOutput.Name = "btnBrowseOutput";
			this.toolTip1.SetToolTip(this.btnBrowseOutput, resources.GetString("btnBrowseOutput.ToolTip"));
			this.btnBrowseOutput.UseVisualStyleBackColor = true;
			this.btnBrowseOutput.Click += new System.EventHandler(this.btnBrowseOutput_Click);
			// 
			// txtOutputPath
			// 
			this.txtOutputPath.AllowDrop = true;
			resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
			this.txtOutputPath.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
			this.txtOutputPath.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
			this.txtOutputPath.Name = "txtOutputPath";
			this.toolTip1.SetToolTip(this.txtOutputPath, resources.GetString("txtOutputPath.ToolTip"));
			this.txtOutputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtOutputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// grpAction
			// 
			this.grpAction.Controls.Add(this.checkBoxAdvancedMode);
			this.grpAction.Controls.Add(this.comboBoxScript);
			this.grpAction.Controls.Add(this.rbActionCorrectFilenames);
			this.grpAction.Controls.Add(this.rbActionCreateCUESheet);
			this.grpAction.Controls.Add(this.rbActionVerifyAndEncode);
			this.grpAction.Controls.Add(this.rbActionVerify);
			this.grpAction.Controls.Add(this.rbActionEncode);
			resources.ApplyResources(this.grpAction, "grpAction");
			this.grpAction.Name = "grpAction";
			this.grpAction.TabStop = false;
			// 
			// checkBoxAdvancedMode
			// 
			resources.ApplyResources(this.checkBoxAdvancedMode, "checkBoxAdvancedMode");
			this.checkBoxAdvancedMode.Name = "checkBoxAdvancedMode";
			this.checkBoxAdvancedMode.UseVisualStyleBackColor = true;
			this.checkBoxAdvancedMode.CheckedChanged += new System.EventHandler(this.checkBoxAdvancedMode_CheckedChanged);
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
			// rbActionVerifyAndEncode
			// 
			resources.ApplyResources(this.rbActionVerifyAndEncode, "rbActionVerifyAndEncode");
			this.rbActionVerifyAndEncode.Name = "rbActionVerifyAndEncode";
			this.rbActionVerifyAndEncode.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionVerifyAndEncode, resources.GetString("rbActionVerifyAndEncode.ToolTip"));
			this.rbActionVerifyAndEncode.UseVisualStyleBackColor = true;
			this.rbActionVerifyAndEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
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
			this.rbActionEncode.Checked = true;
			this.rbActionEncode.Name = "rbActionEncode";
			this.rbActionEncode.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionEncode, resources.GetString("rbActionEncode.ToolTip"));
			this.rbActionEncode.UseVisualStyleBackColor = true;
			this.rbActionEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// groupBoxCorrector
			// 
			this.groupBoxCorrector.Controls.Add(this.rbCorrectorLocateFiles);
			this.groupBoxCorrector.Controls.Add(this.rbCorrectorChangeExtension);
			this.groupBoxCorrector.Controls.Add(this.checkBoxCorrectorOverwrite);
			this.groupBoxCorrector.Controls.Add(this.labelCorrectorFormat);
			this.groupBoxCorrector.Controls.Add(this.comboBoxCorrectorFormat);
			resources.ApplyResources(this.groupBoxCorrector, "groupBoxCorrector");
			this.groupBoxCorrector.Name = "groupBoxCorrector";
			this.groupBoxCorrector.TabStop = false;
			// 
			// rbCorrectorLocateFiles
			// 
			resources.ApplyResources(this.rbCorrectorLocateFiles, "rbCorrectorLocateFiles");
			this.rbCorrectorLocateFiles.Name = "rbCorrectorLocateFiles";
			this.rbCorrectorLocateFiles.TabStop = true;
			this.toolTip1.SetToolTip(this.rbCorrectorLocateFiles, resources.GetString("rbCorrectorLocateFiles.ToolTip"));
			this.rbCorrectorLocateFiles.UseVisualStyleBackColor = true;
			// 
			// rbCorrectorChangeExtension
			// 
			resources.ApplyResources(this.rbCorrectorChangeExtension, "rbCorrectorChangeExtension");
			this.rbCorrectorChangeExtension.Name = "rbCorrectorChangeExtension";
			this.rbCorrectorChangeExtension.TabStop = true;
			this.toolTip1.SetToolTip(this.rbCorrectorChangeExtension, resources.GetString("rbCorrectorChangeExtension.ToolTip"));
			this.rbCorrectorChangeExtension.UseVisualStyleBackColor = true;
			this.rbCorrectorChangeExtension.CheckedChanged += new System.EventHandler(this.rbCorrectorChangeExtension_CheckedChanged);
			// 
			// checkBoxCorrectorOverwrite
			// 
			resources.ApplyResources(this.checkBoxCorrectorOverwrite, "checkBoxCorrectorOverwrite");
			this.checkBoxCorrectorOverwrite.Name = "checkBoxCorrectorOverwrite";
			this.toolTip1.SetToolTip(this.checkBoxCorrectorOverwrite, resources.GetString("checkBoxCorrectorOverwrite.ToolTip"));
			this.checkBoxCorrectorOverwrite.UseVisualStyleBackColor = true;
			// 
			// labelCorrectorFormat
			// 
			resources.ApplyResources(this.labelCorrectorFormat, "labelCorrectorFormat");
			this.labelCorrectorFormat.MinimumSize = new System.Drawing.Size(16, 16);
			this.labelCorrectorFormat.Name = "labelCorrectorFormat";
			// 
			// comboBoxCorrectorFormat
			// 
			this.comboBoxCorrectorFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboBoxCorrectorFormat.FormattingEnabled = true;
			resources.ApplyResources(this.comboBoxCorrectorFormat, "comboBoxCorrectorFormat");
			this.comboBoxCorrectorFormat.Name = "comboBoxCorrectorFormat";
			this.comboBoxCorrectorFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxCorrectorFormat_SelectedIndexChanged);
			// 
			// grpExtra
			// 
			this.grpExtra.Controls.Add(this.numericWriteOffset);
			this.grpExtra.Controls.Add(this.txtPreGapLength);
			this.grpExtra.Controls.Add(this.lblWriteOffset);
			this.grpExtra.Controls.Add(this.labelPregap);
			this.grpExtra.Controls.Add(this.txtDataTrackLength);
			this.grpExtra.Controls.Add(this.labelDataTrack);
			resources.ApplyResources(this.grpExtra, "grpExtra");
			this.grpExtra.Name = "grpExtra";
			this.grpExtra.TabStop = false;
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
			resources.ApplyResources(this.txtPreGapLength, "txtPreGapLength");
			this.txtPreGapLength.Culture = new System.Globalization.CultureInfo("");
			this.txtPreGapLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.txtPreGapLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
			this.txtPreGapLength.Name = "txtPreGapLength";
			this.txtPreGapLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.toolTip1.SetToolTip(this.txtPreGapLength, resources.GetString("txtPreGapLength.ToolTip"));
			// 
			// lblWriteOffset
			// 
			resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
			this.lblWriteOffset.Name = "lblWriteOffset";
			// 
			// labelPregap
			// 
			resources.ApplyResources(this.labelPregap, "labelPregap");
			this.labelPregap.Name = "labelPregap";
			// 
			// txtDataTrackLength
			// 
			resources.ApplyResources(this.txtDataTrackLength, "txtDataTrackLength");
			this.txtDataTrackLength.Culture = new System.Globalization.CultureInfo("");
			this.txtDataTrackLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.txtDataTrackLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
			this.txtDataTrackLength.Name = "txtDataTrackLength";
			this.txtDataTrackLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.toolTip1.SetToolTip(this.txtDataTrackLength, resources.GetString("txtDataTrackLength.ToolTip"));
			// 
			// labelDataTrack
			// 
			resources.ApplyResources(this.labelDataTrack, "labelDataTrack");
			this.labelDataTrack.Name = "labelDataTrack";
			// 
			// tableLayoutPanel3
			// 
			resources.ApplyResources(this.tableLayoutPanel3, "tableLayoutPanel3");
			this.tableLayoutPanel3.Controls.Add(this.labelMotd, 0, 2);
			this.tableLayoutPanel3.Controls.Add(this.grpOutputStyle, 0, 0);
			this.tableLayoutPanel3.Controls.Add(this.grpFreedb, 0, 1);
			this.tableLayoutPanel3.Name = "tableLayoutPanel3";
			// 
			// labelMotd
			// 
			resources.ApplyResources(this.labelMotd, "labelMotd");
			this.labelMotd.MinimumSize = new System.Drawing.Size(112, 38);
			this.labelMotd.Name = "labelMotd";
			// 
			// grpOutputStyle
			// 
			this.grpOutputStyle.Controls.Add(this.rbTracks);
			this.grpOutputStyle.Controls.Add(this.rbEmbedCUE);
			this.grpOutputStyle.Controls.Add(this.rbSingleFile);
			resources.ApplyResources(this.grpOutputStyle, "grpOutputStyle");
			this.grpOutputStyle.Name = "grpOutputStyle";
			this.grpOutputStyle.TabStop = false;
			// 
			// rbTracks
			// 
			resources.ApplyResources(this.rbTracks, "rbTracks");
			this.rbTracks.Name = "rbTracks";
			this.rbTracks.TabStop = true;
			this.toolTip1.SetToolTip(this.rbTracks, resources.GetString("rbTracks.ToolTip"));
			this.rbTracks.UseVisualStyleBackColor = true;
			// 
			// rbEmbedCUE
			// 
			resources.ApplyResources(this.rbEmbedCUE, "rbEmbedCUE");
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
			this.rbSingleFile.Name = "rbSingleFile";
			this.rbSingleFile.TabStop = true;
			this.toolTip1.SetToolTip(this.rbSingleFile, resources.GetString("rbSingleFile.ToolTip"));
			this.rbSingleFile.UseVisualStyleBackColor = true;
			// 
			// grpFreedb
			// 
			this.grpFreedb.Controls.Add(this.rbFreedbAlways);
			this.grpFreedb.Controls.Add(this.rbFreedbIf);
			this.grpFreedb.Controls.Add(this.rbFreedbNever);
			resources.ApplyResources(this.grpFreedb, "grpFreedb");
			this.grpFreedb.Name = "grpFreedb";
			this.grpFreedb.TabStop = false;
			// 
			// rbFreedbAlways
			// 
			resources.ApplyResources(this.rbFreedbAlways, "rbFreedbAlways");
			this.rbFreedbAlways.Name = "rbFreedbAlways";
			this.rbFreedbAlways.TabStop = true;
			this.rbFreedbAlways.UseVisualStyleBackColor = true;
			// 
			// rbFreedbIf
			// 
			resources.ApplyResources(this.rbFreedbIf, "rbFreedbIf");
			this.rbFreedbIf.Name = "rbFreedbIf";
			this.rbFreedbIf.TabStop = true;
			this.rbFreedbIf.UseVisualStyleBackColor = true;
			// 
			// rbFreedbNever
			// 
			resources.ApplyResources(this.rbFreedbNever, "rbFreedbNever");
			this.rbFreedbNever.Name = "rbFreedbNever";
			this.rbFreedbNever.TabStop = true;
			this.rbFreedbNever.UseVisualStyleBackColor = true;
			// 
			// panel1
			// 
			this.panel1.Controls.Add(this.btnAbout);
			this.panel1.Controls.Add(this.btnSettings);
			this.panel1.Controls.Add(this.btnConvert);
			this.panel1.Controls.Add(this.btnStop);
			this.panel1.Controls.Add(this.btnResume);
			this.panel1.Controls.Add(this.btnPause);
			resources.ApplyResources(this.panel1, "panel1");
			this.panel1.Name = "panel1";
			// 
			// btnAbout
			// 
			this.btnAbout.Image = global::JDP.Properties.Resources.information;
			resources.ApplyResources(this.btnAbout, "btnAbout");
			this.btnAbout.Name = "btnAbout";
			this.btnAbout.UseVisualStyleBackColor = true;
			this.btnAbout.Click += new System.EventHandler(this.btnAbout_Click);
			// 
			// btnSettings
			// 
			this.btnSettings.Image = global::JDP.Properties.Resources.cog;
			resources.ApplyResources(this.btnSettings, "btnSettings");
			this.btnSettings.Name = "btnSettings";
			this.btnSettings.UseVisualStyleBackColor = true;
			this.btnSettings.Click += new System.EventHandler(this.btnSettings_Click);
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
			// toolStrip1
			// 
			this.toolStrip1.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
			this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripSeparator1,
            this.toolStripButton1,
            this.toolStripButton2,
            this.toolStripButton3,
            this.toolStripButton4,
            this.toolStripButton5,
            this.toolStripSeparator3});
			resources.ApplyResources(this.toolStrip1, "toolStrip1");
			this.toolStrip1.MaximumSize = new System.Drawing.Size(32, 0);
			this.toolStrip1.Name = "toolStrip1";
			this.toolStrip1.TextDirection = System.Windows.Forms.ToolStripTextDirection.Vertical90;
			// 
			// toolStripSeparator1
			// 
			resources.ApplyResources(this.toolStripSeparator1, "toolStripSeparator1");
			this.toolStripSeparator1.Name = "toolStripSeparator1";
			this.toolStripSeparator1.TextDirection = System.Windows.Forms.ToolStripTextDirection.Vertical90;
			// 
			// toolStripButton1
			// 
			resources.ApplyResources(this.toolStripButton1, "toolStripButton1");
			this.toolStripButton1.BackColor = System.Drawing.SystemColors.Control;
			this.toolStripButton1.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripButton1.Image = global::JDP.Properties.Resources.folder;
			this.toolStripButton1.Name = "toolStripButton1";
			this.toolStripButton1.Click += new System.EventHandler(this.toolStripButton1_Click);
			// 
			// toolStripButton2
			// 
			resources.ApplyResources(this.toolStripButton2, "toolStripButton2");
			this.toolStripButton2.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripButton2.Image = global::JDP.Properties.Resources.folder_add;
			this.toolStripButton2.Name = "toolStripButton2";
			this.toolStripButton2.Click += new System.EventHandler(this.toolStripButton2_Click);
			// 
			// toolStripButton3
			// 
			resources.ApplyResources(this.toolStripButton3, "toolStripButton3");
			this.toolStripButton3.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripButton3.Image = global::JDP.Properties.Resources.folder_feed;
			this.toolStripButton3.Name = "toolStripButton3";
			this.toolStripButton3.Click += new System.EventHandler(this.toolStripButton3_Click);
			// 
			// toolStripButton4
			// 
			resources.ApplyResources(this.toolStripButton4, "toolStripButton4");
			this.toolStripButton4.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripButton4.Image = global::JDP.Properties.Resources.folder_page;
			this.toolStripButton4.Name = "toolStripButton4";
			this.toolStripButton4.Click += new System.EventHandler(this.toolStripButton4_Click);
			// 
			// toolStripButton5
			// 
			resources.ApplyResources(this.toolStripButton5, "toolStripButton5");
			this.toolStripButton5.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripButton5.Image = global::JDP.Properties.Resources.folder_delete;
			this.toolStripButton5.Name = "toolStripButton5";
			this.toolStripButton5.Click += new System.EventHandler(this.toolStripButton5_Click);
			// 
			// toolStripSeparator3
			// 
			resources.ApplyResources(this.toolStripSeparator3, "toolStripSeparator3");
			this.toolStripSeparator3.Name = "toolStripSeparator3";
			this.toolStripSeparator3.TextDirection = System.Windows.Forms.ToolStripTextDirection.Vertical90;
			// 
			// toolTip1
			// 
			this.toolTip1.AutoPopDelay = 15000;
			this.toolTip1.InitialDelay = 500;
			this.toolTip1.ReshowDelay = 100;
			// 
			// contextMenuStripFileTree
			// 
			this.contextMenuStripFileTree.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.SelectedNodeName,
            this.toolStripSeparator2,
            this.setAsMyMusicFolderToolStripMenuItem,
            this.resetToOriginalLocationToolStripMenuItem});
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
			// frmCUETools
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.toolStripContainer1);
			this.MaximizeBox = false;
			this.Name = "frmCUETools";
			this.Load += new System.EventHandler(this.frmCUETools_Load);
			this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUETools_FormClosed);
			this.toolStripContainer1.BottomToolStripPanel.ResumeLayout(false);
			this.toolStripContainer1.BottomToolStripPanel.PerformLayout();
			this.toolStripContainer1.ContentPanel.ResumeLayout(false);
			this.toolStripContainer1.LeftToolStripPanel.ResumeLayout(false);
			this.toolStripContainer1.LeftToolStripPanel.PerformLayout();
			this.toolStripContainer1.ResumeLayout(false);
			this.toolStripContainer1.PerformLayout();
			this.statusStrip2.ResumeLayout(false);
			this.statusStrip2.PerformLayout();
			this.tableLayoutPanel1.ResumeLayout(false);
			this.grpInput.ResumeLayout(false);
			this.grpInput.PerformLayout();
			this.tableLayoutPanel2.ResumeLayout(false);
			this.grpAudioOutput.ResumeLayout(false);
			this.grpAudioOutput.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).EndInit();
			this.grpOutputPathGeneration.ResumeLayout(false);
			this.grpOutputPathGeneration.PerformLayout();
			this.grpAction.ResumeLayout(false);
			this.grpAction.PerformLayout();
			this.groupBoxCorrector.ResumeLayout(false);
			this.groupBoxCorrector.PerformLayout();
			this.grpExtra.ResumeLayout(false);
			this.grpExtra.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
			this.tableLayoutPanel3.ResumeLayout(false);
			this.grpOutputStyle.ResumeLayout(false);
			this.grpOutputStyle.PerformLayout();
			this.grpFreedb.ResumeLayout(false);
			this.grpFreedb.PerformLayout();
			this.panel1.ResumeLayout(false);
			this.toolStrip1.ResumeLayout(false);
			this.toolStrip1.PerformLayout();
			this.contextMenuStripFileTree.ResumeLayout(false);
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Button btnConvert;
		private System.Windows.Forms.Button btnBrowseOutput;
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.GroupBox grpOutputStyle;
		private System.Windows.Forms.Button btnAbout;
		private System.Windows.Forms.RadioButton rbSingleFile;
		private System.Windows.Forms.GroupBox grpOutputPathGeneration;
		private System.Windows.Forms.GroupBox grpAudioOutput;
		private System.Windows.Forms.Button btnSettings;
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
		private System.Windows.Forms.RadioButton rbActionVerifyAndEncode;
		private System.Windows.Forms.GroupBox grpFreedb;
		private System.Windows.Forms.RadioButton rbFreedbAlways;
		private System.Windows.Forms.RadioButton rbFreedbIf;
		private System.Windows.Forms.RadioButton rbFreedbNever;
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
		private System.Windows.Forms.GroupBox groupBoxCorrector;
		private System.Windows.Forms.Label labelCorrectorFormat;
		private System.Windows.Forms.ComboBox comboBoxCorrectorFormat;
		private System.Windows.Forms.CheckBox checkBoxCorrectorOverwrite;
		private System.Windows.Forms.RadioButton rbCorrectorLocateFiles;
		private System.Windows.Forms.RadioButton rbCorrectorChangeExtension;
		private System.Windows.Forms.ComboBox comboBoxScript;
		private System.Windows.Forms.RadioButton radioButtonAudioNone;
		private System.Windows.Forms.RadioButton radioButtonAudioLossy;
		private System.Windows.Forms.RadioButton radioButtonAudioHybrid;
		private System.Windows.Forms.RadioButton radioButtonAudioLossless;
		private System.Windows.Forms.ComboBox comboBoxEncoder;
		private System.Windows.Forms.CheckBox checkBoxAdvancedMode;
		private System.Windows.Forms.ToolStripContainer toolStripContainer1;
		private System.Windows.Forms.ToolStrip toolStrip1;
		private System.Windows.Forms.ToolStripButton toolStripButton1;
		private System.Windows.Forms.ToolStripButton toolStripButton2;
		private System.Windows.Forms.ToolStripButton toolStripButton3;
		private System.Windows.Forms.ToolStripButton toolStripButton4;
		private System.Windows.Forms.ToolStripButton toolStripButton5;
		private System.Windows.Forms.StatusStrip statusStrip2;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelProcessed;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelAR;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar2;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
		private System.Windows.Forms.ComboBox comboBoxOutputFormat;
		private System.Windows.Forms.CheckBox checkBoxDontGenerate;
		private System.Windows.Forms.Label labelOutput;
		private System.Windows.Forms.Label labelInput;
		private System.Windows.Forms.Label labelOutputTemplate;
		private System.Windows.Forms.TrackBar trackBarEncoderMode;
		private System.Windows.Forms.Label labelEncoderMode;
		private System.Windows.Forms.Label labelEncoderMaxMode;
		private System.Windows.Forms.Label labelEncoderMinMode;
		private System.Windows.Forms.RadioButton rbTracks;
		private System.Windows.Forms.Label labelMotd;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel2;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel3;
		private System.Windows.Forms.Panel panel1;
	}
}

