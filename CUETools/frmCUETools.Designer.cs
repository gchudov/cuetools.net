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
			this.btnConvert = new System.Windows.Forms.Button();
			this.btnBrowseOutput = new System.Windows.Forms.Button();
			this.txtOutputPath = new System.Windows.Forms.TextBox();
			this.grpOutputStyle = new System.Windows.Forms.GroupBox();
			this.rbEmbedCUE = new System.Windows.Forms.RadioButton();
			this.rbGapsLeftOut = new System.Windows.Forms.RadioButton();
			this.rbGapsPrepended = new System.Windows.Forms.RadioButton();
			this.rbGapsAppended = new System.Windows.Forms.RadioButton();
			this.rbSingleFile = new System.Windows.Forms.RadioButton();
			this.btnAbout = new System.Windows.Forms.Button();
			this.grpOutputPathGeneration = new System.Windows.Forms.GroupBox();
			this.txtCustomFormat = new System.Windows.Forms.TextBox();
			this.rbCustomFormat = new System.Windows.Forms.RadioButton();
			this.txtCreateSubdirectory = new System.Windows.Forms.TextBox();
			this.rbDontGenerate = new System.Windows.Forms.RadioButton();
			this.rbCreateSubdirectory = new System.Windows.Forms.RadioButton();
			this.rbAppendFilename = new System.Windows.Forms.RadioButton();
			this.txtAppendFilename = new System.Windows.Forms.TextBox();
			this.grpAudioOutput = new System.Windows.Forms.GroupBox();
			this.btnCodec = new System.Windows.Forms.Button();
			this.rbUDC1 = new System.Windows.Forms.RadioButton();
			this.rbTTA = new System.Windows.Forms.RadioButton();
			this.chkLossyWAV = new System.Windows.Forms.CheckBox();
			this.rbAPE = new System.Windows.Forms.RadioButton();
			this.rbNoAudio = new System.Windows.Forms.RadioButton();
			this.rbWavPack = new System.Windows.Forms.RadioButton();
			this.rbWAV = new System.Windows.Forms.RadioButton();
			this.rbFLAC = new System.Windows.Forms.RadioButton();
			this.btnSettings = new System.Windows.Forms.Button();
			this.grpAction = new System.Windows.Forms.GroupBox();
			this.rbActionCorrectFilenames = new System.Windows.Forms.RadioButton();
			this.chkRecursive = new System.Windows.Forms.CheckBox();
			this.rbActionCreateCUESheet = new System.Windows.Forms.RadioButton();
			this.chkMulti = new System.Windows.Forms.CheckBox();
			this.rbActionVerifyAndCRCs = new System.Windows.Forms.RadioButton();
			this.rbActionVerifyAndEncode = new System.Windows.Forms.RadioButton();
			this.rbActionVerifyThenEncode = new System.Windows.Forms.RadioButton();
			this.rbActionVerify = new System.Windows.Forms.RadioButton();
			this.rbActionEncode = new System.Windows.Forms.RadioButton();
			this.txtPreGapLength = new System.Windows.Forms.MaskedTextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.txtDataTrackLength = new System.Windows.Forms.MaskedTextBox();
			this.statusStrip1 = new System.Windows.Forms.StatusStrip();
			this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelProcessed = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelWV = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelFLAC = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelAR = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.btnStop = new System.Windows.Forms.Button();
			this.btnPause = new System.Windows.Forms.Button();
			this.btnResume = new System.Windows.Forms.Button();
			this.grpFreedb = new System.Windows.Forms.GroupBox();
			this.rbFreedbAlways = new System.Windows.Forms.RadioButton();
			this.rbFreedbIf = new System.Windows.Forms.RadioButton();
			this.rbFreedbNever = new System.Windows.Forms.RadioButton();
			this.contextMenuStripUDC = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripMenuItem();
			this.tAKToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripMenuItem();
			this.mP3ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.oGGToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.txtInputPath = new System.Windows.Forms.TextBox();
			this.grpInput = new System.Windows.Forms.GroupBox();
			this.textBatchReport = new System.Windows.Forms.TextBox();
			this.fileSystemTreeView1 = new CUEControls.FileSystemTreeView();
			this.grpExtra = new System.Windows.Forms.GroupBox();
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.contextMenuStripFileTree = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.SelectedNodeName = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.setAsMyMusicFolderToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.resetToOriginalLocationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.panel1 = new System.Windows.Forms.Panel();
			this.grpOutputStyle.SuspendLayout();
			this.grpOutputPathGeneration.SuspendLayout();
			this.grpAudioOutput.SuspendLayout();
			this.grpAction.SuspendLayout();
			this.statusStrip1.SuspendLayout();
			this.grpFreedb.SuspendLayout();
			this.contextMenuStripUDC.SuspendLayout();
			this.grpInput.SuspendLayout();
			this.grpExtra.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.contextMenuStripFileTree.SuspendLayout();
			this.panel1.SuspendLayout();
			this.SuspendLayout();
			// 
			// btnConvert
			// 
			this.btnConvert.AccessibleDescription = null;
			this.btnConvert.AccessibleName = null;
			resources.ApplyResources(this.btnConvert, "btnConvert");
			this.btnConvert.BackgroundImage = null;
			this.btnConvert.Font = null;
			this.btnConvert.Name = "btnConvert";
			this.toolTip1.SetToolTip(this.btnConvert, resources.GetString("btnConvert.ToolTip"));
			this.btnConvert.UseVisualStyleBackColor = true;
			this.btnConvert.Click += new System.EventHandler(this.btnConvert_Click);
			// 
			// btnBrowseOutput
			// 
			this.btnBrowseOutput.AccessibleDescription = null;
			this.btnBrowseOutput.AccessibleName = null;
			resources.ApplyResources(this.btnBrowseOutput, "btnBrowseOutput");
			this.btnBrowseOutput.BackgroundImage = null;
			this.btnBrowseOutput.Font = null;
			this.btnBrowseOutput.Name = "btnBrowseOutput";
			this.toolTip1.SetToolTip(this.btnBrowseOutput, resources.GetString("btnBrowseOutput.ToolTip"));
			this.btnBrowseOutput.UseVisualStyleBackColor = true;
			this.btnBrowseOutput.Click += new System.EventHandler(this.btnBrowseOutput_Click);
			// 
			// txtOutputPath
			// 
			this.txtOutputPath.AccessibleDescription = null;
			this.txtOutputPath.AccessibleName = null;
			this.txtOutputPath.AllowDrop = true;
			resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
			this.txtOutputPath.BackgroundImage = null;
			this.txtOutputPath.Font = null;
			this.txtOutputPath.Name = "txtOutputPath";
			this.toolTip1.SetToolTip(this.txtOutputPath, resources.GetString("txtOutputPath.ToolTip"));
			this.txtOutputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtOutputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// grpOutputStyle
			// 
			this.grpOutputStyle.AccessibleDescription = null;
			this.grpOutputStyle.AccessibleName = null;
			resources.ApplyResources(this.grpOutputStyle, "grpOutputStyle");
			this.grpOutputStyle.BackgroundImage = null;
			this.grpOutputStyle.Controls.Add(this.rbEmbedCUE);
			this.grpOutputStyle.Controls.Add(this.rbGapsLeftOut);
			this.grpOutputStyle.Controls.Add(this.rbGapsPrepended);
			this.grpOutputStyle.Controls.Add(this.rbGapsAppended);
			this.grpOutputStyle.Controls.Add(this.rbSingleFile);
			this.grpOutputStyle.Font = null;
			this.grpOutputStyle.Name = "grpOutputStyle";
			this.grpOutputStyle.TabStop = false;
			this.toolTip1.SetToolTip(this.grpOutputStyle, resources.GetString("grpOutputStyle.ToolTip"));
			// 
			// rbEmbedCUE
			// 
			this.rbEmbedCUE.AccessibleDescription = null;
			this.rbEmbedCUE.AccessibleName = null;
			resources.ApplyResources(this.rbEmbedCUE, "rbEmbedCUE");
			this.rbEmbedCUE.BackgroundImage = null;
			this.rbEmbedCUE.Font = null;
			this.rbEmbedCUE.Name = "rbEmbedCUE";
			this.rbEmbedCUE.TabStop = true;
			this.toolTip1.SetToolTip(this.rbEmbedCUE, resources.GetString("rbEmbedCUE.ToolTip"));
			this.rbEmbedCUE.UseVisualStyleBackColor = true;
			this.rbEmbedCUE.CheckedChanged += new System.EventHandler(this.rbEmbedCUE_CheckedChanged);
			// 
			// rbGapsLeftOut
			// 
			this.rbGapsLeftOut.AccessibleDescription = null;
			this.rbGapsLeftOut.AccessibleName = null;
			resources.ApplyResources(this.rbGapsLeftOut, "rbGapsLeftOut");
			this.rbGapsLeftOut.BackgroundImage = null;
			this.rbGapsLeftOut.Font = null;
			this.rbGapsLeftOut.Name = "rbGapsLeftOut";
			this.toolTip1.SetToolTip(this.rbGapsLeftOut, resources.GetString("rbGapsLeftOut.ToolTip"));
			this.rbGapsLeftOut.UseVisualStyleBackColor = true;
			// 
			// rbGapsPrepended
			// 
			this.rbGapsPrepended.AccessibleDescription = null;
			this.rbGapsPrepended.AccessibleName = null;
			resources.ApplyResources(this.rbGapsPrepended, "rbGapsPrepended");
			this.rbGapsPrepended.BackgroundImage = null;
			this.rbGapsPrepended.Font = null;
			this.rbGapsPrepended.Name = "rbGapsPrepended";
			this.toolTip1.SetToolTip(this.rbGapsPrepended, resources.GetString("rbGapsPrepended.ToolTip"));
			this.rbGapsPrepended.UseVisualStyleBackColor = true;
			// 
			// rbGapsAppended
			// 
			this.rbGapsAppended.AccessibleDescription = null;
			this.rbGapsAppended.AccessibleName = null;
			resources.ApplyResources(this.rbGapsAppended, "rbGapsAppended");
			this.rbGapsAppended.BackgroundImage = null;
			this.rbGapsAppended.Font = null;
			this.rbGapsAppended.Name = "rbGapsAppended";
			this.toolTip1.SetToolTip(this.rbGapsAppended, resources.GetString("rbGapsAppended.ToolTip"));
			this.rbGapsAppended.UseVisualStyleBackColor = true;
			// 
			// rbSingleFile
			// 
			this.rbSingleFile.AccessibleDescription = null;
			this.rbSingleFile.AccessibleName = null;
			resources.ApplyResources(this.rbSingleFile, "rbSingleFile");
			this.rbSingleFile.BackgroundImage = null;
			this.rbSingleFile.Checked = true;
			this.rbSingleFile.Font = null;
			this.rbSingleFile.Name = "rbSingleFile";
			this.rbSingleFile.TabStop = true;
			this.toolTip1.SetToolTip(this.rbSingleFile, resources.GetString("rbSingleFile.ToolTip"));
			this.rbSingleFile.UseVisualStyleBackColor = true;
			// 
			// btnAbout
			// 
			this.btnAbout.AccessibleDescription = null;
			this.btnAbout.AccessibleName = null;
			resources.ApplyResources(this.btnAbout, "btnAbout");
			this.btnAbout.BackgroundImage = null;
			this.btnAbout.Font = null;
			this.btnAbout.Name = "btnAbout";
			this.toolTip1.SetToolTip(this.btnAbout, resources.GetString("btnAbout.ToolTip"));
			this.btnAbout.UseVisualStyleBackColor = true;
			this.btnAbout.Click += new System.EventHandler(this.btnAbout_Click);
			// 
			// grpOutputPathGeneration
			// 
			this.grpOutputPathGeneration.AccessibleDescription = null;
			this.grpOutputPathGeneration.AccessibleName = null;
			resources.ApplyResources(this.grpOutputPathGeneration, "grpOutputPathGeneration");
			this.grpOutputPathGeneration.BackgroundImage = null;
			this.grpOutputPathGeneration.Controls.Add(this.btnBrowseOutput);
			this.grpOutputPathGeneration.Controls.Add(this.txtOutputPath);
			this.grpOutputPathGeneration.Controls.Add(this.txtCustomFormat);
			this.grpOutputPathGeneration.Controls.Add(this.rbCustomFormat);
			this.grpOutputPathGeneration.Controls.Add(this.txtCreateSubdirectory);
			this.grpOutputPathGeneration.Controls.Add(this.rbDontGenerate);
			this.grpOutputPathGeneration.Controls.Add(this.rbCreateSubdirectory);
			this.grpOutputPathGeneration.Controls.Add(this.rbAppendFilename);
			this.grpOutputPathGeneration.Controls.Add(this.txtAppendFilename);
			this.grpOutputPathGeneration.Font = null;
			this.grpOutputPathGeneration.Name = "grpOutputPathGeneration";
			this.grpOutputPathGeneration.TabStop = false;
			this.toolTip1.SetToolTip(this.grpOutputPathGeneration, resources.GetString("grpOutputPathGeneration.ToolTip"));
			// 
			// txtCustomFormat
			// 
			this.txtCustomFormat.AccessibleDescription = null;
			this.txtCustomFormat.AccessibleName = null;
			resources.ApplyResources(this.txtCustomFormat, "txtCustomFormat");
			this.txtCustomFormat.BackgroundImage = null;
			this.txtCustomFormat.Font = null;
			this.txtCustomFormat.Name = "txtCustomFormat";
			this.toolTip1.SetToolTip(this.txtCustomFormat, resources.GetString("txtCustomFormat.ToolTip"));
			this.txtCustomFormat.TextChanged += new System.EventHandler(this.txtCustomFormat_TextChanged);
			// 
			// rbCustomFormat
			// 
			this.rbCustomFormat.AccessibleDescription = null;
			this.rbCustomFormat.AccessibleName = null;
			resources.ApplyResources(this.rbCustomFormat, "rbCustomFormat");
			this.rbCustomFormat.BackgroundImage = null;
			this.rbCustomFormat.Font = null;
			this.rbCustomFormat.Name = "rbCustomFormat";
			this.rbCustomFormat.TabStop = true;
			this.toolTip1.SetToolTip(this.rbCustomFormat, resources.GetString("rbCustomFormat.ToolTip"));
			this.rbCustomFormat.UseVisualStyleBackColor = true;
			this.rbCustomFormat.CheckedChanged += new System.EventHandler(this.rbCustomFormat_CheckedChanged);
			// 
			// txtCreateSubdirectory
			// 
			this.txtCreateSubdirectory.AccessibleDescription = null;
			this.txtCreateSubdirectory.AccessibleName = null;
			resources.ApplyResources(this.txtCreateSubdirectory, "txtCreateSubdirectory");
			this.txtCreateSubdirectory.BackgroundImage = null;
			this.txtCreateSubdirectory.Font = null;
			this.txtCreateSubdirectory.Name = "txtCreateSubdirectory";
			this.toolTip1.SetToolTip(this.txtCreateSubdirectory, resources.GetString("txtCreateSubdirectory.ToolTip"));
			this.txtCreateSubdirectory.TextChanged += new System.EventHandler(this.txtCreateSubdirectory_TextChanged);
			// 
			// rbDontGenerate
			// 
			this.rbDontGenerate.AccessibleDescription = null;
			this.rbDontGenerate.AccessibleName = null;
			resources.ApplyResources(this.rbDontGenerate, "rbDontGenerate");
			this.rbDontGenerate.BackgroundImage = null;
			this.rbDontGenerate.Font = null;
			this.rbDontGenerate.Name = "rbDontGenerate";
			this.toolTip1.SetToolTip(this.rbDontGenerate, resources.GetString("rbDontGenerate.ToolTip"));
			this.rbDontGenerate.UseVisualStyleBackColor = true;
			// 
			// rbCreateSubdirectory
			// 
			this.rbCreateSubdirectory.AccessibleDescription = null;
			this.rbCreateSubdirectory.AccessibleName = null;
			resources.ApplyResources(this.rbCreateSubdirectory, "rbCreateSubdirectory");
			this.rbCreateSubdirectory.BackgroundImage = null;
			this.rbCreateSubdirectory.Checked = true;
			this.rbCreateSubdirectory.Font = null;
			this.rbCreateSubdirectory.Name = "rbCreateSubdirectory";
			this.rbCreateSubdirectory.TabStop = true;
			this.toolTip1.SetToolTip(this.rbCreateSubdirectory, resources.GetString("rbCreateSubdirectory.ToolTip"));
			this.rbCreateSubdirectory.UseVisualStyleBackColor = true;
			this.rbCreateSubdirectory.CheckedChanged += new System.EventHandler(this.rbCreateSubdirectory_CheckedChanged);
			// 
			// rbAppendFilename
			// 
			this.rbAppendFilename.AccessibleDescription = null;
			this.rbAppendFilename.AccessibleName = null;
			resources.ApplyResources(this.rbAppendFilename, "rbAppendFilename");
			this.rbAppendFilename.BackgroundImage = null;
			this.rbAppendFilename.Font = null;
			this.rbAppendFilename.Name = "rbAppendFilename";
			this.toolTip1.SetToolTip(this.rbAppendFilename, resources.GetString("rbAppendFilename.ToolTip"));
			this.rbAppendFilename.UseVisualStyleBackColor = true;
			this.rbAppendFilename.CheckedChanged += new System.EventHandler(this.rbAppendFilename_CheckedChanged);
			// 
			// txtAppendFilename
			// 
			this.txtAppendFilename.AccessibleDescription = null;
			this.txtAppendFilename.AccessibleName = null;
			resources.ApplyResources(this.txtAppendFilename, "txtAppendFilename");
			this.txtAppendFilename.BackgroundImage = null;
			this.txtAppendFilename.Font = null;
			this.txtAppendFilename.Name = "txtAppendFilename";
			this.toolTip1.SetToolTip(this.txtAppendFilename, resources.GetString("txtAppendFilename.ToolTip"));
			this.txtAppendFilename.TextChanged += new System.EventHandler(this.txtAppendFilename_TextChanged);
			// 
			// grpAudioOutput
			// 
			this.grpAudioOutput.AccessibleDescription = null;
			this.grpAudioOutput.AccessibleName = null;
			resources.ApplyResources(this.grpAudioOutput, "grpAudioOutput");
			this.grpAudioOutput.BackgroundImage = null;
			this.grpAudioOutput.Controls.Add(this.btnCodec);
			this.grpAudioOutput.Controls.Add(this.rbUDC1);
			this.grpAudioOutput.Controls.Add(this.rbTTA);
			this.grpAudioOutput.Controls.Add(this.chkLossyWAV);
			this.grpAudioOutput.Controls.Add(this.rbAPE);
			this.grpAudioOutput.Controls.Add(this.rbNoAudio);
			this.grpAudioOutput.Controls.Add(this.rbWavPack);
			this.grpAudioOutput.Controls.Add(this.rbWAV);
			this.grpAudioOutput.Controls.Add(this.rbFLAC);
			this.grpAudioOutput.Font = null;
			this.grpAudioOutput.Name = "grpAudioOutput";
			this.grpAudioOutput.TabStop = false;
			this.toolTip1.SetToolTip(this.grpAudioOutput, resources.GetString("grpAudioOutput.ToolTip"));
			// 
			// btnCodec
			// 
			this.btnCodec.AccessibleDescription = null;
			this.btnCodec.AccessibleName = null;
			resources.ApplyResources(this.btnCodec, "btnCodec");
			this.btnCodec.BackgroundImage = null;
			this.btnCodec.Font = null;
			this.btnCodec.Name = "btnCodec";
			this.toolTip1.SetToolTip(this.btnCodec, resources.GetString("btnCodec.ToolTip"));
			this.btnCodec.UseVisualStyleBackColor = true;
			this.btnCodec.Click += new System.EventHandler(this.btnCodec_Click);
			// 
			// rbUDC1
			// 
			this.rbUDC1.AccessibleDescription = null;
			this.rbUDC1.AccessibleName = null;
			resources.ApplyResources(this.rbUDC1, "rbUDC1");
			this.rbUDC1.BackgroundImage = null;
			this.rbUDC1.Font = null;
			this.rbUDC1.Name = "rbUDC1";
			this.rbUDC1.TabStop = true;
			this.toolTip1.SetToolTip(this.rbUDC1, resources.GetString("rbUDC1.ToolTip"));
			this.rbUDC1.UseVisualStyleBackColor = true;
			this.rbUDC1.CheckedChanged += new System.EventHandler(this.rbUDC1_CheckedChanged);
			// 
			// rbTTA
			// 
			this.rbTTA.AccessibleDescription = null;
			this.rbTTA.AccessibleName = null;
			resources.ApplyResources(this.rbTTA, "rbTTA");
			this.rbTTA.BackgroundImage = null;
			this.rbTTA.Font = null;
			this.rbTTA.Name = "rbTTA";
			this.rbTTA.TabStop = true;
			this.toolTip1.SetToolTip(this.rbTTA, resources.GetString("rbTTA.ToolTip"));
			this.rbTTA.UseVisualStyleBackColor = true;
			this.rbTTA.CheckedChanged += new System.EventHandler(this.rbTTA_CheckedChanged);
			// 
			// chkLossyWAV
			// 
			this.chkLossyWAV.AccessibleDescription = null;
			this.chkLossyWAV.AccessibleName = null;
			resources.ApplyResources(this.chkLossyWAV, "chkLossyWAV");
			this.chkLossyWAV.BackgroundImage = null;
			this.chkLossyWAV.Font = null;
			this.chkLossyWAV.Name = "chkLossyWAV";
			this.toolTip1.SetToolTip(this.chkLossyWAV, resources.GetString("chkLossyWAV.ToolTip"));
			this.chkLossyWAV.UseVisualStyleBackColor = true;
			this.chkLossyWAV.CheckedChanged += new System.EventHandler(this.chkLossyWAV_CheckedChanged);
			// 
			// rbAPE
			// 
			this.rbAPE.AccessibleDescription = null;
			this.rbAPE.AccessibleName = null;
			resources.ApplyResources(this.rbAPE, "rbAPE");
			this.rbAPE.BackgroundImage = null;
			this.rbAPE.Font = null;
			this.rbAPE.Name = "rbAPE";
			this.rbAPE.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPE, resources.GetString("rbAPE.ToolTip"));
			this.rbAPE.UseVisualStyleBackColor = true;
			this.rbAPE.CheckedChanged += new System.EventHandler(this.rbAPE_CheckedChanged);
			// 
			// rbNoAudio
			// 
			this.rbNoAudio.AccessibleDescription = null;
			this.rbNoAudio.AccessibleName = null;
			resources.ApplyResources(this.rbNoAudio, "rbNoAudio");
			this.rbNoAudio.BackgroundImage = null;
			this.rbNoAudio.Font = null;
			this.rbNoAudio.Name = "rbNoAudio";
			this.toolTip1.SetToolTip(this.rbNoAudio, resources.GetString("rbNoAudio.ToolTip"));
			this.rbNoAudio.UseVisualStyleBackColor = true;
			this.rbNoAudio.CheckedChanged += new System.EventHandler(this.rbNoAudio_CheckedChanged);
			// 
			// rbWavPack
			// 
			this.rbWavPack.AccessibleDescription = null;
			this.rbWavPack.AccessibleName = null;
			resources.ApplyResources(this.rbWavPack, "rbWavPack");
			this.rbWavPack.BackgroundImage = null;
			this.rbWavPack.Font = null;
			this.rbWavPack.Name = "rbWavPack";
			this.toolTip1.SetToolTip(this.rbWavPack, resources.GetString("rbWavPack.ToolTip"));
			this.rbWavPack.UseVisualStyleBackColor = true;
			this.rbWavPack.CheckedChanged += new System.EventHandler(this.rbWavPack_CheckedChanged);
			// 
			// rbWAV
			// 
			this.rbWAV.AccessibleDescription = null;
			this.rbWAV.AccessibleName = null;
			resources.ApplyResources(this.rbWAV, "rbWAV");
			this.rbWAV.BackgroundImage = null;
			this.rbWAV.Checked = true;
			this.rbWAV.Font = null;
			this.rbWAV.Name = "rbWAV";
			this.rbWAV.TabStop = true;
			this.toolTip1.SetToolTip(this.rbWAV, resources.GetString("rbWAV.ToolTip"));
			this.rbWAV.UseVisualStyleBackColor = true;
			this.rbWAV.CheckedChanged += new System.EventHandler(this.rbWAV_CheckedChanged);
			// 
			// rbFLAC
			// 
			this.rbFLAC.AccessibleDescription = null;
			this.rbFLAC.AccessibleName = null;
			resources.ApplyResources(this.rbFLAC, "rbFLAC");
			this.rbFLAC.BackgroundImage = null;
			this.rbFLAC.Font = null;
			this.rbFLAC.Name = "rbFLAC";
			this.toolTip1.SetToolTip(this.rbFLAC, resources.GetString("rbFLAC.ToolTip"));
			this.rbFLAC.UseVisualStyleBackColor = true;
			this.rbFLAC.CheckedChanged += new System.EventHandler(this.rbFLAC_CheckedChanged);
			// 
			// btnSettings
			// 
			this.btnSettings.AccessibleDescription = null;
			this.btnSettings.AccessibleName = null;
			resources.ApplyResources(this.btnSettings, "btnSettings");
			this.btnSettings.BackgroundImage = null;
			this.btnSettings.Font = null;
			this.btnSettings.Name = "btnSettings";
			this.toolTip1.SetToolTip(this.btnSettings, resources.GetString("btnSettings.ToolTip"));
			this.btnSettings.UseVisualStyleBackColor = true;
			this.btnSettings.Click += new System.EventHandler(this.btnSettings_Click);
			// 
			// grpAction
			// 
			this.grpAction.AccessibleDescription = null;
			this.grpAction.AccessibleName = null;
			resources.ApplyResources(this.grpAction, "grpAction");
			this.grpAction.BackgroundImage = null;
			this.grpAction.Controls.Add(this.rbActionCorrectFilenames);
			this.grpAction.Controls.Add(this.chkRecursive);
			this.grpAction.Controls.Add(this.rbActionCreateCUESheet);
			this.grpAction.Controls.Add(this.chkMulti);
			this.grpAction.Controls.Add(this.rbActionVerifyAndCRCs);
			this.grpAction.Controls.Add(this.rbActionVerifyAndEncode);
			this.grpAction.Controls.Add(this.rbActionVerifyThenEncode);
			this.grpAction.Controls.Add(this.rbActionVerify);
			this.grpAction.Controls.Add(this.rbActionEncode);
			this.grpAction.Font = null;
			this.grpAction.Name = "grpAction";
			this.grpAction.TabStop = false;
			this.toolTip1.SetToolTip(this.grpAction, resources.GetString("grpAction.ToolTip"));
			// 
			// rbActionCorrectFilenames
			// 
			this.rbActionCorrectFilenames.AccessibleDescription = null;
			this.rbActionCorrectFilenames.AccessibleName = null;
			resources.ApplyResources(this.rbActionCorrectFilenames, "rbActionCorrectFilenames");
			this.rbActionCorrectFilenames.BackgroundImage = null;
			this.rbActionCorrectFilenames.Font = null;
			this.rbActionCorrectFilenames.Name = "rbActionCorrectFilenames";
			this.rbActionCorrectFilenames.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionCorrectFilenames, resources.GetString("rbActionCorrectFilenames.ToolTip"));
			this.rbActionCorrectFilenames.UseVisualStyleBackColor = true;
			this.rbActionCorrectFilenames.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// chkRecursive
			// 
			this.chkRecursive.AccessibleDescription = null;
			this.chkRecursive.AccessibleName = null;
			resources.ApplyResources(this.chkRecursive, "chkRecursive");
			this.chkRecursive.BackgroundImage = null;
			this.chkRecursive.Font = null;
			this.chkRecursive.Name = "chkRecursive";
			this.toolTip1.SetToolTip(this.chkRecursive, resources.GetString("chkRecursive.ToolTip"));
			this.chkRecursive.UseVisualStyleBackColor = true;
			this.chkRecursive.CheckedChanged += new System.EventHandler(this.chkRecursive_CheckedChanged);
			// 
			// rbActionCreateCUESheet
			// 
			this.rbActionCreateCUESheet.AccessibleDescription = null;
			this.rbActionCreateCUESheet.AccessibleName = null;
			resources.ApplyResources(this.rbActionCreateCUESheet, "rbActionCreateCUESheet");
			this.rbActionCreateCUESheet.BackgroundImage = null;
			this.rbActionCreateCUESheet.Font = null;
			this.rbActionCreateCUESheet.Name = "rbActionCreateCUESheet";
			this.rbActionCreateCUESheet.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionCreateCUESheet, resources.GetString("rbActionCreateCUESheet.ToolTip"));
			this.rbActionCreateCUESheet.UseVisualStyleBackColor = true;
			this.rbActionCreateCUESheet.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// chkMulti
			// 
			this.chkMulti.AccessibleDescription = null;
			this.chkMulti.AccessibleName = null;
			resources.ApplyResources(this.chkMulti, "chkMulti");
			this.chkMulti.BackgroundImage = null;
			this.chkMulti.Font = null;
			this.chkMulti.Name = "chkMulti";
			this.toolTip1.SetToolTip(this.chkMulti, resources.GetString("chkMulti.ToolTip"));
			this.chkMulti.UseVisualStyleBackColor = true;
			this.chkMulti.CheckedChanged += new System.EventHandler(this.chkMulti_CheckedChanged);
			// 
			// rbActionVerifyAndCRCs
			// 
			this.rbActionVerifyAndCRCs.AccessibleDescription = null;
			this.rbActionVerifyAndCRCs.AccessibleName = null;
			resources.ApplyResources(this.rbActionVerifyAndCRCs, "rbActionVerifyAndCRCs");
			this.rbActionVerifyAndCRCs.BackgroundImage = null;
			this.rbActionVerifyAndCRCs.Font = null;
			this.rbActionVerifyAndCRCs.Name = "rbActionVerifyAndCRCs";
			this.toolTip1.SetToolTip(this.rbActionVerifyAndCRCs, resources.GetString("rbActionVerifyAndCRCs.ToolTip"));
			this.rbActionVerifyAndCRCs.UseVisualStyleBackColor = true;
			this.rbActionVerifyAndCRCs.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// rbActionVerifyAndEncode
			// 
			this.rbActionVerifyAndEncode.AccessibleDescription = null;
			this.rbActionVerifyAndEncode.AccessibleName = null;
			resources.ApplyResources(this.rbActionVerifyAndEncode, "rbActionVerifyAndEncode");
			this.rbActionVerifyAndEncode.BackgroundImage = null;
			this.rbActionVerifyAndEncode.Font = null;
			this.rbActionVerifyAndEncode.Name = "rbActionVerifyAndEncode";
			this.rbActionVerifyAndEncode.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionVerifyAndEncode, resources.GetString("rbActionVerifyAndEncode.ToolTip"));
			this.rbActionVerifyAndEncode.UseVisualStyleBackColor = true;
			this.rbActionVerifyAndEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// rbActionVerifyThenEncode
			// 
			this.rbActionVerifyThenEncode.AccessibleDescription = null;
			this.rbActionVerifyThenEncode.AccessibleName = null;
			resources.ApplyResources(this.rbActionVerifyThenEncode, "rbActionVerifyThenEncode");
			this.rbActionVerifyThenEncode.BackgroundImage = null;
			this.rbActionVerifyThenEncode.Font = null;
			this.rbActionVerifyThenEncode.Name = "rbActionVerifyThenEncode";
			this.toolTip1.SetToolTip(this.rbActionVerifyThenEncode, resources.GetString("rbActionVerifyThenEncode.ToolTip"));
			this.rbActionVerifyThenEncode.UseVisualStyleBackColor = true;
			this.rbActionVerifyThenEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// rbActionVerify
			// 
			this.rbActionVerify.AccessibleDescription = null;
			this.rbActionVerify.AccessibleName = null;
			resources.ApplyResources(this.rbActionVerify, "rbActionVerify");
			this.rbActionVerify.BackgroundImage = null;
			this.rbActionVerify.Font = null;
			this.rbActionVerify.Name = "rbActionVerify";
			this.toolTip1.SetToolTip(this.rbActionVerify, resources.GetString("rbActionVerify.ToolTip"));
			this.rbActionVerify.UseVisualStyleBackColor = true;
			this.rbActionVerify.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// rbActionEncode
			// 
			this.rbActionEncode.AccessibleDescription = null;
			this.rbActionEncode.AccessibleName = null;
			resources.ApplyResources(this.rbActionEncode, "rbActionEncode");
			this.rbActionEncode.BackgroundImage = null;
			this.rbActionEncode.Checked = true;
			this.rbActionEncode.Font = null;
			this.rbActionEncode.Name = "rbActionEncode";
			this.rbActionEncode.TabStop = true;
			this.toolTip1.SetToolTip(this.rbActionEncode, resources.GetString("rbActionEncode.ToolTip"));
			this.rbActionEncode.UseVisualStyleBackColor = true;
			this.rbActionEncode.CheckedChanged += new System.EventHandler(this.rbAction_CheckedChanged);
			// 
			// txtPreGapLength
			// 
			this.txtPreGapLength.AccessibleDescription = null;
			this.txtPreGapLength.AccessibleName = null;
			resources.ApplyResources(this.txtPreGapLength, "txtPreGapLength");
			this.txtPreGapLength.BackgroundImage = null;
			this.txtPreGapLength.Culture = new System.Globalization.CultureInfo("");
			this.txtPreGapLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.txtPreGapLength.Font = null;
			this.txtPreGapLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
			this.txtPreGapLength.Name = "txtPreGapLength";
			this.txtPreGapLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.toolTip1.SetToolTip(this.txtPreGapLength, resources.GetString("txtPreGapLength.ToolTip"));
			// 
			// label2
			// 
			this.label2.AccessibleDescription = null;
			this.label2.AccessibleName = null;
			resources.ApplyResources(this.label2, "label2");
			this.label2.Font = null;
			this.label2.Name = "label2";
			this.toolTip1.SetToolTip(this.label2, resources.GetString("label2.ToolTip"));
			// 
			// label1
			// 
			this.label1.AccessibleDescription = null;
			this.label1.AccessibleName = null;
			resources.ApplyResources(this.label1, "label1");
			this.label1.Font = null;
			this.label1.Name = "label1";
			this.toolTip1.SetToolTip(this.label1, resources.GetString("label1.ToolTip"));
			// 
			// txtDataTrackLength
			// 
			this.txtDataTrackLength.AccessibleDescription = null;
			this.txtDataTrackLength.AccessibleName = null;
			resources.ApplyResources(this.txtDataTrackLength, "txtDataTrackLength");
			this.txtDataTrackLength.BackgroundImage = null;
			this.txtDataTrackLength.Culture = new System.Globalization.CultureInfo("");
			this.txtDataTrackLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.txtDataTrackLength.Font = null;
			this.txtDataTrackLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
			this.txtDataTrackLength.Name = "txtDataTrackLength";
			this.txtDataTrackLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.toolTip1.SetToolTip(this.txtDataTrackLength, resources.GetString("txtDataTrackLength.ToolTip"));
			// 
			// statusStrip1
			// 
			this.statusStrip1.AccessibleDescription = null;
			this.statusStrip1.AccessibleName = null;
			resources.ApplyResources(this.statusStrip1, "statusStrip1");
			this.statusStrip1.BackgroundImage = null;
			this.statusStrip1.Font = null;
			this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLabelProcessed,
            this.toolStripStatusLabel2,
            this.toolStripStatusLabelWV,
            this.toolStripStatusLabelFLAC,
            this.toolStripStatusLabelAR,
            this.toolStripProgressBar1,
            this.toolStripProgressBar2});
			this.statusStrip1.Name = "statusStrip1";
			this.statusStrip1.ShowItemToolTips = true;
			this.statusStrip1.SizingGrip = false;
			this.toolTip1.SetToolTip(this.statusStrip1, resources.GetString("statusStrip1.ToolTip"));
			// 
			// toolStripStatusLabel1
			// 
			this.toolStripStatusLabel1.AccessibleDescription = null;
			this.toolStripStatusLabel1.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
			this.toolStripStatusLabel1.BackgroundImage = null;
			this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
			this.toolStripStatusLabel1.Spring = true;
			// 
			// toolStripStatusLabelProcessed
			// 
			this.toolStripStatusLabelProcessed.AccessibleDescription = null;
			this.toolStripStatusLabelProcessed.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabelProcessed, "toolStripStatusLabelProcessed");
			this.toolStripStatusLabelProcessed.BackgroundImage = null;
			this.toolStripStatusLabelProcessed.Name = "toolStripStatusLabelProcessed";
			// 
			// toolStripStatusLabel2
			// 
			this.toolStripStatusLabel2.AccessibleDescription = null;
			this.toolStripStatusLabel2.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabel2, "toolStripStatusLabel2");
			this.toolStripStatusLabel2.BackgroundImage = null;
			this.toolStripStatusLabel2.Image = global::JDP.Properties.Resources.wav;
			this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
			this.toolStripStatusLabel2.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
			// 
			// toolStripStatusLabelWV
			// 
			this.toolStripStatusLabelWV.AccessibleDescription = null;
			this.toolStripStatusLabelWV.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabelWV, "toolStripStatusLabelWV");
			this.toolStripStatusLabelWV.BackgroundImage = null;
			this.toolStripStatusLabelWV.Image = global::JDP.Properties.Resources.wv;
			this.toolStripStatusLabelWV.Name = "toolStripStatusLabelWV";
			this.toolStripStatusLabelWV.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
			// 
			// toolStripStatusLabelFLAC
			// 
			this.toolStripStatusLabelFLAC.AccessibleDescription = null;
			this.toolStripStatusLabelFLAC.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabelFLAC, "toolStripStatusLabelFLAC");
			this.toolStripStatusLabelFLAC.BackgroundImage = null;
			this.toolStripStatusLabelFLAC.Image = global::JDP.Properties.Resources.flac;
			this.toolStripStatusLabelFLAC.Name = "toolStripStatusLabelFLAC";
			this.toolStripStatusLabelFLAC.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
			// 
			// toolStripStatusLabelAR
			// 
			this.toolStripStatusLabelAR.AccessibleDescription = null;
			this.toolStripStatusLabelAR.AccessibleName = null;
			resources.ApplyResources(this.toolStripStatusLabelAR, "toolStripStatusLabelAR");
			this.toolStripStatusLabelAR.BackgroundImage = null;
			this.toolStripStatusLabelAR.Name = "toolStripStatusLabelAR";
			this.toolStripStatusLabelAR.Padding = new System.Windows.Forms.Padding(0, 0, 5, 0);
			// 
			// toolStripProgressBar1
			// 
			this.toolStripProgressBar1.AccessibleDescription = null;
			this.toolStripProgressBar1.AccessibleName = null;
			resources.ApplyResources(this.toolStripProgressBar1, "toolStripProgressBar1");
			this.toolStripProgressBar1.AutoToolTip = true;
			this.toolStripProgressBar1.Name = "toolStripProgressBar1";
			this.toolStripProgressBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
			// 
			// toolStripProgressBar2
			// 
			this.toolStripProgressBar2.AccessibleDescription = null;
			this.toolStripProgressBar2.AccessibleName = null;
			resources.ApplyResources(this.toolStripProgressBar2, "toolStripProgressBar2");
			this.toolStripProgressBar2.AutoToolTip = true;
			this.toolStripProgressBar2.Name = "toolStripProgressBar2";
			this.toolStripProgressBar2.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
			// 
			// toolTip1
			// 
			this.toolTip1.AutoPopDelay = 15000;
			this.toolTip1.InitialDelay = 500;
			this.toolTip1.ReshowDelay = 100;
			// 
			// btnStop
			// 
			this.btnStop.AccessibleDescription = null;
			this.btnStop.AccessibleName = null;
			resources.ApplyResources(this.btnStop, "btnStop");
			this.btnStop.BackgroundImage = null;
			this.btnStop.Font = null;
			this.btnStop.Name = "btnStop";
			this.toolTip1.SetToolTip(this.btnStop, resources.GetString("btnStop.ToolTip"));
			this.btnStop.UseVisualStyleBackColor = true;
			this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
			// 
			// btnPause
			// 
			this.btnPause.AccessibleDescription = null;
			this.btnPause.AccessibleName = null;
			resources.ApplyResources(this.btnPause, "btnPause");
			this.btnPause.BackgroundImage = null;
			this.btnPause.Font = null;
			this.btnPause.Name = "btnPause";
			this.toolTip1.SetToolTip(this.btnPause, resources.GetString("btnPause.ToolTip"));
			this.btnPause.UseVisualStyleBackColor = true;
			this.btnPause.Click += new System.EventHandler(this.btnPause_Click);
			// 
			// btnResume
			// 
			this.btnResume.AccessibleDescription = null;
			this.btnResume.AccessibleName = null;
			resources.ApplyResources(this.btnResume, "btnResume");
			this.btnResume.BackgroundImage = null;
			this.btnResume.Font = null;
			this.btnResume.Name = "btnResume";
			this.toolTip1.SetToolTip(this.btnResume, resources.GetString("btnResume.ToolTip"));
			this.btnResume.UseVisualStyleBackColor = true;
			this.btnResume.Click += new System.EventHandler(this.btnPause_Click);
			// 
			// grpFreedb
			// 
			this.grpFreedb.AccessibleDescription = null;
			this.grpFreedb.AccessibleName = null;
			resources.ApplyResources(this.grpFreedb, "grpFreedb");
			this.grpFreedb.BackgroundImage = null;
			this.grpFreedb.Controls.Add(this.rbFreedbAlways);
			this.grpFreedb.Controls.Add(this.rbFreedbIf);
			this.grpFreedb.Controls.Add(this.rbFreedbNever);
			this.grpFreedb.Font = null;
			this.grpFreedb.Name = "grpFreedb";
			this.grpFreedb.TabStop = false;
			this.toolTip1.SetToolTip(this.grpFreedb, resources.GetString("grpFreedb.ToolTip"));
			// 
			// rbFreedbAlways
			// 
			this.rbFreedbAlways.AccessibleDescription = null;
			this.rbFreedbAlways.AccessibleName = null;
			resources.ApplyResources(this.rbFreedbAlways, "rbFreedbAlways");
			this.rbFreedbAlways.BackgroundImage = null;
			this.rbFreedbAlways.Font = null;
			this.rbFreedbAlways.Name = "rbFreedbAlways";
			this.rbFreedbAlways.TabStop = true;
			this.toolTip1.SetToolTip(this.rbFreedbAlways, resources.GetString("rbFreedbAlways.ToolTip"));
			this.rbFreedbAlways.UseVisualStyleBackColor = true;
			// 
			// rbFreedbIf
			// 
			this.rbFreedbIf.AccessibleDescription = null;
			this.rbFreedbIf.AccessibleName = null;
			resources.ApplyResources(this.rbFreedbIf, "rbFreedbIf");
			this.rbFreedbIf.BackgroundImage = null;
			this.rbFreedbIf.Font = null;
			this.rbFreedbIf.Name = "rbFreedbIf";
			this.rbFreedbIf.TabStop = true;
			this.toolTip1.SetToolTip(this.rbFreedbIf, resources.GetString("rbFreedbIf.ToolTip"));
			this.rbFreedbIf.UseVisualStyleBackColor = true;
			// 
			// rbFreedbNever
			// 
			this.rbFreedbNever.AccessibleDescription = null;
			this.rbFreedbNever.AccessibleName = null;
			resources.ApplyResources(this.rbFreedbNever, "rbFreedbNever");
			this.rbFreedbNever.BackgroundImage = null;
			this.rbFreedbNever.Font = null;
			this.rbFreedbNever.Name = "rbFreedbNever";
			this.rbFreedbNever.TabStop = true;
			this.toolTip1.SetToolTip(this.rbFreedbNever, resources.GetString("rbFreedbNever.ToolTip"));
			this.rbFreedbNever.UseVisualStyleBackColor = true;
			// 
			// contextMenuStripUDC
			// 
			this.contextMenuStripUDC.AccessibleDescription = null;
			this.contextMenuStripUDC.AccessibleName = null;
			resources.ApplyResources(this.contextMenuStripUDC, "contextMenuStripUDC");
			this.contextMenuStripUDC.BackgroundImage = null;
			this.contextMenuStripUDC.Font = null;
			this.contextMenuStripUDC.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripMenuItem2,
            this.tAKToolStripMenuItem,
            this.toolStripMenuItem1,
            this.toolStripSeparator1,
            this.toolStripMenuItem3,
            this.mP3ToolStripMenuItem,
            this.oGGToolStripMenuItem});
			this.contextMenuStripUDC.Name = "contextMenuStripUDC";
			this.toolTip1.SetToolTip(this.contextMenuStripUDC, resources.GetString("contextMenuStripUDC.ToolTip"));
			this.contextMenuStripUDC.ItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.contextMenuStripUDC_ItemClicked);
			// 
			// toolStripMenuItem2
			// 
			this.toolStripMenuItem2.AccessibleDescription = null;
			this.toolStripMenuItem2.AccessibleName = null;
			resources.ApplyResources(this.toolStripMenuItem2, "toolStripMenuItem2");
			this.toolStripMenuItem2.BackgroundImage = null;
			this.toolStripMenuItem2.Name = "toolStripMenuItem2";
			this.toolStripMenuItem2.ShortcutKeyDisplayString = null;
			// 
			// tAKToolStripMenuItem
			// 
			this.tAKToolStripMenuItem.AccessibleDescription = null;
			this.tAKToolStripMenuItem.AccessibleName = null;
			resources.ApplyResources(this.tAKToolStripMenuItem, "tAKToolStripMenuItem");
			this.tAKToolStripMenuItem.BackgroundImage = null;
			this.tAKToolStripMenuItem.Name = "tAKToolStripMenuItem";
			this.tAKToolStripMenuItem.ShortcutKeyDisplayString = null;
			// 
			// toolStripMenuItem1
			// 
			this.toolStripMenuItem1.AccessibleDescription = null;
			this.toolStripMenuItem1.AccessibleName = null;
			resources.ApplyResources(this.toolStripMenuItem1, "toolStripMenuItem1");
			this.toolStripMenuItem1.BackgroundImage = null;
			this.toolStripMenuItem1.Name = "toolStripMenuItem1";
			this.toolStripMenuItem1.ShortcutKeyDisplayString = null;
			// 
			// toolStripSeparator1
			// 
			this.toolStripSeparator1.AccessibleDescription = null;
			this.toolStripSeparator1.AccessibleName = null;
			resources.ApplyResources(this.toolStripSeparator1, "toolStripSeparator1");
			this.toolStripSeparator1.Name = "toolStripSeparator1";
			// 
			// toolStripMenuItem3
			// 
			this.toolStripMenuItem3.AccessibleDescription = null;
			this.toolStripMenuItem3.AccessibleName = null;
			resources.ApplyResources(this.toolStripMenuItem3, "toolStripMenuItem3");
			this.toolStripMenuItem3.BackgroundImage = null;
			this.toolStripMenuItem3.Name = "toolStripMenuItem3";
			this.toolStripMenuItem3.ShortcutKeyDisplayString = null;
			// 
			// mP3ToolStripMenuItem
			// 
			this.mP3ToolStripMenuItem.AccessibleDescription = null;
			this.mP3ToolStripMenuItem.AccessibleName = null;
			resources.ApplyResources(this.mP3ToolStripMenuItem, "mP3ToolStripMenuItem");
			this.mP3ToolStripMenuItem.BackgroundImage = null;
			this.mP3ToolStripMenuItem.Name = "mP3ToolStripMenuItem";
			this.mP3ToolStripMenuItem.ShortcutKeyDisplayString = null;
			// 
			// oGGToolStripMenuItem
			// 
			this.oGGToolStripMenuItem.AccessibleDescription = null;
			this.oGGToolStripMenuItem.AccessibleName = null;
			resources.ApplyResources(this.oGGToolStripMenuItem, "oGGToolStripMenuItem");
			this.oGGToolStripMenuItem.BackgroundImage = null;
			this.oGGToolStripMenuItem.Name = "oGGToolStripMenuItem";
			this.oGGToolStripMenuItem.ShortcutKeyDisplayString = null;
			// 
			// txtInputPath
			// 
			this.txtInputPath.AccessibleDescription = null;
			this.txtInputPath.AccessibleName = null;
			this.txtInputPath.AllowDrop = true;
			resources.ApplyResources(this.txtInputPath, "txtInputPath");
			this.txtInputPath.BackColor = System.Drawing.SystemColors.Control;
			this.txtInputPath.BackgroundImage = null;
			this.txtInputPath.Font = null;
			this.txtInputPath.Name = "txtInputPath";
			this.toolTip1.SetToolTip(this.txtInputPath, resources.GetString("txtInputPath.ToolTip"));
			this.txtInputPath.TextChanged += new System.EventHandler(this.txtInputPath_TextChanged);
			this.txtInputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtInputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// grpInput
			// 
			this.grpInput.AccessibleDescription = null;
			this.grpInput.AccessibleName = null;
			resources.ApplyResources(this.grpInput, "grpInput");
			this.grpInput.BackgroundImage = null;
			this.grpInput.Controls.Add(this.textBatchReport);
			this.grpInput.Controls.Add(this.fileSystemTreeView1);
			this.grpInput.Controls.Add(this.txtInputPath);
			this.grpInput.Font = null;
			this.grpInput.Name = "grpInput";
			this.grpInput.TabStop = false;
			this.toolTip1.SetToolTip(this.grpInput, resources.GetString("grpInput.ToolTip"));
			// 
			// textBatchReport
			// 
			this.textBatchReport.AccessibleDescription = null;
			this.textBatchReport.AccessibleName = null;
			resources.ApplyResources(this.textBatchReport, "textBatchReport");
			this.textBatchReport.BackgroundImage = null;
			this.textBatchReport.Font = null;
			this.textBatchReport.Name = "textBatchReport";
			this.textBatchReport.ReadOnly = true;
			this.textBatchReport.TabStop = false;
			this.toolTip1.SetToolTip(this.textBatchReport, resources.GetString("textBatchReport.ToolTip"));
			// 
			// fileSystemTreeView1
			// 
			this.fileSystemTreeView1.AccessibleDescription = null;
			this.fileSystemTreeView1.AccessibleName = null;
			this.fileSystemTreeView1.AllowDrop = true;
			resources.ApplyResources(this.fileSystemTreeView1, "fileSystemTreeView1");
			this.fileSystemTreeView1.BackColor = System.Drawing.SystemColors.Control;
			this.fileSystemTreeView1.BackgroundImage = null;
			this.fileSystemTreeView1.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.fileSystemTreeView1.CheckBoxes = true;
			this.fileSystemTreeView1.Font = null;
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
			this.toolTip1.SetToolTip(this.fileSystemTreeView1, resources.GetString("fileSystemTreeView1.ToolTip"));
			this.fileSystemTreeView1.AfterCheck += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterCheck);
			this.fileSystemTreeView1.NodeAttributes += new CUEControls.FileSystemTreeViewNodeAttributesHandler(this.fileSystemTreeView1_NodeAttributes);
			this.fileSystemTreeView1.DragDrop += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragDrop);
			this.fileSystemTreeView1.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterSelect);
			this.fileSystemTreeView1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.fileSystemTreeView1_MouseDown);
			this.fileSystemTreeView1.DragEnter += new System.Windows.Forms.DragEventHandler(this.fileSystemTreeView1_DragEnter);
			this.fileSystemTreeView1.AfterExpand += new System.Windows.Forms.TreeViewEventHandler(this.fileSystemTreeView1_AfterExpand);
			// 
			// grpExtra
			// 
			this.grpExtra.AccessibleDescription = null;
			this.grpExtra.AccessibleName = null;
			resources.ApplyResources(this.grpExtra, "grpExtra");
			this.grpExtra.BackgroundImage = null;
			this.grpExtra.Controls.Add(this.numericWriteOffset);
			this.grpExtra.Controls.Add(this.txtPreGapLength);
			this.grpExtra.Controls.Add(this.lblWriteOffset);
			this.grpExtra.Controls.Add(this.label2);
			this.grpExtra.Controls.Add(this.txtDataTrackLength);
			this.grpExtra.Controls.Add(this.label1);
			this.grpExtra.Font = null;
			this.grpExtra.Name = "grpExtra";
			this.grpExtra.TabStop = false;
			this.toolTip1.SetToolTip(this.grpExtra, resources.GetString("grpExtra.ToolTip"));
			// 
			// numericWriteOffset
			// 
			this.numericWriteOffset.AccessibleDescription = null;
			this.numericWriteOffset.AccessibleName = null;
			resources.ApplyResources(this.numericWriteOffset, "numericWriteOffset");
			this.numericWriteOffset.Font = null;
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
			// lblWriteOffset
			// 
			this.lblWriteOffset.AccessibleDescription = null;
			this.lblWriteOffset.AccessibleName = null;
			resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
			this.lblWriteOffset.Font = null;
			this.lblWriteOffset.Name = "lblWriteOffset";
			this.toolTip1.SetToolTip(this.lblWriteOffset, resources.GetString("lblWriteOffset.ToolTip"));
			// 
			// contextMenuStripFileTree
			// 
			this.contextMenuStripFileTree.AccessibleDescription = null;
			this.contextMenuStripFileTree.AccessibleName = null;
			resources.ApplyResources(this.contextMenuStripFileTree, "contextMenuStripFileTree");
			this.contextMenuStripFileTree.BackgroundImage = null;
			this.contextMenuStripFileTree.Font = null;
			this.contextMenuStripFileTree.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.SelectedNodeName,
            this.toolStripSeparator2,
            this.setAsMyMusicFolderToolStripMenuItem,
            this.resetToOriginalLocationToolStripMenuItem});
			this.contextMenuStripFileTree.Name = "contextMenuStripFileTree";
			this.toolTip1.SetToolTip(this.contextMenuStripFileTree, resources.GetString("contextMenuStripFileTree.ToolTip"));
			// 
			// SelectedNodeName
			// 
			this.SelectedNodeName.AccessibleDescription = null;
			this.SelectedNodeName.AccessibleName = null;
			resources.ApplyResources(this.SelectedNodeName, "SelectedNodeName");
			this.SelectedNodeName.BackgroundImage = null;
			this.SelectedNodeName.Name = "SelectedNodeName";
			this.SelectedNodeName.ShortcutKeyDisplayString = null;
			// 
			// toolStripSeparator2
			// 
			this.toolStripSeparator2.AccessibleDescription = null;
			this.toolStripSeparator2.AccessibleName = null;
			resources.ApplyResources(this.toolStripSeparator2, "toolStripSeparator2");
			this.toolStripSeparator2.Name = "toolStripSeparator2";
			// 
			// setAsMyMusicFolderToolStripMenuItem
			// 
			this.setAsMyMusicFolderToolStripMenuItem.AccessibleDescription = null;
			this.setAsMyMusicFolderToolStripMenuItem.AccessibleName = null;
			resources.ApplyResources(this.setAsMyMusicFolderToolStripMenuItem, "setAsMyMusicFolderToolStripMenuItem");
			this.setAsMyMusicFolderToolStripMenuItem.BackgroundImage = null;
			this.setAsMyMusicFolderToolStripMenuItem.Name = "setAsMyMusicFolderToolStripMenuItem";
			this.setAsMyMusicFolderToolStripMenuItem.ShortcutKeyDisplayString = null;
			this.setAsMyMusicFolderToolStripMenuItem.Click += new System.EventHandler(this.setAsMyMusicFolderToolStripMenuItem_Click);
			// 
			// resetToOriginalLocationToolStripMenuItem
			// 
			this.resetToOriginalLocationToolStripMenuItem.AccessibleDescription = null;
			this.resetToOriginalLocationToolStripMenuItem.AccessibleName = null;
			resources.ApplyResources(this.resetToOriginalLocationToolStripMenuItem, "resetToOriginalLocationToolStripMenuItem");
			this.resetToOriginalLocationToolStripMenuItem.BackgroundImage = null;
			this.resetToOriginalLocationToolStripMenuItem.Name = "resetToOriginalLocationToolStripMenuItem";
			this.resetToOriginalLocationToolStripMenuItem.ShortcutKeyDisplayString = null;
			this.resetToOriginalLocationToolStripMenuItem.Click += new System.EventHandler(this.resetToOriginalLocationToolStripMenuItem_Click);
			// 
			// panel1
			// 
			this.panel1.AccessibleDescription = null;
			this.panel1.AccessibleName = null;
			resources.ApplyResources(this.panel1, "panel1");
			this.panel1.BackgroundImage = null;
			this.panel1.Controls.Add(this.grpOutputPathGeneration);
			this.panel1.Controls.Add(this.btnStop);
			this.panel1.Controls.Add(this.btnConvert);
			this.panel1.Controls.Add(this.btnSettings);
			this.panel1.Controls.Add(this.grpExtra);
			this.panel1.Controls.Add(this.btnAbout);
			this.panel1.Controls.Add(this.grpOutputStyle);
			this.panel1.Controls.Add(this.grpFreedb);
			this.panel1.Controls.Add(this.grpAudioOutput);
			this.panel1.Controls.Add(this.grpAction);
			this.panel1.Controls.Add(this.btnPause);
			this.panel1.Controls.Add(this.btnResume);
			this.panel1.Font = null;
			this.panel1.Name = "panel1";
			this.toolTip1.SetToolTip(this.panel1, resources.GetString("panel1.ToolTip"));
			// 
			// frmCUETools
			// 
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.Controls.Add(this.panel1);
			this.Controls.Add(this.grpInput);
			this.Controls.Add(this.statusStrip1);
			this.MaximizeBox = false;
			this.Name = "frmCUETools";
			this.toolTip1.SetToolTip(this, resources.GetString("$this.ToolTip"));
			this.Load += new System.EventHandler(this.frmCUETools_Load);
			this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUETools_FormClosed);
			this.grpOutputStyle.ResumeLayout(false);
			this.grpOutputStyle.PerformLayout();
			this.grpOutputPathGeneration.ResumeLayout(false);
			this.grpOutputPathGeneration.PerformLayout();
			this.grpAudioOutput.ResumeLayout(false);
			this.grpAudioOutput.PerformLayout();
			this.grpAction.ResumeLayout(false);
			this.grpAction.PerformLayout();
			this.statusStrip1.ResumeLayout(false);
			this.statusStrip1.PerformLayout();
			this.grpFreedb.ResumeLayout(false);
			this.grpFreedb.PerformLayout();
			this.contextMenuStripUDC.ResumeLayout(false);
			this.grpInput.ResumeLayout(false);
			this.grpInput.PerformLayout();
			this.grpExtra.ResumeLayout(false);
			this.grpExtra.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
			this.contextMenuStripFileTree.ResumeLayout(false);
			this.panel1.ResumeLayout(false);
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button btnConvert;
		private System.Windows.Forms.Button btnBrowseOutput;
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.GroupBox grpOutputStyle;
		private System.Windows.Forms.Button btnAbout;
		private System.Windows.Forms.RadioButton rbGapsLeftOut;
		private System.Windows.Forms.RadioButton rbGapsPrepended;
		private System.Windows.Forms.RadioButton rbGapsAppended;
		private System.Windows.Forms.RadioButton rbSingleFile;
		private System.Windows.Forms.GroupBox grpOutputPathGeneration;
		private System.Windows.Forms.RadioButton rbDontGenerate;
		private System.Windows.Forms.RadioButton rbCreateSubdirectory;
		private System.Windows.Forms.RadioButton rbAppendFilename;
		private System.Windows.Forms.TextBox txtAppendFilename;
		private System.Windows.Forms.TextBox txtCreateSubdirectory;
		private System.Windows.Forms.GroupBox grpAudioOutput;
		private System.Windows.Forms.RadioButton rbFLAC;
		private System.Windows.Forms.RadioButton rbWAV;
		private System.Windows.Forms.RadioButton rbWavPack;
		private System.Windows.Forms.RadioButton rbCustomFormat;
		private System.Windows.Forms.TextBox txtCustomFormat;
		private System.Windows.Forms.Button btnSettings;
		private System.Windows.Forms.RadioButton rbNoAudio;
		private System.Windows.Forms.GroupBox grpAction;
		private System.Windows.Forms.RadioButton rbActionVerifyThenEncode;
		private System.Windows.Forms.RadioButton rbActionVerify;
		private System.Windows.Forms.RadioButton rbActionEncode;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar2;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.RadioButton rbEmbedCUE;
		private System.Windows.Forms.MaskedTextBox txtDataTrackLength;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.RadioButton rbAPE;
		private System.Windows.Forms.Button btnStop;
		private System.Windows.Forms.Button btnPause;
		private System.Windows.Forms.Button btnResume;
		private System.Windows.Forms.CheckBox chkLossyWAV;
		private System.Windows.Forms.RadioButton rbActionVerifyAndEncode;
		private System.Windows.Forms.RadioButton rbTTA;
		private System.Windows.Forms.GroupBox grpFreedb;
		private System.Windows.Forms.RadioButton rbFreedbAlways;
		private System.Windows.Forms.RadioButton rbFreedbIf;
		private System.Windows.Forms.RadioButton rbFreedbNever;
		private System.Windows.Forms.RadioButton rbUDC1;
		private System.Windows.Forms.Button btnCodec;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripUDC;
		private System.Windows.Forms.ToolStripMenuItem tAKToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem mP3ToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem oGGToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem1;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem2;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem3;
		private System.Windows.Forms.RadioButton rbActionVerifyAndCRCs;
		private System.Windows.Forms.MaskedTextBox txtPreGapLength;
		private System.Windows.Forms.Label label2;
		private CUEControls.FileSystemTreeView fileSystemTreeView1;
		private System.Windows.Forms.TextBox txtInputPath;
		private System.Windows.Forms.CheckBox chkMulti;
		private System.Windows.Forms.CheckBox chkRecursive;
		private System.Windows.Forms.GroupBox grpInput;
		private System.Windows.Forms.GroupBox grpExtra;
		private System.Windows.Forms.RadioButton rbActionCorrectFilenames;
		private System.Windows.Forms.RadioButton rbActionCreateCUESheet;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripFileTree;
		private System.Windows.Forms.ToolStripMenuItem setAsMyMusicFolderToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem SelectedNodeName;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
		private System.Windows.Forms.ToolStripMenuItem resetToOriginalLocationToolStripMenuItem;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelAR;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelFLAC;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelWV;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.Label lblWriteOffset;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelProcessed;
		private System.Windows.Forms.TextBox textBatchReport;
		private System.Windows.Forms.Panel panel1;
	}
}

