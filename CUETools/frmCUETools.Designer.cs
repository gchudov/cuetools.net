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
			this.grpCUEPaths = new System.Windows.Forms.GroupBox();
			this.btnBrowseOutput = new System.Windows.Forms.Button();
			this.btnBrowseInput = new System.Windows.Forms.Button();
			this.lblOutput = new System.Windows.Forms.Label();
			this.lblInput = new System.Windows.Forms.Label();
			this.txtOutputPath = new System.Windows.Forms.TextBox();
			this.txtInputPath = new System.Windows.Forms.TextBox();
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
			this.rbAPE = new System.Windows.Forms.RadioButton();
			this.rbNoAudio = new System.Windows.Forms.RadioButton();
			this.rbWavPack = new System.Windows.Forms.RadioButton();
			this.rbFLAC = new System.Windows.Forms.RadioButton();
			this.rbWAV = new System.Windows.Forms.RadioButton();
			this.btnBatch = new System.Windows.Forms.Button();
			this.btnFilenameCorrector = new System.Windows.Forms.Button();
			this.btnSettings = new System.Windows.Forms.Button();
			this.grpAccurateRip = new System.Windows.Forms.GroupBox();
			this.label1 = new System.Windows.Forms.Label();
			this.txtDataTrackLength = new System.Windows.Forms.MaskedTextBox();
			this.rbArApplyOffset = new System.Windows.Forms.RadioButton();
			this.rbArVerify = new System.Windows.Forms.RadioButton();
			this.rbArNone = new System.Windows.Forms.RadioButton();
			this.statusStrip1 = new System.Windows.Forms.StatusStrip();
			this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.btnCUECreator = new System.Windows.Forms.Button();
			this.grpCUEPaths.SuspendLayout();
			this.grpOutputStyle.SuspendLayout();
			this.grpOutputPathGeneration.SuspendLayout();
			this.grpAudioOutput.SuspendLayout();
			this.grpAccurateRip.SuspendLayout();
			this.statusStrip1.SuspendLayout();
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
			// grpCUEPaths
			// 
			this.grpCUEPaths.AccessibleDescription = null;
			this.grpCUEPaths.AccessibleName = null;
			resources.ApplyResources(this.grpCUEPaths, "grpCUEPaths");
			this.grpCUEPaths.BackgroundImage = null;
			this.grpCUEPaths.Controls.Add(this.btnBrowseOutput);
			this.grpCUEPaths.Controls.Add(this.btnBrowseInput);
			this.grpCUEPaths.Controls.Add(this.lblOutput);
			this.grpCUEPaths.Controls.Add(this.lblInput);
			this.grpCUEPaths.Controls.Add(this.txtOutputPath);
			this.grpCUEPaths.Controls.Add(this.txtInputPath);
			this.grpCUEPaths.Font = null;
			this.grpCUEPaths.Name = "grpCUEPaths";
			this.grpCUEPaths.TabStop = false;
			this.toolTip1.SetToolTip(this.grpCUEPaths, resources.GetString("grpCUEPaths.ToolTip"));
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
			// btnBrowseInput
			// 
			this.btnBrowseInput.AccessibleDescription = null;
			this.btnBrowseInput.AccessibleName = null;
			resources.ApplyResources(this.btnBrowseInput, "btnBrowseInput");
			this.btnBrowseInput.BackgroundImage = null;
			this.btnBrowseInput.Font = null;
			this.btnBrowseInput.Name = "btnBrowseInput";
			this.toolTip1.SetToolTip(this.btnBrowseInput, resources.GetString("btnBrowseInput.ToolTip"));
			this.btnBrowseInput.UseVisualStyleBackColor = true;
			this.btnBrowseInput.Click += new System.EventHandler(this.btnBrowseInput_Click);
			// 
			// lblOutput
			// 
			this.lblOutput.AccessibleDescription = null;
			this.lblOutput.AccessibleName = null;
			resources.ApplyResources(this.lblOutput, "lblOutput");
			this.lblOutput.Font = null;
			this.lblOutput.Name = "lblOutput";
			this.toolTip1.SetToolTip(this.lblOutput, resources.GetString("lblOutput.ToolTip"));
			// 
			// lblInput
			// 
			this.lblInput.AccessibleDescription = null;
			this.lblInput.AccessibleName = null;
			resources.ApplyResources(this.lblInput, "lblInput");
			this.lblInput.Font = null;
			this.lblInput.Name = "lblInput";
			this.toolTip1.SetToolTip(this.lblInput, resources.GetString("lblInput.ToolTip"));
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
			// txtInputPath
			// 
			this.txtInputPath.AccessibleDescription = null;
			this.txtInputPath.AccessibleName = null;
			this.txtInputPath.AllowDrop = true;
			resources.ApplyResources(this.txtInputPath, "txtInputPath");
			this.txtInputPath.BackgroundImage = null;
			this.txtInputPath.Font = null;
			this.txtInputPath.Name = "txtInputPath";
			this.toolTip1.SetToolTip(this.txtInputPath, resources.GetString("txtInputPath.ToolTip"));
			this.txtInputPath.TextChanged += new System.EventHandler(this.txtInputPath_TextChanged);
			this.txtInputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtInputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
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
			this.grpAudioOutput.Controls.Add(this.rbAPE);
			this.grpAudioOutput.Controls.Add(this.rbNoAudio);
			this.grpAudioOutput.Controls.Add(this.rbWavPack);
			this.grpAudioOutput.Controls.Add(this.rbFLAC);
			this.grpAudioOutput.Controls.Add(this.rbWAV);
			this.grpAudioOutput.Font = null;
			this.grpAudioOutput.Name = "grpAudioOutput";
			this.grpAudioOutput.TabStop = false;
			this.toolTip1.SetToolTip(this.grpAudioOutput, resources.GetString("grpAudioOutput.ToolTip"));
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
			// btnBatch
			// 
			this.btnBatch.AccessibleDescription = null;
			this.btnBatch.AccessibleName = null;
			resources.ApplyResources(this.btnBatch, "btnBatch");
			this.btnBatch.BackgroundImage = null;
			this.btnBatch.Font = null;
			this.btnBatch.Name = "btnBatch";
			this.toolTip1.SetToolTip(this.btnBatch, resources.GetString("btnBatch.ToolTip"));
			this.btnBatch.UseVisualStyleBackColor = true;
			this.btnBatch.Click += new System.EventHandler(this.btnBatch_Click);
			// 
			// btnFilenameCorrector
			// 
			this.btnFilenameCorrector.AccessibleDescription = null;
			this.btnFilenameCorrector.AccessibleName = null;
			resources.ApplyResources(this.btnFilenameCorrector, "btnFilenameCorrector");
			this.btnFilenameCorrector.BackgroundImage = null;
			this.btnFilenameCorrector.Font = null;
			this.btnFilenameCorrector.Name = "btnFilenameCorrector";
			this.toolTip1.SetToolTip(this.btnFilenameCorrector, resources.GetString("btnFilenameCorrector.ToolTip"));
			this.btnFilenameCorrector.UseVisualStyleBackColor = true;
			this.btnFilenameCorrector.Click += new System.EventHandler(this.btnFilenameCorrector_Click);
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
			// grpAccurateRip
			// 
			this.grpAccurateRip.AccessibleDescription = null;
			this.grpAccurateRip.AccessibleName = null;
			resources.ApplyResources(this.grpAccurateRip, "grpAccurateRip");
			this.grpAccurateRip.BackgroundImage = null;
			this.grpAccurateRip.Controls.Add(this.label1);
			this.grpAccurateRip.Controls.Add(this.txtDataTrackLength);
			this.grpAccurateRip.Controls.Add(this.rbArApplyOffset);
			this.grpAccurateRip.Controls.Add(this.rbArVerify);
			this.grpAccurateRip.Controls.Add(this.rbArNone);
			this.grpAccurateRip.Font = null;
			this.grpAccurateRip.Name = "grpAccurateRip";
			this.grpAccurateRip.TabStop = false;
			this.toolTip1.SetToolTip(this.grpAccurateRip, resources.GetString("grpAccurateRip.ToolTip"));
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
			// rbArApplyOffset
			// 
			this.rbArApplyOffset.AccessibleDescription = null;
			this.rbArApplyOffset.AccessibleName = null;
			resources.ApplyResources(this.rbArApplyOffset, "rbArApplyOffset");
			this.rbArApplyOffset.BackgroundImage = null;
			this.rbArApplyOffset.Font = null;
			this.rbArApplyOffset.Name = "rbArApplyOffset";
			this.toolTip1.SetToolTip(this.rbArApplyOffset, resources.GetString("rbArApplyOffset.ToolTip"));
			this.rbArApplyOffset.UseVisualStyleBackColor = true;
			// 
			// rbArVerify
			// 
			this.rbArVerify.AccessibleDescription = null;
			this.rbArVerify.AccessibleName = null;
			resources.ApplyResources(this.rbArVerify, "rbArVerify");
			this.rbArVerify.BackgroundImage = null;
			this.rbArVerify.Font = null;
			this.rbArVerify.Name = "rbArVerify";
			this.toolTip1.SetToolTip(this.rbArVerify, resources.GetString("rbArVerify.ToolTip"));
			this.rbArVerify.UseVisualStyleBackColor = true;
			this.rbArVerify.CheckedChanged += new System.EventHandler(this.rbArVerify_CheckedChanged);
			// 
			// rbArNone
			// 
			this.rbArNone.AccessibleDescription = null;
			this.rbArNone.AccessibleName = null;
			resources.ApplyResources(this.rbArNone, "rbArNone");
			this.rbArNone.BackgroundImage = null;
			this.rbArNone.Checked = true;
			this.rbArNone.Font = null;
			this.rbArNone.Name = "rbArNone";
			this.rbArNone.TabStop = true;
			this.toolTip1.SetToolTip(this.rbArNone, resources.GetString("rbArNone.ToolTip"));
			this.rbArNone.UseVisualStyleBackColor = true;
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
            this.toolStripProgressBar1,
            this.toolStripProgressBar2});
			this.statusStrip1.Name = "statusStrip1";
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
			// btnCUECreator
			// 
			this.btnCUECreator.AccessibleDescription = null;
			this.btnCUECreator.AccessibleName = null;
			resources.ApplyResources(this.btnCUECreator, "btnCUECreator");
			this.btnCUECreator.BackgroundImage = null;
			this.btnCUECreator.Font = null;
			this.btnCUECreator.Name = "btnCUECreator";
			this.toolTip1.SetToolTip(this.btnCUECreator, resources.GetString("btnCUECreator.ToolTip"));
			this.btnCUECreator.UseVisualStyleBackColor = true;
			this.btnCUECreator.Click += new System.EventHandler(this.btnCUECreator_Click);
			// 
			// frmCUETools
			// 
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.Controls.Add(this.btnCUECreator);
			this.Controls.Add(this.statusStrip1);
			this.Controls.Add(this.grpAccurateRip);
			this.Controls.Add(this.btnSettings);
			this.Controls.Add(this.btnFilenameCorrector);
			this.Controls.Add(this.btnBatch);
			this.Controls.Add(this.grpAudioOutput);
			this.Controls.Add(this.grpOutputPathGeneration);
			this.Controls.Add(this.btnAbout);
			this.Controls.Add(this.grpOutputStyle);
			this.Controls.Add(this.grpCUEPaths);
			this.Controls.Add(this.btnConvert);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.Icon = null;
			this.MaximizeBox = false;
			this.Name = "frmCUETools";
			this.toolTip1.SetToolTip(this, resources.GetString("$this.ToolTip"));
			this.Load += new System.EventHandler(this.frmCUETools_Load);
			this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUETools_FormClosed);
			this.grpCUEPaths.ResumeLayout(false);
			this.grpCUEPaths.PerformLayout();
			this.grpOutputStyle.ResumeLayout(false);
			this.grpOutputStyle.PerformLayout();
			this.grpOutputPathGeneration.ResumeLayout(false);
			this.grpOutputPathGeneration.PerformLayout();
			this.grpAudioOutput.ResumeLayout(false);
			this.grpAudioOutput.PerformLayout();
			this.grpAccurateRip.ResumeLayout(false);
			this.grpAccurateRip.PerformLayout();
			this.statusStrip1.ResumeLayout(false);
			this.statusStrip1.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button btnConvert;
		private System.Windows.Forms.GroupBox grpCUEPaths;
		private System.Windows.Forms.Button btnBrowseOutput;
		private System.Windows.Forms.Button btnBrowseInput;
		private System.Windows.Forms.Label lblOutput;
		private System.Windows.Forms.Label lblInput;
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.TextBox txtInputPath;
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
		private System.Windows.Forms.Button btnBatch;
		private System.Windows.Forms.Button btnFilenameCorrector;
		private System.Windows.Forms.Button btnSettings;
		private System.Windows.Forms.RadioButton rbNoAudio;
		private System.Windows.Forms.GroupBox grpAccurateRip;
		private System.Windows.Forms.RadioButton rbArApplyOffset;
		private System.Windows.Forms.RadioButton rbArVerify;
		private System.Windows.Forms.RadioButton rbArNone;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar2;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.RadioButton rbEmbedCUE;
        private System.Windows.Forms.Button btnCUECreator;
		private System.Windows.Forms.MaskedTextBox txtDataTrackLength;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.RadioButton rbAPE;
	}
}

