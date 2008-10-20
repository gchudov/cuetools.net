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
			this.btnStop = new System.Windows.Forms.Button();
			this.btnPause = new System.Windows.Forms.Button();
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
			resources.ApplyResources(this.btnConvert, "btnConvert");
			this.btnConvert.Name = "btnConvert";
			this.btnConvert.UseVisualStyleBackColor = true;
			this.btnConvert.Click += new System.EventHandler(this.btnConvert_Click);
			// 
			// grpCUEPaths
			// 
			this.grpCUEPaths.Controls.Add(this.btnBrowseOutput);
			this.grpCUEPaths.Controls.Add(this.btnBrowseInput);
			this.grpCUEPaths.Controls.Add(this.lblOutput);
			this.grpCUEPaths.Controls.Add(this.lblInput);
			this.grpCUEPaths.Controls.Add(this.txtOutputPath);
			this.grpCUEPaths.Controls.Add(this.txtInputPath);
			resources.ApplyResources(this.grpCUEPaths, "grpCUEPaths");
			this.grpCUEPaths.Name = "grpCUEPaths";
			this.grpCUEPaths.TabStop = false;
			// 
			// btnBrowseOutput
			// 
			resources.ApplyResources(this.btnBrowseOutput, "btnBrowseOutput");
			this.btnBrowseOutput.Name = "btnBrowseOutput";
			this.btnBrowseOutput.UseVisualStyleBackColor = true;
			this.btnBrowseOutput.Click += new System.EventHandler(this.btnBrowseOutput_Click);
			// 
			// btnBrowseInput
			// 
			resources.ApplyResources(this.btnBrowseInput, "btnBrowseInput");
			this.btnBrowseInput.Name = "btnBrowseInput";
			this.btnBrowseInput.UseVisualStyleBackColor = true;
			this.btnBrowseInput.Click += new System.EventHandler(this.btnBrowseInput_Click);
			// 
			// lblOutput
			// 
			resources.ApplyResources(this.lblOutput, "lblOutput");
			this.lblOutput.Name = "lblOutput";
			// 
			// lblInput
			// 
			resources.ApplyResources(this.lblInput, "lblInput");
			this.lblInput.Name = "lblInput";
			// 
			// txtOutputPath
			// 
			this.txtOutputPath.AllowDrop = true;
			resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
			this.txtOutputPath.Name = "txtOutputPath";
			this.txtOutputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtOutputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// txtInputPath
			// 
			this.txtInputPath.AllowDrop = true;
			resources.ApplyResources(this.txtInputPath, "txtInputPath");
			this.txtInputPath.Name = "txtInputPath";
			this.txtInputPath.TextChanged += new System.EventHandler(this.txtInputPath_TextChanged);
			this.txtInputPath.DragDrop += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragDrop);
			this.txtInputPath.DragEnter += new System.Windows.Forms.DragEventHandler(this.PathTextBox_DragEnter);
			// 
			// grpOutputStyle
			// 
			this.grpOutputStyle.Controls.Add(this.rbEmbedCUE);
			this.grpOutputStyle.Controls.Add(this.rbGapsLeftOut);
			this.grpOutputStyle.Controls.Add(this.rbGapsPrepended);
			this.grpOutputStyle.Controls.Add(this.rbGapsAppended);
			this.grpOutputStyle.Controls.Add(this.rbSingleFile);
			resources.ApplyResources(this.grpOutputStyle, "grpOutputStyle");
			this.grpOutputStyle.Name = "grpOutputStyle";
			this.grpOutputStyle.TabStop = false;
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
			// rbGapsLeftOut
			// 
			resources.ApplyResources(this.rbGapsLeftOut, "rbGapsLeftOut");
			this.rbGapsLeftOut.Name = "rbGapsLeftOut";
			this.toolTip1.SetToolTip(this.rbGapsLeftOut, resources.GetString("rbGapsLeftOut.ToolTip"));
			this.rbGapsLeftOut.UseVisualStyleBackColor = true;
			// 
			// rbGapsPrepended
			// 
			resources.ApplyResources(this.rbGapsPrepended, "rbGapsPrepended");
			this.rbGapsPrepended.Name = "rbGapsPrepended";
			this.toolTip1.SetToolTip(this.rbGapsPrepended, resources.GetString("rbGapsPrepended.ToolTip"));
			this.rbGapsPrepended.UseVisualStyleBackColor = true;
			// 
			// rbGapsAppended
			// 
			resources.ApplyResources(this.rbGapsAppended, "rbGapsAppended");
			this.rbGapsAppended.Name = "rbGapsAppended";
			this.toolTip1.SetToolTip(this.rbGapsAppended, resources.GetString("rbGapsAppended.ToolTip"));
			this.rbGapsAppended.UseVisualStyleBackColor = true;
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
			// btnAbout
			// 
			resources.ApplyResources(this.btnAbout, "btnAbout");
			this.btnAbout.Name = "btnAbout";
			this.btnAbout.UseVisualStyleBackColor = true;
			this.btnAbout.Click += new System.EventHandler(this.btnAbout_Click);
			// 
			// grpOutputPathGeneration
			// 
			this.grpOutputPathGeneration.Controls.Add(this.txtCustomFormat);
			this.grpOutputPathGeneration.Controls.Add(this.rbCustomFormat);
			this.grpOutputPathGeneration.Controls.Add(this.txtCreateSubdirectory);
			this.grpOutputPathGeneration.Controls.Add(this.rbDontGenerate);
			this.grpOutputPathGeneration.Controls.Add(this.rbCreateSubdirectory);
			this.grpOutputPathGeneration.Controls.Add(this.rbAppendFilename);
			this.grpOutputPathGeneration.Controls.Add(this.txtAppendFilename);
			resources.ApplyResources(this.grpOutputPathGeneration, "grpOutputPathGeneration");
			this.grpOutputPathGeneration.Name = "grpOutputPathGeneration";
			this.grpOutputPathGeneration.TabStop = false;
			// 
			// txtCustomFormat
			// 
			resources.ApplyResources(this.txtCustomFormat, "txtCustomFormat");
			this.txtCustomFormat.Name = "txtCustomFormat";
			this.txtCustomFormat.TextChanged += new System.EventHandler(this.txtCustomFormat_TextChanged);
			// 
			// rbCustomFormat
			// 
			resources.ApplyResources(this.rbCustomFormat, "rbCustomFormat");
			this.rbCustomFormat.Name = "rbCustomFormat";
			this.rbCustomFormat.TabStop = true;
			this.rbCustomFormat.UseVisualStyleBackColor = true;
			this.rbCustomFormat.CheckedChanged += new System.EventHandler(this.rbCustomFormat_CheckedChanged);
			// 
			// txtCreateSubdirectory
			// 
			resources.ApplyResources(this.txtCreateSubdirectory, "txtCreateSubdirectory");
			this.txtCreateSubdirectory.Name = "txtCreateSubdirectory";
			this.txtCreateSubdirectory.TextChanged += new System.EventHandler(this.txtCreateSubdirectory_TextChanged);
			// 
			// rbDontGenerate
			// 
			resources.ApplyResources(this.rbDontGenerate, "rbDontGenerate");
			this.rbDontGenerate.Name = "rbDontGenerate";
			this.rbDontGenerate.UseVisualStyleBackColor = true;
			// 
			// rbCreateSubdirectory
			// 
			resources.ApplyResources(this.rbCreateSubdirectory, "rbCreateSubdirectory");
			this.rbCreateSubdirectory.Checked = true;
			this.rbCreateSubdirectory.Name = "rbCreateSubdirectory";
			this.rbCreateSubdirectory.TabStop = true;
			this.rbCreateSubdirectory.UseVisualStyleBackColor = true;
			this.rbCreateSubdirectory.CheckedChanged += new System.EventHandler(this.rbCreateSubdirectory_CheckedChanged);
			// 
			// rbAppendFilename
			// 
			resources.ApplyResources(this.rbAppendFilename, "rbAppendFilename");
			this.rbAppendFilename.Name = "rbAppendFilename";
			this.rbAppendFilename.UseVisualStyleBackColor = true;
			this.rbAppendFilename.CheckedChanged += new System.EventHandler(this.rbAppendFilename_CheckedChanged);
			// 
			// txtAppendFilename
			// 
			resources.ApplyResources(this.txtAppendFilename, "txtAppendFilename");
			this.txtAppendFilename.Name = "txtAppendFilename";
			this.txtAppendFilename.TextChanged += new System.EventHandler(this.txtAppendFilename_TextChanged);
			// 
			// grpAudioOutput
			// 
			this.grpAudioOutput.Controls.Add(this.rbAPE);
			this.grpAudioOutput.Controls.Add(this.rbNoAudio);
			this.grpAudioOutput.Controls.Add(this.rbWavPack);
			this.grpAudioOutput.Controls.Add(this.rbFLAC);
			this.grpAudioOutput.Controls.Add(this.rbWAV);
			resources.ApplyResources(this.grpAudioOutput, "grpAudioOutput");
			this.grpAudioOutput.Name = "grpAudioOutput";
			this.grpAudioOutput.TabStop = false;
			// 
			// rbAPE
			// 
			resources.ApplyResources(this.rbAPE, "rbAPE");
			this.rbAPE.Name = "rbAPE";
			this.rbAPE.TabStop = true;
			this.rbAPE.UseVisualStyleBackColor = true;
			this.rbAPE.CheckedChanged += new System.EventHandler(this.rbAPE_CheckedChanged);
			// 
			// rbNoAudio
			// 
			resources.ApplyResources(this.rbNoAudio, "rbNoAudio");
			this.rbNoAudio.Name = "rbNoAudio";
			this.toolTip1.SetToolTip(this.rbNoAudio, resources.GetString("rbNoAudio.ToolTip"));
			this.rbNoAudio.UseVisualStyleBackColor = true;
			this.rbNoAudio.CheckedChanged += new System.EventHandler(this.rbNoAudio_CheckedChanged);
			// 
			// rbWavPack
			// 
			resources.ApplyResources(this.rbWavPack, "rbWavPack");
			this.rbWavPack.Name = "rbWavPack";
			this.rbWavPack.UseVisualStyleBackColor = true;
			this.rbWavPack.CheckedChanged += new System.EventHandler(this.rbWavPack_CheckedChanged);
			// 
			// rbFLAC
			// 
			resources.ApplyResources(this.rbFLAC, "rbFLAC");
			this.rbFLAC.Name = "rbFLAC";
			this.rbFLAC.UseVisualStyleBackColor = true;
			this.rbFLAC.CheckedChanged += new System.EventHandler(this.rbFLAC_CheckedChanged);
			// 
			// rbWAV
			// 
			resources.ApplyResources(this.rbWAV, "rbWAV");
			this.rbWAV.Checked = true;
			this.rbWAV.Name = "rbWAV";
			this.rbWAV.TabStop = true;
			this.rbWAV.UseVisualStyleBackColor = true;
			this.rbWAV.CheckedChanged += new System.EventHandler(this.rbWAV_CheckedChanged);
			// 
			// btnBatch
			// 
			resources.ApplyResources(this.btnBatch, "btnBatch");
			this.btnBatch.Name = "btnBatch";
			this.btnBatch.UseVisualStyleBackColor = true;
			this.btnBatch.Click += new System.EventHandler(this.btnBatch_Click);
			// 
			// btnFilenameCorrector
			// 
			resources.ApplyResources(this.btnFilenameCorrector, "btnFilenameCorrector");
			this.btnFilenameCorrector.Name = "btnFilenameCorrector";
			this.btnFilenameCorrector.UseVisualStyleBackColor = true;
			this.btnFilenameCorrector.Click += new System.EventHandler(this.btnFilenameCorrector_Click);
			// 
			// btnSettings
			// 
			resources.ApplyResources(this.btnSettings, "btnSettings");
			this.btnSettings.Name = "btnSettings";
			this.btnSettings.UseVisualStyleBackColor = true;
			this.btnSettings.Click += new System.EventHandler(this.btnSettings_Click);
			// 
			// grpAccurateRip
			// 
			this.grpAccurateRip.Controls.Add(this.label1);
			this.grpAccurateRip.Controls.Add(this.txtDataTrackLength);
			this.grpAccurateRip.Controls.Add(this.rbArApplyOffset);
			this.grpAccurateRip.Controls.Add(this.rbArVerify);
			this.grpAccurateRip.Controls.Add(this.rbArNone);
			resources.ApplyResources(this.grpAccurateRip, "grpAccurateRip");
			this.grpAccurateRip.Name = "grpAccurateRip";
			this.grpAccurateRip.TabStop = false;
			// 
			// label1
			// 
			resources.ApplyResources(this.label1, "label1");
			this.label1.Name = "label1";
			// 
			// txtDataTrackLength
			// 
			this.txtDataTrackLength.Culture = new System.Globalization.CultureInfo("");
			this.txtDataTrackLength.CutCopyMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.txtDataTrackLength.InsertKeyMode = System.Windows.Forms.InsertKeyMode.Overwrite;
			resources.ApplyResources(this.txtDataTrackLength, "txtDataTrackLength");
			this.txtDataTrackLength.Name = "txtDataTrackLength";
			this.txtDataTrackLength.TextMaskFormat = System.Windows.Forms.MaskFormat.IncludePromptAndLiterals;
			this.toolTip1.SetToolTip(this.txtDataTrackLength, resources.GetString("txtDataTrackLength.ToolTip"));
			// 
			// rbArApplyOffset
			// 
			resources.ApplyResources(this.rbArApplyOffset, "rbArApplyOffset");
			this.rbArApplyOffset.Name = "rbArApplyOffset";
			this.toolTip1.SetToolTip(this.rbArApplyOffset, resources.GetString("rbArApplyOffset.ToolTip"));
			this.rbArApplyOffset.UseVisualStyleBackColor = true;
			// 
			// rbArVerify
			// 
			resources.ApplyResources(this.rbArVerify, "rbArVerify");
			this.rbArVerify.Name = "rbArVerify";
			this.toolTip1.SetToolTip(this.rbArVerify, resources.GetString("rbArVerify.ToolTip"));
			this.rbArVerify.UseVisualStyleBackColor = true;
			this.rbArVerify.CheckedChanged += new System.EventHandler(this.rbArVerify_CheckedChanged);
			// 
			// rbArNone
			// 
			resources.ApplyResources(this.rbArNone, "rbArNone");
			this.rbArNone.Checked = true;
			this.rbArNone.Name = "rbArNone";
			this.rbArNone.TabStop = true;
			this.toolTip1.SetToolTip(this.rbArNone, resources.GetString("rbArNone.ToolTip"));
			this.rbArNone.UseVisualStyleBackColor = true;
			// 
			// statusStrip1
			// 
			this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripProgressBar1,
            this.toolStripProgressBar2});
			resources.ApplyResources(this.statusStrip1, "statusStrip1");
			this.statusStrip1.Name = "statusStrip1";
			this.statusStrip1.SizingGrip = false;
			// 
			// toolStripStatusLabel1
			// 
			this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
			resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
			this.toolStripStatusLabel1.Spring = true;
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
			// toolTip1
			// 
			this.toolTip1.AutoPopDelay = 15000;
			this.toolTip1.InitialDelay = 500;
			this.toolTip1.ReshowDelay = 100;
			// 
			// btnCUECreator
			// 
			resources.ApplyResources(this.btnCUECreator, "btnCUECreator");
			this.btnCUECreator.Name = "btnCUECreator";
			this.btnCUECreator.UseVisualStyleBackColor = true;
			this.btnCUECreator.Click += new System.EventHandler(this.btnCUECreator_Click);
			// 
			// btnStop
			// 
			resources.ApplyResources(this.btnStop, "btnStop");
			this.btnStop.Name = "btnStop";
			this.btnStop.UseVisualStyleBackColor = true;
			this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
			// 
			// btnPause
			// 
			resources.ApplyResources(this.btnPause, "btnPause");
			this.btnPause.Name = "btnPause";
			this.btnPause.UseVisualStyleBackColor = true;
			this.btnPause.Click += new System.EventHandler(this.btnPause_Click);
			// 
			// frmCUETools
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.btnPause);
			this.Controls.Add(this.btnStop);
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
			this.MaximizeBox = false;
			this.Name = "frmCUETools";
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
		private System.Windows.Forms.Button btnStop;
		private System.Windows.Forms.Button btnPause;
	}
}

