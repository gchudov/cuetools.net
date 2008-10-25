namespace JDP {
	partial class frmSettings {
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
			System.Windows.Forms.Button btnCancel;
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmSettings));
			this.grpGeneral = new System.Windows.Forms.GroupBox();
			this.chkFillUpCUE = new System.Windows.Forms.CheckBox();
			this.chkEmbedLog = new System.Windows.Forms.CheckBox();
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.chkAutoCorrectFilenames = new System.Windows.Forms.CheckBox();
			this.chkPreserveHTOA = new System.Windows.Forms.CheckBox();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.grpFLAC = new System.Windows.Forms.GroupBox();
			this.numericFLACCompressionLevel = new System.Windows.Forms.NumericUpDown();
			this.lblFLACCompressionLevel = new System.Windows.Forms.Label();
			this.chkFLACVerify = new System.Windows.Forms.CheckBox();
			this.btnOK = new System.Windows.Forms.Button();
			this.grpWavPack = new System.Windows.Forms.GroupBox();
			this.numWVExtraMode = new System.Windows.Forms.NumericUpDown();
			this.chkWVExtraMode = new System.Windows.Forms.CheckBox();
			this.rbWVVeryHigh = new System.Windows.Forms.RadioButton();
			this.rbWVHigh = new System.Windows.Forms.RadioButton();
			this.rbWVNormal = new System.Windows.Forms.RadioButton();
			this.rbWVFast = new System.Windows.Forms.RadioButton();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.chkArFixOffset = new System.Windows.Forms.CheckBox();
			this.label4 = new System.Windows.Forms.Label();
			this.numEncodeWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.label3 = new System.Windows.Forms.Label();
			this.numEncodeWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.chkArNoUnverifiedAudio = new System.Windows.Forms.CheckBox();
			this.chkArSaveLog = new System.Windows.Forms.CheckBox();
			this.label2 = new System.Windows.Forms.Label();
			this.numFixWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.label1 = new System.Windows.Forms.Label();
			this.numFixWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.chkArAddCRCs = new System.Windows.Forms.CheckBox();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.chkFilenamesANSISafe = new System.Windows.Forms.CheckBox();
			this.grpAudioFilenames = new System.Windows.Forms.GroupBox();
			this.chkKeepOriginalFilenames = new System.Windows.Forms.CheckBox();
			this.txtSpecialExceptions = new System.Windows.Forms.TextBox();
			this.chkRemoveSpecial = new System.Windows.Forms.CheckBox();
			this.chkReplaceSpaces = new System.Windows.Forms.CheckBox();
			this.txtTrackFilenameFormat = new System.Windows.Forms.TextBox();
			this.lblTrackFilenameFormat = new System.Windows.Forms.Label();
			this.lblSingleFilenameFormat = new System.Windows.Forms.Label();
			this.txtSingleFilenameFormat = new System.Windows.Forms.TextBox();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.rbAPEinsane = new System.Windows.Forms.RadioButton();
			this.rbAPEextrahigh = new System.Windows.Forms.RadioButton();
			this.rbAPEhigh = new System.Windows.Forms.RadioButton();
			this.rbAPEnormal = new System.Windows.Forms.RadioButton();
			this.rbAPEfast = new System.Windows.Forms.RadioButton();
			btnCancel = new System.Windows.Forms.Button();
			this.grpGeneral.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.grpFLAC.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericFLACCompressionLevel)).BeginInit();
			this.grpWavPack.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numWVExtraMode)).BeginInit();
			this.groupBox1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).BeginInit();
			this.grpAudioFilenames.SuspendLayout();
			this.groupBox2.SuspendLayout();
			this.SuspendLayout();
			// 
			// btnCancel
			// 
			btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			resources.ApplyResources(btnCancel, "btnCancel");
			btnCancel.Name = "btnCancel";
			btnCancel.UseVisualStyleBackColor = true;
			// 
			// grpGeneral
			// 
			this.grpGeneral.Controls.Add(this.chkFillUpCUE);
			this.grpGeneral.Controls.Add(this.chkEmbedLog);
			this.grpGeneral.Controls.Add(this.numericWriteOffset);
			this.grpGeneral.Controls.Add(this.chkAutoCorrectFilenames);
			this.grpGeneral.Controls.Add(this.chkPreserveHTOA);
			this.grpGeneral.Controls.Add(this.lblWriteOffset);
			resources.ApplyResources(this.grpGeneral, "grpGeneral");
			this.grpGeneral.Name = "grpGeneral";
			this.grpGeneral.TabStop = false;
			// 
			// chkFillUpCUE
			// 
			resources.ApplyResources(this.chkFillUpCUE, "chkFillUpCUE");
			this.chkFillUpCUE.Name = "chkFillUpCUE";
			this.chkFillUpCUE.UseVisualStyleBackColor = true;
			// 
			// chkEmbedLog
			// 
			resources.ApplyResources(this.chkEmbedLog, "chkEmbedLog");
			this.chkEmbedLog.Name = "chkEmbedLog";
			this.toolTip1.SetToolTip(this.chkEmbedLog, resources.GetString("chkEmbedLog.ToolTip"));
			this.chkEmbedLog.UseVisualStyleBackColor = true;
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
			// 
			// chkAutoCorrectFilenames
			// 
			resources.ApplyResources(this.chkAutoCorrectFilenames, "chkAutoCorrectFilenames");
			this.chkAutoCorrectFilenames.Name = "chkAutoCorrectFilenames";
			this.toolTip1.SetToolTip(this.chkAutoCorrectFilenames, resources.GetString("chkAutoCorrectFilenames.ToolTip"));
			this.chkAutoCorrectFilenames.UseVisualStyleBackColor = true;
			// 
			// chkPreserveHTOA
			// 
			resources.ApplyResources(this.chkPreserveHTOA, "chkPreserveHTOA");
			this.chkPreserveHTOA.Name = "chkPreserveHTOA";
			this.chkPreserveHTOA.UseVisualStyleBackColor = true;
			// 
			// lblWriteOffset
			// 
			resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
			this.lblWriteOffset.Name = "lblWriteOffset";
			// 
			// grpFLAC
			// 
			this.grpFLAC.Controls.Add(this.numericFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.lblFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.chkFLACVerify);
			resources.ApplyResources(this.grpFLAC, "grpFLAC");
			this.grpFLAC.Name = "grpFLAC";
			this.grpFLAC.TabStop = false;
			// 
			// numericFLACCompressionLevel
			// 
			resources.ApplyResources(this.numericFLACCompressionLevel, "numericFLACCompressionLevel");
			this.numericFLACCompressionLevel.Maximum = new decimal(new int[] {
            8,
            0,
            0,
            0});
			this.numericFLACCompressionLevel.Name = "numericFLACCompressionLevel";
			this.numericFLACCompressionLevel.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
			// 
			// lblFLACCompressionLevel
			// 
			resources.ApplyResources(this.lblFLACCompressionLevel, "lblFLACCompressionLevel");
			this.lblFLACCompressionLevel.Name = "lblFLACCompressionLevel";
			// 
			// chkFLACVerify
			// 
			resources.ApplyResources(this.chkFLACVerify, "chkFLACVerify");
			this.chkFLACVerify.Name = "chkFLACVerify";
			this.chkFLACVerify.UseVisualStyleBackColor = true;
			// 
			// btnOK
			// 
			this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			resources.ApplyResources(this.btnOK, "btnOK");
			this.btnOK.Name = "btnOK";
			this.btnOK.UseVisualStyleBackColor = true;
			this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
			// 
			// grpWavPack
			// 
			this.grpWavPack.Controls.Add(this.numWVExtraMode);
			this.grpWavPack.Controls.Add(this.chkWVExtraMode);
			this.grpWavPack.Controls.Add(this.rbWVVeryHigh);
			this.grpWavPack.Controls.Add(this.rbWVHigh);
			this.grpWavPack.Controls.Add(this.rbWVNormal);
			this.grpWavPack.Controls.Add(this.rbWVFast);
			resources.ApplyResources(this.grpWavPack, "grpWavPack");
			this.grpWavPack.Name = "grpWavPack";
			this.grpWavPack.TabStop = false;
			// 
			// numWVExtraMode
			// 
			resources.ApplyResources(this.numWVExtraMode, "numWVExtraMode");
			this.numWVExtraMode.Maximum = new decimal(new int[] {
            6,
            0,
            0,
            0});
			this.numWVExtraMode.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numWVExtraMode.Name = "numWVExtraMode";
			this.numWVExtraMode.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkWVExtraMode
			// 
			resources.ApplyResources(this.chkWVExtraMode, "chkWVExtraMode");
			this.chkWVExtraMode.Name = "chkWVExtraMode";
			this.chkWVExtraMode.UseVisualStyleBackColor = true;
			this.chkWVExtraMode.CheckedChanged += new System.EventHandler(this.chkWVExtraMode_CheckedChanged);
			// 
			// rbWVVeryHigh
			// 
			resources.ApplyResources(this.rbWVVeryHigh, "rbWVVeryHigh");
			this.rbWVVeryHigh.Name = "rbWVVeryHigh";
			this.rbWVVeryHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVHigh
			// 
			resources.ApplyResources(this.rbWVHigh, "rbWVHigh");
			this.rbWVHigh.Name = "rbWVHigh";
			this.rbWVHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVNormal
			// 
			resources.ApplyResources(this.rbWVNormal, "rbWVNormal");
			this.rbWVNormal.Checked = true;
			this.rbWVNormal.Name = "rbWVNormal";
			this.rbWVNormal.TabStop = true;
			this.rbWVNormal.UseVisualStyleBackColor = true;
			// 
			// rbWVFast
			// 
			resources.ApplyResources(this.rbWVFast, "rbWVFast");
			this.rbWVFast.Name = "rbWVFast";
			this.rbWVFast.UseVisualStyleBackColor = true;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.chkArFixOffset);
			this.groupBox1.Controls.Add(this.label4);
			this.groupBox1.Controls.Add(this.numEncodeWhenPercent);
			this.groupBox1.Controls.Add(this.label3);
			this.groupBox1.Controls.Add(this.numEncodeWhenConfidence);
			this.groupBox1.Controls.Add(this.chkArNoUnverifiedAudio);
			this.groupBox1.Controls.Add(this.chkArSaveLog);
			this.groupBox1.Controls.Add(this.label2);
			this.groupBox1.Controls.Add(this.numFixWhenConfidence);
			this.groupBox1.Controls.Add(this.label1);
			this.groupBox1.Controls.Add(this.numFixWhenPercent);
			this.groupBox1.Controls.Add(this.chkArAddCRCs);
			resources.ApplyResources(this.groupBox1, "groupBox1");
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.TabStop = false;
			// 
			// chkArFixOffset
			// 
			resources.ApplyResources(this.chkArFixOffset, "chkArFixOffset");
			this.chkArFixOffset.Checked = true;
			this.chkArFixOffset.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArFixOffset.Name = "chkArFixOffset";
			this.chkArFixOffset.UseVisualStyleBackColor = true;
			this.chkArFixOffset.CheckedChanged += new System.EventHandler(this.chkArFixOffset_CheckedChanged);
			// 
			// label4
			// 
			resources.ApplyResources(this.label4, "label4");
			this.label4.Name = "label4";
			// 
			// numEncodeWhenPercent
			// 
			this.numEncodeWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			resources.ApplyResources(this.numEncodeWhenPercent, "numEncodeWhenPercent");
			this.numEncodeWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenPercent.Name = "numEncodeWhenPercent";
			this.numEncodeWhenPercent.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
			// 
			// label3
			// 
			resources.ApplyResources(this.label3, "label3");
			this.label3.Name = "label3";
			// 
			// numEncodeWhenConfidence
			// 
			resources.ApplyResources(this.numEncodeWhenConfidence, "numEncodeWhenConfidence");
			this.numEncodeWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenConfidence.Name = "numEncodeWhenConfidence";
			this.numEncodeWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkArNoUnverifiedAudio
			// 
			resources.ApplyResources(this.chkArNoUnverifiedAudio, "chkArNoUnverifiedAudio");
			this.chkArNoUnverifiedAudio.Checked = true;
			this.chkArNoUnverifiedAudio.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArNoUnverifiedAudio.Name = "chkArNoUnverifiedAudio";
			this.chkArNoUnverifiedAudio.UseVisualStyleBackColor = true;
			this.chkArNoUnverifiedAudio.CheckedChanged += new System.EventHandler(this.chkArNoUnverifiedAudio_CheckedChanged);
			// 
			// chkArSaveLog
			// 
			resources.ApplyResources(this.chkArSaveLog, "chkArSaveLog");
			this.chkArSaveLog.Checked = true;
			this.chkArSaveLog.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArSaveLog.Name = "chkArSaveLog";
			this.chkArSaveLog.UseVisualStyleBackColor = true;
			// 
			// label2
			// 
			resources.ApplyResources(this.label2, "label2");
			this.label2.Name = "label2";
			// 
			// numFixWhenConfidence
			// 
			resources.ApplyResources(this.numFixWhenConfidence, "numFixWhenConfidence");
			this.numFixWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenConfidence.Name = "numFixWhenConfidence";
			this.numFixWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// label1
			// 
			resources.ApplyResources(this.label1, "label1");
			this.label1.Name = "label1";
			// 
			// numFixWhenPercent
			// 
			this.numFixWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			resources.ApplyResources(this.numFixWhenPercent, "numFixWhenPercent");
			this.numFixWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenPercent.Name = "numFixWhenPercent";
			this.numFixWhenPercent.Value = new decimal(new int[] {
            51,
            0,
            0,
            0});
			// 
			// chkArAddCRCs
			// 
			resources.ApplyResources(this.chkArAddCRCs, "chkArAddCRCs");
			this.chkArAddCRCs.Checked = true;
			this.chkArAddCRCs.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArAddCRCs.Name = "chkArAddCRCs";
			this.toolTip1.SetToolTip(this.chkArAddCRCs, resources.GetString("chkArAddCRCs.ToolTip"));
			this.chkArAddCRCs.UseVisualStyleBackColor = true;
			// 
			// toolTip1
			// 
			this.toolTip1.AutoPopDelay = 15000;
			this.toolTip1.InitialDelay = 500;
			this.toolTip1.ReshowDelay = 100;
			// 
			// chkFilenamesANSISafe
			// 
			resources.ApplyResources(this.chkFilenamesANSISafe, "chkFilenamesANSISafe");
			this.chkFilenamesANSISafe.Name = "chkFilenamesANSISafe";
			this.toolTip1.SetToolTip(this.chkFilenamesANSISafe, resources.GetString("chkFilenamesANSISafe.ToolTip"));
			this.chkFilenamesANSISafe.UseVisualStyleBackColor = true;
			// 
			// grpAudioFilenames
			// 
			this.grpAudioFilenames.Controls.Add(this.chkFilenamesANSISafe);
			this.grpAudioFilenames.Controls.Add(this.chkKeepOriginalFilenames);
			this.grpAudioFilenames.Controls.Add(this.txtSpecialExceptions);
			this.grpAudioFilenames.Controls.Add(this.chkRemoveSpecial);
			this.grpAudioFilenames.Controls.Add(this.chkReplaceSpaces);
			this.grpAudioFilenames.Controls.Add(this.txtTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblSingleFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.txtSingleFilenameFormat);
			resources.ApplyResources(this.grpAudioFilenames, "grpAudioFilenames");
			this.grpAudioFilenames.Name = "grpAudioFilenames";
			this.grpAudioFilenames.TabStop = false;
			// 
			// chkKeepOriginalFilenames
			// 
			resources.ApplyResources(this.chkKeepOriginalFilenames, "chkKeepOriginalFilenames");
			this.chkKeepOriginalFilenames.Checked = true;
			this.chkKeepOriginalFilenames.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkKeepOriginalFilenames.Name = "chkKeepOriginalFilenames";
			this.chkKeepOriginalFilenames.UseVisualStyleBackColor = true;
			// 
			// txtSpecialExceptions
			// 
			resources.ApplyResources(this.txtSpecialExceptions, "txtSpecialExceptions");
			this.txtSpecialExceptions.Name = "txtSpecialExceptions";
			// 
			// chkRemoveSpecial
			// 
			resources.ApplyResources(this.chkRemoveSpecial, "chkRemoveSpecial");
			this.chkRemoveSpecial.Checked = true;
			this.chkRemoveSpecial.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkRemoveSpecial.Name = "chkRemoveSpecial";
			this.chkRemoveSpecial.UseVisualStyleBackColor = true;
			// 
			// chkReplaceSpaces
			// 
			resources.ApplyResources(this.chkReplaceSpaces, "chkReplaceSpaces");
			this.chkReplaceSpaces.Checked = true;
			this.chkReplaceSpaces.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkReplaceSpaces.Name = "chkReplaceSpaces";
			this.chkReplaceSpaces.UseVisualStyleBackColor = true;
			// 
			// txtTrackFilenameFormat
			// 
			resources.ApplyResources(this.txtTrackFilenameFormat, "txtTrackFilenameFormat");
			this.txtTrackFilenameFormat.Name = "txtTrackFilenameFormat";
			// 
			// lblTrackFilenameFormat
			// 
			resources.ApplyResources(this.lblTrackFilenameFormat, "lblTrackFilenameFormat");
			this.lblTrackFilenameFormat.Name = "lblTrackFilenameFormat";
			// 
			// lblSingleFilenameFormat
			// 
			resources.ApplyResources(this.lblSingleFilenameFormat, "lblSingleFilenameFormat");
			this.lblSingleFilenameFormat.Name = "lblSingleFilenameFormat";
			// 
			// txtSingleFilenameFormat
			// 
			resources.ApplyResources(this.txtSingleFilenameFormat, "txtSingleFilenameFormat");
			this.txtSingleFilenameFormat.Name = "txtSingleFilenameFormat";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.Add(this.rbAPEinsane);
			this.groupBox2.Controls.Add(this.rbAPEextrahigh);
			this.groupBox2.Controls.Add(this.rbAPEhigh);
			this.groupBox2.Controls.Add(this.rbAPEnormal);
			this.groupBox2.Controls.Add(this.rbAPEfast);
			resources.ApplyResources(this.groupBox2, "groupBox2");
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.TabStop = false;
			// 
			// rbAPEinsane
			// 
			resources.ApplyResources(this.rbAPEinsane, "rbAPEinsane");
			this.rbAPEinsane.Name = "rbAPEinsane";
			this.rbAPEinsane.TabStop = true;
			this.rbAPEinsane.UseVisualStyleBackColor = true;
			// 
			// rbAPEextrahigh
			// 
			resources.ApplyResources(this.rbAPEextrahigh, "rbAPEextrahigh");
			this.rbAPEextrahigh.Name = "rbAPEextrahigh";
			this.rbAPEextrahigh.TabStop = true;
			this.rbAPEextrahigh.UseVisualStyleBackColor = true;
			// 
			// rbAPEhigh
			// 
			resources.ApplyResources(this.rbAPEhigh, "rbAPEhigh");
			this.rbAPEhigh.Name = "rbAPEhigh";
			this.rbAPEhigh.TabStop = true;
			this.rbAPEhigh.UseVisualStyleBackColor = true;
			// 
			// rbAPEnormal
			// 
			resources.ApplyResources(this.rbAPEnormal, "rbAPEnormal");
			this.rbAPEnormal.Name = "rbAPEnormal";
			this.rbAPEnormal.TabStop = true;
			this.rbAPEnormal.UseVisualStyleBackColor = true;
			// 
			// rbAPEfast
			// 
			resources.ApplyResources(this.rbAPEfast, "rbAPEfast");
			this.rbAPEfast.Name = "rbAPEfast";
			this.rbAPEfast.TabStop = true;
			this.rbAPEfast.UseVisualStyleBackColor = true;
			// 
			// frmSettings
			// 
			this.AcceptButton = this.btnOK;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = btnCancel;
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.grpFLAC);
			this.Controls.Add(this.grpAudioFilenames);
			this.Controls.Add(btnCancel);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.grpWavPack);
			this.Controls.Add(this.btnOK);
			this.Controls.Add(this.grpGeneral);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "frmSettings";
			this.Load += new System.EventHandler(this.frmSettings_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmSettings_FormClosing);
			this.grpGeneral.ResumeLayout(false);
			this.grpGeneral.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
			this.grpFLAC.ResumeLayout(false);
			this.grpFLAC.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericFLACCompressionLevel)).EndInit();
			this.grpWavPack.ResumeLayout(false);
			this.grpWavPack.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numWVExtraMode)).EndInit();
			this.groupBox1.ResumeLayout(false);
			this.groupBox1.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).EndInit();
			this.grpAudioFilenames.ResumeLayout(false);
			this.grpAudioFilenames.PerformLayout();
			this.groupBox2.ResumeLayout(false);
			this.groupBox2.PerformLayout();
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.GroupBox grpGeneral;
		private System.Windows.Forms.CheckBox chkPreserveHTOA;
		private System.Windows.Forms.Label lblWriteOffset;
		private System.Windows.Forms.GroupBox grpFLAC;
		private System.Windows.Forms.Label lblFLACCompressionLevel;
		private System.Windows.Forms.CheckBox chkFLACVerify;
		private System.Windows.Forms.Button btnOK;
		private System.Windows.Forms.GroupBox grpWavPack;
		private System.Windows.Forms.RadioButton rbWVVeryHigh;
		private System.Windows.Forms.RadioButton rbWVHigh;
		private System.Windows.Forms.RadioButton rbWVNormal;
		private System.Windows.Forms.RadioButton rbWVFast;
		private System.Windows.Forms.CheckBox chkWVExtraMode;
		private System.Windows.Forms.CheckBox chkAutoCorrectFilenames;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.CheckBox chkArAddCRCs;
		private System.Windows.Forms.NumericUpDown numericFLACCompressionLevel;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.NumericUpDown numFixWhenPercent;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.NumericUpDown numFixWhenConfidence;
		private System.Windows.Forms.GroupBox grpAudioFilenames;
		private System.Windows.Forms.CheckBox chkKeepOriginalFilenames;
		private System.Windows.Forms.TextBox txtSpecialExceptions;
		private System.Windows.Forms.CheckBox chkRemoveSpecial;
		private System.Windows.Forms.CheckBox chkReplaceSpaces;
		private System.Windows.Forms.TextBox txtTrackFilenameFormat;
		private System.Windows.Forms.Label lblTrackFilenameFormat;
		private System.Windows.Forms.Label lblSingleFilenameFormat;
		private System.Windows.Forms.TextBox txtSingleFilenameFormat;
		private System.Windows.Forms.CheckBox chkArNoUnverifiedAudio;
		private System.Windows.Forms.CheckBox chkArSaveLog;
		private System.Windows.Forms.NumericUpDown numWVExtraMode;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.NumericUpDown numEncodeWhenConfidence;
		private System.Windows.Forms.NumericUpDown numEncodeWhenPercent;
		private System.Windows.Forms.CheckBox chkArFixOffset;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.CheckBox chkEmbedLog;
		private System.Windows.Forms.CheckBox chkFillUpCUE;
		private System.Windows.Forms.CheckBox chkFilenamesANSISafe;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.RadioButton rbAPEinsane;
		private System.Windows.Forms.RadioButton rbAPEextrahigh;
		private System.Windows.Forms.RadioButton rbAPEhigh;
		private System.Windows.Forms.RadioButton rbAPEnormal;
		private System.Windows.Forms.RadioButton rbAPEfast;

	}
}