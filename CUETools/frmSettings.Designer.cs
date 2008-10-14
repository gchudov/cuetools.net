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
			this.grpGeneral = new System.Windows.Forms.GroupBox();
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
			this.grpAudioFilenames = new System.Windows.Forms.GroupBox();
			this.chkKeepOriginalFilenames = new System.Windows.Forms.CheckBox();
			this.txtSpecialExceptions = new System.Windows.Forms.TextBox();
			this.chkRemoveSpecial = new System.Windows.Forms.CheckBox();
			this.chkReplaceSpaces = new System.Windows.Forms.CheckBox();
			this.txtTrackFilenameFormat = new System.Windows.Forms.TextBox();
			this.lblTrackFilenameFormat = new System.Windows.Forms.Label();
			this.lblSingleFilenameFormat = new System.Windows.Forms.Label();
			this.txtSingleFilenameFormat = new System.Windows.Forms.TextBox();
			this.chkEmbedLog = new System.Windows.Forms.CheckBox();
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
			this.SuspendLayout();
			// 
			// btnCancel
			// 
			btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			btnCancel.Location = new System.Drawing.Point(267, 373);
			btnCancel.Name = "btnCancel";
			btnCancel.Size = new System.Drawing.Size(73, 23);
			btnCancel.TabIndex = 5;
			btnCancel.Text = "Cancel";
			btnCancel.UseVisualStyleBackColor = true;
			// 
			// grpGeneral
			// 
			this.grpGeneral.Controls.Add(this.chkEmbedLog);
			this.grpGeneral.Controls.Add(this.numericWriteOffset);
			this.grpGeneral.Controls.Add(this.chkAutoCorrectFilenames);
			this.grpGeneral.Controls.Add(this.chkPreserveHTOA);
			this.grpGeneral.Controls.Add(this.lblWriteOffset);
			this.grpGeneral.Location = new System.Drawing.Point(8, 4);
			this.grpGeneral.Name = "grpGeneral";
			this.grpGeneral.Size = new System.Drawing.Size(246, 144);
			this.grpGeneral.TabIndex = 0;
			this.grpGeneral.TabStop = false;
			this.grpGeneral.Text = "General";
			// 
			// numericWriteOffset
			// 
			this.numericWriteOffset.Location = new System.Drawing.Point(133, 20);
			this.numericWriteOffset.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
			this.numericWriteOffset.Minimum = new decimal(new int[] {
            5000,
            0,
            0,
            -2147483648});
			this.numericWriteOffset.Name = "numericWriteOffset";
			this.numericWriteOffset.Size = new System.Drawing.Size(62, 21);
			this.numericWriteOffset.TabIndex = 5;
			this.numericWriteOffset.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// chkAutoCorrectFilenames
			// 
			this.chkAutoCorrectFilenames.Location = new System.Drawing.Point(12, 61);
			this.chkAutoCorrectFilenames.Name = "chkAutoCorrectFilenames";
			this.chkAutoCorrectFilenames.Size = new System.Drawing.Size(232, 17);
			this.chkAutoCorrectFilenames.TabIndex = 3;
			this.chkAutoCorrectFilenames.Text = "Locate audio files if missing";
			this.toolTip1.SetToolTip(this.chkAutoCorrectFilenames, "Preprocess with filename corrector if unable to locate audio files");
			this.chkAutoCorrectFilenames.UseVisualStyleBackColor = true;
			// 
			// chkPreserveHTOA
			// 
			this.chkPreserveHTOA.AutoSize = true;
			this.chkPreserveHTOA.Location = new System.Drawing.Point(12, 44);
			this.chkPreserveHTOA.Name = "chkPreserveHTOA";
			this.chkPreserveHTOA.Size = new System.Drawing.Size(229, 17);
			this.chkPreserveHTOA.TabIndex = 2;
			this.chkPreserveHTOA.Text = "Preserve HTOA for gaps appended output";
			this.chkPreserveHTOA.UseVisualStyleBackColor = true;
			// 
			// lblWriteOffset
			// 
			this.lblWriteOffset.AutoSize = true;
			this.lblWriteOffset.Location = new System.Drawing.Point(9, 23);
			this.lblWriteOffset.Name = "lblWriteOffset";
			this.lblWriteOffset.Size = new System.Drawing.Size(118, 13);
			this.lblWriteOffset.TabIndex = 0;
			this.lblWriteOffset.Text = "Write offset (samples):";
			// 
			// grpFLAC
			// 
			this.grpFLAC.Controls.Add(this.numericFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.lblFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.chkFLACVerify);
			this.grpFLAC.Location = new System.Drawing.Point(267, 282);
			this.grpFLAC.Name = "grpFLAC";
			this.grpFLAC.Size = new System.Drawing.Size(259, 48);
			this.grpFLAC.TabIndex = 1;
			this.grpFLAC.TabStop = false;
			this.grpFLAC.Text = "FLAC";
			// 
			// numericFLACCompressionLevel
			// 
			this.numericFLACCompressionLevel.Location = new System.Drawing.Point(213, 15);
			this.numericFLACCompressionLevel.Maximum = new decimal(new int[] {
            8,
            0,
            0,
            0});
			this.numericFLACCompressionLevel.Name = "numericFLACCompressionLevel";
			this.numericFLACCompressionLevel.Size = new System.Drawing.Size(35, 21);
			this.numericFLACCompressionLevel.TabIndex = 4;
			this.numericFLACCompressionLevel.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.numericFLACCompressionLevel.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
			// 
			// lblFLACCompressionLevel
			// 
			this.lblFLACCompressionLevel.AutoSize = true;
			this.lblFLACCompressionLevel.Location = new System.Drawing.Point(99, 17);
			this.lblFLACCompressionLevel.Name = "lblFLACCompressionLevel";
			this.lblFLACCompressionLevel.Size = new System.Drawing.Size(97, 13);
			this.lblFLACCompressionLevel.TabIndex = 0;
			this.lblFLACCompressionLevel.Text = "Compression level:";
			// 
			// chkFLACVerify
			// 
			this.chkFLACVerify.AutoSize = true;
			this.chkFLACVerify.Location = new System.Drawing.Point(12, 16);
			this.chkFLACVerify.Name = "chkFLACVerify";
			this.chkFLACVerify.Size = new System.Drawing.Size(54, 17);
			this.chkFLACVerify.TabIndex = 2;
			this.chkFLACVerify.Text = "Verify";
			this.chkFLACVerify.UseVisualStyleBackColor = true;
			// 
			// btnOK
			// 
			this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.btnOK.Location = new System.Drawing.Point(185, 373);
			this.btnOK.Name = "btnOK";
			this.btnOK.Size = new System.Drawing.Size(73, 23);
			this.btnOK.TabIndex = 3;
			this.btnOK.Text = "OK";
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
			this.grpWavPack.Location = new System.Drawing.Point(267, 182);
			this.grpWavPack.Name = "grpWavPack";
			this.grpWavPack.Size = new System.Drawing.Size(259, 94);
			this.grpWavPack.TabIndex = 2;
			this.grpWavPack.TabStop = false;
			this.grpWavPack.Text = "WavPack";
			// 
			// numWVExtraMode
			// 
			this.numWVExtraMode.Location = new System.Drawing.Point(212, 19);
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
			this.numWVExtraMode.Size = new System.Drawing.Size(29, 21);
			this.numWVExtraMode.TabIndex = 5;
			this.numWVExtraMode.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkWVExtraMode
			// 
			this.chkWVExtraMode.AutoSize = true;
			this.chkWVExtraMode.Location = new System.Drawing.Point(94, 20);
			this.chkWVExtraMode.Name = "chkWVExtraMode";
			this.chkWVExtraMode.Size = new System.Drawing.Size(112, 17);
			this.chkWVExtraMode.TabIndex = 4;
			this.chkWVExtraMode.Text = "Extra mode (1-6):";
			this.chkWVExtraMode.UseVisualStyleBackColor = true;
			this.chkWVExtraMode.CheckedChanged += new System.EventHandler(this.chkWVExtraMode_CheckedChanged);
			// 
			// rbWVVeryHigh
			// 
			this.rbWVVeryHigh.AutoSize = true;
			this.rbWVVeryHigh.Location = new System.Drawing.Point(13, 70);
			this.rbWVVeryHigh.Name = "rbWVVeryHigh";
			this.rbWVVeryHigh.Size = new System.Drawing.Size(71, 17);
			this.rbWVVeryHigh.TabIndex = 3;
			this.rbWVVeryHigh.Text = "Very High";
			this.rbWVVeryHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVHigh
			// 
			this.rbWVHigh.AutoSize = true;
			this.rbWVHigh.Location = new System.Drawing.Point(13, 53);
			this.rbWVHigh.Name = "rbWVHigh";
			this.rbWVHigh.Size = new System.Drawing.Size(46, 17);
			this.rbWVHigh.TabIndex = 1;
			this.rbWVHigh.Text = "High";
			this.rbWVHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVNormal
			// 
			this.rbWVNormal.AutoSize = true;
			this.rbWVNormal.Checked = true;
			this.rbWVNormal.Location = new System.Drawing.Point(13, 36);
			this.rbWVNormal.Name = "rbWVNormal";
			this.rbWVNormal.Size = new System.Drawing.Size(58, 17);
			this.rbWVNormal.TabIndex = 2;
			this.rbWVNormal.TabStop = true;
			this.rbWVNormal.Text = "Normal";
			this.rbWVNormal.UseVisualStyleBackColor = true;
			// 
			// rbWVFast
			// 
			this.rbWVFast.AutoSize = true;
			this.rbWVFast.Location = new System.Drawing.Point(13, 19);
			this.rbWVFast.Name = "rbWVFast";
			this.rbWVFast.Size = new System.Drawing.Size(46, 17);
			this.rbWVFast.TabIndex = 0;
			this.rbWVFast.Text = "Fast";
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
			this.groupBox1.Location = new System.Drawing.Point(267, 4);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(259, 172);
			this.groupBox1.TabIndex = 4;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "AccurateRip";
			// 
			// chkArFixOffset
			// 
			this.chkArFixOffset.AutoSize = true;
			this.chkArFixOffset.Checked = true;
			this.chkArFixOffset.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArFixOffset.Location = new System.Drawing.Point(12, 105);
			this.chkArFixOffset.Name = "chkArFixOffset";
			this.chkArFixOffset.Size = new System.Drawing.Size(81, 17);
			this.chkArFixOffset.TabIndex = 12;
			this.chkArFixOffset.Text = "Fix offset if";
			this.chkArFixOffset.UseVisualStyleBackColor = true;
			this.chkArFixOffset.CheckedChanged += new System.EventHandler(this.chkArFixOffset_CheckedChanged);
			// 
			// label4
			// 
			this.label4.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point(41, 68);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(121, 13);
			this.label4.TabIndex = 11;
			this.label4.Text = "% of verified tracks >=";
			// 
			// numEncodeWhenPercent
			// 
			this.numEncodeWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			this.numEncodeWhenPercent.Location = new System.Drawing.Point(168, 66);
			this.numEncodeWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenPercent.Name = "numEncodeWhenPercent";
			this.numEncodeWhenPercent.Size = new System.Drawing.Size(38, 21);
			this.numEncodeWhenPercent.TabIndex = 10;
			this.numEncodeWhenPercent.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
			// 
			// label3
			// 
			this.label3.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(62, 89);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(101, 13);
			this.label3.TabIndex = 9;
			this.label3.Text = "with confidence >=";
			// 
			// numEncodeWhenConfidence
			// 
			this.numEncodeWhenConfidence.Location = new System.Drawing.Point(168, 87);
			this.numEncodeWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenConfidence.Name = "numEncodeWhenConfidence";
			this.numEncodeWhenConfidence.Size = new System.Drawing.Size(38, 21);
			this.numEncodeWhenConfidence.TabIndex = 8;
			this.numEncodeWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkArNoUnverifiedAudio
			// 
			this.chkArNoUnverifiedAudio.AutoSize = true;
			this.chkArNoUnverifiedAudio.Checked = true;
			this.chkArNoUnverifiedAudio.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArNoUnverifiedAudio.Location = new System.Drawing.Point(12, 44);
			this.chkArNoUnverifiedAudio.Name = "chkArNoUnverifiedAudio";
			this.chkArNoUnverifiedAudio.Size = new System.Drawing.Size(93, 17);
			this.chkArNoUnverifiedAudio.TabIndex = 7;
			this.chkArNoUnverifiedAudio.Text = "Encode only if";
			this.chkArNoUnverifiedAudio.UseVisualStyleBackColor = true;
			this.chkArNoUnverifiedAudio.CheckedChanged += new System.EventHandler(this.chkArNoUnverifiedAudio_CheckedChanged);
			// 
			// chkArSaveLog
			// 
			this.chkArSaveLog.AutoSize = true;
			this.chkArSaveLog.Checked = true;
			this.chkArSaveLog.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArSaveLog.Location = new System.Drawing.Point(94, 21);
			this.chkArSaveLog.Name = "chkArSaveLog";
			this.chkArSaveLog.Size = new System.Drawing.Size(69, 17);
			this.chkArSaveLog.TabIndex = 6;
			this.chkArSaveLog.Text = "Write log";
			this.chkArSaveLog.UseVisualStyleBackColor = true;
			// 
			// label2
			// 
			this.label2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(62, 146);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(101, 13);
			this.label2.TabIndex = 5;
			this.label2.Text = "with confidence >=";
			// 
			// numFixWhenConfidence
			// 
			this.numFixWhenConfidence.Location = new System.Drawing.Point(168, 144);
			this.numFixWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenConfidence.Name = "numFixWhenConfidence";
			this.numFixWhenConfidence.Size = new System.Drawing.Size(37, 21);
			this.numFixWhenConfidence.TabIndex = 4;
			this.numFixWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// label1
			// 
			this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(42, 125);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(121, 13);
			this.label1.TabIndex = 3;
			this.label1.Text = "% of verified tracks >=";
			// 
			// numFixWhenPercent
			// 
			this.numFixWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			this.numFixWhenPercent.Location = new System.Drawing.Point(168, 123);
			this.numFixWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenPercent.Name = "numFixWhenPercent";
			this.numFixWhenPercent.Size = new System.Drawing.Size(38, 21);
			this.numFixWhenPercent.TabIndex = 2;
			this.numFixWhenPercent.Value = new decimal(new int[] {
            51,
            0,
            0,
            0});
			// 
			// chkArAddCRCs
			// 
			this.chkArAddCRCs.AutoSize = true;
			this.chkArAddCRCs.Checked = true;
			this.chkArAddCRCs.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArAddCRCs.Location = new System.Drawing.Point(12, 21);
			this.chkArAddCRCs.Name = "chkArAddCRCs";
			this.chkArAddCRCs.Size = new System.Drawing.Size(76, 17);
			this.chkArAddCRCs.TabIndex = 1;
			this.chkArAddCRCs.Text = "Write tags";
			this.toolTip1.SetToolTip(this.chkArAddCRCs, "When using \"apply offset\" AccurateRip mode, also add ACCURATERIPCOUNT tag to flac" +
					" files. You can set up foobar2000 to show this value, and see if your music was " +
					"ripped correctly or how popular it is.");
			this.chkArAddCRCs.UseVisualStyleBackColor = true;
			// 
			// toolTip1
			// 
			this.toolTip1.AutoPopDelay = 15000;
			this.toolTip1.InitialDelay = 500;
			this.toolTip1.ReshowDelay = 100;
			// 
			// grpAudioFilenames
			// 
			this.grpAudioFilenames.Controls.Add(this.chkKeepOriginalFilenames);
			this.grpAudioFilenames.Controls.Add(this.txtSpecialExceptions);
			this.grpAudioFilenames.Controls.Add(this.chkRemoveSpecial);
			this.grpAudioFilenames.Controls.Add(this.chkReplaceSpaces);
			this.grpAudioFilenames.Controls.Add(this.txtTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblSingleFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.txtSingleFilenameFormat);
			this.grpAudioFilenames.Location = new System.Drawing.Point(12, 150);
			this.grpAudioFilenames.Name = "grpAudioFilenames";
			this.grpAudioFilenames.Size = new System.Drawing.Size(246, 180);
			this.grpAudioFilenames.TabIndex = 6;
			this.grpAudioFilenames.TabStop = false;
			this.grpAudioFilenames.Text = "Audio Filenames";
			// 
			// chkKeepOriginalFilenames
			// 
			this.chkKeepOriginalFilenames.AutoSize = true;
			this.chkKeepOriginalFilenames.Checked = true;
			this.chkKeepOriginalFilenames.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkKeepOriginalFilenames.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.chkKeepOriginalFilenames.Location = new System.Drawing.Point(12, 20);
			this.chkKeepOriginalFilenames.Name = "chkKeepOriginalFilenames";
			this.chkKeepOriginalFilenames.Size = new System.Drawing.Size(135, 17);
			this.chkKeepOriginalFilenames.TabIndex = 0;
			this.chkKeepOriginalFilenames.Text = "Keep original filenames";
			this.chkKeepOriginalFilenames.UseVisualStyleBackColor = true;
			// 
			// txtSpecialExceptions
			// 
			this.txtSpecialExceptions.Location = new System.Drawing.Point(92, 123);
			this.txtSpecialExceptions.Name = "txtSpecialExceptions";
			this.txtSpecialExceptions.Size = new System.Drawing.Size(136, 21);
			this.txtSpecialExceptions.TabIndex = 6;
			this.txtSpecialExceptions.Text = "-()";
			// 
			// chkRemoveSpecial
			// 
			this.chkRemoveSpecial.AutoSize = true;
			this.chkRemoveSpecial.Checked = true;
			this.chkRemoveSpecial.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkRemoveSpecial.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.chkRemoveSpecial.Location = new System.Drawing.Point(12, 100);
			this.chkRemoveSpecial.Name = "chkRemoveSpecial";
			this.chkRemoveSpecial.Size = new System.Drawing.Size(194, 17);
			this.chkRemoveSpecial.TabIndex = 5;
			this.chkRemoveSpecial.Text = "Remove special characters except:";
			this.chkRemoveSpecial.UseVisualStyleBackColor = true;
			// 
			// chkReplaceSpaces
			// 
			this.chkReplaceSpaces.AutoSize = true;
			this.chkReplaceSpaces.Checked = true;
			this.chkReplaceSpaces.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkReplaceSpaces.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.chkReplaceSpaces.Location = new System.Drawing.Point(12, 150);
			this.chkReplaceSpaces.Name = "chkReplaceSpaces";
			this.chkReplaceSpaces.Size = new System.Drawing.Size(185, 17);
			this.chkReplaceSpaces.TabIndex = 7;
			this.chkReplaceSpaces.Text = "Replace spaces with underscores";
			this.chkReplaceSpaces.UseVisualStyleBackColor = true;
			// 
			// txtTrackFilenameFormat
			// 
			this.txtTrackFilenameFormat.Location = new System.Drawing.Point(92, 72);
			this.txtTrackFilenameFormat.Name = "txtTrackFilenameFormat";
			this.txtTrackFilenameFormat.Size = new System.Drawing.Size(136, 21);
			this.txtTrackFilenameFormat.TabIndex = 4;
			this.txtTrackFilenameFormat.Text = "%N-%A-%T";
			// 
			// lblTrackFilenameFormat
			// 
			this.lblTrackFilenameFormat.AutoSize = true;
			this.lblTrackFilenameFormat.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.lblTrackFilenameFormat.Location = new System.Drawing.Point(10, 76);
			this.lblTrackFilenameFormat.Name = "lblTrackFilenameFormat";
			this.lblTrackFilenameFormat.Size = new System.Drawing.Size(72, 13);
			this.lblTrackFilenameFormat.TabIndex = 3;
			this.lblTrackFilenameFormat.Text = "Track format:";
			// 
			// lblSingleFilenameFormat
			// 
			this.lblSingleFilenameFormat.AutoSize = true;
			this.lblSingleFilenameFormat.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.lblSingleFilenameFormat.Location = new System.Drawing.Point(10, 48);
			this.lblSingleFilenameFormat.Name = "lblSingleFilenameFormat";
			this.lblSingleFilenameFormat.Size = new System.Drawing.Size(74, 13);
			this.lblSingleFilenameFormat.TabIndex = 1;
			this.lblSingleFilenameFormat.Text = "Single format:";
			// 
			// txtSingleFilenameFormat
			// 
			this.txtSingleFilenameFormat.Location = new System.Drawing.Point(92, 44);
			this.txtSingleFilenameFormat.Name = "txtSingleFilenameFormat";
			this.txtSingleFilenameFormat.Size = new System.Drawing.Size(136, 21);
			this.txtSingleFilenameFormat.TabIndex = 2;
			this.txtSingleFilenameFormat.Text = "%F";
			// 
			// chkEmbedLog
			// 
			this.chkEmbedLog.AutoSize = true;
			this.chkEmbedLog.Location = new System.Drawing.Point(12, 78);
			this.chkEmbedLog.Name = "chkEmbedLog";
			this.chkEmbedLog.Size = new System.Drawing.Size(134, 17);
			this.chkEmbedLog.TabIndex = 6;
			this.chkEmbedLog.Text = "Embed log file as a tag";
			this.toolTip1.SetToolTip(this.chkEmbedLog, "File should be in the same directory as source file and have a .log extension");
			this.chkEmbedLog.UseVisualStyleBackColor = true;
			// 
			// frmSettings
			// 
			this.AcceptButton = this.btnOK;
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = btnCancel;
			this.ClientSize = new System.Drawing.Size(540, 430);
			this.Controls.Add(this.grpAudioFilenames);
			this.Controls.Add(btnCancel);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.grpWavPack);
			this.Controls.Add(this.btnOK);
			this.Controls.Add(this.grpFLAC);
			this.Controls.Add(this.grpGeneral);
			this.Font = new System.Drawing.Font("Tahoma", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "frmSettings";
			this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
			this.Text = "Advanced Settings";
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

	}
}