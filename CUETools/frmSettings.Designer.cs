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
			this.labelLanguage = new System.Windows.Forms.Label();
			this.comboLanguage = new System.Windows.Forms.ComboBox();
			this.chkSingleInstance = new System.Windows.Forms.CheckBox();
			this.chkOverwriteTags = new System.Windows.Forms.CheckBox();
			this.chkExtractLog = new System.Windows.Forms.CheckBox();
			this.chkReducePriority = new System.Windows.Forms.CheckBox();
			this.chkTruncateExtra4206Samples = new System.Windows.Forms.CheckBox();
			this.chkCreateCUEFileWhenEmbedded = new System.Windows.Forms.CheckBox();
			this.chkCreateM3U = new System.Windows.Forms.CheckBox();
			this.chkFillUpCUE = new System.Windows.Forms.CheckBox();
			this.chkEmbedLog = new System.Windows.Forms.CheckBox();
			this.chkAutoCorrectFilenames = new System.Windows.Forms.CheckBox();
			this.chkPreserveHTOA = new System.Windows.Forms.CheckBox();
			this.numericFLACCompressionLevel = new System.Windows.Forms.NumericUpDown();
			this.lblFLACCompressionLevel = new System.Windows.Forms.Label();
			this.chkFLACVerify = new System.Windows.Forms.CheckBox();
			this.btnOK = new System.Windows.Forms.Button();
			this.chkWVStoreMD5 = new System.Windows.Forms.CheckBox();
			this.numWVExtraMode = new System.Windows.Forms.NumericUpDown();
			this.chkWVExtraMode = new System.Windows.Forms.CheckBox();
			this.rbWVVeryHigh = new System.Windows.Forms.RadioButton();
			this.rbWVHigh = new System.Windows.Forms.RadioButton();
			this.rbWVNormal = new System.Windows.Forms.RadioButton();
			this.rbWVFast = new System.Windows.Forms.RadioButton();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.chkEncodeWhenZeroOffset = new System.Windows.Forms.CheckBox();
			this.chkArFixOffset = new System.Windows.Forms.CheckBox();
			this.chkWriteArLogOnConvert = new System.Windows.Forms.CheckBox();
			this.chkWriteArTagsOnConvert = new System.Windows.Forms.CheckBox();
			this.numEncodeWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.labelEncodeWhenConfidence = new System.Windows.Forms.Label();
			this.numEncodeWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.chkArNoUnverifiedAudio = new System.Windows.Forms.CheckBox();
			this.labelFixWhenConfidence = new System.Windows.Forms.Label();
			this.numFixWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.numFixWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.chkFilenamesANSISafe = new System.Windows.Forms.CheckBox();
			this.chkWriteARTagsOnVerify = new System.Windows.Forms.CheckBox();
			this.chkHDCDDecode = new System.Windows.Forms.CheckBox();
			this.chkHDCDStopLooking = new System.Windows.Forms.CheckBox();
			this.chkHDCD24bit = new System.Windows.Forms.CheckBox();
			this.chkHDCDLW16 = new System.Windows.Forms.CheckBox();
			this.grpAudioFilenames = new System.Windows.Forms.GroupBox();
			this.chkKeepOriginalFilenames = new System.Windows.Forms.CheckBox();
			this.txtSpecialExceptions = new System.Windows.Forms.TextBox();
			this.chkRemoveSpecial = new System.Windows.Forms.CheckBox();
			this.chkReplaceSpaces = new System.Windows.Forms.CheckBox();
			this.txtTrackFilenameFormat = new System.Windows.Forms.TextBox();
			this.lblTrackFilenameFormat = new System.Windows.Forms.Label();
			this.lblSingleFilenameFormat = new System.Windows.Forms.Label();
			this.txtSingleFilenameFormat = new System.Windows.Forms.TextBox();
			this.rbAPEinsane = new System.Windows.Forms.RadioButton();
			this.rbAPEextrahigh = new System.Windows.Forms.RadioButton();
			this.rbAPEhigh = new System.Windows.Forms.RadioButton();
			this.rbAPEnormal = new System.Windows.Forms.RadioButton();
			this.rbAPEfast = new System.Windows.Forms.RadioButton();
			this.tabControl1 = new System.Windows.Forms.TabControl();
			this.tabPage1 = new System.Windows.Forms.TabPage();
			this.tabPage2 = new System.Windows.Forms.TabPage();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.chkWriteARLogOnVerify = new System.Windows.Forms.CheckBox();
			this.tabPage3 = new System.Windows.Forms.TabPage();
			this.tabControl2 = new System.Windows.Forms.TabControl();
			this.tabPage5 = new System.Windows.Forms.TabPage();
			this.tabPage6 = new System.Windows.Forms.TabPage();
			this.tabPage7 = new System.Windows.Forms.TabPage();
			this.tabPage8 = new System.Windows.Forms.TabPage();
			this.label1 = new System.Windows.Forms.Label();
			this.numericLossyWAVQuality = new System.Windows.Forms.NumericUpDown();
			this.tabPage9 = new System.Windows.Forms.TabPage();
			this.chkUDC1ID3v2 = new System.Windows.Forms.CheckBox();
			this.chkUDC1APEv2 = new System.Windows.Forms.CheckBox();
			this.label6 = new System.Windows.Forms.Label();
			this.label5 = new System.Windows.Forms.Label();
			this.textUDC1EncParams = new System.Windows.Forms.TextBox();
			this.textUDC1Encoder = new System.Windows.Forms.TextBox();
			this.textUDC1Params = new System.Windows.Forms.TextBox();
			this.textUDC1Decoder = new System.Windows.Forms.TextBox();
			this.textUDC1Extension = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.label3 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.imageList1 = new System.Windows.Forms.ImageList(this.components);
			this.tabPage4 = new System.Windows.Forms.TabPage();
			this.grpHDCD = new System.Windows.Forms.GroupBox();
			this.chkHDCDDetect = new System.Windows.Forms.CheckBox();
			btnCancel = new System.Windows.Forms.Button();
			this.grpGeneral.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericFLACCompressionLevel)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numWVExtraMode)).BeginInit();
			this.groupBox1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).BeginInit();
			this.grpAudioFilenames.SuspendLayout();
			this.tabControl1.SuspendLayout();
			this.tabPage1.SuspendLayout();
			this.tabPage2.SuspendLayout();
			this.groupBox3.SuspendLayout();
			this.tabPage3.SuspendLayout();
			this.tabControl2.SuspendLayout();
			this.tabPage5.SuspendLayout();
			this.tabPage6.SuspendLayout();
			this.tabPage7.SuspendLayout();
			this.tabPage8.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericLossyWAVQuality)).BeginInit();
			this.tabPage9.SuspendLayout();
			this.tabPage4.SuspendLayout();
			this.grpHDCD.SuspendLayout();
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
			this.grpGeneral.Controls.Add(this.labelLanguage);
			this.grpGeneral.Controls.Add(this.comboLanguage);
			this.grpGeneral.Controls.Add(this.chkSingleInstance);
			this.grpGeneral.Controls.Add(this.chkOverwriteTags);
			this.grpGeneral.Controls.Add(this.chkExtractLog);
			this.grpGeneral.Controls.Add(this.chkReducePriority);
			this.grpGeneral.Controls.Add(this.chkTruncateExtra4206Samples);
			this.grpGeneral.Controls.Add(this.chkCreateCUEFileWhenEmbedded);
			this.grpGeneral.Controls.Add(this.chkCreateM3U);
			this.grpGeneral.Controls.Add(this.chkFillUpCUE);
			this.grpGeneral.Controls.Add(this.chkEmbedLog);
			this.grpGeneral.Controls.Add(this.chkAutoCorrectFilenames);
			this.grpGeneral.Controls.Add(this.chkPreserveHTOA);
			resources.ApplyResources(this.grpGeneral, "grpGeneral");
			this.grpGeneral.Name = "grpGeneral";
			this.grpGeneral.TabStop = false;
			// 
			// labelLanguage
			// 
			resources.ApplyResources(this.labelLanguage, "labelLanguage");
			this.labelLanguage.Name = "labelLanguage";
			// 
			// comboLanguage
			// 
			this.comboLanguage.DisplayMember = "EnglishName";
			this.comboLanguage.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboLanguage.FormattingEnabled = true;
			resources.ApplyResources(this.comboLanguage, "comboLanguage");
			this.comboLanguage.Name = "comboLanguage";
			// 
			// chkSingleInstance
			// 
			resources.ApplyResources(this.chkSingleInstance, "chkSingleInstance");
			this.chkSingleInstance.Name = "chkSingleInstance";
			this.chkSingleInstance.UseVisualStyleBackColor = true;
			// 
			// chkOverwriteTags
			// 
			resources.ApplyResources(this.chkOverwriteTags, "chkOverwriteTags");
			this.chkOverwriteTags.Name = "chkOverwriteTags";
			this.chkOverwriteTags.UseVisualStyleBackColor = true;
			// 
			// chkExtractLog
			// 
			resources.ApplyResources(this.chkExtractLog, "chkExtractLog");
			this.chkExtractLog.Name = "chkExtractLog";
			this.chkExtractLog.UseVisualStyleBackColor = true;
			// 
			// chkReducePriority
			// 
			resources.ApplyResources(this.chkReducePriority, "chkReducePriority");
			this.chkReducePriority.Name = "chkReducePriority";
			this.chkReducePriority.UseVisualStyleBackColor = true;
			// 
			// chkTruncateExtra4206Samples
			// 
			resources.ApplyResources(this.chkTruncateExtra4206Samples, "chkTruncateExtra4206Samples");
			this.chkTruncateExtra4206Samples.Name = "chkTruncateExtra4206Samples";
			this.toolTip1.SetToolTip(this.chkTruncateExtra4206Samples, resources.GetString("chkTruncateExtra4206Samples.ToolTip"));
			this.chkTruncateExtra4206Samples.UseVisualStyleBackColor = true;
			// 
			// chkCreateCUEFileWhenEmbedded
			// 
			resources.ApplyResources(this.chkCreateCUEFileWhenEmbedded, "chkCreateCUEFileWhenEmbedded");
			this.chkCreateCUEFileWhenEmbedded.Name = "chkCreateCUEFileWhenEmbedded";
			this.chkCreateCUEFileWhenEmbedded.UseVisualStyleBackColor = true;
			// 
			// chkCreateM3U
			// 
			resources.ApplyResources(this.chkCreateM3U, "chkCreateM3U");
			this.chkCreateM3U.Name = "chkCreateM3U";
			this.chkCreateM3U.UseVisualStyleBackColor = true;
			// 
			// chkFillUpCUE
			// 
			resources.ApplyResources(this.chkFillUpCUE, "chkFillUpCUE");
			this.chkFillUpCUE.Name = "chkFillUpCUE";
			this.chkFillUpCUE.UseVisualStyleBackColor = true;
			this.chkFillUpCUE.CheckedChanged += new System.EventHandler(this.chkFillUpCUE_CheckedChanged);
			// 
			// chkEmbedLog
			// 
			resources.ApplyResources(this.chkEmbedLog, "chkEmbedLog");
			this.chkEmbedLog.Name = "chkEmbedLog";
			this.toolTip1.SetToolTip(this.chkEmbedLog, resources.GetString("chkEmbedLog.ToolTip"));
			this.chkEmbedLog.UseVisualStyleBackColor = true;
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
			// chkWVStoreMD5
			// 
			resources.ApplyResources(this.chkWVStoreMD5, "chkWVStoreMD5");
			this.chkWVStoreMD5.Name = "chkWVStoreMD5";
			this.chkWVStoreMD5.UseVisualStyleBackColor = true;
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
			this.groupBox1.Controls.Add(this.chkEncodeWhenZeroOffset);
			this.groupBox1.Controls.Add(this.chkArFixOffset);
			this.groupBox1.Controls.Add(this.chkWriteArLogOnConvert);
			this.groupBox1.Controls.Add(this.chkWriteArTagsOnConvert);
			this.groupBox1.Controls.Add(this.numEncodeWhenPercent);
			this.groupBox1.Controls.Add(this.labelEncodeWhenConfidence);
			this.groupBox1.Controls.Add(this.numEncodeWhenConfidence);
			this.groupBox1.Controls.Add(this.chkArNoUnverifiedAudio);
			this.groupBox1.Controls.Add(this.labelFixWhenConfidence);
			this.groupBox1.Controls.Add(this.numFixWhenConfidence);
			this.groupBox1.Controls.Add(this.numFixWhenPercent);
			resources.ApplyResources(this.groupBox1, "groupBox1");
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.TabStop = false;
			// 
			// chkEncodeWhenZeroOffset
			// 
			resources.ApplyResources(this.chkEncodeWhenZeroOffset, "chkEncodeWhenZeroOffset");
			this.chkEncodeWhenZeroOffset.Name = "chkEncodeWhenZeroOffset";
			this.chkEncodeWhenZeroOffset.UseVisualStyleBackColor = true;
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
			// chkWriteArLogOnConvert
			// 
			resources.ApplyResources(this.chkWriteArLogOnConvert, "chkWriteArLogOnConvert");
			this.chkWriteArLogOnConvert.Checked = true;
			this.chkWriteArLogOnConvert.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteArLogOnConvert.Name = "chkWriteArLogOnConvert";
			this.chkWriteArLogOnConvert.UseVisualStyleBackColor = true;
			// 
			// chkWriteArTagsOnConvert
			// 
			resources.ApplyResources(this.chkWriteArTagsOnConvert, "chkWriteArTagsOnConvert");
			this.chkWriteArTagsOnConvert.Checked = true;
			this.chkWriteArTagsOnConvert.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteArTagsOnConvert.Name = "chkWriteArTagsOnConvert";
			this.toolTip1.SetToolTip(this.chkWriteArTagsOnConvert, resources.GetString("chkWriteArTagsOnConvert.ToolTip"));
			this.chkWriteArTagsOnConvert.UseVisualStyleBackColor = true;
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
			// labelEncodeWhenConfidence
			// 
			resources.ApplyResources(this.labelEncodeWhenConfidence, "labelEncodeWhenConfidence");
			this.labelEncodeWhenConfidence.Name = "labelEncodeWhenConfidence";
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
			// labelFixWhenConfidence
			// 
			resources.ApplyResources(this.labelFixWhenConfidence, "labelFixWhenConfidence");
			this.labelFixWhenConfidence.Name = "labelFixWhenConfidence";
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
			this.chkFilenamesANSISafe.CheckedChanged += new System.EventHandler(this.chkFilenamesANSISafe_CheckedChanged);
			// 
			// chkWriteARTagsOnVerify
			// 
			resources.ApplyResources(this.chkWriteARTagsOnVerify, "chkWriteARTagsOnVerify");
			this.chkWriteARTagsOnVerify.Checked = true;
			this.chkWriteARTagsOnVerify.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteARTagsOnVerify.Name = "chkWriteARTagsOnVerify";
			this.toolTip1.SetToolTip(this.chkWriteARTagsOnVerify, resources.GetString("chkWriteARTagsOnVerify.ToolTip"));
			this.chkWriteARTagsOnVerify.UseVisualStyleBackColor = true;
			// 
			// chkHDCDDecode
			// 
			resources.ApplyResources(this.chkHDCDDecode, "chkHDCDDecode");
			this.chkHDCDDecode.Name = "chkHDCDDecode";
			this.toolTip1.SetToolTip(this.chkHDCDDecode, resources.GetString("chkHDCDDecode.ToolTip"));
			this.chkHDCDDecode.UseVisualStyleBackColor = true;
			this.chkHDCDDecode.CheckedChanged += new System.EventHandler(this.chkHDCDDecode_CheckedChanged);
			// 
			// chkHDCDStopLooking
			// 
			resources.ApplyResources(this.chkHDCDStopLooking, "chkHDCDStopLooking");
			this.chkHDCDStopLooking.Name = "chkHDCDStopLooking";
			this.toolTip1.SetToolTip(this.chkHDCDStopLooking, resources.GetString("chkHDCDStopLooking.ToolTip"));
			this.chkHDCDStopLooking.UseVisualStyleBackColor = true;
			// 
			// chkHDCD24bit
			// 
			resources.ApplyResources(this.chkHDCD24bit, "chkHDCD24bit");
			this.chkHDCD24bit.Name = "chkHDCD24bit";
			this.toolTip1.SetToolTip(this.chkHDCD24bit, resources.GetString("chkHDCD24bit.ToolTip"));
			this.chkHDCD24bit.UseVisualStyleBackColor = true;
			// 
			// chkHDCDLW16
			// 
			resources.ApplyResources(this.chkHDCDLW16, "chkHDCDLW16");
			this.chkHDCDLW16.Name = "chkHDCDLW16";
			this.toolTip1.SetToolTip(this.chkHDCDLW16, resources.GetString("chkHDCDLW16.ToolTip"));
			this.chkHDCDLW16.UseVisualStyleBackColor = true;
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
			this.chkRemoveSpecial.CheckedChanged += new System.EventHandler(this.chkRemoveSpecial_CheckedChanged);
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
			// tabControl1
			// 
			resources.ApplyResources(this.tabControl1, "tabControl1");
			this.tabControl1.Controls.Add(this.tabPage1);
			this.tabControl1.Controls.Add(this.tabPage2);
			this.tabControl1.Controls.Add(this.tabPage3);
			this.tabControl1.Controls.Add(this.tabPage4);
			this.tabControl1.HotTrack = true;
			this.tabControl1.Multiline = true;
			this.tabControl1.Name = "tabControl1";
			this.tabControl1.SelectedIndex = 0;
			// 
			// tabPage1
			// 
			this.tabPage1.BackColor = System.Drawing.Color.Transparent;
			this.tabPage1.Controls.Add(this.grpGeneral);
			this.tabPage1.Controls.Add(this.grpAudioFilenames);
			resources.ApplyResources(this.tabPage1, "tabPage1");
			this.tabPage1.Name = "tabPage1";
			// 
			// tabPage2
			// 
			this.tabPage2.BackColor = System.Drawing.SystemColors.Control;
			this.tabPage2.Controls.Add(this.groupBox3);
			this.tabPage2.Controls.Add(this.groupBox1);
			resources.ApplyResources(this.tabPage2, "tabPage2");
			this.tabPage2.Name = "tabPage2";
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.Add(this.chkWriteARLogOnVerify);
			this.groupBox3.Controls.Add(this.chkWriteARTagsOnVerify);
			resources.ApplyResources(this.groupBox3, "groupBox3");
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.TabStop = false;
			// 
			// chkWriteARLogOnVerify
			// 
			resources.ApplyResources(this.chkWriteARLogOnVerify, "chkWriteARLogOnVerify");
			this.chkWriteARLogOnVerify.Checked = true;
			this.chkWriteARLogOnVerify.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteARLogOnVerify.Name = "chkWriteARLogOnVerify";
			this.chkWriteARLogOnVerify.UseVisualStyleBackColor = true;
			// 
			// tabPage3
			// 
			this.tabPage3.BackColor = System.Drawing.SystemColors.Control;
			this.tabPage3.Controls.Add(this.tabControl2);
			resources.ApplyResources(this.tabPage3, "tabPage3");
			this.tabPage3.Name = "tabPage3";
			// 
			// tabControl2
			// 
			this.tabControl2.Controls.Add(this.tabPage5);
			this.tabControl2.Controls.Add(this.tabPage6);
			this.tabControl2.Controls.Add(this.tabPage7);
			this.tabControl2.Controls.Add(this.tabPage8);
			this.tabControl2.Controls.Add(this.tabPage9);
			this.tabControl2.ImageList = this.imageList1;
			resources.ApplyResources(this.tabControl2, "tabControl2");
			this.tabControl2.Multiline = true;
			this.tabControl2.Name = "tabControl2";
			this.tabControl2.SelectedIndex = 0;
			// 
			// tabPage5
			// 
			this.tabPage5.BackColor = System.Drawing.Color.Transparent;
			this.tabPage5.Controls.Add(this.numericFLACCompressionLevel);
			this.tabPage5.Controls.Add(this.lblFLACCompressionLevel);
			this.tabPage5.Controls.Add(this.chkFLACVerify);
			resources.ApplyResources(this.tabPage5, "tabPage5");
			this.tabPage5.Name = "tabPage5";
			this.tabPage5.UseVisualStyleBackColor = true;
			// 
			// tabPage6
			// 
			this.tabPage6.Controls.Add(this.chkWVStoreMD5);
			this.tabPage6.Controls.Add(this.numWVExtraMode);
			this.tabPage6.Controls.Add(this.rbWVFast);
			this.tabPage6.Controls.Add(this.chkWVExtraMode);
			this.tabPage6.Controls.Add(this.rbWVNormal);
			this.tabPage6.Controls.Add(this.rbWVVeryHigh);
			this.tabPage6.Controls.Add(this.rbWVHigh);
			resources.ApplyResources(this.tabPage6, "tabPage6");
			this.tabPage6.Name = "tabPage6";
			this.tabPage6.UseVisualStyleBackColor = true;
			// 
			// tabPage7
			// 
			this.tabPage7.Controls.Add(this.rbAPEinsane);
			this.tabPage7.Controls.Add(this.rbAPEextrahigh);
			this.tabPage7.Controls.Add(this.rbAPEfast);
			this.tabPage7.Controls.Add(this.rbAPEhigh);
			this.tabPage7.Controls.Add(this.rbAPEnormal);
			resources.ApplyResources(this.tabPage7, "tabPage7");
			this.tabPage7.Name = "tabPage7";
			this.tabPage7.UseVisualStyleBackColor = true;
			// 
			// tabPage8
			// 
			this.tabPage8.Controls.Add(this.label1);
			this.tabPage8.Controls.Add(this.numericLossyWAVQuality);
			resources.ApplyResources(this.tabPage8, "tabPage8");
			this.tabPage8.Name = "tabPage8";
			this.tabPage8.UseVisualStyleBackColor = true;
			// 
			// label1
			// 
			resources.ApplyResources(this.label1, "label1");
			this.label1.Name = "label1";
			// 
			// numericLossyWAVQuality
			// 
			resources.ApplyResources(this.numericLossyWAVQuality, "numericLossyWAVQuality");
			this.numericLossyWAVQuality.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
			this.numericLossyWAVQuality.Name = "numericLossyWAVQuality";
			this.numericLossyWAVQuality.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
			// 
			// tabPage9
			// 
			this.tabPage9.Controls.Add(this.chkUDC1ID3v2);
			this.tabPage9.Controls.Add(this.chkUDC1APEv2);
			this.tabPage9.Controls.Add(this.label6);
			this.tabPage9.Controls.Add(this.label5);
			this.tabPage9.Controls.Add(this.textUDC1EncParams);
			this.tabPage9.Controls.Add(this.textUDC1Encoder);
			this.tabPage9.Controls.Add(this.textUDC1Params);
			this.tabPage9.Controls.Add(this.textUDC1Decoder);
			this.tabPage9.Controls.Add(this.textUDC1Extension);
			this.tabPage9.Controls.Add(this.label4);
			this.tabPage9.Controls.Add(this.label3);
			this.tabPage9.Controls.Add(this.label2);
			resources.ApplyResources(this.tabPage9, "tabPage9");
			this.tabPage9.Name = "tabPage9";
			this.tabPage9.UseVisualStyleBackColor = true;
			// 
			// chkUDC1ID3v2
			// 
			resources.ApplyResources(this.chkUDC1ID3v2, "chkUDC1ID3v2");
			this.chkUDC1ID3v2.Name = "chkUDC1ID3v2";
			this.chkUDC1ID3v2.UseVisualStyleBackColor = true;
			// 
			// chkUDC1APEv2
			// 
			resources.ApplyResources(this.chkUDC1APEv2, "chkUDC1APEv2");
			this.chkUDC1APEv2.Name = "chkUDC1APEv2";
			this.chkUDC1APEv2.UseVisualStyleBackColor = true;
			// 
			// label6
			// 
			resources.ApplyResources(this.label6, "label6");
			this.label6.Name = "label6";
			// 
			// label5
			// 
			resources.ApplyResources(this.label5, "label5");
			this.label5.Name = "label5";
			// 
			// textUDC1EncParams
			// 
			resources.ApplyResources(this.textUDC1EncParams, "textUDC1EncParams");
			this.textUDC1EncParams.Name = "textUDC1EncParams";
			// 
			// textUDC1Encoder
			// 
			resources.ApplyResources(this.textUDC1Encoder, "textUDC1Encoder");
			this.textUDC1Encoder.Name = "textUDC1Encoder";
			// 
			// textUDC1Params
			// 
			resources.ApplyResources(this.textUDC1Params, "textUDC1Params");
			this.textUDC1Params.Name = "textUDC1Params";
			// 
			// textUDC1Decoder
			// 
			resources.ApplyResources(this.textUDC1Decoder, "textUDC1Decoder");
			this.textUDC1Decoder.Name = "textUDC1Decoder";
			// 
			// textUDC1Extension
			// 
			resources.ApplyResources(this.textUDC1Extension, "textUDC1Extension");
			this.textUDC1Extension.Name = "textUDC1Extension";
			// 
			// label4
			// 
			resources.ApplyResources(this.label4, "label4");
			this.label4.Name = "label4";
			// 
			// label3
			// 
			resources.ApplyResources(this.label3, "label3");
			this.label3.Name = "label3";
			// 
			// label2
			// 
			resources.ApplyResources(this.label2, "label2");
			this.label2.Name = "label2";
			// 
			// imageList1
			// 
			this.imageList1.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList1.ImageStream")));
			this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
			this.imageList1.Images.SetKeyName(0, "flac");
			this.imageList1.Images.SetKeyName(1, "wv");
			// 
			// tabPage4
			// 
			this.tabPage4.BackColor = System.Drawing.SystemColors.Control;
			this.tabPage4.Controls.Add(this.grpHDCD);
			this.tabPage4.Controls.Add(this.chkHDCDDetect);
			resources.ApplyResources(this.tabPage4, "tabPage4");
			this.tabPage4.Name = "tabPage4";
			// 
			// grpHDCD
			// 
			this.grpHDCD.Controls.Add(this.chkHDCD24bit);
			this.grpHDCD.Controls.Add(this.chkHDCDLW16);
			this.grpHDCD.Controls.Add(this.chkHDCDStopLooking);
			this.grpHDCD.Controls.Add(this.chkHDCDDecode);
			resources.ApplyResources(this.grpHDCD, "grpHDCD");
			this.grpHDCD.Name = "grpHDCD";
			this.grpHDCD.TabStop = false;
			// 
			// chkHDCDDetect
			// 
			resources.ApplyResources(this.chkHDCDDetect, "chkHDCDDetect");
			this.chkHDCDDetect.Name = "chkHDCDDetect";
			this.chkHDCDDetect.UseVisualStyleBackColor = true;
			this.chkHDCDDetect.CheckedChanged += new System.EventHandler(this.chkHDCDDetect_CheckedChanged);
			// 
			// frmSettings
			// 
			this.AcceptButton = this.btnOK;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = btnCancel;
			this.ControlBox = false;
			this.Controls.Add(this.tabControl1);
			this.Controls.Add(btnCancel);
			this.Controls.Add(this.btnOK);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "frmSettings";
			this.ShowIcon = false;
			this.Load += new System.EventHandler(this.frmSettings_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmSettings_FormClosing);
			this.grpGeneral.ResumeLayout(false);
			this.grpGeneral.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericFLACCompressionLevel)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numWVExtraMode)).EndInit();
			this.groupBox1.ResumeLayout(false);
			this.groupBox1.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).EndInit();
			this.grpAudioFilenames.ResumeLayout(false);
			this.grpAudioFilenames.PerformLayout();
			this.tabControl1.ResumeLayout(false);
			this.tabPage1.ResumeLayout(false);
			this.tabPage2.ResumeLayout(false);
			this.groupBox3.ResumeLayout(false);
			this.groupBox3.PerformLayout();
			this.tabPage3.ResumeLayout(false);
			this.tabControl2.ResumeLayout(false);
			this.tabPage5.ResumeLayout(false);
			this.tabPage5.PerformLayout();
			this.tabPage6.ResumeLayout(false);
			this.tabPage6.PerformLayout();
			this.tabPage7.ResumeLayout(false);
			this.tabPage7.PerformLayout();
			this.tabPage8.ResumeLayout(false);
			this.tabPage8.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericLossyWAVQuality)).EndInit();
			this.tabPage9.ResumeLayout(false);
			this.tabPage9.PerformLayout();
			this.tabPage4.ResumeLayout(false);
			this.tabPage4.PerformLayout();
			this.grpHDCD.ResumeLayout(false);
			this.grpHDCD.PerformLayout();
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.GroupBox grpGeneral;
		private System.Windows.Forms.CheckBox chkPreserveHTOA;
		private System.Windows.Forms.Label lblFLACCompressionLevel;
		private System.Windows.Forms.CheckBox chkFLACVerify;
		private System.Windows.Forms.Button btnOK;
		private System.Windows.Forms.RadioButton rbWVVeryHigh;
		private System.Windows.Forms.RadioButton rbWVHigh;
		private System.Windows.Forms.RadioButton rbWVNormal;
		private System.Windows.Forms.RadioButton rbWVFast;
		private System.Windows.Forms.CheckBox chkWVExtraMode;
		private System.Windows.Forms.CheckBox chkAutoCorrectFilenames;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.CheckBox chkWriteArTagsOnConvert;
		private System.Windows.Forms.NumericUpDown numericFLACCompressionLevel;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.NumericUpDown numFixWhenPercent;
		private System.Windows.Forms.Label labelFixWhenConfidence;
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
		private System.Windows.Forms.CheckBox chkWriteArLogOnConvert;
		private System.Windows.Forms.NumericUpDown numWVExtraMode;
		private System.Windows.Forms.Label labelEncodeWhenConfidence;
		private System.Windows.Forms.NumericUpDown numEncodeWhenConfidence;
		private System.Windows.Forms.NumericUpDown numEncodeWhenPercent;
		private System.Windows.Forms.CheckBox chkArFixOffset;
		private System.Windows.Forms.CheckBox chkEmbedLog;
		private System.Windows.Forms.CheckBox chkFillUpCUE;
		private System.Windows.Forms.CheckBox chkFilenamesANSISafe;
		private System.Windows.Forms.RadioButton rbAPEinsane;
		private System.Windows.Forms.RadioButton rbAPEextrahigh;
		private System.Windows.Forms.RadioButton rbAPEhigh;
		private System.Windows.Forms.RadioButton rbAPEnormal;
		private System.Windows.Forms.RadioButton rbAPEfast;
		private System.Windows.Forms.TabControl tabControl1;
		private System.Windows.Forms.TabPage tabPage1;
		private System.Windows.Forms.TabPage tabPage2;
		private System.Windows.Forms.TabPage tabPage3;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.CheckBox chkWriteARLogOnVerify;
		private System.Windows.Forms.CheckBox chkWriteARTagsOnVerify;
		private System.Windows.Forms.CheckBox chkEncodeWhenZeroOffset;
		private System.Windows.Forms.TabPage tabPage4;
		private System.Windows.Forms.CheckBox chkHDCDDecode;
		private System.Windows.Forms.CheckBox chkHDCDDetect;
		private System.Windows.Forms.CheckBox chkWVStoreMD5;
		private System.Windows.Forms.GroupBox grpHDCD;
		private System.Windows.Forms.CheckBox chkHDCDStopLooking;
		private System.Windows.Forms.CheckBox chkCreateM3U;
		private System.Windows.Forms.CheckBox chkCreateCUEFileWhenEmbedded;
		private System.Windows.Forms.CheckBox chkTruncateExtra4206Samples;
		private System.Windows.Forms.CheckBox chkReducePriority;
		private System.Windows.Forms.NumericUpDown numericLossyWAVQuality;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.CheckBox chkHDCDLW16;
		private System.Windows.Forms.CheckBox chkHDCD24bit;
		private System.Windows.Forms.CheckBox chkExtractLog;
		private System.Windows.Forms.CheckBox chkOverwriteTags;
		private System.Windows.Forms.TabControl tabControl2;
		private System.Windows.Forms.TabPage tabPage5;
		private System.Windows.Forms.TabPage tabPage6;
		private System.Windows.Forms.TabPage tabPage7;
		private System.Windows.Forms.TabPage tabPage8;
		private System.Windows.Forms.TabPage tabPage9;
		private System.Windows.Forms.TextBox textUDC1Params;
		private System.Windows.Forms.TextBox textUDC1Decoder;
		private System.Windows.Forms.TextBox textUDC1Extension;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox textUDC1EncParams;
		private System.Windows.Forms.TextBox textUDC1Encoder;
		private System.Windows.Forms.CheckBox chkUDC1APEv2;
		private System.Windows.Forms.CheckBox chkUDC1ID3v2;
		private System.Windows.Forms.ImageList imageList1;
		private System.Windows.Forms.CheckBox chkSingleInstance;
		private System.Windows.Forms.Label labelLanguage;
		private System.Windows.Forms.ComboBox comboLanguage;

	}
}