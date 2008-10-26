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
			this.chkEncodeWhenZeroOffset = new System.Windows.Forms.CheckBox();
			this.chkArFixOffset = new System.Windows.Forms.CheckBox();
			this.chkWriteArLogOnConvert = new System.Windows.Forms.CheckBox();
			this.chkWriteArTagsOnConvert = new System.Windows.Forms.CheckBox();
			this.labelEncodeWhenPercent = new System.Windows.Forms.Label();
			this.numEncodeWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.labelEncodeWhenConfidence = new System.Windows.Forms.Label();
			this.numEncodeWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.chkArNoUnverifiedAudio = new System.Windows.Forms.CheckBox();
			this.labelFixWhenConfidence = new System.Windows.Forms.Label();
			this.numFixWhenConfidence = new System.Windows.Forms.NumericUpDown();
			this.labelFixWhenPercent = new System.Windows.Forms.Label();
			this.numFixWhenPercent = new System.Windows.Forms.NumericUpDown();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.chkFilenamesANSISafe = new System.Windows.Forms.CheckBox();
			this.chkWriteARTagsOnVerify = new System.Windows.Forms.CheckBox();
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
			this.tabControl1 = new System.Windows.Forms.TabControl();
			this.tabPage1 = new System.Windows.Forms.TabPage();
			this.tabPage2 = new System.Windows.Forms.TabPage();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.chkWriteARLogOnVerify = new System.Windows.Forms.CheckBox();
			this.tabPage3 = new System.Windows.Forms.TabPage();
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
			this.tabControl1.SuspendLayout();
			this.tabPage1.SuspendLayout();
			this.tabPage2.SuspendLayout();
			this.groupBox3.SuspendLayout();
			this.tabPage3.SuspendLayout();
			this.SuspendLayout();
			// 
			// btnCancel
			// 
			btnCancel.AccessibleDescription = null;
			btnCancel.AccessibleName = null;
			resources.ApplyResources(btnCancel, "btnCancel");
			btnCancel.BackgroundImage = null;
			btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			btnCancel.Font = null;
			btnCancel.Name = "btnCancel";
			this.toolTip1.SetToolTip(btnCancel, resources.GetString("btnCancel.ToolTip"));
			btnCancel.UseVisualStyleBackColor = true;
			// 
			// grpGeneral
			// 
			this.grpGeneral.AccessibleDescription = null;
			this.grpGeneral.AccessibleName = null;
			resources.ApplyResources(this.grpGeneral, "grpGeneral");
			this.grpGeneral.BackgroundImage = null;
			this.grpGeneral.Controls.Add(this.chkFillUpCUE);
			this.grpGeneral.Controls.Add(this.chkEmbedLog);
			this.grpGeneral.Controls.Add(this.numericWriteOffset);
			this.grpGeneral.Controls.Add(this.chkAutoCorrectFilenames);
			this.grpGeneral.Controls.Add(this.chkPreserveHTOA);
			this.grpGeneral.Controls.Add(this.lblWriteOffset);
			this.grpGeneral.Font = null;
			this.grpGeneral.Name = "grpGeneral";
			this.grpGeneral.TabStop = false;
			this.toolTip1.SetToolTip(this.grpGeneral, resources.GetString("grpGeneral.ToolTip"));
			// 
			// chkFillUpCUE
			// 
			this.chkFillUpCUE.AccessibleDescription = null;
			this.chkFillUpCUE.AccessibleName = null;
			resources.ApplyResources(this.chkFillUpCUE, "chkFillUpCUE");
			this.chkFillUpCUE.BackgroundImage = null;
			this.chkFillUpCUE.Font = null;
			this.chkFillUpCUE.Name = "chkFillUpCUE";
			this.toolTip1.SetToolTip(this.chkFillUpCUE, resources.GetString("chkFillUpCUE.ToolTip"));
			this.chkFillUpCUE.UseVisualStyleBackColor = true;
			// 
			// chkEmbedLog
			// 
			this.chkEmbedLog.AccessibleDescription = null;
			this.chkEmbedLog.AccessibleName = null;
			resources.ApplyResources(this.chkEmbedLog, "chkEmbedLog");
			this.chkEmbedLog.BackgroundImage = null;
			this.chkEmbedLog.Font = null;
			this.chkEmbedLog.Name = "chkEmbedLog";
			this.toolTip1.SetToolTip(this.chkEmbedLog, resources.GetString("chkEmbedLog.ToolTip"));
			this.chkEmbedLog.UseVisualStyleBackColor = true;
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
			// chkAutoCorrectFilenames
			// 
			this.chkAutoCorrectFilenames.AccessibleDescription = null;
			this.chkAutoCorrectFilenames.AccessibleName = null;
			resources.ApplyResources(this.chkAutoCorrectFilenames, "chkAutoCorrectFilenames");
			this.chkAutoCorrectFilenames.BackgroundImage = null;
			this.chkAutoCorrectFilenames.Font = null;
			this.chkAutoCorrectFilenames.Name = "chkAutoCorrectFilenames";
			this.toolTip1.SetToolTip(this.chkAutoCorrectFilenames, resources.GetString("chkAutoCorrectFilenames.ToolTip"));
			this.chkAutoCorrectFilenames.UseVisualStyleBackColor = true;
			// 
			// chkPreserveHTOA
			// 
			this.chkPreserveHTOA.AccessibleDescription = null;
			this.chkPreserveHTOA.AccessibleName = null;
			resources.ApplyResources(this.chkPreserveHTOA, "chkPreserveHTOA");
			this.chkPreserveHTOA.BackgroundImage = null;
			this.chkPreserveHTOA.Font = null;
			this.chkPreserveHTOA.Name = "chkPreserveHTOA";
			this.toolTip1.SetToolTip(this.chkPreserveHTOA, resources.GetString("chkPreserveHTOA.ToolTip"));
			this.chkPreserveHTOA.UseVisualStyleBackColor = true;
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
			// grpFLAC
			// 
			this.grpFLAC.AccessibleDescription = null;
			this.grpFLAC.AccessibleName = null;
			resources.ApplyResources(this.grpFLAC, "grpFLAC");
			this.grpFLAC.BackgroundImage = null;
			this.grpFLAC.Controls.Add(this.numericFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.lblFLACCompressionLevel);
			this.grpFLAC.Controls.Add(this.chkFLACVerify);
			this.grpFLAC.Font = null;
			this.grpFLAC.Name = "grpFLAC";
			this.grpFLAC.TabStop = false;
			this.toolTip1.SetToolTip(this.grpFLAC, resources.GetString("grpFLAC.ToolTip"));
			// 
			// numericFLACCompressionLevel
			// 
			this.numericFLACCompressionLevel.AccessibleDescription = null;
			this.numericFLACCompressionLevel.AccessibleName = null;
			resources.ApplyResources(this.numericFLACCompressionLevel, "numericFLACCompressionLevel");
			this.numericFLACCompressionLevel.Font = null;
			this.numericFLACCompressionLevel.Maximum = new decimal(new int[] {
            8,
            0,
            0,
            0});
			this.numericFLACCompressionLevel.Name = "numericFLACCompressionLevel";
			this.toolTip1.SetToolTip(this.numericFLACCompressionLevel, resources.GetString("numericFLACCompressionLevel.ToolTip"));
			this.numericFLACCompressionLevel.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
			// 
			// lblFLACCompressionLevel
			// 
			this.lblFLACCompressionLevel.AccessibleDescription = null;
			this.lblFLACCompressionLevel.AccessibleName = null;
			resources.ApplyResources(this.lblFLACCompressionLevel, "lblFLACCompressionLevel");
			this.lblFLACCompressionLevel.Font = null;
			this.lblFLACCompressionLevel.Name = "lblFLACCompressionLevel";
			this.toolTip1.SetToolTip(this.lblFLACCompressionLevel, resources.GetString("lblFLACCompressionLevel.ToolTip"));
			// 
			// chkFLACVerify
			// 
			this.chkFLACVerify.AccessibleDescription = null;
			this.chkFLACVerify.AccessibleName = null;
			resources.ApplyResources(this.chkFLACVerify, "chkFLACVerify");
			this.chkFLACVerify.BackgroundImage = null;
			this.chkFLACVerify.Font = null;
			this.chkFLACVerify.Name = "chkFLACVerify";
			this.toolTip1.SetToolTip(this.chkFLACVerify, resources.GetString("chkFLACVerify.ToolTip"));
			this.chkFLACVerify.UseVisualStyleBackColor = true;
			// 
			// btnOK
			// 
			this.btnOK.AccessibleDescription = null;
			this.btnOK.AccessibleName = null;
			resources.ApplyResources(this.btnOK, "btnOK");
			this.btnOK.BackgroundImage = null;
			this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.btnOK.Font = null;
			this.btnOK.Name = "btnOK";
			this.toolTip1.SetToolTip(this.btnOK, resources.GetString("btnOK.ToolTip"));
			this.btnOK.UseVisualStyleBackColor = true;
			this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
			// 
			// grpWavPack
			// 
			this.grpWavPack.AccessibleDescription = null;
			this.grpWavPack.AccessibleName = null;
			resources.ApplyResources(this.grpWavPack, "grpWavPack");
			this.grpWavPack.BackgroundImage = null;
			this.grpWavPack.Controls.Add(this.numWVExtraMode);
			this.grpWavPack.Controls.Add(this.chkWVExtraMode);
			this.grpWavPack.Controls.Add(this.rbWVVeryHigh);
			this.grpWavPack.Controls.Add(this.rbWVHigh);
			this.grpWavPack.Controls.Add(this.rbWVNormal);
			this.grpWavPack.Controls.Add(this.rbWVFast);
			this.grpWavPack.Font = null;
			this.grpWavPack.Name = "grpWavPack";
			this.grpWavPack.TabStop = false;
			this.toolTip1.SetToolTip(this.grpWavPack, resources.GetString("grpWavPack.ToolTip"));
			// 
			// numWVExtraMode
			// 
			this.numWVExtraMode.AccessibleDescription = null;
			this.numWVExtraMode.AccessibleName = null;
			resources.ApplyResources(this.numWVExtraMode, "numWVExtraMode");
			this.numWVExtraMode.Font = null;
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
			this.toolTip1.SetToolTip(this.numWVExtraMode, resources.GetString("numWVExtraMode.ToolTip"));
			this.numWVExtraMode.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkWVExtraMode
			// 
			this.chkWVExtraMode.AccessibleDescription = null;
			this.chkWVExtraMode.AccessibleName = null;
			resources.ApplyResources(this.chkWVExtraMode, "chkWVExtraMode");
			this.chkWVExtraMode.BackgroundImage = null;
			this.chkWVExtraMode.Font = null;
			this.chkWVExtraMode.Name = "chkWVExtraMode";
			this.toolTip1.SetToolTip(this.chkWVExtraMode, resources.GetString("chkWVExtraMode.ToolTip"));
			this.chkWVExtraMode.UseVisualStyleBackColor = true;
			this.chkWVExtraMode.CheckedChanged += new System.EventHandler(this.chkWVExtraMode_CheckedChanged);
			// 
			// rbWVVeryHigh
			// 
			this.rbWVVeryHigh.AccessibleDescription = null;
			this.rbWVVeryHigh.AccessibleName = null;
			resources.ApplyResources(this.rbWVVeryHigh, "rbWVVeryHigh");
			this.rbWVVeryHigh.BackgroundImage = null;
			this.rbWVVeryHigh.Font = null;
			this.rbWVVeryHigh.Name = "rbWVVeryHigh";
			this.toolTip1.SetToolTip(this.rbWVVeryHigh, resources.GetString("rbWVVeryHigh.ToolTip"));
			this.rbWVVeryHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVHigh
			// 
			this.rbWVHigh.AccessibleDescription = null;
			this.rbWVHigh.AccessibleName = null;
			resources.ApplyResources(this.rbWVHigh, "rbWVHigh");
			this.rbWVHigh.BackgroundImage = null;
			this.rbWVHigh.Font = null;
			this.rbWVHigh.Name = "rbWVHigh";
			this.toolTip1.SetToolTip(this.rbWVHigh, resources.GetString("rbWVHigh.ToolTip"));
			this.rbWVHigh.UseVisualStyleBackColor = true;
			// 
			// rbWVNormal
			// 
			this.rbWVNormal.AccessibleDescription = null;
			this.rbWVNormal.AccessibleName = null;
			resources.ApplyResources(this.rbWVNormal, "rbWVNormal");
			this.rbWVNormal.BackgroundImage = null;
			this.rbWVNormal.Checked = true;
			this.rbWVNormal.Font = null;
			this.rbWVNormal.Name = "rbWVNormal";
			this.rbWVNormal.TabStop = true;
			this.toolTip1.SetToolTip(this.rbWVNormal, resources.GetString("rbWVNormal.ToolTip"));
			this.rbWVNormal.UseVisualStyleBackColor = true;
			// 
			// rbWVFast
			// 
			this.rbWVFast.AccessibleDescription = null;
			this.rbWVFast.AccessibleName = null;
			resources.ApplyResources(this.rbWVFast, "rbWVFast");
			this.rbWVFast.BackgroundImage = null;
			this.rbWVFast.Font = null;
			this.rbWVFast.Name = "rbWVFast";
			this.toolTip1.SetToolTip(this.rbWVFast, resources.GetString("rbWVFast.ToolTip"));
			this.rbWVFast.UseVisualStyleBackColor = true;
			// 
			// groupBox1
			// 
			this.groupBox1.AccessibleDescription = null;
			this.groupBox1.AccessibleName = null;
			resources.ApplyResources(this.groupBox1, "groupBox1");
			this.groupBox1.BackgroundImage = null;
			this.groupBox1.Controls.Add(this.chkEncodeWhenZeroOffset);
			this.groupBox1.Controls.Add(this.chkArFixOffset);
			this.groupBox1.Controls.Add(this.chkWriteArLogOnConvert);
			this.groupBox1.Controls.Add(this.chkWriteArTagsOnConvert);
			this.groupBox1.Controls.Add(this.labelEncodeWhenPercent);
			this.groupBox1.Controls.Add(this.numEncodeWhenPercent);
			this.groupBox1.Controls.Add(this.labelEncodeWhenConfidence);
			this.groupBox1.Controls.Add(this.numEncodeWhenConfidence);
			this.groupBox1.Controls.Add(this.chkArNoUnverifiedAudio);
			this.groupBox1.Controls.Add(this.labelFixWhenConfidence);
			this.groupBox1.Controls.Add(this.numFixWhenConfidence);
			this.groupBox1.Controls.Add(this.labelFixWhenPercent);
			this.groupBox1.Controls.Add(this.numFixWhenPercent);
			this.groupBox1.Font = null;
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.TabStop = false;
			this.toolTip1.SetToolTip(this.groupBox1, resources.GetString("groupBox1.ToolTip"));
			// 
			// chkEncodeWhenZeroOffset
			// 
			this.chkEncodeWhenZeroOffset.AccessibleDescription = null;
			this.chkEncodeWhenZeroOffset.AccessibleName = null;
			resources.ApplyResources(this.chkEncodeWhenZeroOffset, "chkEncodeWhenZeroOffset");
			this.chkEncodeWhenZeroOffset.BackgroundImage = null;
			this.chkEncodeWhenZeroOffset.Font = null;
			this.chkEncodeWhenZeroOffset.Name = "chkEncodeWhenZeroOffset";
			this.toolTip1.SetToolTip(this.chkEncodeWhenZeroOffset, resources.GetString("chkEncodeWhenZeroOffset.ToolTip"));
			this.chkEncodeWhenZeroOffset.UseVisualStyleBackColor = true;
			// 
			// chkArFixOffset
			// 
			this.chkArFixOffset.AccessibleDescription = null;
			this.chkArFixOffset.AccessibleName = null;
			resources.ApplyResources(this.chkArFixOffset, "chkArFixOffset");
			this.chkArFixOffset.BackgroundImage = null;
			this.chkArFixOffset.Checked = true;
			this.chkArFixOffset.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArFixOffset.Font = null;
			this.chkArFixOffset.Name = "chkArFixOffset";
			this.toolTip1.SetToolTip(this.chkArFixOffset, resources.GetString("chkArFixOffset.ToolTip"));
			this.chkArFixOffset.UseVisualStyleBackColor = true;
			this.chkArFixOffset.CheckedChanged += new System.EventHandler(this.chkArFixOffset_CheckedChanged);
			// 
			// chkWriteArLogOnConvert
			// 
			this.chkWriteArLogOnConvert.AccessibleDescription = null;
			this.chkWriteArLogOnConvert.AccessibleName = null;
			resources.ApplyResources(this.chkWriteArLogOnConvert, "chkWriteArLogOnConvert");
			this.chkWriteArLogOnConvert.BackgroundImage = null;
			this.chkWriteArLogOnConvert.Checked = true;
			this.chkWriteArLogOnConvert.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteArLogOnConvert.Font = null;
			this.chkWriteArLogOnConvert.Name = "chkWriteArLogOnConvert";
			this.toolTip1.SetToolTip(this.chkWriteArLogOnConvert, resources.GetString("chkWriteArLogOnConvert.ToolTip"));
			this.chkWriteArLogOnConvert.UseVisualStyleBackColor = true;
			// 
			// chkWriteArTagsOnConvert
			// 
			this.chkWriteArTagsOnConvert.AccessibleDescription = null;
			this.chkWriteArTagsOnConvert.AccessibleName = null;
			resources.ApplyResources(this.chkWriteArTagsOnConvert, "chkWriteArTagsOnConvert");
			this.chkWriteArTagsOnConvert.BackgroundImage = null;
			this.chkWriteArTagsOnConvert.Checked = true;
			this.chkWriteArTagsOnConvert.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteArTagsOnConvert.Font = null;
			this.chkWriteArTagsOnConvert.Name = "chkWriteArTagsOnConvert";
			this.toolTip1.SetToolTip(this.chkWriteArTagsOnConvert, resources.GetString("chkWriteArTagsOnConvert.ToolTip"));
			this.chkWriteArTagsOnConvert.UseVisualStyleBackColor = true;
			// 
			// labelEncodeWhenPercent
			// 
			this.labelEncodeWhenPercent.AccessibleDescription = null;
			this.labelEncodeWhenPercent.AccessibleName = null;
			resources.ApplyResources(this.labelEncodeWhenPercent, "labelEncodeWhenPercent");
			this.labelEncodeWhenPercent.Font = null;
			this.labelEncodeWhenPercent.Name = "labelEncodeWhenPercent";
			this.toolTip1.SetToolTip(this.labelEncodeWhenPercent, resources.GetString("labelEncodeWhenPercent.ToolTip"));
			// 
			// numEncodeWhenPercent
			// 
			this.numEncodeWhenPercent.AccessibleDescription = null;
			this.numEncodeWhenPercent.AccessibleName = null;
			resources.ApplyResources(this.numEncodeWhenPercent, "numEncodeWhenPercent");
			this.numEncodeWhenPercent.Font = null;
			this.numEncodeWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			this.numEncodeWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenPercent.Name = "numEncodeWhenPercent";
			this.toolTip1.SetToolTip(this.numEncodeWhenPercent, resources.GetString("numEncodeWhenPercent.ToolTip"));
			this.numEncodeWhenPercent.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
			// 
			// labelEncodeWhenConfidence
			// 
			this.labelEncodeWhenConfidence.AccessibleDescription = null;
			this.labelEncodeWhenConfidence.AccessibleName = null;
			resources.ApplyResources(this.labelEncodeWhenConfidence, "labelEncodeWhenConfidence");
			this.labelEncodeWhenConfidence.Font = null;
			this.labelEncodeWhenConfidence.Name = "labelEncodeWhenConfidence";
			this.toolTip1.SetToolTip(this.labelEncodeWhenConfidence, resources.GetString("labelEncodeWhenConfidence.ToolTip"));
			// 
			// numEncodeWhenConfidence
			// 
			this.numEncodeWhenConfidence.AccessibleDescription = null;
			this.numEncodeWhenConfidence.AccessibleName = null;
			resources.ApplyResources(this.numEncodeWhenConfidence, "numEncodeWhenConfidence");
			this.numEncodeWhenConfidence.Font = null;
			this.numEncodeWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numEncodeWhenConfidence.Name = "numEncodeWhenConfidence";
			this.toolTip1.SetToolTip(this.numEncodeWhenConfidence, resources.GetString("numEncodeWhenConfidence.ToolTip"));
			this.numEncodeWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// chkArNoUnverifiedAudio
			// 
			this.chkArNoUnverifiedAudio.AccessibleDescription = null;
			this.chkArNoUnverifiedAudio.AccessibleName = null;
			resources.ApplyResources(this.chkArNoUnverifiedAudio, "chkArNoUnverifiedAudio");
			this.chkArNoUnverifiedAudio.BackgroundImage = null;
			this.chkArNoUnverifiedAudio.Checked = true;
			this.chkArNoUnverifiedAudio.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkArNoUnverifiedAudio.Font = null;
			this.chkArNoUnverifiedAudio.Name = "chkArNoUnverifiedAudio";
			this.toolTip1.SetToolTip(this.chkArNoUnverifiedAudio, resources.GetString("chkArNoUnverifiedAudio.ToolTip"));
			this.chkArNoUnverifiedAudio.UseVisualStyleBackColor = true;
			this.chkArNoUnverifiedAudio.CheckedChanged += new System.EventHandler(this.chkArNoUnverifiedAudio_CheckedChanged);
			// 
			// labelFixWhenConfidence
			// 
			this.labelFixWhenConfidence.AccessibleDescription = null;
			this.labelFixWhenConfidence.AccessibleName = null;
			resources.ApplyResources(this.labelFixWhenConfidence, "labelFixWhenConfidence");
			this.labelFixWhenConfidence.Font = null;
			this.labelFixWhenConfidence.Name = "labelFixWhenConfidence";
			this.toolTip1.SetToolTip(this.labelFixWhenConfidence, resources.GetString("labelFixWhenConfidence.ToolTip"));
			// 
			// numFixWhenConfidence
			// 
			this.numFixWhenConfidence.AccessibleDescription = null;
			this.numFixWhenConfidence.AccessibleName = null;
			resources.ApplyResources(this.numFixWhenConfidence, "numFixWhenConfidence");
			this.numFixWhenConfidence.Font = null;
			this.numFixWhenConfidence.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenConfidence.Name = "numFixWhenConfidence";
			this.toolTip1.SetToolTip(this.numFixWhenConfidence, resources.GetString("numFixWhenConfidence.ToolTip"));
			this.numFixWhenConfidence.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
			// 
			// labelFixWhenPercent
			// 
			this.labelFixWhenPercent.AccessibleDescription = null;
			this.labelFixWhenPercent.AccessibleName = null;
			resources.ApplyResources(this.labelFixWhenPercent, "labelFixWhenPercent");
			this.labelFixWhenPercent.Font = null;
			this.labelFixWhenPercent.Name = "labelFixWhenPercent";
			this.toolTip1.SetToolTip(this.labelFixWhenPercent, resources.GetString("labelFixWhenPercent.ToolTip"));
			// 
			// numFixWhenPercent
			// 
			this.numFixWhenPercent.AccessibleDescription = null;
			this.numFixWhenPercent.AccessibleName = null;
			resources.ApplyResources(this.numFixWhenPercent, "numFixWhenPercent");
			this.numFixWhenPercent.Font = null;
			this.numFixWhenPercent.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
			this.numFixWhenPercent.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
			this.numFixWhenPercent.Name = "numFixWhenPercent";
			this.toolTip1.SetToolTip(this.numFixWhenPercent, resources.GetString("numFixWhenPercent.ToolTip"));
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
			this.chkFilenamesANSISafe.AccessibleDescription = null;
			this.chkFilenamesANSISafe.AccessibleName = null;
			resources.ApplyResources(this.chkFilenamesANSISafe, "chkFilenamesANSISafe");
			this.chkFilenamesANSISafe.BackgroundImage = null;
			this.chkFilenamesANSISafe.Font = null;
			this.chkFilenamesANSISafe.Name = "chkFilenamesANSISafe";
			this.toolTip1.SetToolTip(this.chkFilenamesANSISafe, resources.GetString("chkFilenamesANSISafe.ToolTip"));
			this.chkFilenamesANSISafe.UseVisualStyleBackColor = true;
			// 
			// chkWriteARTagsOnVerify
			// 
			this.chkWriteARTagsOnVerify.AccessibleDescription = null;
			this.chkWriteARTagsOnVerify.AccessibleName = null;
			resources.ApplyResources(this.chkWriteARTagsOnVerify, "chkWriteARTagsOnVerify");
			this.chkWriteARTagsOnVerify.BackgroundImage = null;
			this.chkWriteARTagsOnVerify.Checked = true;
			this.chkWriteARTagsOnVerify.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteARTagsOnVerify.Font = null;
			this.chkWriteARTagsOnVerify.Name = "chkWriteARTagsOnVerify";
			this.toolTip1.SetToolTip(this.chkWriteARTagsOnVerify, resources.GetString("chkWriteARTagsOnVerify.ToolTip"));
			this.chkWriteARTagsOnVerify.UseVisualStyleBackColor = true;
			// 
			// grpAudioFilenames
			// 
			this.grpAudioFilenames.AccessibleDescription = null;
			this.grpAudioFilenames.AccessibleName = null;
			resources.ApplyResources(this.grpAudioFilenames, "grpAudioFilenames");
			this.grpAudioFilenames.BackgroundImage = null;
			this.grpAudioFilenames.Controls.Add(this.chkFilenamesANSISafe);
			this.grpAudioFilenames.Controls.Add(this.chkKeepOriginalFilenames);
			this.grpAudioFilenames.Controls.Add(this.txtSpecialExceptions);
			this.grpAudioFilenames.Controls.Add(this.chkRemoveSpecial);
			this.grpAudioFilenames.Controls.Add(this.chkReplaceSpaces);
			this.grpAudioFilenames.Controls.Add(this.txtTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblTrackFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.lblSingleFilenameFormat);
			this.grpAudioFilenames.Controls.Add(this.txtSingleFilenameFormat);
			this.grpAudioFilenames.Font = null;
			this.grpAudioFilenames.Name = "grpAudioFilenames";
			this.grpAudioFilenames.TabStop = false;
			this.toolTip1.SetToolTip(this.grpAudioFilenames, resources.GetString("grpAudioFilenames.ToolTip"));
			// 
			// chkKeepOriginalFilenames
			// 
			this.chkKeepOriginalFilenames.AccessibleDescription = null;
			this.chkKeepOriginalFilenames.AccessibleName = null;
			resources.ApplyResources(this.chkKeepOriginalFilenames, "chkKeepOriginalFilenames");
			this.chkKeepOriginalFilenames.BackgroundImage = null;
			this.chkKeepOriginalFilenames.Checked = true;
			this.chkKeepOriginalFilenames.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkKeepOriginalFilenames.Font = null;
			this.chkKeepOriginalFilenames.Name = "chkKeepOriginalFilenames";
			this.toolTip1.SetToolTip(this.chkKeepOriginalFilenames, resources.GetString("chkKeepOriginalFilenames.ToolTip"));
			this.chkKeepOriginalFilenames.UseVisualStyleBackColor = true;
			// 
			// txtSpecialExceptions
			// 
			this.txtSpecialExceptions.AccessibleDescription = null;
			this.txtSpecialExceptions.AccessibleName = null;
			resources.ApplyResources(this.txtSpecialExceptions, "txtSpecialExceptions");
			this.txtSpecialExceptions.BackgroundImage = null;
			this.txtSpecialExceptions.Font = null;
			this.txtSpecialExceptions.Name = "txtSpecialExceptions";
			this.toolTip1.SetToolTip(this.txtSpecialExceptions, resources.GetString("txtSpecialExceptions.ToolTip"));
			// 
			// chkRemoveSpecial
			// 
			this.chkRemoveSpecial.AccessibleDescription = null;
			this.chkRemoveSpecial.AccessibleName = null;
			resources.ApplyResources(this.chkRemoveSpecial, "chkRemoveSpecial");
			this.chkRemoveSpecial.BackgroundImage = null;
			this.chkRemoveSpecial.Checked = true;
			this.chkRemoveSpecial.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkRemoveSpecial.Font = null;
			this.chkRemoveSpecial.Name = "chkRemoveSpecial";
			this.toolTip1.SetToolTip(this.chkRemoveSpecial, resources.GetString("chkRemoveSpecial.ToolTip"));
			this.chkRemoveSpecial.UseVisualStyleBackColor = true;
			// 
			// chkReplaceSpaces
			// 
			this.chkReplaceSpaces.AccessibleDescription = null;
			this.chkReplaceSpaces.AccessibleName = null;
			resources.ApplyResources(this.chkReplaceSpaces, "chkReplaceSpaces");
			this.chkReplaceSpaces.BackgroundImage = null;
			this.chkReplaceSpaces.Checked = true;
			this.chkReplaceSpaces.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkReplaceSpaces.Font = null;
			this.chkReplaceSpaces.Name = "chkReplaceSpaces";
			this.toolTip1.SetToolTip(this.chkReplaceSpaces, resources.GetString("chkReplaceSpaces.ToolTip"));
			this.chkReplaceSpaces.UseVisualStyleBackColor = true;
			// 
			// txtTrackFilenameFormat
			// 
			this.txtTrackFilenameFormat.AccessibleDescription = null;
			this.txtTrackFilenameFormat.AccessibleName = null;
			resources.ApplyResources(this.txtTrackFilenameFormat, "txtTrackFilenameFormat");
			this.txtTrackFilenameFormat.BackgroundImage = null;
			this.txtTrackFilenameFormat.Font = null;
			this.txtTrackFilenameFormat.Name = "txtTrackFilenameFormat";
			this.toolTip1.SetToolTip(this.txtTrackFilenameFormat, resources.GetString("txtTrackFilenameFormat.ToolTip"));
			// 
			// lblTrackFilenameFormat
			// 
			this.lblTrackFilenameFormat.AccessibleDescription = null;
			this.lblTrackFilenameFormat.AccessibleName = null;
			resources.ApplyResources(this.lblTrackFilenameFormat, "lblTrackFilenameFormat");
			this.lblTrackFilenameFormat.Font = null;
			this.lblTrackFilenameFormat.Name = "lblTrackFilenameFormat";
			this.toolTip1.SetToolTip(this.lblTrackFilenameFormat, resources.GetString("lblTrackFilenameFormat.ToolTip"));
			// 
			// lblSingleFilenameFormat
			// 
			this.lblSingleFilenameFormat.AccessibleDescription = null;
			this.lblSingleFilenameFormat.AccessibleName = null;
			resources.ApplyResources(this.lblSingleFilenameFormat, "lblSingleFilenameFormat");
			this.lblSingleFilenameFormat.Font = null;
			this.lblSingleFilenameFormat.Name = "lblSingleFilenameFormat";
			this.toolTip1.SetToolTip(this.lblSingleFilenameFormat, resources.GetString("lblSingleFilenameFormat.ToolTip"));
			// 
			// txtSingleFilenameFormat
			// 
			this.txtSingleFilenameFormat.AccessibleDescription = null;
			this.txtSingleFilenameFormat.AccessibleName = null;
			resources.ApplyResources(this.txtSingleFilenameFormat, "txtSingleFilenameFormat");
			this.txtSingleFilenameFormat.BackgroundImage = null;
			this.txtSingleFilenameFormat.Font = null;
			this.txtSingleFilenameFormat.Name = "txtSingleFilenameFormat";
			this.toolTip1.SetToolTip(this.txtSingleFilenameFormat, resources.GetString("txtSingleFilenameFormat.ToolTip"));
			// 
			// groupBox2
			// 
			this.groupBox2.AccessibleDescription = null;
			this.groupBox2.AccessibleName = null;
			resources.ApplyResources(this.groupBox2, "groupBox2");
			this.groupBox2.BackgroundImage = null;
			this.groupBox2.Controls.Add(this.rbAPEinsane);
			this.groupBox2.Controls.Add(this.rbAPEextrahigh);
			this.groupBox2.Controls.Add(this.rbAPEhigh);
			this.groupBox2.Controls.Add(this.rbAPEnormal);
			this.groupBox2.Controls.Add(this.rbAPEfast);
			this.groupBox2.Font = null;
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.TabStop = false;
			this.toolTip1.SetToolTip(this.groupBox2, resources.GetString("groupBox2.ToolTip"));
			// 
			// rbAPEinsane
			// 
			this.rbAPEinsane.AccessibleDescription = null;
			this.rbAPEinsane.AccessibleName = null;
			resources.ApplyResources(this.rbAPEinsane, "rbAPEinsane");
			this.rbAPEinsane.BackgroundImage = null;
			this.rbAPEinsane.Font = null;
			this.rbAPEinsane.Name = "rbAPEinsane";
			this.rbAPEinsane.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPEinsane, resources.GetString("rbAPEinsane.ToolTip"));
			this.rbAPEinsane.UseVisualStyleBackColor = true;
			// 
			// rbAPEextrahigh
			// 
			this.rbAPEextrahigh.AccessibleDescription = null;
			this.rbAPEextrahigh.AccessibleName = null;
			resources.ApplyResources(this.rbAPEextrahigh, "rbAPEextrahigh");
			this.rbAPEextrahigh.BackgroundImage = null;
			this.rbAPEextrahigh.Font = null;
			this.rbAPEextrahigh.Name = "rbAPEextrahigh";
			this.rbAPEextrahigh.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPEextrahigh, resources.GetString("rbAPEextrahigh.ToolTip"));
			this.rbAPEextrahigh.UseVisualStyleBackColor = true;
			// 
			// rbAPEhigh
			// 
			this.rbAPEhigh.AccessibleDescription = null;
			this.rbAPEhigh.AccessibleName = null;
			resources.ApplyResources(this.rbAPEhigh, "rbAPEhigh");
			this.rbAPEhigh.BackgroundImage = null;
			this.rbAPEhigh.Font = null;
			this.rbAPEhigh.Name = "rbAPEhigh";
			this.rbAPEhigh.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPEhigh, resources.GetString("rbAPEhigh.ToolTip"));
			this.rbAPEhigh.UseVisualStyleBackColor = true;
			// 
			// rbAPEnormal
			// 
			this.rbAPEnormal.AccessibleDescription = null;
			this.rbAPEnormal.AccessibleName = null;
			resources.ApplyResources(this.rbAPEnormal, "rbAPEnormal");
			this.rbAPEnormal.BackgroundImage = null;
			this.rbAPEnormal.Font = null;
			this.rbAPEnormal.Name = "rbAPEnormal";
			this.rbAPEnormal.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPEnormal, resources.GetString("rbAPEnormal.ToolTip"));
			this.rbAPEnormal.UseVisualStyleBackColor = true;
			// 
			// rbAPEfast
			// 
			this.rbAPEfast.AccessibleDescription = null;
			this.rbAPEfast.AccessibleName = null;
			resources.ApplyResources(this.rbAPEfast, "rbAPEfast");
			this.rbAPEfast.BackgroundImage = null;
			this.rbAPEfast.Font = null;
			this.rbAPEfast.Name = "rbAPEfast";
			this.rbAPEfast.TabStop = true;
			this.toolTip1.SetToolTip(this.rbAPEfast, resources.GetString("rbAPEfast.ToolTip"));
			this.rbAPEfast.UseVisualStyleBackColor = true;
			// 
			// tabControl1
			// 
			this.tabControl1.AccessibleDescription = null;
			this.tabControl1.AccessibleName = null;
			resources.ApplyResources(this.tabControl1, "tabControl1");
			this.tabControl1.BackgroundImage = null;
			this.tabControl1.Controls.Add(this.tabPage1);
			this.tabControl1.Controls.Add(this.tabPage2);
			this.tabControl1.Controls.Add(this.tabPage3);
			this.tabControl1.Font = null;
			this.tabControl1.HotTrack = true;
			this.tabControl1.Multiline = true;
			this.tabControl1.Name = "tabControl1";
			this.tabControl1.SelectedIndex = 0;
			this.toolTip1.SetToolTip(this.tabControl1, resources.GetString("tabControl1.ToolTip"));
			// 
			// tabPage1
			// 
			this.tabPage1.AccessibleDescription = null;
			this.tabPage1.AccessibleName = null;
			resources.ApplyResources(this.tabPage1, "tabPage1");
			this.tabPage1.BackColor = System.Drawing.Color.Transparent;
			this.tabPage1.BackgroundImage = null;
			this.tabPage1.Controls.Add(this.grpGeneral);
			this.tabPage1.Controls.Add(this.grpAudioFilenames);
			this.tabPage1.Font = null;
			this.tabPage1.Name = "tabPage1";
			this.toolTip1.SetToolTip(this.tabPage1, resources.GetString("tabPage1.ToolTip"));
			// 
			// tabPage2
			// 
			this.tabPage2.AccessibleDescription = null;
			this.tabPage2.AccessibleName = null;
			resources.ApplyResources(this.tabPage2, "tabPage2");
			this.tabPage2.BackColor = System.Drawing.SystemColors.Control;
			this.tabPage2.BackgroundImage = null;
			this.tabPage2.Controls.Add(this.groupBox3);
			this.tabPage2.Controls.Add(this.groupBox1);
			this.tabPage2.Font = null;
			this.tabPage2.Name = "tabPage2";
			this.toolTip1.SetToolTip(this.tabPage2, resources.GetString("tabPage2.ToolTip"));
			// 
			// groupBox3
			// 
			this.groupBox3.AccessibleDescription = null;
			this.groupBox3.AccessibleName = null;
			resources.ApplyResources(this.groupBox3, "groupBox3");
			this.groupBox3.BackgroundImage = null;
			this.groupBox3.Controls.Add(this.chkWriteARLogOnVerify);
			this.groupBox3.Controls.Add(this.chkWriteARTagsOnVerify);
			this.groupBox3.Font = null;
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.TabStop = false;
			this.toolTip1.SetToolTip(this.groupBox3, resources.GetString("groupBox3.ToolTip"));
			// 
			// chkWriteARLogOnVerify
			// 
			this.chkWriteARLogOnVerify.AccessibleDescription = null;
			this.chkWriteARLogOnVerify.AccessibleName = null;
			resources.ApplyResources(this.chkWriteARLogOnVerify, "chkWriteARLogOnVerify");
			this.chkWriteARLogOnVerify.BackgroundImage = null;
			this.chkWriteARLogOnVerify.Checked = true;
			this.chkWriteARLogOnVerify.CheckState = System.Windows.Forms.CheckState.Checked;
			this.chkWriteARLogOnVerify.Font = null;
			this.chkWriteARLogOnVerify.Name = "chkWriteARLogOnVerify";
			this.toolTip1.SetToolTip(this.chkWriteARLogOnVerify, resources.GetString("chkWriteARLogOnVerify.ToolTip"));
			this.chkWriteARLogOnVerify.UseVisualStyleBackColor = true;
			// 
			// tabPage3
			// 
			this.tabPage3.AccessibleDescription = null;
			this.tabPage3.AccessibleName = null;
			resources.ApplyResources(this.tabPage3, "tabPage3");
			this.tabPage3.BackColor = System.Drawing.SystemColors.Control;
			this.tabPage3.BackgroundImage = null;
			this.tabPage3.Controls.Add(this.grpWavPack);
			this.tabPage3.Controls.Add(this.groupBox2);
			this.tabPage3.Controls.Add(this.grpFLAC);
			this.tabPage3.Font = null;
			this.tabPage3.Name = "tabPage3";
			this.toolTip1.SetToolTip(this.tabPage3, resources.GetString("tabPage3.ToolTip"));
			// 
			// frmSettings
			// 
			this.AcceptButton = this.btnOK;
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.CancelButton = btnCancel;
			this.ControlBox = false;
			this.Controls.Add(this.tabControl1);
			this.Controls.Add(btnCancel);
			this.Controls.Add(this.btnOK);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Icon = null;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "frmSettings";
			this.ShowIcon = false;
			this.ShowInTaskbar = false;
			this.toolTip1.SetToolTip(this, resources.GetString("$this.ToolTip"));
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
			this.tabControl1.ResumeLayout(false);
			this.tabPage1.ResumeLayout(false);
			this.tabPage2.ResumeLayout(false);
			this.groupBox3.ResumeLayout(false);
			this.groupBox3.PerformLayout();
			this.tabPage3.ResumeLayout(false);
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
		private System.Windows.Forms.CheckBox chkWriteArTagsOnConvert;
		private System.Windows.Forms.NumericUpDown numericFLACCompressionLevel;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.ToolTip toolTip1;
		private System.Windows.Forms.NumericUpDown numFixWhenPercent;
		private System.Windows.Forms.Label labelFixWhenPercent;
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
		private System.Windows.Forms.Label labelEncodeWhenPercent;
		private System.Windows.Forms.CheckBox chkEmbedLog;
		private System.Windows.Forms.CheckBox chkFillUpCUE;
		private System.Windows.Forms.CheckBox chkFilenamesANSISafe;
		private System.Windows.Forms.GroupBox groupBox2;
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

	}
}