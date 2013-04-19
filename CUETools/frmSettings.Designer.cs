namespace JDP
{
    partial class frmSettings
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.Windows.Forms.Button btnCancel;
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmSettings));
            this.grpGeneral = new System.Windows.Forms.GroupBox();
            this.checkBoxSeparateDecodingThread = new System.Windows.Forms.CheckBox();
            this.checkBoxCheckForUpdates = new System.Windows.Forms.CheckBox();
            this.chkAllowMultipleInstances = new System.Windows.Forms.CheckBox();
            this.chkReducePriority = new System.Windows.Forms.CheckBox();
            this.chkTruncateExtra4206Samples = new System.Windows.Forms.CheckBox();
            this.chkCreateCUEFileWhenEmbedded = new System.Windows.Forms.CheckBox();
            this.chkCreateM3U = new System.Windows.Forms.CheckBox();
            this.chkAutoCorrectFilenames = new System.Windows.Forms.CheckBox();
            this.labelLanguage = new System.Windows.Forms.Label();
            this.comboLanguage = new System.Windows.Forms.ComboBox();
            this.btnOK = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.chkWriteArLogOnConvert = new System.Windows.Forms.CheckBox();
            this.chkWriteArTagsOnConvert = new System.Windows.Forms.CheckBox();
            this.chkEncodeWhenZeroOffset = new System.Windows.Forms.CheckBox();
            this.numEncodeWhenPercent = new System.Windows.Forms.NumericUpDown();
            this.labelEncodeWhenConfidence = new System.Windows.Forms.Label();
            this.numEncodeWhenConfidence = new System.Windows.Forms.NumericUpDown();
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
            this.chkEmbedLog = new System.Windows.Forms.CheckBox();
            this.chkKeepOriginalFilenames = new System.Windows.Forms.CheckBox();
            this.txtTrackFilenameFormat = new System.Windows.Forms.TextBox();
            this.txtSingleFilenameFormat = new System.Windows.Forms.TextBox();
            this.textBoxEncoderModes = new System.Windows.Forms.TextBox();
            this.encodersBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.cUEConfigBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.checkBoxEncoderLossless = new System.Windows.Forms.CheckBox();
            this.textBoxEncoderPath = new System.Windows.Forms.TextBox();
            this.textBoxEncoderParameters = new System.Windows.Forms.TextBox();
            this.rbGapsLeftOut = new System.Windows.Forms.RadioButton();
            this.rbGapsPrepended = new System.Windows.Forms.RadioButton();
            this.rbGapsAppended = new System.Windows.Forms.RadioButton();
            this.rbGapsPlusHTOA = new System.Windows.Forms.RadioButton();
            this.textBoxEncoderName = new System.Windows.Forms.TextBox();
            this.textBoxDecoderName = new System.Windows.Forms.TextBox();
            this.bindingSourceDecoders = new System.Windows.Forms.BindingSource(this.components);
            this.grpAudioFilenames = new System.Windows.Forms.GroupBox();
            this.txtSpecialExceptions = new System.Windows.Forms.TextBox();
            this.chkRemoveSpecial = new System.Windows.Forms.CheckBox();
            this.chkReplaceSpaces = new System.Windows.Forms.CheckBox();
            this.lblTrackFilenameFormat = new System.Windows.Forms.Label();
            this.lblSingleFilenameFormat = new System.Windows.Forms.Label();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.groupBoxGaps = new System.Windows.Forms.GroupBox();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.groupBoxAlbumArt = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
            this.checkBoxCopyAlbumArt = new System.Windows.Forms.CheckBox();
            this.checkBoxExtractAlbumArt = new System.Windows.Forms.CheckBox();
            this.checkBoxEmbedAlbumArt = new System.Windows.Forms.CheckBox();
            this.labelAlbumArtMaximumResolution = new System.Windows.Forms.Label();
            this.numericUpDownMaxResolution = new System.Windows.Forms.NumericUpDown();
            this.textBoxAlArtFilenameFormat = new System.Windows.Forms.TextBox();
            this.groupBoxTagging = new System.Windows.Forms.GroupBox();
            this.chkExtractLog = new System.Windows.Forms.CheckBox();
            this.checkBoxCopyBasicTags = new System.Windows.Forms.CheckBox();
            this.checkBoxWriteCUETags = new System.Windows.Forms.CheckBox();
            this.checkBoxCopyUnknownTags = new System.Windows.Forms.CheckBox();
            this.chkOverwriteTags = new System.Windows.Forms.CheckBox();
            this.chkFillUpCUE = new System.Windows.Forms.CheckBox();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.groupBoxARLog = new System.Windows.Forms.GroupBox();
            this.textBoxARLogExtension = new System.Windows.Forms.TextBox();
            this.labelLogFileExtension = new System.Windows.Forms.Label();
            this.checkBoxARLogVerbose = new System.Windows.Forms.CheckBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.label2 = new System.Windows.Forms.Label();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.checkBoxFixToNearest = new System.Windows.Forms.CheckBox();
            this.label3 = new System.Windows.Forms.Label();
            this.groupBoxVerify = new System.Windows.Forms.GroupBox();
            this.checkBoxARVerifyUseSourceFolder = new System.Windows.Forms.CheckBox();
            this.chkWriteARLogOnVerify = new System.Windows.Forms.CheckBox();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.groupBoxFormat = new System.Windows.Forms.GroupBox();
            this.comboFormatLossyEncoder = new System.Windows.Forms.ComboBox();
            this.labelFormatLossyEncoder = new System.Windows.Forms.Label();
            this.checkBoxFormatAllowLossy = new System.Windows.Forms.CheckBox();
            this.comboFormatLosslessEncoder = new System.Windows.Forms.ComboBox();
            this.labelFormatLosslessEncoder = new System.Windows.Forms.Label();
            this.checkBoxFormatEmbedCUESheet = new System.Windows.Forms.CheckBox();
            this.comboFormatDecoder = new System.Windows.Forms.ComboBox();
            this.checkBoxFormatAllowLossless = new System.Windows.Forms.CheckBox();
            this.labelFormatDefaultDecoder = new System.Windows.Forms.Label();
            this.labelFormatTagger = new System.Windows.Forms.Label();
            this.comboBoxFormatTagger = new System.Windows.Forms.ComboBox();
            this.listViewFormats = new System.Windows.Forms.ListView();
            this.columnHeader1 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.tabPageEncoders = new System.Windows.Forms.TabPage();
            this.tableLayoutPanel4 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.groupBoxExternalEncoder = new System.Windows.Forms.GroupBox();
            this.labelEncoderName = new System.Windows.Forms.Label();
            this.labelEncoderModes = new System.Windows.Forms.Label();
            this.labelEncoderPath = new System.Windows.Forms.Label();
            this.labelEncoderParameters = new System.Windows.Forms.Label();
            this.propertyGridEncoderSettings = new System.Windows.Forms.PropertyGrid();
            this.panel3 = new System.Windows.Forms.Panel();
            this.buttonEncoderDelete = new System.Windows.Forms.Button();
            this.labelEncoderExtension = new System.Windows.Forms.Label();
            this.buttonEncoderAdd = new System.Windows.Forms.Button();
            this.comboBoxEncoderExtension = new System.Windows.Forms.ComboBox();
            this.listBoxEncoders = new System.Windows.Forms.ListBox();
            this.tabPage11 = new System.Windows.Forms.TabPage();
            this.tableLayoutPanel5 = new System.Windows.Forms.TableLayoutPanel();
            this.groupBoxExternalDecoder = new System.Windows.Forms.GroupBox();
            this.label4 = new System.Windows.Forms.Label();
            this.textBoxDecoderPath = new System.Windows.Forms.TextBox();
            this.labelDecoderPath = new System.Windows.Forms.Label();
            this.labelDecoderParameters = new System.Windows.Forms.Label();
            this.textBoxDecoderParameters = new System.Windows.Forms.TextBox();
            this.panel2 = new System.Windows.Forms.Panel();
            this.buttonDecoderDelete = new System.Windows.Forms.Button();
            this.buttonDecoderAdd = new System.Windows.Forms.Button();
            this.comboBoxDecoderExtension = new System.Windows.Forms.ComboBox();
            this.labelDecoderExtension = new System.Windows.Forms.Label();
            this.listBoxDecoders = new System.Windows.Forms.ListBox();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.grpHDCD = new System.Windows.Forms.GroupBox();
            this.chkHDCDDetect = new System.Windows.Forms.CheckBox();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.richTextBoxScript = new System.Windows.Forms.RichTextBox();
            this.buttonScriptCompile = new System.Windows.Forms.Button();
            this.groupBoxScriptConditions = new System.Windows.Forms.GroupBox();
            this.listViewScriptConditions = new System.Windows.Forms.ListView();
            this.columnHeader6 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.listViewScripts = new System.Windows.Forms.ListView();
            this.columnHeader5 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.labelFormatDecoder = new System.Windows.Forms.Label();
            this.labelFormatEncoder = new System.Windows.Forms.Label();
            this.columnHeader2 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            btnCancel = new System.Windows.Forms.Button();
            this.grpGeneral.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.encodersBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.cUEConfigBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.bindingSourceDecoders)).BeginInit();
            this.grpAudioFilenames.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.groupBoxGaps.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.groupBoxAlbumArt.SuspendLayout();
            this.tableLayoutPanel3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxResolution)).BeginInit();
            this.groupBoxTagging.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.groupBoxARLog.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.groupBoxVerify.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.groupBoxFormat.SuspendLayout();
            this.tabPageEncoders.SuspendLayout();
            this.tableLayoutPanel4.SuspendLayout();
            this.panel1.SuspendLayout();
            this.groupBoxExternalEncoder.SuspendLayout();
            this.panel3.SuspendLayout();
            this.tabPage11.SuspendLayout();
            this.tableLayoutPanel5.SuspendLayout();
            this.groupBoxExternalDecoder.SuspendLayout();
            this.panel2.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.grpHDCD.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.groupBoxScriptConditions.SuspendLayout();
            this.tabPage7.SuspendLayout();
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
            this.grpGeneral.Controls.Add(this.checkBoxSeparateDecodingThread);
            this.grpGeneral.Controls.Add(this.checkBoxCheckForUpdates);
            this.grpGeneral.Controls.Add(this.chkAllowMultipleInstances);
            this.grpGeneral.Controls.Add(this.chkReducePriority);
            this.grpGeneral.Controls.Add(this.chkTruncateExtra4206Samples);
            this.grpGeneral.Controls.Add(this.chkCreateCUEFileWhenEmbedded);
            this.grpGeneral.Controls.Add(this.chkCreateM3U);
            this.grpGeneral.Controls.Add(this.chkAutoCorrectFilenames);
            resources.ApplyResources(this.grpGeneral, "grpGeneral");
            this.grpGeneral.Name = "grpGeneral";
            this.grpGeneral.TabStop = false;
            // 
            // checkBoxSeparateDecodingThread
            // 
            resources.ApplyResources(this.checkBoxSeparateDecodingThread, "checkBoxSeparateDecodingThread");
            this.checkBoxSeparateDecodingThread.Name = "checkBoxSeparateDecodingThread";
            this.toolTip1.SetToolTip(this.checkBoxSeparateDecodingThread, resources.GetString("checkBoxSeparateDecodingThread.ToolTip"));
            this.checkBoxSeparateDecodingThread.UseVisualStyleBackColor = true;
            // 
            // checkBoxCheckForUpdates
            // 
            resources.ApplyResources(this.checkBoxCheckForUpdates, "checkBoxCheckForUpdates");
            this.checkBoxCheckForUpdates.Name = "checkBoxCheckForUpdates";
            this.toolTip1.SetToolTip(this.checkBoxCheckForUpdates, resources.GetString("checkBoxCheckForUpdates.ToolTip"));
            this.checkBoxCheckForUpdates.UseVisualStyleBackColor = true;
            // 
            // chkAllowMultipleInstances
            // 
            resources.ApplyResources(this.chkAllowMultipleInstances, "chkAllowMultipleInstances");
            this.chkAllowMultipleInstances.Name = "chkAllowMultipleInstances";
            this.toolTip1.SetToolTip(this.chkAllowMultipleInstances, resources.GetString("chkAllowMultipleInstances.ToolTip"));
            this.chkAllowMultipleInstances.UseVisualStyleBackColor = true;
            // 
            // chkReducePriority
            // 
            resources.ApplyResources(this.chkReducePriority, "chkReducePriority");
            this.chkReducePriority.Name = "chkReducePriority";
            this.toolTip1.SetToolTip(this.chkReducePriority, resources.GetString("chkReducePriority.ToolTip"));
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
            this.toolTip1.SetToolTip(this.chkCreateCUEFileWhenEmbedded, resources.GetString("chkCreateCUEFileWhenEmbedded.ToolTip"));
            this.chkCreateCUEFileWhenEmbedded.UseVisualStyleBackColor = true;
            // 
            // chkCreateM3U
            // 
            resources.ApplyResources(this.chkCreateM3U, "chkCreateM3U");
            this.chkCreateM3U.Name = "chkCreateM3U";
            this.toolTip1.SetToolTip(this.chkCreateM3U, resources.GetString("chkCreateM3U.ToolTip"));
            this.chkCreateM3U.UseVisualStyleBackColor = true;
            // 
            // chkAutoCorrectFilenames
            // 
            resources.ApplyResources(this.chkAutoCorrectFilenames, "chkAutoCorrectFilenames");
            this.chkAutoCorrectFilenames.Name = "chkAutoCorrectFilenames";
            this.toolTip1.SetToolTip(this.chkAutoCorrectFilenames, resources.GetString("chkAutoCorrectFilenames.ToolTip"));
            this.chkAutoCorrectFilenames.UseVisualStyleBackColor = true;
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
            this.toolTip1.SetToolTip(this.comboLanguage, resources.GetString("comboLanguage.ToolTip"));
            // 
            // btnOK
            // 
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            resources.ApplyResources(this.btnOK, "btnOK");
            this.btnOK.Name = "btnOK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.chkWriteArLogOnConvert);
            this.groupBox1.Controls.Add(this.chkWriteArTagsOnConvert);
            resources.ApplyResources(this.groupBox1, "groupBox1");
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.TabStop = false;
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
            // chkEncodeWhenZeroOffset
            // 
            resources.ApplyResources(this.chkEncodeWhenZeroOffset, "chkEncodeWhenZeroOffset");
            this.chkEncodeWhenZeroOffset.Name = "chkEncodeWhenZeroOffset";
            // 
            // numEncodeWhenPercent
            // 
            resources.ApplyResources(this.numEncodeWhenPercent, "numEncodeWhenPercent");
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
            resources.ApplyResources(this.numFixWhenPercent, "numFixWhenPercent");
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
            // chkEmbedLog
            // 
            resources.ApplyResources(this.chkEmbedLog, "chkEmbedLog");
            this.chkEmbedLog.Name = "chkEmbedLog";
            this.toolTip1.SetToolTip(this.chkEmbedLog, resources.GetString("chkEmbedLog.ToolTip"));
            this.chkEmbedLog.UseVisualStyleBackColor = true;
            // 
            // chkKeepOriginalFilenames
            // 
            resources.ApplyResources(this.chkKeepOriginalFilenames, "chkKeepOriginalFilenames");
            this.chkKeepOriginalFilenames.Checked = true;
            this.chkKeepOriginalFilenames.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkKeepOriginalFilenames.Name = "chkKeepOriginalFilenames";
            this.toolTip1.SetToolTip(this.chkKeepOriginalFilenames, resources.GetString("chkKeepOriginalFilenames.ToolTip"));
            this.chkKeepOriginalFilenames.UseVisualStyleBackColor = true;
            // 
            // txtTrackFilenameFormat
            // 
            this.txtTrackFilenameFormat.AutoCompleteCustomSource.AddRange(new string[] {
            resources.GetString("txtTrackFilenameFormat.AutoCompleteCustomSource"),
            resources.GetString("txtTrackFilenameFormat.AutoCompleteCustomSource1"),
            resources.GetString("txtTrackFilenameFormat.AutoCompleteCustomSource2"),
            resources.GetString("txtTrackFilenameFormat.AutoCompleteCustomSource3")});
            this.txtTrackFilenameFormat.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.txtTrackFilenameFormat.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.CustomSource;
            resources.ApplyResources(this.txtTrackFilenameFormat, "txtTrackFilenameFormat");
            this.txtTrackFilenameFormat.Name = "txtTrackFilenameFormat";
            this.toolTip1.SetToolTip(this.txtTrackFilenameFormat, resources.GetString("txtTrackFilenameFormat.ToolTip"));
            // 
            // txtSingleFilenameFormat
            // 
            this.txtSingleFilenameFormat.AutoCompleteCustomSource.AddRange(new string[] {
            resources.GetString("txtSingleFilenameFormat.AutoCompleteCustomSource")});
            resources.ApplyResources(this.txtSingleFilenameFormat, "txtSingleFilenameFormat");
            this.txtSingleFilenameFormat.Name = "txtSingleFilenameFormat";
            this.toolTip1.SetToolTip(this.txtSingleFilenameFormat, resources.GetString("txtSingleFilenameFormat.ToolTip"));
            // 
            // textBoxEncoderModes
            // 
            resources.ApplyResources(this.textBoxEncoderModes, "textBoxEncoderModes");
            this.textBoxEncoderModes.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.encodersBindingSource, "SupportedModesStr", true));
            this.textBoxEncoderModes.Name = "textBoxEncoderModes";
            this.toolTip1.SetToolTip(this.textBoxEncoderModes, resources.GetString("textBoxEncoderModes.ToolTip"));
            // 
            // encodersBindingSource
            // 
            this.encodersBindingSource.AllowNew = true;
            this.encodersBindingSource.DataMember = "Encoders";
            this.encodersBindingSource.DataSource = this.cUEConfigBindingSource;
            this.encodersBindingSource.CurrentItemChanged += new System.EventHandler(this.encodersBindingSource_CurrentItemChanged);
            // 
            // cUEConfigBindingSource
            // 
            this.cUEConfigBindingSource.DataSource = typeof(CUETools.Processor.CUEConfig);
            // 
            // checkBoxEncoderLossless
            // 
            resources.ApplyResources(this.checkBoxEncoderLossless, "checkBoxEncoderLossless");
            this.checkBoxEncoderLossless.DataBindings.Add(new System.Windows.Forms.Binding("Checked", this.encodersBindingSource, "Lossless", true));
            this.checkBoxEncoderLossless.Name = "checkBoxEncoderLossless";
            this.toolTip1.SetToolTip(this.checkBoxEncoderLossless, resources.GetString("checkBoxEncoderLossless.ToolTip"));
            this.checkBoxEncoderLossless.UseVisualStyleBackColor = true;
            // 
            // textBoxEncoderPath
            // 
            resources.ApplyResources(this.textBoxEncoderPath, "textBoxEncoderPath");
            this.textBoxEncoderPath.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.encodersBindingSource, "Path", true));
            this.textBoxEncoderPath.Name = "textBoxEncoderPath";
            this.toolTip1.SetToolTip(this.textBoxEncoderPath, resources.GetString("textBoxEncoderPath.ToolTip"));
            // 
            // textBoxEncoderParameters
            // 
            resources.ApplyResources(this.textBoxEncoderParameters, "textBoxEncoderParameters");
            this.textBoxEncoderParameters.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.encodersBindingSource, "Parameters", true));
            this.textBoxEncoderParameters.Name = "textBoxEncoderParameters";
            this.toolTip1.SetToolTip(this.textBoxEncoderParameters, resources.GetString("textBoxEncoderParameters.ToolTip"));
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
            // rbGapsPlusHTOA
            // 
            resources.ApplyResources(this.rbGapsPlusHTOA, "rbGapsPlusHTOA");
            this.rbGapsPlusHTOA.Name = "rbGapsPlusHTOA";
            this.rbGapsPlusHTOA.TabStop = true;
            this.toolTip1.SetToolTip(this.rbGapsPlusHTOA, resources.GetString("rbGapsPlusHTOA.ToolTip"));
            this.rbGapsPlusHTOA.UseVisualStyleBackColor = true;
            // 
            // textBoxEncoderName
            // 
            resources.ApplyResources(this.textBoxEncoderName, "textBoxEncoderName");
            this.textBoxEncoderName.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.encodersBindingSource, "Name", true));
            this.textBoxEncoderName.Name = "textBoxEncoderName";
            this.toolTip1.SetToolTip(this.textBoxEncoderName, resources.GetString("textBoxEncoderName.ToolTip"));
            // 
            // textBoxDecoderName
            // 
            resources.ApplyResources(this.textBoxDecoderName, "textBoxDecoderName");
            this.textBoxDecoderName.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.bindingSourceDecoders, "Name", true));
            this.textBoxDecoderName.Name = "textBoxDecoderName";
            this.toolTip1.SetToolTip(this.textBoxDecoderName, resources.GetString("textBoxDecoderName.ToolTip"));
            // 
            // bindingSourceDecoders
            // 
            this.bindingSourceDecoders.AllowNew = true;
            this.bindingSourceDecoders.DataMember = "Decoders";
            this.bindingSourceDecoders.DataSource = this.cUEConfigBindingSource;
            this.bindingSourceDecoders.Sort = "";
            this.bindingSourceDecoders.CurrentItemChanged += new System.EventHandler(this.bindingSourceDecoders_CurrentItemChanged);
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
            // tabControl1
            // 
            resources.ApplyResources(this.tabControl1, "tabControl1");
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage6);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPageEncoders);
            this.tabControl1.Controls.Add(this.tabPage11);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage7);
            this.tabControl1.HotTrack = true;
            this.tabControl1.Multiline = true;
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Deselecting += new System.Windows.Forms.TabControlCancelEventHandler(this.tabControl1_Deselecting);
            // 
            // tabPage1
            // 
            this.tabPage1.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage1.Controls.Add(this.labelLanguage);
            this.tabPage1.Controls.Add(this.comboLanguage);
            this.tabPage1.Controls.Add(this.groupBoxGaps);
            this.tabPage1.Controls.Add(this.grpGeneral);
            this.tabPage1.Controls.Add(this.grpAudioFilenames);
            resources.ApplyResources(this.tabPage1, "tabPage1");
            this.tabPage1.Name = "tabPage1";
            // 
            // groupBoxGaps
            // 
            this.groupBoxGaps.Controls.Add(this.rbGapsPlusHTOA);
            this.groupBoxGaps.Controls.Add(this.rbGapsAppended);
            this.groupBoxGaps.Controls.Add(this.rbGapsLeftOut);
            this.groupBoxGaps.Controls.Add(this.rbGapsPrepended);
            resources.ApplyResources(this.groupBoxGaps, "groupBoxGaps");
            this.groupBoxGaps.Name = "groupBoxGaps";
            this.groupBoxGaps.TabStop = false;
            // 
            // tabPage6
            // 
            this.tabPage6.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage6.Controls.Add(this.groupBoxAlbumArt);
            this.tabPage6.Controls.Add(this.groupBoxTagging);
            resources.ApplyResources(this.tabPage6, "tabPage6");
            this.tabPage6.Name = "tabPage6";
            // 
            // groupBoxAlbumArt
            // 
            this.groupBoxAlbumArt.Controls.Add(this.tableLayoutPanel3);
            resources.ApplyResources(this.groupBoxAlbumArt, "groupBoxAlbumArt");
            this.groupBoxAlbumArt.Name = "groupBoxAlbumArt";
            this.groupBoxAlbumArt.TabStop = false;
            // 
            // tableLayoutPanel3
            // 
            resources.ApplyResources(this.tableLayoutPanel3, "tableLayoutPanel3");
            this.tableLayoutPanel3.Controls.Add(this.checkBoxCopyAlbumArt, 0, 0);
            this.tableLayoutPanel3.Controls.Add(this.checkBoxExtractAlbumArt, 0, 2);
            this.tableLayoutPanel3.Controls.Add(this.checkBoxEmbedAlbumArt, 0, 1);
            this.tableLayoutPanel3.Controls.Add(this.labelAlbumArtMaximumResolution, 0, 3);
            this.tableLayoutPanel3.Controls.Add(this.numericUpDownMaxResolution, 2, 3);
            this.tableLayoutPanel3.Controls.Add(this.textBoxAlArtFilenameFormat, 1, 2);
            this.tableLayoutPanel3.Name = "tableLayoutPanel3";
            // 
            // checkBoxCopyAlbumArt
            // 
            resources.ApplyResources(this.checkBoxCopyAlbumArt, "checkBoxCopyAlbumArt");
            this.checkBoxCopyAlbumArt.DataBindings.Add(new System.Windows.Forms.Binding("Checked", this.cUEConfigBindingSource, "CopyAlbumArt", true));
            this.checkBoxCopyAlbumArt.Name = "checkBoxCopyAlbumArt";
            this.checkBoxCopyAlbumArt.UseVisualStyleBackColor = true;
            // 
            // checkBoxExtractAlbumArt
            // 
            resources.ApplyResources(this.checkBoxExtractAlbumArt, "checkBoxExtractAlbumArt");
            this.checkBoxExtractAlbumArt.Name = "checkBoxExtractAlbumArt";
            this.checkBoxExtractAlbumArt.UseVisualStyleBackColor = true;
            // 
            // checkBoxEmbedAlbumArt
            // 
            resources.ApplyResources(this.checkBoxEmbedAlbumArt, "checkBoxEmbedAlbumArt");
            this.checkBoxEmbedAlbumArt.Name = "checkBoxEmbedAlbumArt";
            this.checkBoxEmbedAlbumArt.UseVisualStyleBackColor = true;
            // 
            // labelAlbumArtMaximumResolution
            // 
            this.tableLayoutPanel3.SetColumnSpan(this.labelAlbumArtMaximumResolution, 2);
            resources.ApplyResources(this.labelAlbumArtMaximumResolution, "labelAlbumArtMaximumResolution");
            this.labelAlbumArtMaximumResolution.Name = "labelAlbumArtMaximumResolution";
            // 
            // numericUpDownMaxResolution
            // 
            resources.ApplyResources(this.numericUpDownMaxResolution, "numericUpDownMaxResolution");
            this.numericUpDownMaxResolution.Increment = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numericUpDownMaxResolution.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numericUpDownMaxResolution.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numericUpDownMaxResolution.Name = "numericUpDownMaxResolution";
            this.numericUpDownMaxResolution.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
            // 
            // textBoxAlArtFilenameFormat
            // 
            this.textBoxAlArtFilenameFormat.AutoCompleteCustomSource.AddRange(new string[] {
            resources.GetString("textBoxAlArtFilenameFormat.AutoCompleteCustomSource"),
            resources.GetString("textBoxAlArtFilenameFormat.AutoCompleteCustomSource1"),
            resources.GetString("textBoxAlArtFilenameFormat.AutoCompleteCustomSource2"),
            resources.GetString("textBoxAlArtFilenameFormat.AutoCompleteCustomSource3")});
            this.textBoxAlArtFilenameFormat.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.textBoxAlArtFilenameFormat.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.CustomSource;
            this.tableLayoutPanel3.SetColumnSpan(this.textBoxAlArtFilenameFormat, 2);
            this.textBoxAlArtFilenameFormat.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.cUEConfigBindingSource, "AlArtFilenameFormat", true));
            resources.ApplyResources(this.textBoxAlArtFilenameFormat, "textBoxAlArtFilenameFormat");
            this.textBoxAlArtFilenameFormat.Name = "textBoxAlArtFilenameFormat";
            // 
            // groupBoxTagging
            // 
            this.groupBoxTagging.Controls.Add(this.chkExtractLog);
            this.groupBoxTagging.Controls.Add(this.checkBoxCopyBasicTags);
            this.groupBoxTagging.Controls.Add(this.checkBoxWriteCUETags);
            this.groupBoxTagging.Controls.Add(this.checkBoxCopyUnknownTags);
            this.groupBoxTagging.Controls.Add(this.chkOverwriteTags);
            this.groupBoxTagging.Controls.Add(this.chkFillUpCUE);
            this.groupBoxTagging.Controls.Add(this.chkEmbedLog);
            resources.ApplyResources(this.groupBoxTagging, "groupBoxTagging");
            this.groupBoxTagging.Name = "groupBoxTagging";
            this.groupBoxTagging.TabStop = false;
            // 
            // chkExtractLog
            // 
            resources.ApplyResources(this.chkExtractLog, "chkExtractLog");
            this.chkExtractLog.Name = "chkExtractLog";
            this.chkExtractLog.UseVisualStyleBackColor = true;
            // 
            // checkBoxCopyBasicTags
            // 
            resources.ApplyResources(this.checkBoxCopyBasicTags, "checkBoxCopyBasicTags");
            this.checkBoxCopyBasicTags.Name = "checkBoxCopyBasicTags";
            this.checkBoxCopyBasicTags.UseVisualStyleBackColor = true;
            // 
            // checkBoxWriteCUETags
            // 
            resources.ApplyResources(this.checkBoxWriteCUETags, "checkBoxWriteCUETags");
            this.checkBoxWriteCUETags.Name = "checkBoxWriteCUETags";
            this.checkBoxWriteCUETags.UseVisualStyleBackColor = true;
            // 
            // checkBoxCopyUnknownTags
            // 
            resources.ApplyResources(this.checkBoxCopyUnknownTags, "checkBoxCopyUnknownTags");
            this.checkBoxCopyUnknownTags.Name = "checkBoxCopyUnknownTags";
            this.checkBoxCopyUnknownTags.UseVisualStyleBackColor = true;
            // 
            // chkOverwriteTags
            // 
            resources.ApplyResources(this.chkOverwriteTags, "chkOverwriteTags");
            this.chkOverwriteTags.Name = "chkOverwriteTags";
            this.chkOverwriteTags.UseVisualStyleBackColor = true;
            // 
            // chkFillUpCUE
            // 
            resources.ApplyResources(this.chkFillUpCUE, "chkFillUpCUE");
            this.chkFillUpCUE.Name = "chkFillUpCUE";
            this.chkFillUpCUE.UseVisualStyleBackColor = true;
            // 
            // tabPage2
            // 
            this.tabPage2.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage2.Controls.Add(this.groupBoxARLog);
            this.tabPage2.Controls.Add(this.groupBox5);
            this.tabPage2.Controls.Add(this.groupBox4);
            this.tabPage2.Controls.Add(this.groupBoxVerify);
            this.tabPage2.Controls.Add(this.groupBox1);
            resources.ApplyResources(this.tabPage2, "tabPage2");
            this.tabPage2.Name = "tabPage2";
            // 
            // groupBoxARLog
            // 
            this.groupBoxARLog.Controls.Add(this.textBoxARLogExtension);
            this.groupBoxARLog.Controls.Add(this.labelLogFileExtension);
            this.groupBoxARLog.Controls.Add(this.checkBoxARLogVerbose);
            resources.ApplyResources(this.groupBoxARLog, "groupBoxARLog");
            this.groupBoxARLog.Name = "groupBoxARLog";
            this.groupBoxARLog.TabStop = false;
            // 
            // textBoxARLogExtension
            // 
            this.textBoxARLogExtension.AutoCompleteCustomSource.AddRange(new string[] {
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource1"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource2"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource3"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource4"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource5"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource6"),
            resources.GetString("textBoxARLogExtension.AutoCompleteCustomSource7")});
            this.textBoxARLogExtension.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.textBoxARLogExtension.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.CustomSource;
            this.textBoxARLogExtension.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.cUEConfigBindingSource, "ArLogFilenameFormat", true));
            resources.ApplyResources(this.textBoxARLogExtension, "textBoxARLogExtension");
            this.textBoxARLogExtension.Name = "textBoxARLogExtension";
            // 
            // labelLogFileExtension
            // 
            resources.ApplyResources(this.labelLogFileExtension, "labelLogFileExtension");
            this.labelLogFileExtension.Name = "labelLogFileExtension";
            // 
            // checkBoxARLogVerbose
            // 
            resources.ApplyResources(this.checkBoxARLogVerbose, "checkBoxARLogVerbose");
            this.checkBoxARLogVerbose.Name = "checkBoxARLogVerbose";
            this.checkBoxARLogVerbose.UseVisualStyleBackColor = true;
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.tableLayoutPanel2);
            resources.ApplyResources(this.groupBox5, "groupBox5");
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.TabStop = false;
            // 
            // tableLayoutPanel2
            // 
            resources.ApplyResources(this.tableLayoutPanel2, "tableLayoutPanel2");
            this.tableLayoutPanel2.Controls.Add(this.label2, 0, 0);
            this.tableLayoutPanel2.Controls.Add(this.chkEncodeWhenZeroOffset, 0, 2);
            this.tableLayoutPanel2.Controls.Add(this.numEncodeWhenPercent, 1, 0);
            this.tableLayoutPanel2.Controls.Add(this.labelEncodeWhenConfidence, 0, 1);
            this.tableLayoutPanel2.Controls.Add(this.numEncodeWhenConfidence, 1, 1);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            // 
            // label2
            // 
            resources.ApplyResources(this.label2, "label2");
            this.label2.Name = "label2";
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.tableLayoutPanel1);
            resources.ApplyResources(this.groupBox4, "groupBox4");
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.TabStop = false;
            // 
            // tableLayoutPanel1
            // 
            resources.ApplyResources(this.tableLayoutPanel1, "tableLayoutPanel1");
            this.tableLayoutPanel1.Controls.Add(this.checkBoxFixToNearest, 0, 2);
            this.tableLayoutPanel1.Controls.Add(this.label3, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.numFixWhenConfidence, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.labelFixWhenConfidence, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.numFixWhenPercent, 1, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            // 
            // checkBoxFixToNearest
            // 
            resources.ApplyResources(this.checkBoxFixToNearest, "checkBoxFixToNearest");
            this.tableLayoutPanel1.SetColumnSpan(this.checkBoxFixToNearest, 2);
            this.checkBoxFixToNearest.Name = "checkBoxFixToNearest";
            this.checkBoxFixToNearest.UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            resources.ApplyResources(this.label3, "label3");
            this.label3.Name = "label3";
            // 
            // groupBoxVerify
            // 
            this.groupBoxVerify.Controls.Add(this.checkBoxARVerifyUseSourceFolder);
            this.groupBoxVerify.Controls.Add(this.chkWriteARLogOnVerify);
            this.groupBoxVerify.Controls.Add(this.chkWriteARTagsOnVerify);
            resources.ApplyResources(this.groupBoxVerify, "groupBoxVerify");
            this.groupBoxVerify.Name = "groupBoxVerify";
            this.groupBoxVerify.TabStop = false;
            // 
            // checkBoxARVerifyUseSourceFolder
            // 
            resources.ApplyResources(this.checkBoxARVerifyUseSourceFolder, "checkBoxARVerifyUseSourceFolder");
            this.checkBoxARVerifyUseSourceFolder.Name = "checkBoxARVerifyUseSourceFolder";
            this.checkBoxARVerifyUseSourceFolder.UseVisualStyleBackColor = true;
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
            this.tabPage3.Controls.Add(this.groupBoxFormat);
            this.tabPage3.Controls.Add(this.listViewFormats);
            resources.ApplyResources(this.tabPage3, "tabPage3");
            this.tabPage3.Name = "tabPage3";
            // 
            // groupBoxFormat
            // 
            this.groupBoxFormat.Controls.Add(this.comboFormatLossyEncoder);
            this.groupBoxFormat.Controls.Add(this.labelFormatLossyEncoder);
            this.groupBoxFormat.Controls.Add(this.checkBoxFormatAllowLossy);
            this.groupBoxFormat.Controls.Add(this.comboFormatLosslessEncoder);
            this.groupBoxFormat.Controls.Add(this.labelFormatLosslessEncoder);
            this.groupBoxFormat.Controls.Add(this.checkBoxFormatEmbedCUESheet);
            this.groupBoxFormat.Controls.Add(this.comboFormatDecoder);
            this.groupBoxFormat.Controls.Add(this.checkBoxFormatAllowLossless);
            this.groupBoxFormat.Controls.Add(this.labelFormatDefaultDecoder);
            this.groupBoxFormat.Controls.Add(this.labelFormatTagger);
            this.groupBoxFormat.Controls.Add(this.comboBoxFormatTagger);
            resources.ApplyResources(this.groupBoxFormat, "groupBoxFormat");
            this.groupBoxFormat.Name = "groupBoxFormat";
            this.groupBoxFormat.TabStop = false;
            // 
            // comboFormatLossyEncoder
            // 
            this.comboFormatLossyEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboFormatLossyEncoder.FormattingEnabled = true;
            resources.ApplyResources(this.comboFormatLossyEncoder, "comboFormatLossyEncoder");
            this.comboFormatLossyEncoder.Name = "comboFormatLossyEncoder";
            // 
            // labelFormatLossyEncoder
            // 
            resources.ApplyResources(this.labelFormatLossyEncoder, "labelFormatLossyEncoder");
            this.labelFormatLossyEncoder.Name = "labelFormatLossyEncoder";
            // 
            // checkBoxFormatAllowLossy
            // 
            resources.ApplyResources(this.checkBoxFormatAllowLossy, "checkBoxFormatAllowLossy");
            this.checkBoxFormatAllowLossy.Name = "checkBoxFormatAllowLossy";
            this.checkBoxFormatAllowLossy.UseVisualStyleBackColor = true;
            // 
            // comboFormatLosslessEncoder
            // 
            this.comboFormatLosslessEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboFormatLosslessEncoder.FormattingEnabled = true;
            resources.ApplyResources(this.comboFormatLosslessEncoder, "comboFormatLosslessEncoder");
            this.comboFormatLosslessEncoder.Name = "comboFormatLosslessEncoder";
            // 
            // labelFormatLosslessEncoder
            // 
            resources.ApplyResources(this.labelFormatLosslessEncoder, "labelFormatLosslessEncoder");
            this.labelFormatLosslessEncoder.Name = "labelFormatLosslessEncoder";
            // 
            // checkBoxFormatEmbedCUESheet
            // 
            resources.ApplyResources(this.checkBoxFormatEmbedCUESheet, "checkBoxFormatEmbedCUESheet");
            this.checkBoxFormatEmbedCUESheet.Name = "checkBoxFormatEmbedCUESheet";
            this.checkBoxFormatEmbedCUESheet.UseVisualStyleBackColor = true;
            // 
            // comboFormatDecoder
            // 
            this.comboFormatDecoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboFormatDecoder.FormattingEnabled = true;
            resources.ApplyResources(this.comboFormatDecoder, "comboFormatDecoder");
            this.comboFormatDecoder.Name = "comboFormatDecoder";
            // 
            // checkBoxFormatAllowLossless
            // 
            resources.ApplyResources(this.checkBoxFormatAllowLossless, "checkBoxFormatAllowLossless");
            this.checkBoxFormatAllowLossless.Name = "checkBoxFormatAllowLossless";
            this.checkBoxFormatAllowLossless.UseVisualStyleBackColor = true;
            // 
            // labelFormatDefaultDecoder
            // 
            resources.ApplyResources(this.labelFormatDefaultDecoder, "labelFormatDefaultDecoder");
            this.labelFormatDefaultDecoder.Name = "labelFormatDefaultDecoder";
            // 
            // labelFormatTagger
            // 
            resources.ApplyResources(this.labelFormatTagger, "labelFormatTagger");
            this.labelFormatTagger.Name = "labelFormatTagger";
            // 
            // comboBoxFormatTagger
            // 
            this.comboBoxFormatTagger.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxFormatTagger.FormattingEnabled = true;
            resources.ApplyResources(this.comboBoxFormatTagger, "comboBoxFormatTagger");
            this.comboBoxFormatTagger.Name = "comboBoxFormatTagger";
            // 
            // listViewFormats
            // 
            this.listViewFormats.BackColor = System.Drawing.SystemColors.Control;
            this.listViewFormats.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.listViewFormats.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1});
            this.listViewFormats.FullRowSelect = true;
            this.listViewFormats.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.listViewFormats.HideSelection = false;
            this.listViewFormats.LabelEdit = true;
            resources.ApplyResources(this.listViewFormats, "listViewFormats");
            this.listViewFormats.MultiSelect = false;
            this.listViewFormats.Name = "listViewFormats";
            this.listViewFormats.UseCompatibleStateImageBehavior = false;
            this.listViewFormats.View = System.Windows.Forms.View.Details;
            this.listViewFormats.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listViewFormats_AfterLabelEdit);
            this.listViewFormats.BeforeLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listViewFormats_BeforeLabelEdit);
            this.listViewFormats.ItemSelectionChanged += new System.Windows.Forms.ListViewItemSelectionChangedEventHandler(this.listViewFormats_ItemSelectionChanged);
            this.listViewFormats.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listViewFormats_KeyDown);
            // 
            // columnHeader1
            // 
            resources.ApplyResources(this.columnHeader1, "columnHeader1");
            // 
            // tabPageEncoders
            // 
            resources.ApplyResources(this.tabPageEncoders, "tabPageEncoders");
            this.tabPageEncoders.BackColor = System.Drawing.SystemColors.Control;
            this.tabPageEncoders.Controls.Add(this.tableLayoutPanel4);
            this.tabPageEncoders.Name = "tabPageEncoders";
            // 
            // tableLayoutPanel4
            // 
            resources.ApplyResources(this.tableLayoutPanel4, "tableLayoutPanel4");
            this.tableLayoutPanel4.Controls.Add(this.panel1, 1, 1);
            this.tableLayoutPanel4.Controls.Add(this.panel3, 1, 0);
            this.tableLayoutPanel4.Controls.Add(this.listBoxEncoders, 0, 0);
            this.tableLayoutPanel4.Name = "tableLayoutPanel4";
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.groupBoxExternalEncoder);
            this.panel1.Controls.Add(this.propertyGridEncoderSettings);
            resources.ApplyResources(this.panel1, "panel1");
            this.panel1.Name = "panel1";
            // 
            // groupBoxExternalEncoder
            // 
            resources.ApplyResources(this.groupBoxExternalEncoder, "groupBoxExternalEncoder");
            this.groupBoxExternalEncoder.Controls.Add(this.labelEncoderName);
            this.groupBoxExternalEncoder.Controls.Add(this.textBoxEncoderName);
            this.groupBoxExternalEncoder.Controls.Add(this.textBoxEncoderPath);
            this.groupBoxExternalEncoder.Controls.Add(this.labelEncoderModes);
            this.groupBoxExternalEncoder.Controls.Add(this.textBoxEncoderModes);
            this.groupBoxExternalEncoder.Controls.Add(this.checkBoxEncoderLossless);
            this.groupBoxExternalEncoder.Controls.Add(this.textBoxEncoderParameters);
            this.groupBoxExternalEncoder.Controls.Add(this.labelEncoderPath);
            this.groupBoxExternalEncoder.Controls.Add(this.labelEncoderParameters);
            this.groupBoxExternalEncoder.Name = "groupBoxExternalEncoder";
            this.groupBoxExternalEncoder.TabStop = false;
            // 
            // labelEncoderName
            // 
            resources.ApplyResources(this.labelEncoderName, "labelEncoderName");
            this.labelEncoderName.Name = "labelEncoderName";
            // 
            // labelEncoderModes
            // 
            resources.ApplyResources(this.labelEncoderModes, "labelEncoderModes");
            this.labelEncoderModes.Name = "labelEncoderModes";
            // 
            // labelEncoderPath
            // 
            resources.ApplyResources(this.labelEncoderPath, "labelEncoderPath");
            this.labelEncoderPath.Name = "labelEncoderPath";
            // 
            // labelEncoderParameters
            // 
            resources.ApplyResources(this.labelEncoderParameters, "labelEncoderParameters");
            this.labelEncoderParameters.Name = "labelEncoderParameters";
            // 
            // propertyGridEncoderSettings
            // 
            resources.ApplyResources(this.propertyGridEncoderSettings, "propertyGridEncoderSettings");
            this.propertyGridEncoderSettings.Name = "propertyGridEncoderSettings";
            this.propertyGridEncoderSettings.PropertySort = System.Windows.Forms.PropertySort.Categorized;
            this.propertyGridEncoderSettings.ToolbarVisible = false;
            // 
            // panel3
            // 
            this.panel3.Controls.Add(this.buttonEncoderDelete);
            this.panel3.Controls.Add(this.labelEncoderExtension);
            this.panel3.Controls.Add(this.buttonEncoderAdd);
            this.panel3.Controls.Add(this.comboBoxEncoderExtension);
            resources.ApplyResources(this.panel3, "panel3");
            this.panel3.Name = "panel3";
            // 
            // buttonEncoderDelete
            // 
            this.buttonEncoderDelete.DataBindings.Add(new System.Windows.Forms.Binding("Enabled", this.encodersBindingSource, "CanBeDeleted", true));
            resources.ApplyResources(this.buttonEncoderDelete, "buttonEncoderDelete");
            this.buttonEncoderDelete.Name = "buttonEncoderDelete";
            this.buttonEncoderDelete.UseVisualStyleBackColor = true;
            this.buttonEncoderDelete.Click += new System.EventHandler(this.buttonEncoderDelete_Click);
            // 
            // labelEncoderExtension
            // 
            resources.ApplyResources(this.labelEncoderExtension, "labelEncoderExtension");
            this.labelEncoderExtension.DataBindings.Add(new System.Windows.Forms.Binding("ImageKey", this.encodersBindingSource, "DotExtension", true, System.Windows.Forms.DataSourceUpdateMode.Never));
            this.labelEncoderExtension.Name = "labelEncoderExtension";
            // 
            // buttonEncoderAdd
            // 
            resources.ApplyResources(this.buttonEncoderAdd, "buttonEncoderAdd");
            this.buttonEncoderAdd.Name = "buttonEncoderAdd";
            this.buttonEncoderAdd.UseVisualStyleBackColor = true;
            this.buttonEncoderAdd.Click += new System.EventHandler(this.buttonEncoderAdd_Click);
            // 
            // comboBoxEncoderExtension
            // 
            resources.ApplyResources(this.comboBoxEncoderExtension, "comboBoxEncoderExtension");
            this.comboBoxEncoderExtension.DataBindings.Add(new System.Windows.Forms.Binding("SelectedItem", this.encodersBindingSource, "Extension", true));
            this.comboBoxEncoderExtension.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxEncoderExtension.FormattingEnabled = true;
            this.comboBoxEncoderExtension.Name = "comboBoxEncoderExtension";
            this.comboBoxEncoderExtension.SelectedIndexChanged += new System.EventHandler(this.comboBoxEncoderExtension_SelectedIndexChanged);
            // 
            // listBoxEncoders
            // 
            this.listBoxEncoders.BackColor = System.Drawing.SystemColors.Control;
            this.listBoxEncoders.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.listBoxEncoders.DataSource = this.encodersBindingSource;
            this.listBoxEncoders.DisplayMember = "FullName";
            resources.ApplyResources(this.listBoxEncoders, "listBoxEncoders");
            this.listBoxEncoders.Name = "listBoxEncoders";
            this.tableLayoutPanel4.SetRowSpan(this.listBoxEncoders, 2);
            this.listBoxEncoders.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listBoxEncoders_KeyDown);
            // 
            // tabPage11
            // 
            this.tabPage11.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage11.Controls.Add(this.tableLayoutPanel5);
            resources.ApplyResources(this.tabPage11, "tabPage11");
            this.tabPage11.Name = "tabPage11";
            // 
            // tableLayoutPanel5
            // 
            resources.ApplyResources(this.tableLayoutPanel5, "tableLayoutPanel5");
            this.tableLayoutPanel5.Controls.Add(this.groupBoxExternalDecoder, 1, 1);
            this.tableLayoutPanel5.Controls.Add(this.panel2, 1, 0);
            this.tableLayoutPanel5.Controls.Add(this.listBoxDecoders, 0, 0);
            this.tableLayoutPanel5.Name = "tableLayoutPanel5";
            // 
            // groupBoxExternalDecoder
            // 
            this.groupBoxExternalDecoder.Controls.Add(this.label4);
            this.groupBoxExternalDecoder.Controls.Add(this.textBoxDecoderName);
            this.groupBoxExternalDecoder.Controls.Add(this.textBoxDecoderPath);
            this.groupBoxExternalDecoder.Controls.Add(this.labelDecoderPath);
            this.groupBoxExternalDecoder.Controls.Add(this.labelDecoderParameters);
            this.groupBoxExternalDecoder.Controls.Add(this.textBoxDecoderParameters);
            this.groupBoxExternalDecoder.DataBindings.Add(new System.Windows.Forms.Binding("Visible", this.bindingSourceDecoders, "CanBeDeleted", true));
            resources.ApplyResources(this.groupBoxExternalDecoder, "groupBoxExternalDecoder");
            this.groupBoxExternalDecoder.Name = "groupBoxExternalDecoder";
            this.groupBoxExternalDecoder.TabStop = false;
            // 
            // label4
            // 
            resources.ApplyResources(this.label4, "label4");
            this.label4.Name = "label4";
            // 
            // textBoxDecoderPath
            // 
            resources.ApplyResources(this.textBoxDecoderPath, "textBoxDecoderPath");
            this.textBoxDecoderPath.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.bindingSourceDecoders, "Path", true));
            this.textBoxDecoderPath.Name = "textBoxDecoderPath";
            // 
            // labelDecoderPath
            // 
            resources.ApplyResources(this.labelDecoderPath, "labelDecoderPath");
            this.labelDecoderPath.Name = "labelDecoderPath";
            // 
            // labelDecoderParameters
            // 
            resources.ApplyResources(this.labelDecoderParameters, "labelDecoderParameters");
            this.labelDecoderParameters.Name = "labelDecoderParameters";
            // 
            // textBoxDecoderParameters
            // 
            resources.ApplyResources(this.textBoxDecoderParameters, "textBoxDecoderParameters");
            this.textBoxDecoderParameters.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.bindingSourceDecoders, "Parameters", true));
            this.textBoxDecoderParameters.Name = "textBoxDecoderParameters";
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.buttonDecoderDelete);
            this.panel2.Controls.Add(this.buttonDecoderAdd);
            this.panel2.Controls.Add(this.comboBoxDecoderExtension);
            this.panel2.Controls.Add(this.labelDecoderExtension);
            resources.ApplyResources(this.panel2, "panel2");
            this.panel2.Name = "panel2";
            // 
            // buttonDecoderDelete
            // 
            this.buttonDecoderDelete.DataBindings.Add(new System.Windows.Forms.Binding("Enabled", this.bindingSourceDecoders, "CanBeDeleted", true));
            resources.ApplyResources(this.buttonDecoderDelete, "buttonDecoderDelete");
            this.buttonDecoderDelete.Name = "buttonDecoderDelete";
            this.buttonDecoderDelete.UseVisualStyleBackColor = true;
            this.buttonDecoderDelete.Click += new System.EventHandler(this.buttonDecoderDelete_Click);
            // 
            // buttonDecoderAdd
            // 
            resources.ApplyResources(this.buttonDecoderAdd, "buttonDecoderAdd");
            this.buttonDecoderAdd.Name = "buttonDecoderAdd";
            this.buttonDecoderAdd.UseVisualStyleBackColor = true;
            this.buttonDecoderAdd.Click += new System.EventHandler(this.buttonDecoderAdd_Click);
            // 
            // comboBoxDecoderExtension
            // 
            resources.ApplyResources(this.comboBoxDecoderExtension, "comboBoxDecoderExtension");
            this.comboBoxDecoderExtension.DataBindings.Add(new System.Windows.Forms.Binding("SelectedItem", this.bindingSourceDecoders, "Extension", true));
            this.comboBoxDecoderExtension.DataBindings.Add(new System.Windows.Forms.Binding("Enabled", this.bindingSourceDecoders, "CanBeDeleted", true));
            this.comboBoxDecoderExtension.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxDecoderExtension.FormattingEnabled = true;
            this.comboBoxDecoderExtension.Name = "comboBoxDecoderExtension";
            this.comboBoxDecoderExtension.SelectedIndexChanged += new System.EventHandler(this.comboBoxDecoderExtension_SelectedIndexChanged);
            // 
            // labelDecoderExtension
            // 
            resources.ApplyResources(this.labelDecoderExtension, "labelDecoderExtension");
            this.labelDecoderExtension.Name = "labelDecoderExtension";
            // 
            // listBoxDecoders
            // 
            this.listBoxDecoders.BackColor = System.Drawing.SystemColors.Control;
            this.listBoxDecoders.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.listBoxDecoders.DataSource = this.bindingSourceDecoders;
            this.listBoxDecoders.DisplayMember = "FullName";
            resources.ApplyResources(this.listBoxDecoders, "listBoxDecoders");
            this.listBoxDecoders.FormattingEnabled = true;
            this.listBoxDecoders.Name = "listBoxDecoders";
            this.tableLayoutPanel5.SetRowSpan(this.listBoxDecoders, 2);
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
            // tabPage5
            // 
            this.tabPage5.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage5.Controls.Add(this.richTextBoxScript);
            this.tabPage5.Controls.Add(this.buttonScriptCompile);
            this.tabPage5.Controls.Add(this.groupBoxScriptConditions);
            this.tabPage5.Controls.Add(this.listViewScripts);
            resources.ApplyResources(this.tabPage5, "tabPage5");
            this.tabPage5.Name = "tabPage5";
            // 
            // richTextBoxScript
            // 
            this.richTextBoxScript.AcceptsTab = true;
            this.richTextBoxScript.DetectUrls = false;
            resources.ApplyResources(this.richTextBoxScript, "richTextBoxScript");
            this.richTextBoxScript.Name = "richTextBoxScript";
            // 
            // buttonScriptCompile
            // 
            resources.ApplyResources(this.buttonScriptCompile, "buttonScriptCompile");
            this.buttonScriptCompile.Name = "buttonScriptCompile";
            this.buttonScriptCompile.UseVisualStyleBackColor = true;
            this.buttonScriptCompile.Click += new System.EventHandler(this.buttonScriptCompile_Click);
            // 
            // groupBoxScriptConditions
            // 
            this.groupBoxScriptConditions.Controls.Add(this.listViewScriptConditions);
            resources.ApplyResources(this.groupBoxScriptConditions, "groupBoxScriptConditions");
            this.groupBoxScriptConditions.Name = "groupBoxScriptConditions";
            this.groupBoxScriptConditions.TabStop = false;
            // 
            // listViewScriptConditions
            // 
            this.listViewScriptConditions.BackColor = System.Drawing.SystemColors.Control;
            this.listViewScriptConditions.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.listViewScriptConditions.CheckBoxes = true;
            this.listViewScriptConditions.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader6});
            this.listViewScriptConditions.FullRowSelect = true;
            this.listViewScriptConditions.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            resources.ApplyResources(this.listViewScriptConditions, "listViewScriptConditions");
            this.listViewScriptConditions.MultiSelect = false;
            this.listViewScriptConditions.Name = "listViewScriptConditions";
            this.listViewScriptConditions.UseCompatibleStateImageBehavior = false;
            this.listViewScriptConditions.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader6
            // 
            resources.ApplyResources(this.columnHeader6, "columnHeader6");
            // 
            // listViewScripts
            // 
            this.listViewScripts.BackColor = System.Drawing.SystemColors.Control;
            this.listViewScripts.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.listViewScripts.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader5});
            this.listViewScripts.FullRowSelect = true;
            this.listViewScripts.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.listViewScripts.HideSelection = false;
            this.listViewScripts.LabelEdit = true;
            resources.ApplyResources(this.listViewScripts, "listViewScripts");
            this.listViewScripts.MultiSelect = false;
            this.listViewScripts.Name = "listViewScripts";
            this.listViewScripts.UseCompatibleStateImageBehavior = false;
            this.listViewScripts.View = System.Windows.Forms.View.Details;
            this.listViewScripts.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listViewScripts_AfterLabelEdit);
            this.listViewScripts.BeforeLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listViewScripts_BeforeLabelEdit);
            this.listViewScripts.ItemSelectionChanged += new System.Windows.Forms.ListViewItemSelectionChangedEventHandler(this.listViewScripts_ItemSelectionChanged);
            this.listViewScripts.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listViewScripts_KeyDown);
            // 
            // columnHeader5
            // 
            resources.ApplyResources(this.columnHeader5, "columnHeader5");
            // 
            // tabPage7
            // 
            this.tabPage7.BackColor = System.Drawing.SystemColors.Control;
            this.tabPage7.Controls.Add(this.propertyGrid1);
            resources.ApplyResources(this.tabPage7, "tabPage7");
            this.tabPage7.Name = "tabPage7";
            // 
            // propertyGrid1
            // 
            resources.ApplyResources(this.propertyGrid1, "propertyGrid1");
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.PropertySort = System.Windows.Forms.PropertySort.Categorized;
            this.propertyGrid1.ToolbarVisible = false;
            // 
            // labelFormatDecoder
            // 
            resources.ApplyResources(this.labelFormatDecoder, "labelFormatDecoder");
            this.labelFormatDecoder.Name = "labelFormatDecoder";
            // 
            // labelFormatEncoder
            // 
            resources.ApplyResources(this.labelFormatEncoder, "labelFormatEncoder");
            this.labelFormatEncoder.Name = "labelFormatEncoder";
            // 
            // columnHeader2
            // 
            resources.ApplyResources(this.columnHeader2, "columnHeader2");
            // 
            // checkBox1
            // 
            resources.ApplyResources(this.checkBox1, "checkBox1");
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.UseVisualStyleBackColor = true;
            // 
            // frmSettings
            // 
            this.AcceptButton = this.btnOK;
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = btnCancel;
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "frmSettings";
            this.ShowIcon = false;
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmSettings_FormClosing);
            this.Load += new System.EventHandler(this.frmSettings_Load);
            this.grpGeneral.ResumeLayout(false);
            this.grpGeneral.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenPercent)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEncodeWhenConfidence)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFixWhenConfidence)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFixWhenPercent)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.encodersBindingSource)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.cUEConfigBindingSource)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.bindingSourceDecoders)).EndInit();
            this.grpAudioFilenames.ResumeLayout(false);
            this.grpAudioFilenames.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.groupBoxGaps.ResumeLayout(false);
            this.groupBoxGaps.PerformLayout();
            this.tabPage6.ResumeLayout(false);
            this.groupBoxAlbumArt.ResumeLayout(false);
            this.tableLayoutPanel3.ResumeLayout(false);
            this.tableLayoutPanel3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownMaxResolution)).EndInit();
            this.groupBoxTagging.ResumeLayout(false);
            this.groupBoxTagging.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.groupBoxARLog.ResumeLayout(false);
            this.groupBoxARLog.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.tableLayoutPanel2.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.groupBoxVerify.ResumeLayout(false);
            this.groupBoxVerify.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.groupBoxFormat.ResumeLayout(false);
            this.groupBoxFormat.PerformLayout();
            this.tabPageEncoders.ResumeLayout(false);
            this.tableLayoutPanel4.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.groupBoxExternalEncoder.ResumeLayout(false);
            this.groupBoxExternalEncoder.PerformLayout();
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.tabPage11.ResumeLayout(false);
            this.tableLayoutPanel5.ResumeLayout(false);
            this.groupBoxExternalDecoder.ResumeLayout(false);
            this.groupBoxExternalDecoder.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.grpHDCD.ResumeLayout(false);
            this.grpHDCD.PerformLayout();
            this.tabPage5.ResumeLayout(false);
            this.groupBoxScriptConditions.ResumeLayout(false);
            this.tabPage7.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox grpGeneral;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.CheckBox chkAutoCorrectFilenames;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox chkWriteArTagsOnConvert;
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
        private System.Windows.Forms.CheckBox chkWriteArLogOnConvert;
        private System.Windows.Forms.Label labelEncodeWhenConfidence;
        private System.Windows.Forms.NumericUpDown numEncodeWhenConfidence;
        private System.Windows.Forms.NumericUpDown numEncodeWhenPercent;
        private System.Windows.Forms.CheckBox chkFilenamesANSISafe;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.GroupBox groupBoxVerify;
        private System.Windows.Forms.CheckBox chkWriteARLogOnVerify;
        private System.Windows.Forms.CheckBox chkWriteARTagsOnVerify;
        private System.Windows.Forms.CheckBox chkEncodeWhenZeroOffset;
        private System.Windows.Forms.TabPage tabPage4;
        private System.Windows.Forms.CheckBox chkHDCDDecode;
        private System.Windows.Forms.CheckBox chkHDCDDetect;
        private System.Windows.Forms.GroupBox grpHDCD;
        private System.Windows.Forms.CheckBox chkHDCDStopLooking;
        private System.Windows.Forms.CheckBox chkCreateM3U;
        private System.Windows.Forms.CheckBox chkCreateCUEFileWhenEmbedded;
        private System.Windows.Forms.CheckBox chkTruncateExtra4206Samples;
        private System.Windows.Forms.CheckBox chkReducePriority;
        private System.Windows.Forms.CheckBox chkHDCDLW16;
        private System.Windows.Forms.CheckBox chkHDCD24bit;
        private System.Windows.Forms.CheckBox chkAllowMultipleInstances;
        private System.Windows.Forms.Label labelLanguage;
        private System.Windows.Forms.ComboBox comboLanguage;
        private System.Windows.Forms.GroupBox groupBoxExternalEncoder;
        private System.Windows.Forms.TextBox textBoxEncoderPath;
        private System.Windows.Forms.TextBox textBoxEncoderParameters;
        private System.Windows.Forms.Label labelEncoderPath;
        private System.Windows.Forms.Label labelEncoderParameters;
        private System.Windows.Forms.Label labelEncoderExtension;
        private System.Windows.Forms.TabPage tabPage11;
        private System.Windows.Forms.TextBox textBoxDecoderParameters;
        private System.Windows.Forms.TextBox textBoxDecoderPath;
        private System.Windows.Forms.Label labelDecoderParameters;
        private System.Windows.Forms.Label labelDecoderPath;
        private System.Windows.Forms.Label labelDecoderExtension;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Label labelFormatDefaultDecoder;
        private System.Windows.Forms.Label labelFormatLosslessEncoder;
        private System.Windows.Forms.ListView listViewFormats;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ComboBox comboFormatDecoder;
        private System.Windows.Forms.Label labelFormatDecoder;
        private System.Windows.Forms.ComboBox comboFormatLosslessEncoder;
        private System.Windows.Forms.Label labelFormatEncoder;
        private System.Windows.Forms.ComboBox comboBoxEncoderExtension;
        private System.Windows.Forms.GroupBox groupBoxExternalDecoder;
        private System.Windows.Forms.ComboBox comboBoxDecoderExtension;
        private System.Windows.Forms.Label labelFormatTagger;
        private System.Windows.Forms.ComboBox comboBoxFormatTagger;
        private System.Windows.Forms.CheckBox checkBoxFormatEmbedCUESheet;
        private System.Windows.Forms.CheckBox checkBoxFormatAllowLossless;
        private System.Windows.Forms.GroupBox groupBoxFormat;
        private System.Windows.Forms.CheckBox checkBoxEncoderLossless;
        private System.Windows.Forms.CheckBox checkBoxFormatAllowLossy;
        private System.Windows.Forms.ComboBox comboFormatLossyEncoder;
        private System.Windows.Forms.Label labelFormatLossyEncoder;
        private System.Windows.Forms.TabPage tabPage5;
        private System.Windows.Forms.GroupBox groupBoxScriptConditions;
        private System.Windows.Forms.ListView listViewScripts;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ListView listViewScriptConditions;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.Button buttonScriptCompile;
        private System.Windows.Forms.RichTextBox richTextBoxScript;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.GroupBox groupBoxARLog;
        private System.Windows.Forms.TextBox textBoxARLogExtension;
        private System.Windows.Forms.Label labelLogFileExtension;
        private System.Windows.Forms.CheckBox checkBoxARLogVerbose;
        private System.Windows.Forms.CheckBox checkBoxARVerifyUseSourceFolder;
        private System.Windows.Forms.TabPage tabPage6;
        private System.Windows.Forms.GroupBox groupBoxTagging;
        private System.Windows.Forms.CheckBox checkBoxCopyUnknownTags;
        private System.Windows.Forms.CheckBox chkOverwriteTags;
        private System.Windows.Forms.CheckBox chkFillUpCUE;
        private System.Windows.Forms.CheckBox chkEmbedLog;
        private System.Windows.Forms.CheckBox checkBoxWriteCUETags;
        private System.Windows.Forms.CheckBox checkBoxCopyAlbumArt;
        private System.Windows.Forms.CheckBox checkBoxCopyBasicTags;
        private System.Windows.Forms.CheckBox checkBoxFixToNearest;
        private System.Windows.Forms.CheckBox checkBoxEmbedAlbumArt;
        private System.Windows.Forms.GroupBox groupBoxAlbumArt;
        private System.Windows.Forms.Label labelAlbumArtMaximumResolution;
        private System.Windows.Forms.NumericUpDown numericUpDownMaxResolution;
        private System.Windows.Forms.CheckBox chkExtractLog;
        private System.Windows.Forms.CheckBox checkBoxExtractAlbumArt;
        private System.Windows.Forms.Label labelEncoderModes;
        private System.Windows.Forms.TextBox textBoxEncoderModes;
        private System.Windows.Forms.GroupBox groupBoxGaps;
        private System.Windows.Forms.RadioButton rbGapsPlusHTOA;
        private System.Windows.Forms.RadioButton rbGapsAppended;
        private System.Windows.Forms.RadioButton rbGapsLeftOut;
        private System.Windows.Forms.RadioButton rbGapsPrepended;
        private System.Windows.Forms.CheckBox checkBoxCheckForUpdates;
        private System.Windows.Forms.BindingSource cUEConfigBindingSource;
        private System.Windows.Forms.BindingSource encodersBindingSource;
        private System.Windows.Forms.ListBox listBoxEncoders;
        private System.Windows.Forms.TextBox textBoxEncoderName;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel2;
        private System.Windows.Forms.Button buttonEncoderDelete;
        private System.Windows.Forms.Button buttonEncoderAdd;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel3;
        private System.Windows.Forms.TextBox textBoxAlArtFilenameFormat;
        private System.Windows.Forms.Label labelEncoderName;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.CheckBox checkBoxSeparateDecodingThread;
        private System.Windows.Forms.TabPage tabPage7;
        private System.Windows.Forms.PropertyGrid propertyGrid1;
        private System.Windows.Forms.PropertyGrid propertyGridEncoderSettings;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel4;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel5;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Button buttonDecoderDelete;
        private System.Windows.Forms.Button buttonDecoderAdd;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox textBoxDecoderName;
        private System.Windows.Forms.BindingSource bindingSourceDecoders;
        private System.Windows.Forms.ListBox listBoxDecoders;
        public System.Windows.Forms.TabControl tabControl1;
        public System.Windows.Forms.TabPage tabPageEncoders;

    }
}