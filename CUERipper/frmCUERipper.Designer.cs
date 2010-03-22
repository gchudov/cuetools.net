namespace CUERipper
{
	partial class frmCUERipper
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmCUERipper));
			this.statusStrip1 = new System.Windows.Forms.StatusStrip();
			this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusLabelMusicBrainz = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusCTDB = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripStatusAr = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
			this.listTracks = new System.Windows.Forms.ListView();
			this.Title = new System.Windows.Forms.ColumnHeader();
			this.TrackNo = new System.Windows.Forms.ColumnHeader();
			this.Start = new System.Windows.Forms.ColumnHeader();
			this.Length = new System.Windows.Forms.ColumnHeader();
			this.buttonGo = new System.Windows.Forms.Button();
			this.buttonAbort = new System.Windows.Forms.Button();
			this.buttonPause = new System.Windows.Forms.Button();
			this.contextMenuStripRelease = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.editToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.checkBoxEACMode = new System.Windows.Forms.CheckBox();
			this.groupBoxSettings = new System.Windows.Forms.GroupBox();
			this.bnComboBoxLosslessOrNot = new BBBNOVA.BNComboBox();
			this.losslessOrNotBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.bindingSourceCR = new System.Windows.Forms.BindingSource(this.components);
			this.bnComboBoxEncoder = new BBBNOVA.BNComboBox();
			this.encodersBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.labelSecureMode = new System.Windows.Forms.Label();
			this.bnComboBoxFormat = new BBBNOVA.BNComboBox();
			this.formatsBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.labelEncoderMinMode = new System.Windows.Forms.Label();
			this.bnComboBoxImage = new BBBNOVA.BNComboBox();
			this.cUEStylesBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.labelEncoderMaxMode = new System.Windows.Forms.Label();
			this.labelEncoderMode = new System.Windows.Forms.Label();
			this.trackBarEncoderMode = new System.Windows.Forms.TrackBar();
			this.trackBarSecureMode = new System.Windows.Forms.TrackBar();
			this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
			this.progressBarErrors = new ProgressODoom.ProgressBarEx();
			this.plainBackgroundPainter1 = new ProgressODoom.PlainBackgroundPainter();
			this.styledBorderPainter1 = new ProgressODoom.StyledBorderPainter();
			this.plainProgressPainter1 = new ProgressODoom.PlainProgressPainter();
			this.gradientGlossPainter1 = new ProgressODoom.GradientGlossPainter();
			this.progressBarCD = new ProgressODoom.ProgressBarEx();
			this.plainProgressPainter2 = new ProgressODoom.PlainProgressPainter();
			this.comboBoxOutputFormat = new System.Windows.Forms.ComboBox();
			this.txtOutputPath = new System.Windows.Forms.TextBox();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.bnComboBoxRelease = new BBBNOVA.BNComboBox();
			this.releasesBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.imageListMetadataSource = new System.Windows.Forms.ImageList(this.components);
			this.bnComboBoxDrives = new BBBNOVA.BNComboBox();
			this.drivesBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.statusStrip1.SuspendLayout();
			this.contextMenuStripRelease.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.groupBoxSettings.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.losslessOrNotBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.bindingSourceCR)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.encodersBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.formatsBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.cUEStylesBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarSecureMode)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.releasesBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.drivesBindingSource)).BeginInit();
			this.SuspendLayout();
			// 
			// statusStrip1
			// 
			this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLabelMusicBrainz,
            this.toolStripStatusCTDB,
            this.toolStripStatusAr,
            this.toolStripProgressBar1,
            this.toolStripStatusLabel2});
			resources.ApplyResources(this.statusStrip1, "statusStrip1");
			this.statusStrip1.Name = "statusStrip1";
			this.statusStrip1.ShowItemToolTips = true;
			this.statusStrip1.SizingGrip = false;
			// 
			// toolStripStatusLabel1
			// 
			this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
			resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
			this.toolStripStatusLabel1.Spring = true;
			// 
			// toolStripStatusLabelMusicBrainz
			// 
			this.toolStripStatusLabelMusicBrainz.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
			this.toolStripStatusLabelMusicBrainz.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenInner;
			this.toolStripStatusLabelMusicBrainz.Image = global::CUERipper.Properties.Resources.musicbrainz;
			this.toolStripStatusLabelMusicBrainz.Name = "toolStripStatusLabelMusicBrainz";
			resources.ApplyResources(this.toolStripStatusLabelMusicBrainz, "toolStripStatusLabelMusicBrainz");
			this.toolStripStatusLabelMusicBrainz.Click += new System.EventHandler(this.toolStripStatusLabelMusicBrainz_Click);
			// 
			// toolStripStatusCTDB
			// 
			this.toolStripStatusCTDB.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
			this.toolStripStatusCTDB.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenInner;
			this.toolStripStatusCTDB.Image = global::CUERipper.Properties.Resources.cdrepair;
			this.toolStripStatusCTDB.Name = "toolStripStatusCTDB";
			resources.ApplyResources(this.toolStripStatusCTDB, "toolStripStatusCTDB");
			// 
			// toolStripStatusAr
			// 
			this.toolStripStatusAr.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)
						| System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
			this.toolStripStatusAr.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenInner;
			resources.ApplyResources(this.toolStripStatusAr, "toolStripStatusAr");
			this.toolStripStatusAr.Name = "toolStripStatusAr";
			// 
			// toolStripProgressBar1
			// 
			this.toolStripProgressBar1.AutoToolTip = true;
			this.toolStripProgressBar1.MarqueeAnimationSpeed = 500;
			this.toolStripProgressBar1.Name = "toolStripProgressBar1";
			resources.ApplyResources(this.toolStripProgressBar1, "toolStripProgressBar1");
			this.toolStripProgressBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
			// 
			// toolStripStatusLabel2
			// 
			this.toolStripStatusLabel2.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
			this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
			resources.ApplyResources(this.toolStripStatusLabel2, "toolStripStatusLabel2");
			// 
			// listTracks
			// 
			resources.ApplyResources(this.listTracks, "listTracks");
			this.listTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.Title,
            this.TrackNo,
            this.Start,
            this.Length});
			this.listTracks.FullRowSelect = true;
			this.listTracks.GridLines = true;
			this.listTracks.LabelEdit = true;
			this.listTracks.Name = "listTracks";
			this.toolTip1.SetToolTip(this.listTracks, resources.GetString("listTracks.ToolTip"));
			this.listTracks.UseCompatibleStateImageBehavior = false;
			this.listTracks.View = System.Windows.Forms.View.Details;
			this.listTracks.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listTracks_AfterLabelEdit);
			this.listTracks.DoubleClick += new System.EventHandler(this.listTracks_DoubleClick);
			this.listTracks.PreviewKeyDown += new System.Windows.Forms.PreviewKeyDownEventHandler(this.listTracks_PreviewKeyDown);
			this.listTracks.BeforeLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listTracks_BeforeLabelEdit);
			this.listTracks.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listTracks_KeyDown);
			// 
			// Title
			// 
			resources.ApplyResources(this.Title, "Title");
			// 
			// TrackNo
			// 
			resources.ApplyResources(this.TrackNo, "TrackNo");
			// 
			// Start
			// 
			resources.ApplyResources(this.Start, "Start");
			// 
			// Length
			// 
			resources.ApplyResources(this.Length, "Length");
			// 
			// buttonGo
			// 
			resources.ApplyResources(this.buttonGo, "buttonGo");
			this.buttonGo.Name = "buttonGo";
			this.buttonGo.UseVisualStyleBackColor = true;
			this.buttonGo.Click += new System.EventHandler(this.buttonGo_Click);
			// 
			// buttonAbort
			// 
			resources.ApplyResources(this.buttonAbort, "buttonAbort");
			this.buttonAbort.Name = "buttonAbort";
			this.buttonAbort.UseVisualStyleBackColor = true;
			this.buttonAbort.Click += new System.EventHandler(this.buttonAbort_Click);
			// 
			// buttonPause
			// 
			resources.ApplyResources(this.buttonPause, "buttonPause");
			this.buttonPause.Name = "buttonPause";
			this.buttonPause.UseVisualStyleBackColor = true;
			this.buttonPause.Click += new System.EventHandler(this.buttonPause_Click);
			// 
			// contextMenuStripRelease
			// 
			this.contextMenuStripRelease.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.editToolStripMenuItem});
			this.contextMenuStripRelease.Name = "contextMenuStripRelease";
			resources.ApplyResources(this.contextMenuStripRelease, "contextMenuStripRelease");
			// 
			// editToolStripMenuItem
			// 
			this.editToolStripMenuItem.Name = "editToolStripMenuItem";
			resources.ApplyResources(this.editToolStripMenuItem, "editToolStripMenuItem");
			this.editToolStripMenuItem.Click += new System.EventHandler(this.editToolStripMenuItem_Click);
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
			// lblWriteOffset
			// 
			resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
			this.lblWriteOffset.Name = "lblWriteOffset";
			// 
			// checkBoxEACMode
			// 
			resources.ApplyResources(this.checkBoxEACMode, "checkBoxEACMode");
			this.checkBoxEACMode.Name = "checkBoxEACMode";
			this.checkBoxEACMode.UseVisualStyleBackColor = true;
			this.checkBoxEACMode.CheckedChanged += new System.EventHandler(this.checkBoxEACMode_CheckedChanged);
			// 
			// groupBoxSettings
			// 
			this.groupBoxSettings.Controls.Add(this.bnComboBoxLosslessOrNot);
			this.groupBoxSettings.Controls.Add(this.bnComboBoxEncoder);
			this.groupBoxSettings.Controls.Add(this.labelSecureMode);
			this.groupBoxSettings.Controls.Add(this.bnComboBoxFormat);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMinMode);
			this.groupBoxSettings.Controls.Add(this.bnComboBoxImage);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMaxMode);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMode);
			this.groupBoxSettings.Controls.Add(this.trackBarEncoderMode);
			this.groupBoxSettings.Controls.Add(this.trackBarSecureMode);
			this.groupBoxSettings.Controls.Add(this.lblWriteOffset);
			this.groupBoxSettings.Controls.Add(this.checkBoxEACMode);
			this.groupBoxSettings.Controls.Add(this.numericWriteOffset);
			resources.ApplyResources(this.groupBoxSettings, "groupBoxSettings");
			this.groupBoxSettings.Name = "groupBoxSettings";
			this.groupBoxSettings.TabStop = false;
			// 
			// bnComboBoxLosslessOrNot
			// 
			this.bnComboBoxLosslessOrNot.BackColor = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxLosslessOrNot.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxLosslessOrNot.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxLosslessOrNot.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxLosslessOrNot.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxLosslessOrNot.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxLosslessOrNot.DataSource = this.losslessOrNotBindingSource;
			this.bnComboBoxLosslessOrNot.DropDownHeight = 200;
			this.bnComboBoxLosslessOrNot.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxLosslessOrNot.DropDownWidth = 80;
			this.bnComboBoxLosslessOrNot.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxLosslessOrNot.ImageList = null;
			this.bnComboBoxLosslessOrNot.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxLosslessOrNot, "bnComboBoxLosslessOrNot");
			this.bnComboBoxLosslessOrNot.MaxDropDownItems = 8;
			this.bnComboBoxLosslessOrNot.MinimumSize = new System.Drawing.Size(40, 21);
			this.bnComboBoxLosslessOrNot.Name = "bnComboBoxLosslessOrNot";
			this.bnComboBoxLosslessOrNot.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxLosslessOrNot.Radius")));
			this.bnComboBoxLosslessOrNot.SelectedIndex = -1;
			this.bnComboBoxLosslessOrNot.SelectedItem = null;
			this.bnComboBoxLosslessOrNot.Sorted = false;
			this.bnComboBoxLosslessOrNot.SelectedIndexChanged += new System.EventHandler(this.bnComboBoxLosslessOrNot_SelectedIndexChanged);
			// 
			// losslessOrNotBindingSource
			// 
			this.losslessOrNotBindingSource.DataMember = "LosslessOrNot";
			this.losslessOrNotBindingSource.DataSource = this.bindingSourceCR;
			// 
			// bindingSourceCR
			// 
			this.bindingSourceCR.DataSource = typeof(CUERipper.frmCUERipper);
			// 
			// bnComboBoxEncoder
			// 
			this.bnComboBoxEncoder.BackColor = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxEncoder.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxEncoder.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxEncoder.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxEncoder.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxEncoder.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxEncoder.DataSource = this.encodersBindingSource;
			this.bnComboBoxEncoder.DropDownHeight = 200;
			this.bnComboBoxEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxEncoder.DropDownWidth = 80;
			this.bnComboBoxEncoder.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxEncoder.ImageList = null;
			this.bnComboBoxEncoder.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxEncoder, "bnComboBoxEncoder");
			this.bnComboBoxEncoder.MaxDropDownItems = 8;
			this.bnComboBoxEncoder.MinimumSize = new System.Drawing.Size(40, 21);
			this.bnComboBoxEncoder.Name = "bnComboBoxEncoder";
			this.bnComboBoxEncoder.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxEncoder.Radius")));
			this.bnComboBoxEncoder.SelectedIndex = -1;
			this.bnComboBoxEncoder.SelectedItem = null;
			this.bnComboBoxEncoder.Sorted = false;
			this.bnComboBoxEncoder.SelectedIndexChanged += new System.EventHandler(this.comboBoxEncoder_SelectedIndexChanged);
			// 
			// encodersBindingSource
			// 
			this.encodersBindingSource.DataMember = "Encoders";
			this.encodersBindingSource.DataSource = this.bindingSourceCR;
			// 
			// labelSecureMode
			// 
			resources.ApplyResources(this.labelSecureMode, "labelSecureMode");
			this.labelSecureMode.Name = "labelSecureMode";
			// 
			// bnComboBoxFormat
			// 
			this.bnComboBoxFormat.BackColor = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxFormat.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxFormat.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxFormat.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxFormat.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxFormat.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxFormat.DataSource = this.formatsBindingSource;
			this.bnComboBoxFormat.DropDownHeight = 200;
			this.bnComboBoxFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxFormat.DropDownWidth = 80;
			this.bnComboBoxFormat.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxFormat.ImageKeyMember = "DotExtension";
			this.bnComboBoxFormat.ImageList = null;
			this.bnComboBoxFormat.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxFormat, "bnComboBoxFormat");
			this.bnComboBoxFormat.MaxDropDownItems = 8;
			this.bnComboBoxFormat.MinimumSize = new System.Drawing.Size(40, 21);
			this.bnComboBoxFormat.Name = "bnComboBoxFormat";
			this.bnComboBoxFormat.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxFormat.Radius")));
			this.bnComboBoxFormat.SelectedIndex = -1;
			this.bnComboBoxFormat.SelectedItem = null;
			this.bnComboBoxFormat.Sorted = false;
			this.bnComboBoxFormat.SelectedIndexChanged += new System.EventHandler(this.bnComboBoxFormat_SelectedIndexChanged);
			// 
			// formatsBindingSource
			// 
			this.formatsBindingSource.DataMember = "Formats";
			this.formatsBindingSource.DataSource = this.bindingSourceCR;
			// 
			// labelEncoderMinMode
			// 
			resources.ApplyResources(this.labelEncoderMinMode, "labelEncoderMinMode");
			this.labelEncoderMinMode.Name = "labelEncoderMinMode";
			// 
			// bnComboBoxImage
			// 
			this.bnComboBoxImage.BackColor = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxImage.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxImage.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxImage.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxImage.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxImage.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxImage.DataSource = this.cUEStylesBindingSource;
			this.bnComboBoxImage.DropDownHeight = 200;
			this.bnComboBoxImage.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxImage.DropDownWidth = 80;
			this.bnComboBoxImage.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxImage.ImageList = null;
			this.bnComboBoxImage.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxImage, "bnComboBoxImage");
			this.bnComboBoxImage.MaxDropDownItems = 8;
			this.bnComboBoxImage.MinimumSize = new System.Drawing.Size(40, 21);
			this.bnComboBoxImage.Name = "bnComboBoxImage";
			this.bnComboBoxImage.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxImage.Radius")));
			this.bnComboBoxImage.SelectedIndex = -1;
			this.bnComboBoxImage.SelectedItem = null;
			this.bnComboBoxImage.Sorted = false;
			// 
			// cUEStylesBindingSource
			// 
			this.cUEStylesBindingSource.DataMember = "CUEStyles";
			this.cUEStylesBindingSource.DataSource = this.bindingSourceCR;
			// 
			// labelEncoderMaxMode
			// 
			resources.ApplyResources(this.labelEncoderMaxMode, "labelEncoderMaxMode");
			this.labelEncoderMaxMode.Name = "labelEncoderMaxMode";
			// 
			// labelEncoderMode
			// 
			resources.ApplyResources(this.labelEncoderMode, "labelEncoderMode");
			this.labelEncoderMode.Name = "labelEncoderMode";
			// 
			// trackBarEncoderMode
			// 
			resources.ApplyResources(this.trackBarEncoderMode, "trackBarEncoderMode");
			this.trackBarEncoderMode.LargeChange = 1;
			this.trackBarEncoderMode.Name = "trackBarEncoderMode";
			this.trackBarEncoderMode.Scroll += new System.EventHandler(this.trackBarEncoderMode_Scroll);
			// 
			// trackBarSecureMode
			// 
			resources.ApplyResources(this.trackBarSecureMode, "trackBarSecureMode");
			this.trackBarSecureMode.LargeChange = 3;
			this.trackBarSecureMode.Maximum = 2;
			this.trackBarSecureMode.Name = "trackBarSecureMode";
			this.trackBarSecureMode.Scroll += new System.EventHandler(this.trackBarSecureMode_Scroll);
			// 
			// toolStripMenuItem1
			// 
			this.toolStripMenuItem1.Image = global::CUERipper.Properties.Resources.cddb;
			this.toolStripMenuItem1.Name = "toolStripMenuItem1";
			resources.ApplyResources(this.toolStripMenuItem1, "toolStripMenuItem1");
			// 
			// progressBarErrors
			// 
			this.progressBarErrors.BackgroundPainter = this.plainBackgroundPainter1;
			this.progressBarErrors.BorderPainter = this.styledBorderPainter1;
			resources.ApplyResources(this.progressBarErrors, "progressBarErrors");
			this.progressBarErrors.MarqueePercentage = 25;
			this.progressBarErrors.MarqueeSpeed = 30;
			this.progressBarErrors.MarqueeStep = 1;
			this.progressBarErrors.Maximum = 100;
			this.progressBarErrors.Minimum = 0;
			this.progressBarErrors.Name = "progressBarErrors";
			this.progressBarErrors.ProgressPadding = 0;
			this.progressBarErrors.ProgressPainter = this.plainProgressPainter1;
			this.progressBarErrors.ProgressType = ProgressODoom.ProgressType.Smooth;
			this.progressBarErrors.ShowPercentage = false;
			this.progressBarErrors.Value = 10;
			// 
			// plainBackgroundPainter1
			// 
			this.plainBackgroundPainter1.Color = System.Drawing.SystemColors.Control;
			this.plainBackgroundPainter1.GlossPainter = null;
			// 
			// styledBorderPainter1
			// 
			this.styledBorderPainter1.Border3D = System.Windows.Forms.Border3DStyle.Etched;
			// 
			// plainProgressPainter1
			// 
			this.plainProgressPainter1.Color = System.Drawing.Color.Red;
			this.plainProgressPainter1.GlossPainter = this.gradientGlossPainter1;
			this.plainProgressPainter1.LeadingEdge = System.Drawing.Color.Transparent;
			this.plainProgressPainter1.ProgressBorderPainter = null;
			// 
			// gradientGlossPainter1
			// 
			this.gradientGlossPainter1.AlphaHigh = 235;
			this.gradientGlossPainter1.AlphaLow = 0;
			this.gradientGlossPainter1.Angle = 90F;
			this.gradientGlossPainter1.Color = System.Drawing.SystemColors.Control;
			this.gradientGlossPainter1.PercentageCovered = 100;
			this.gradientGlossPainter1.Style = ProgressODoom.GlossStyle.Top;
			this.gradientGlossPainter1.Successor = null;
			// 
			// progressBarCD
			// 
			this.progressBarCD.BackgroundPainter = this.plainBackgroundPainter1;
			this.progressBarCD.BorderPainter = this.styledBorderPainter1;
			resources.ApplyResources(this.progressBarCD, "progressBarCD");
			this.progressBarCD.MarqueePercentage = 25;
			this.progressBarCD.MarqueeSpeed = 30;
			this.progressBarCD.MarqueeStep = 1;
			this.progressBarCD.Maximum = 100;
			this.progressBarCD.Minimum = 0;
			this.progressBarCD.Name = "progressBarCD";
			this.progressBarCD.ProgressPadding = 0;
			this.progressBarCD.ProgressPainter = this.plainProgressPainter2;
			this.progressBarCD.ProgressType = ProgressODoom.ProgressType.Smooth;
			this.progressBarCD.ShowPercentage = true;
			this.progressBarCD.Value = 10;
			// 
			// plainProgressPainter2
			// 
			this.plainProgressPainter2.Color = System.Drawing.Color.Lime;
			this.plainProgressPainter2.GlossPainter = this.gradientGlossPainter1;
			this.plainProgressPainter2.LeadingEdge = System.Drawing.Color.Transparent;
			this.plainProgressPainter2.ProgressBorderPainter = null;
			// 
			// comboBoxOutputFormat
			// 
			this.comboBoxOutputFormat.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
			this.comboBoxOutputFormat.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.ListItems;
			this.comboBoxOutputFormat.FormattingEnabled = true;
			resources.ApplyResources(this.comboBoxOutputFormat, "comboBoxOutputFormat");
			this.comboBoxOutputFormat.Name = "comboBoxOutputFormat";
			this.toolTip1.SetToolTip(this.comboBoxOutputFormat, resources.GetString("comboBoxOutputFormat.ToolTip"));
			this.comboBoxOutputFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxOutputFormat_SelectedIndexChanged);
			this.comboBoxOutputFormat.MouseLeave += new System.EventHandler(this.comboBoxOutputFormat_MouseLeave);
			this.comboBoxOutputFormat.TextUpdate += new System.EventHandler(this.comboBoxOutputFormat_TextUpdate);
			// 
			// txtOutputPath
			// 
			this.txtOutputPath.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
			this.txtOutputPath.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
			resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
			this.txtOutputPath.Name = "txtOutputPath";
			this.txtOutputPath.ReadOnly = true;
			this.toolTip1.SetToolTip(this.txtOutputPath, resources.GetString("txtOutputPath.ToolTip"));
			this.txtOutputPath.Enter += new System.EventHandler(this.txtOutputPath_Enter);
			// 
			// bnComboBoxRelease
			// 
			this.bnComboBoxRelease.BackColor = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxRelease.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxRelease.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxRelease.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxRelease.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxRelease.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxRelease.ContextMenuStrip = this.contextMenuStripRelease;
			this.bnComboBoxRelease.DataSource = this.releasesBindingSource;
			this.bnComboBoxRelease.DropDownHeight = 200;
			this.bnComboBoxRelease.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxRelease.DropDownWidth = 552;
			this.bnComboBoxRelease.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxRelease.ImageKeyMember = "ImageKey";
			this.bnComboBoxRelease.ImageList = this.imageListMetadataSource;
			this.bnComboBoxRelease.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxRelease, "bnComboBoxRelease");
			this.bnComboBoxRelease.MaxDropDownItems = 8;
			this.bnComboBoxRelease.MinimumSize = new System.Drawing.Size(61, 21);
			this.bnComboBoxRelease.Name = "bnComboBoxRelease";
			this.bnComboBoxRelease.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxRelease.Radius")));
			this.bnComboBoxRelease.SelectedIndex = -1;
			this.bnComboBoxRelease.SelectedItem = null;
			this.bnComboBoxRelease.Sorted = false;
			this.toolTip1.SetToolTip(this.bnComboBoxRelease, resources.GetString("bnComboBoxRelease.ToolTip"));
			this.bnComboBoxRelease.SelectedIndexChanged += new System.EventHandler(this.bnComboBoxRelease_SelectedIndexChanged);
			// 
			// releasesBindingSource
			// 
			this.releasesBindingSource.DataMember = "Releases";
			this.releasesBindingSource.DataSource = this.bindingSourceCR;
			// 
			// imageListMetadataSource
			// 
			this.imageListMetadataSource.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageListMetadataSource.ImageStream")));
			this.imageListMetadataSource.TransparentColor = System.Drawing.Color.Transparent;
			this.imageListMetadataSource.Images.SetKeyName(0, "musicbrainz.ico");
			this.imageListMetadataSource.Images.SetKeyName(1, "freedb16.png");
			// 
			// bnComboBoxDrives
			// 
			this.bnComboBoxDrives.BackColor = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxDrives.Border = System.Windows.Forms.BorderStyle.FixedSingle;
			this.bnComboBoxDrives.Color1 = System.Drawing.SystemColors.Control;
			this.bnComboBoxDrives.Color2 = System.Drawing.SystemColors.ControlDark;
			this.bnComboBoxDrives.Color3 = System.Drawing.Color.Maroon;
			this.bnComboBoxDrives.Color4 = System.Drawing.SystemColors.ControlDarkDark;
			this.bnComboBoxDrives.DataSource = this.drivesBindingSource;
			this.bnComboBoxDrives.DropDownHeight = 200;
			this.bnComboBoxDrives.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.bnComboBoxDrives.DropDownWidth = 552;
			this.bnComboBoxDrives.ForeColor = System.Drawing.SystemColors.ControlText;
			this.bnComboBoxDrives.ImageKeyMember = "ImageKey";
			this.bnComboBoxDrives.ImageList = this.imageListMetadataSource;
			this.bnComboBoxDrives.IsDroppedDown = false;
			resources.ApplyResources(this.bnComboBoxDrives, "bnComboBoxDrives");
			this.bnComboBoxDrives.MaxDropDownItems = 8;
			this.bnComboBoxDrives.MinimumSize = new System.Drawing.Size(61, 21);
			this.bnComboBoxDrives.Name = "bnComboBoxDrives";
			this.bnComboBoxDrives.Radius = ((BBBNOVA.BNRadius)(resources.GetObject("bnComboBoxDrives.Radius")));
			this.bnComboBoxDrives.SelectedIndex = -1;
			this.bnComboBoxDrives.SelectedItem = null;
			this.bnComboBoxDrives.Sorted = false;
			this.bnComboBoxDrives.SelectedIndexChanged += new System.EventHandler(this.bnComboBoxDrives_SelectedIndexChanged);
			// 
			// drivesBindingSource
			// 
			this.drivesBindingSource.DataMember = "Drives";
			this.drivesBindingSource.DataSource = this.bindingSourceCR;
			// 
			// frmCUERipper
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.bnComboBoxRelease);
			this.Controls.Add(this.bnComboBoxDrives);
			this.Controls.Add(this.txtOutputPath);
			this.Controls.Add(this.comboBoxOutputFormat);
			this.Controls.Add(this.progressBarErrors);
			this.Controls.Add(this.progressBarCD);
			this.Controls.Add(this.groupBoxSettings);
			this.Controls.Add(this.listTracks);
			this.Controls.Add(this.buttonGo);
			this.Controls.Add(this.statusStrip1);
			this.Controls.Add(this.buttonAbort);
			this.Controls.Add(this.buttonPause);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.KeyPreview = true;
			this.MaximizeBox = false;
			this.Name = "frmCUERipper";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Load += new System.EventHandler(this.frmCUERipper_Load);
			this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUERipper_FormClosed);
			this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.frmCUERipper_KeyDown);
			this.statusStrip1.ResumeLayout(false);
			this.statusStrip1.PerformLayout();
			this.contextMenuStripRelease.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
			this.groupBoxSettings.ResumeLayout(false);
			this.groupBoxSettings.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.losslessOrNotBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.bindingSourceCR)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.encodersBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.formatsBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.cUEStylesBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarSecureMode)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.releasesBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.drivesBindingSource)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ListView listTracks;
		private System.Windows.Forms.ColumnHeader TrackNo;
		private System.Windows.Forms.ColumnHeader Title;
		private System.Windows.Forms.ColumnHeader Start;
		private System.Windows.Forms.ColumnHeader Length;
		private System.Windows.Forms.Button buttonGo;
		private System.Windows.Forms.Button buttonAbort;
		private System.Windows.Forms.Button buttonPause;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripRelease;
		private System.Windows.Forms.ToolStripMenuItem editToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusAr;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.Label lblWriteOffset;
		private System.Windows.Forms.CheckBox checkBoxEACMode;
		private System.Windows.Forms.GroupBox groupBoxSettings;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusCTDB;
		private System.Windows.Forms.TrackBar trackBarEncoderMode;
		private System.Windows.Forms.Label labelEncoderMode;
		private System.Windows.Forms.Label labelEncoderMaxMode;
		private System.Windows.Forms.Label labelEncoderMinMode;
		private System.Windows.Forms.TrackBar trackBarSecureMode;
		private System.Windows.Forms.Label labelSecureMode;
		private ProgressODoom.ProgressBarEx progressBarErrors;
		private ProgressODoom.StyledBorderPainter styledBorderPainter1;
		private ProgressODoom.PlainProgressPainter plainProgressPainter1;
		private ProgressODoom.PlainBackgroundPainter plainBackgroundPainter1;
		private ProgressODoom.GradientGlossPainter gradientGlossPainter1;
		private ProgressODoom.ProgressBarEx progressBarCD;
		private ProgressODoom.PlainProgressPainter plainProgressPainter2;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelMusicBrainz;
		private System.Windows.Forms.ComboBox comboBoxOutputFormat;
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.ToolTip toolTip1;
		private BBBNOVA.BNComboBox bnComboBoxImage;
		private System.Windows.Forms.BindingSource bindingSourceCR;
		private System.Windows.Forms.ImageList imageListMetadataSource;
		private System.Windows.Forms.BindingSource cUEStylesBindingSource;
		private BBBNOVA.BNComboBox bnComboBoxRelease;
		private System.Windows.Forms.BindingSource releasesBindingSource;
		private BBBNOVA.BNComboBox bnComboBoxDrives;
		private System.Windows.Forms.BindingSource drivesBindingSource;
		private BBBNOVA.BNComboBox bnComboBoxFormat;
		private System.Windows.Forms.BindingSource formatsBindingSource;
		private BBBNOVA.BNComboBox bnComboBoxEncoder;
		private System.Windows.Forms.BindingSource encodersBindingSource;
		private BBBNOVA.BNComboBox bnComboBoxLosslessOrNot;
		private System.Windows.Forms.BindingSource losslessOrNotBindingSource;
	}
}

