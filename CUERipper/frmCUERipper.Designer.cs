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
			this.comboDrives = new System.Windows.Forms.ComboBox();
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
			this.comboBoxAudioFormat = new System.Windows.Forms.ComboBox();
			this.comboImage = new System.Windows.Forms.ComboBox();
			this.buttonAbort = new System.Windows.Forms.Button();
			this.buttonPause = new System.Windows.Forms.Button();
			this.comboRelease = new System.Windows.Forms.ComboBox();
			this.contextMenuStripRelease = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.editToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.comboBoxEncoder = new System.Windows.Forms.ComboBox();
			this.radioButtonAudioLossy = new System.Windows.Forms.RadioButton();
			this.radioButtonAudioHybrid = new System.Windows.Forms.RadioButton();
			this.radioButtonAudioLossless = new System.Windows.Forms.RadioButton();
			this.checkBoxEACMode = new System.Windows.Forms.CheckBox();
			this.groupBoxSettings = new System.Windows.Forms.GroupBox();
			this.labelSecureMode = new System.Windows.Forms.Label();
			this.labelEncoderMinMode = new System.Windows.Forms.Label();
			this.labelEncoderMaxMode = new System.Windows.Forms.Label();
			this.labelEncoderMode = new System.Windows.Forms.Label();
			this.trackBarEncoderMode = new System.Windows.Forms.TrackBar();
			this.trackBarSecureMode = new System.Windows.Forms.TrackBar();
			this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
			this.releaseBindingSource = new System.Windows.Forms.BindingSource(this.components);
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
			this.statusStrip1.SuspendLayout();
			this.contextMenuStripRelease.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.groupBoxSettings.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarSecureMode)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.releaseBindingSource)).BeginInit();
			this.SuspendLayout();
			// 
			// comboDrives
			// 
			resources.ApplyResources(this.comboDrives, "comboDrives");
			this.comboDrives.DisplayMember = "Path";
			this.comboDrives.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboDrives.FormattingEnabled = true;
			this.comboDrives.Name = "comboDrives";
			this.comboDrives.ValueMember = "Path";
			this.comboDrives.SelectedIndexChanged += new System.EventHandler(this.comboDrives_SelectedIndexChanged);
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
			// comboBoxAudioFormat
			// 
			this.comboBoxAudioFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboBoxAudioFormat.FormattingEnabled = true;
			this.comboBoxAudioFormat.Items.AddRange(new object[] {
            resources.GetString("comboBoxAudioFormat.Items"),
            resources.GetString("comboBoxAudioFormat.Items1"),
            resources.GetString("comboBoxAudioFormat.Items2"),
            resources.GetString("comboBoxAudioFormat.Items3"),
            resources.GetString("comboBoxAudioFormat.Items4")});
			resources.ApplyResources(this.comboBoxAudioFormat, "comboBoxAudioFormat");
			this.comboBoxAudioFormat.Name = "comboBoxAudioFormat";
			this.comboBoxAudioFormat.SelectedIndexChanged += new System.EventHandler(this.comboBoxAudioFormat_SelectedIndexChanged);
			// 
			// comboImage
			// 
			this.comboImage.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboImage.FormattingEnabled = true;
			this.comboImage.Items.AddRange(new object[] {
            resources.GetString("comboImage.Items"),
            resources.GetString("comboImage.Items1")});
			resources.ApplyResources(this.comboImage, "comboImage");
			this.comboImage.Name = "comboImage";
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
			// comboRelease
			// 
			resources.ApplyResources(this.comboRelease, "comboRelease");
			this.comboRelease.BackColor = System.Drawing.SystemColors.Control;
			this.comboRelease.ContextMenuStrip = this.contextMenuStripRelease;
			this.comboRelease.DrawMode = System.Windows.Forms.DrawMode.OwnerDrawFixed;
			this.comboRelease.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboRelease.FormattingEnabled = true;
			this.comboRelease.Name = "comboRelease";
			this.toolTip1.SetToolTip(this.comboRelease, resources.GetString("comboRelease.ToolTip"));
			this.comboRelease.DrawItem += new System.Windows.Forms.DrawItemEventHandler(this.comboRelease_DrawItem);
			this.comboRelease.SelectedIndexChanged += new System.EventHandler(this.comboRelease_SelectedIndexChanged);
			this.comboRelease.Format += new System.Windows.Forms.ListControlConvertEventHandler(this.comboRelease_Format);
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
			// comboBoxEncoder
			// 
			this.comboBoxEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboBoxEncoder.FormattingEnabled = true;
			resources.ApplyResources(this.comboBoxEncoder, "comboBoxEncoder");
			this.comboBoxEncoder.Name = "comboBoxEncoder";
			this.comboBoxEncoder.SelectedIndexChanged += new System.EventHandler(this.comboBoxEncoder_SelectedIndexChanged);
			// 
			// radioButtonAudioLossy
			// 
			resources.ApplyResources(this.radioButtonAudioLossy, "radioButtonAudioLossy");
			this.radioButtonAudioLossy.Name = "radioButtonAudioLossy";
			this.radioButtonAudioLossy.TabStop = true;
			this.radioButtonAudioLossy.UseVisualStyleBackColor = true;
			this.radioButtonAudioLossy.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
			// 
			// radioButtonAudioHybrid
			// 
			resources.ApplyResources(this.radioButtonAudioHybrid, "radioButtonAudioHybrid");
			this.radioButtonAudioHybrid.Name = "radioButtonAudioHybrid";
			this.radioButtonAudioHybrid.TabStop = true;
			this.radioButtonAudioHybrid.UseVisualStyleBackColor = true;
			this.radioButtonAudioHybrid.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
			// 
			// radioButtonAudioLossless
			// 
			resources.ApplyResources(this.radioButtonAudioLossless, "radioButtonAudioLossless");
			this.radioButtonAudioLossless.Name = "radioButtonAudioLossless";
			this.radioButtonAudioLossless.TabStop = true;
			this.radioButtonAudioLossless.UseVisualStyleBackColor = true;
			this.radioButtonAudioLossless.CheckedChanged += new System.EventHandler(this.radioButtonAudioLossless_CheckedChanged);
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
			this.groupBoxSettings.Controls.Add(this.labelSecureMode);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMinMode);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMaxMode);
			this.groupBoxSettings.Controls.Add(this.labelEncoderMode);
			this.groupBoxSettings.Controls.Add(this.trackBarEncoderMode);
			this.groupBoxSettings.Controls.Add(this.trackBarSecureMode);
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioLossless);
			this.groupBoxSettings.Controls.Add(this.comboBoxAudioFormat);
			this.groupBoxSettings.Controls.Add(this.lblWriteOffset);
			this.groupBoxSettings.Controls.Add(this.checkBoxEACMode);
			this.groupBoxSettings.Controls.Add(this.comboImage);
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioLossy);
			this.groupBoxSettings.Controls.Add(this.comboBoxEncoder);
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioHybrid);
			this.groupBoxSettings.Controls.Add(this.numericWriteOffset);
			resources.ApplyResources(this.groupBoxSettings, "groupBoxSettings");
			this.groupBoxSettings.Name = "groupBoxSettings";
			this.groupBoxSettings.TabStop = false;
			// 
			// labelSecureMode
			// 
			resources.ApplyResources(this.labelSecureMode, "labelSecureMode");
			this.labelSecureMode.Name = "labelSecureMode";
			// 
			// labelEncoderMinMode
			// 
			resources.ApplyResources(this.labelEncoderMinMode, "labelEncoderMinMode");
			this.labelEncoderMinMode.Name = "labelEncoderMinMode";
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
			this.trackBarSecureMode.LargeChange = 3;
			resources.ApplyResources(this.trackBarSecureMode, "trackBarSecureMode");
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
			// releaseBindingSource
			// 
			this.releaseBindingSource.DataSource = typeof(MusicBrainz.Release);
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
			// frmCUERipper
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.txtOutputPath);
			this.Controls.Add(this.comboBoxOutputFormat);
			this.Controls.Add(this.progressBarErrors);
			this.Controls.Add(this.progressBarCD);
			this.Controls.Add(this.groupBoxSettings);
			this.Controls.Add(this.comboRelease);
			this.Controls.Add(this.listTracks);
			this.Controls.Add(this.buttonGo);
			this.Controls.Add(this.statusStrip1);
			this.Controls.Add(this.buttonAbort);
			this.Controls.Add(this.comboDrives);
			this.Controls.Add(this.buttonPause);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
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
			((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBarSecureMode)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.releaseBindingSource)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.ComboBox comboDrives;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ListView listTracks;
		private System.Windows.Forms.ColumnHeader TrackNo;
		private System.Windows.Forms.ColumnHeader Title;
		private System.Windows.Forms.ColumnHeader Start;
		private System.Windows.Forms.ColumnHeader Length;
		private System.Windows.Forms.Button buttonGo;
		private System.Windows.Forms.ComboBox comboBoxAudioFormat;
		private System.Windows.Forms.ComboBox comboImage;
		private System.Windows.Forms.Button buttonAbort;
		private System.Windows.Forms.Button buttonPause;
		private System.Windows.Forms.ComboBox comboRelease;
		private System.Windows.Forms.BindingSource releaseBindingSource;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripRelease;
		private System.Windows.Forms.ToolStripMenuItem editToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusAr;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
		private System.Windows.Forms.Label lblWriteOffset;
		private System.Windows.Forms.ComboBox comboBoxEncoder;
		private System.Windows.Forms.RadioButton radioButtonAudioLossy;
		private System.Windows.Forms.RadioButton radioButtonAudioHybrid;
		private System.Windows.Forms.RadioButton radioButtonAudioLossless;
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
	}
}

