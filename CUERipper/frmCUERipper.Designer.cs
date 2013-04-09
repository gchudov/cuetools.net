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
            CUEControls.RectRadius rectRadius1 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius2 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius3 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius4 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius5 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius6 = new CUEControls.RectRadius();
            CUEControls.RectRadius rectRadius7 = new CUEControls.RectRadius();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabelMusicBrainz = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusCTDB = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusAr = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.listTracks = new System.Windows.Forms.ListView();
            this.Title = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.TrackNo = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.columnHeaderArtist = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.Start = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.Length = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.buttonGo = new System.Windows.Forms.Button();
            this.buttonAbort = new System.Windows.Forms.Button();
            this.buttonPause = new System.Windows.Forms.Button();
            this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
            this.lblWriteOffset = new System.Windows.Forms.Label();
            this.groupBoxSettings = new System.Windows.Forms.GroupBox();
            this.buttonEncoderSettings = new System.Windows.Forms.Button();
            this.checkBoxTestAndCopy = new System.Windows.Forms.CheckBox();
            this.bnComboBoxLosslessOrNot = new CUEControls.ImgComboBox();
            this.losslessOrNotBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.bindingSourceCR = new System.Windows.Forms.BindingSource(this.components);
            this.bnComboBoxEncoder = new CUEControls.ImgComboBox();
            this.encodersBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.labelSecureMode = new System.Windows.Forms.Label();
            this.bnComboBoxFormat = new CUEControls.ImgComboBox();
            this.formatsBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.labelEncoderMinMode = new System.Windows.Forms.Label();
            this.bnComboBoxImage = new CUEControls.ImgComboBox();
            this.cUEStylesBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.labelEncoderMaxMode = new System.Windows.Forms.Label();
            this.labelEncoderMode = new System.Windows.Forms.Label();
            this.trackBarEncoderMode = new System.Windows.Forms.TrackBar();
            this.trackBarSecureMode = new System.Windows.Forms.TrackBar();
            this.drivesBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.imageListChecked = new System.Windows.Forms.ImageList(this.components);
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.progressBarErrors = new ProgressODoom.ProgressBarEx();
            this.plainBackgroundPainter1 = new ProgressODoom.PlainBackgroundPainter();
            this.styledBorderPainter1 = new ProgressODoom.StyledBorderPainter();
            this.plainProgressPainter1 = new ProgressODoom.PlainProgressPainter();
            this.gradientGlossPainter1 = new ProgressODoom.GradientGlossPainter();
            this.progressBarCD = new ProgressODoom.ProgressBarEx();
            this.plainProgressPainter2 = new ProgressODoom.PlainProgressPainter();
            this.txtOutputPath = new System.Windows.Forms.TextBox();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.bnComboBoxRelease = new CUEControls.ImgComboBox();
            this.releasesBindingSource = new System.Windows.Forms.BindingSource(this.components);
            this.imageListMetadataSource = new System.Windows.Forms.ImageList(this.components);
            this.bnComboBoxDrives = new CUEControls.ImgComboBox();
            this.bnComboBoxOutputFormat = new CUEControls.ImgComboBox();
            this.listMetadata = new System.Windows.Forms.ListView();
            this.columnHeaderValue = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.columnHeaderName = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.buttonTrackMetadata = new System.Windows.Forms.Button();
            this.buttonMetadata = new System.Windows.Forms.Button();
            this.buttonVA = new System.Windows.Forms.Button();
            this.buttonReload = new System.Windows.Forms.Button();
            this.buttonEncoding = new System.Windows.Forms.Button();
            this.buttonTracks = new System.Windows.Forms.Button();
            this.buttonFreedbSubmit = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.backgroundWorkerArtwork = new System.ComponentModel.BackgroundWorker();
            this.buttonSettings = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel7 = new System.Windows.Forms.Panel();
            this.panel3 = new System.Windows.Forms.Panel();
            this.panel4 = new System.Windows.Forms.Panel();
            this.panel5 = new System.Windows.Forms.Panel();
            this.panel6 = new System.Windows.Forms.Panel();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
            this.groupBoxSettings.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.losslessOrNotBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.bindingSourceCR)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.encodersBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.formatsBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.cUEStylesBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarEncoderMode)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarSecureMode)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.drivesBindingSource)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.releasesBindingSource)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.panel2.SuspendLayout();
            this.panel7.SuspendLayout();
            this.panel3.SuspendLayout();
            this.panel4.SuspendLayout();
            this.panel5.SuspendLayout();
            this.panel6.SuspendLayout();
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
            this.toolStripStatusLabelMusicBrainz.IsLink = true;
            this.toolStripStatusLabelMusicBrainz.LinkBehavior = System.Windows.Forms.LinkBehavior.NeverUnderline;
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
            this.listTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.Title,
            this.TrackNo,
            this.columnHeaderArtist,
            this.Start,
            this.Length});
            resources.ApplyResources(this.listTracks, "listTracks");
            this.listTracks.FullRowSelect = true;
            this.listTracks.GridLines = true;
            this.listTracks.LabelEdit = true;
            this.listTracks.Name = "listTracks";
            this.toolTip1.SetToolTip(this.listTracks, resources.GetString("listTracks.ToolTip"));
            this.listTracks.UseCompatibleStateImageBehavior = false;
            this.listTracks.View = System.Windows.Forms.View.Details;
            this.listTracks.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listTracks_AfterLabelEdit);
            this.listTracks.BeforeLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listTracks_BeforeLabelEdit);
            this.listTracks.Click += new System.EventHandler(this.listTracks_Click);
            this.listTracks.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listTracks_KeyDown);
            this.listTracks.PreviewKeyDown += new System.Windows.Forms.PreviewKeyDownEventHandler(this.listTracks_PreviewKeyDown);
            // 
            // Title
            // 
            resources.ApplyResources(this.Title, "Title");
            // 
            // TrackNo
            // 
            resources.ApplyResources(this.TrackNo, "TrackNo");
            // 
            // columnHeaderArtist
            // 
            resources.ApplyResources(this.columnHeaderArtist, "columnHeaderArtist");
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
            this.buttonGo.BackColor = System.Drawing.Color.Transparent;
            this.buttonGo.Name = "buttonGo";
            this.buttonGo.UseVisualStyleBackColor = false;
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
            this.numericWriteOffset.ValueChanged += new System.EventHandler(this.numericWriteOffset_ValueChanged);
            // 
            // lblWriteOffset
            // 
            resources.ApplyResources(this.lblWriteOffset, "lblWriteOffset");
            this.lblWriteOffset.Name = "lblWriteOffset";
            // 
            // groupBoxSettings
            // 
            this.groupBoxSettings.Controls.Add(this.buttonEncoderSettings);
            this.groupBoxSettings.Controls.Add(this.checkBoxTestAndCopy);
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
            this.groupBoxSettings.Controls.Add(this.numericWriteOffset);
            resources.ApplyResources(this.groupBoxSettings, "groupBoxSettings");
            this.groupBoxSettings.Name = "groupBoxSettings";
            this.groupBoxSettings.TabStop = false;
            // 
            // buttonEncoderSettings
            // 
            this.buttonEncoderSettings.BackgroundImage = global::CUERipper.Properties.Resources.cog;
            resources.ApplyResources(this.buttonEncoderSettings, "buttonEncoderSettings");
            this.buttonEncoderSettings.Name = "buttonEncoderSettings";
            this.buttonEncoderSettings.UseVisualStyleBackColor = true;
            this.buttonEncoderSettings.Click += new System.EventHandler(this.buttonEncoderSettings_Click);
            // 
            // checkBoxTestAndCopy
            // 
            resources.ApplyResources(this.checkBoxTestAndCopy, "checkBoxTestAndCopy");
            this.checkBoxTestAndCopy.Name = "checkBoxTestAndCopy";
            this.checkBoxTestAndCopy.UseVisualStyleBackColor = true;
            this.checkBoxTestAndCopy.Click += new System.EventHandler(this.checkBoxTestAndCopy_Click);
            // 
            // bnComboBoxLosslessOrNot
            // 
            this.bnComboBoxLosslessOrNot.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxLosslessOrNot.DataSource = this.losslessOrNotBindingSource;
            this.bnComboBoxLosslessOrNot.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxLosslessOrNot.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxLosslessOrNot.ImageKeyMember = "ImageKey";
            this.bnComboBoxLosslessOrNot.ImageList = null;
            resources.ApplyResources(this.bnComboBoxLosslessOrNot, "bnComboBoxLosslessOrNot");
            this.bnComboBoxLosslessOrNot.Name = "bnComboBoxLosslessOrNot";
            rectRadius1.BottomLeft = 2;
            rectRadius1.BottomRight = 2;
            rectRadius1.TopLeft = 2;
            rectRadius1.TopRight = 6;
            this.bnComboBoxLosslessOrNot.Radius = rectRadius1;
            this.bnComboBoxLosslessOrNot.SelectedValueChanged += new System.EventHandler(this.bnComboBoxLosslessOrNot_SelectedValueChanged);
            // 
            // losslessOrNotBindingSource
            // 
            this.losslessOrNotBindingSource.DataMember = "LosslessOrNot";
            this.losslessOrNotBindingSource.DataSource = this.bindingSourceCR;
            // 
            // bindingSourceCR
            // 
            this.bindingSourceCR.DataSource = typeof(CUERipper.CUERipperData);
            // 
            // bnComboBoxEncoder
            // 
            this.bnComboBoxEncoder.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxEncoder.DataSource = this.encodersBindingSource;
            this.bnComboBoxEncoder.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxEncoder.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxEncoder.ImageList = null;
            resources.ApplyResources(this.bnComboBoxEncoder, "bnComboBoxEncoder");
            this.bnComboBoxEncoder.Name = "bnComboBoxEncoder";
            rectRadius2.BottomLeft = 2;
            rectRadius2.BottomRight = 2;
            rectRadius2.TopLeft = 2;
            rectRadius2.TopRight = 6;
            this.bnComboBoxEncoder.Radius = rectRadius2;
            this.bnComboBoxEncoder.SelectedValueChanged += new System.EventHandler(this.bnComboBoxEncoder_SelectedValueChanged);
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
            this.bnComboBoxFormat.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxFormat.DataSource = this.formatsBindingSource;
            this.bnComboBoxFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxFormat.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxFormat.ImageKeyMember = "DotExtension";
            this.bnComboBoxFormat.ImageList = null;
            resources.ApplyResources(this.bnComboBoxFormat, "bnComboBoxFormat");
            this.bnComboBoxFormat.Name = "bnComboBoxFormat";
            rectRadius3.BottomLeft = 2;
            rectRadius3.BottomRight = 2;
            rectRadius3.TopLeft = 2;
            rectRadius3.TopRight = 6;
            this.bnComboBoxFormat.Radius = rectRadius3;
            this.bnComboBoxFormat.SelectedValueChanged += new System.EventHandler(this.bnComboBoxFormat_SelectedValueChanged);
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
            this.bnComboBoxImage.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxImage.DataSource = this.cUEStylesBindingSource;
            this.bnComboBoxImage.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxImage.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxImage.ImageList = null;
            resources.ApplyResources(this.bnComboBoxImage, "bnComboBoxImage");
            this.bnComboBoxImage.Name = "bnComboBoxImage";
            rectRadius4.BottomLeft = 2;
            rectRadius4.BottomRight = 2;
            rectRadius4.TopLeft = 2;
            rectRadius4.TopRight = 6;
            this.bnComboBoxImage.Radius = rectRadius4;
            this.bnComboBoxImage.SelectedValueChanged += new System.EventHandler(this.bnComboBoxImage_SelectedValueChanged);
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
            this.trackBarSecureMode.LargeChange = 1;
            this.trackBarSecureMode.Maximum = 2;
            this.trackBarSecureMode.Name = "trackBarSecureMode";
            this.trackBarSecureMode.Scroll += new System.EventHandler(this.trackBarSecureMode_Scroll);
            // 
            // drivesBindingSource
            // 
            this.drivesBindingSource.DataMember = "Drives";
            this.drivesBindingSource.DataSource = this.bindingSourceCR;
            // 
            // imageListChecked
            // 
            this.imageListChecked.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageListChecked.ImageStream")));
            this.imageListChecked.TransparentColor = System.Drawing.Color.Transparent;
            this.imageListChecked.Images.SetKeyName(0, "checked");
            this.imageListChecked.Images.SetKeyName(1, "unchecked");
            this.imageListChecked.Images.SetKeyName(2, "disabled");
            this.imageListChecked.Images.SetKeyName(3, "mix");
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
            this.progressBarErrors.TabStop = false;
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
            this.progressBarCD.TabStop = false;
            this.progressBarCD.Value = 10;
            // 
            // plainProgressPainter2
            // 
            this.plainProgressPainter2.Color = System.Drawing.Color.Lime;
            this.plainProgressPainter2.GlossPainter = this.gradientGlossPainter1;
            this.plainProgressPainter2.LeadingEdge = System.Drawing.Color.Transparent;
            this.plainProgressPainter2.ProgressBorderPainter = null;
            // 
            // txtOutputPath
            // 
            resources.ApplyResources(this.txtOutputPath, "txtOutputPath");
            this.txtOutputPath.Name = "txtOutputPath";
            this.txtOutputPath.ReadOnly = true;
            this.toolTip1.SetToolTip(this.txtOutputPath, resources.GetString("txtOutputPath.ToolTip"));
            this.txtOutputPath.Enter += new System.EventHandler(this.txtOutputPath_Enter);
            // 
            // bnComboBoxRelease
            // 
            resources.ApplyResources(this.bnComboBoxRelease, "bnComboBoxRelease");
            this.bnComboBoxRelease.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxRelease.DataSource = this.releasesBindingSource;
            this.bnComboBoxRelease.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxRelease.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxRelease.ImageKeyMember = "ImageKey";
            this.bnComboBoxRelease.ImageList = this.imageListMetadataSource;
            this.bnComboBoxRelease.Name = "bnComboBoxRelease";
            rectRadius5.BottomLeft = 2;
            rectRadius5.BottomRight = 2;
            rectRadius5.TopLeft = 2;
            rectRadius5.TopRight = 6;
            this.bnComboBoxRelease.Radius = rectRadius5;
            this.bnComboBoxRelease.SelectedValueChanged += new System.EventHandler(this.bnComboBoxRelease_SelectedValueChanged);
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
            this.imageListMetadataSource.Images.SetKeyName(0, "musicbrainz");
            this.imageListMetadataSource.Images.SetKeyName(1, "freedb");
            this.imageListMetadataSource.Images.SetKeyName(2, "local");
            this.imageListMetadataSource.Images.SetKeyName(3, "localshadow");
            this.imageListMetadataSource.Images.SetKeyName(4, "tracks");
            this.imageListMetadataSource.Images.SetKeyName(5, "tracks1");
            this.imageListMetadataSource.Images.SetKeyName(6, "album");
            this.imageListMetadataSource.Images.SetKeyName(7, "track");
            this.imageListMetadataSource.Images.SetKeyName(8, "ctdb");
            this.imageListMetadataSource.Images.SetKeyName(9, "discogs");
            this.imageListMetadataSource.Images.SetKeyName(10, "cdstub");
            // 
            // bnComboBoxDrives
            // 
            resources.ApplyResources(this.bnComboBoxDrives, "bnComboBoxDrives");
            this.bnComboBoxDrives.BackColor = System.Drawing.Color.Transparent;
            this.bnComboBoxDrives.DataSource = this.drivesBindingSource;
            this.bnComboBoxDrives.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.bnComboBoxDrives.ForeColor = System.Drawing.SystemColors.ControlText;
            this.bnComboBoxDrives.ImageKeyMember = "ImageKey";
            this.bnComboBoxDrives.ImageList = this.imageListMetadataSource;
            this.bnComboBoxDrives.Name = "bnComboBoxDrives";
            rectRadius6.BottomLeft = 2;
            rectRadius6.BottomRight = 2;
            rectRadius6.TopLeft = 2;
            rectRadius6.TopRight = 6;
            this.bnComboBoxDrives.Radius = rectRadius6;
            this.bnComboBoxDrives.SelectedIndexChanged += new System.EventHandler(this.bnComboBoxDrives_SelectedIndexChanged);
            // 
            // bnComboBoxOutputFormat
            // 
            this.bnComboBoxOutputFormat.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.bnComboBoxOutputFormat.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.ListItems;
            this.bnComboBoxOutputFormat.BackColor = System.Drawing.Color.Transparent;
            resources.ApplyResources(this.bnComboBoxOutputFormat, "bnComboBoxOutputFormat");
            this.bnComboBoxOutputFormat.ImageList = null;
            this.bnComboBoxOutputFormat.Name = "bnComboBoxOutputFormat";
            rectRadius7.BottomLeft = 2;
            rectRadius7.BottomRight = 2;
            rectRadius7.TopLeft = 2;
            rectRadius7.TopRight = 6;
            this.bnComboBoxOutputFormat.Radius = rectRadius7;
            this.bnComboBoxOutputFormat.TabStop = false;
            this.bnComboBoxOutputFormat.DropDown += new System.EventHandler(this.bnComboBoxOutputFormat_DroppedDown);
            this.bnComboBoxOutputFormat.TextChanged += new System.EventHandler(this.bnComboBoxOutputFormat_TextChanged);
            this.bnComboBoxOutputFormat.Leave += new System.EventHandler(this.bnComboBoxOutputFormat_Leave);
            this.bnComboBoxOutputFormat.MouseLeave += new System.EventHandler(this.bnComboBoxOutputFormat_MouseLeave);
            // 
            // listMetadata
            // 
            this.listMetadata.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeaderValue,
            this.columnHeaderName});
            resources.ApplyResources(this.listMetadata, "listMetadata");
            this.listMetadata.FullRowSelect = true;
            this.listMetadata.GridLines = true;
            this.listMetadata.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.listMetadata.LabelEdit = true;
            this.listMetadata.Name = "listMetadata";
            this.listMetadata.UseCompatibleStateImageBehavior = false;
            this.listMetadata.View = System.Windows.Forms.View.Details;
            this.listMetadata.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listMetadata_AfterLabelEdit);
            this.listMetadata.BeforeLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listMetadata_BeforeLabelEdit);
            this.listMetadata.Click += new System.EventHandler(this.listMetadata_Click);
            // 
            // columnHeaderValue
            // 
            resources.ApplyResources(this.columnHeaderValue, "columnHeaderValue");
            // 
            // columnHeaderName
            // 
            resources.ApplyResources(this.columnHeaderName, "columnHeaderName");
            // 
            // buttonTrackMetadata
            // 
            this.buttonTrackMetadata.BackColor = System.Drawing.Color.Transparent;
            this.buttonTrackMetadata.FlatAppearance.BorderSize = 0;
            this.buttonTrackMetadata.FlatAppearance.MouseDownBackColor = System.Drawing.SystemColors.Control;
            this.buttonTrackMetadata.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            resources.ApplyResources(this.buttonTrackMetadata, "buttonTrackMetadata");
            this.buttonTrackMetadata.ForeColor = System.Drawing.SystemColors.ControlText;
            this.buttonTrackMetadata.ImageList = this.imageListChecked;
            this.buttonTrackMetadata.Name = "buttonTrackMetadata";
            this.buttonTrackMetadata.UseVisualStyleBackColor = false;
            // 
            // buttonMetadata
            // 
            this.buttonMetadata.Image = global::CUERipper.Properties.Resources.tag_label;
            resources.ApplyResources(this.buttonMetadata, "buttonMetadata");
            this.buttonMetadata.Name = "buttonMetadata";
            this.buttonMetadata.UseVisualStyleBackColor = true;
            this.buttonMetadata.Click += new System.EventHandler(this.buttonMetadata_Click);
            // 
            // buttonVA
            // 
            this.buttonVA.Image = global::CUERipper.Properties.Resources.users__arrow;
            resources.ApplyResources(this.buttonVA, "buttonVA");
            this.buttonVA.Name = "buttonVA";
            this.buttonVA.UseVisualStyleBackColor = true;
            this.buttonVA.Click += new System.EventHandler(this.buttonVA_Click);
            // 
            // buttonReload
            // 
            this.buttonReload.Image = global::CUERipper.Properties.Resources.arrow_circle_double;
            resources.ApplyResources(this.buttonReload, "buttonReload");
            this.buttonReload.Name = "buttonReload";
            this.buttonReload.UseVisualStyleBackColor = true;
            this.buttonReload.Click += new System.EventHandler(this.buttonReload_Click);
            // 
            // buttonEncoding
            // 
            this.buttonEncoding.Image = global::CUERipper.Properties.Resources.spellcheck;
            resources.ApplyResources(this.buttonEncoding, "buttonEncoding");
            this.buttonEncoding.Name = "buttonEncoding";
            this.buttonEncoding.UseVisualStyleBackColor = true;
            this.buttonEncoding.Click += new System.EventHandler(this.buttonEncoding_Click);
            // 
            // buttonTracks
            // 
            this.buttonTracks.Image = global::CUERipper.Properties.Resources.edit_list_order;
            resources.ApplyResources(this.buttonTracks, "buttonTracks");
            this.buttonTracks.Name = "buttonTracks";
            this.buttonTracks.UseVisualStyleBackColor = true;
            this.buttonTracks.Click += new System.EventHandler(this.buttonTracks_Click);
            // 
            // buttonFreedbSubmit
            // 
            this.buttonFreedbSubmit.Image = global::CUERipper.Properties.Resources.freedb16;
            resources.ApplyResources(this.buttonFreedbSubmit, "buttonFreedbSubmit");
            this.buttonFreedbSubmit.Name = "buttonFreedbSubmit";
            this.buttonFreedbSubmit.UseVisualStyleBackColor = true;
            this.buttonFreedbSubmit.Click += new System.EventHandler(this.buttonFreedbSubmit_Click);
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.buttonGo);
            this.panel1.Controls.Add(this.buttonPause);
            this.panel1.Controls.Add(this.buttonAbort);
            this.panel1.Controls.Add(this.progressBarCD);
            this.panel1.Controls.Add(this.progressBarErrors);
            resources.ApplyResources(this.panel1, "panel1");
            this.panel1.Name = "panel1";
            // 
            // pictureBox1
            // 
            resources.ApplyResources(this.pictureBox1, "pictureBox1");
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.TabStop = false;
            this.pictureBox1.MouseClick += new System.Windows.Forms.MouseEventHandler(this.pictureBox1_MouseClick);
            // 
            // backgroundWorkerArtwork
            // 
            this.backgroundWorkerArtwork.WorkerReportsProgress = true;
            this.backgroundWorkerArtwork.WorkerSupportsCancellation = true;
            this.backgroundWorkerArtwork.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorkerArtwork_DoWork);
            this.backgroundWorkerArtwork.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.backgroundWorkerArtwork_ProgressChanged);
            this.backgroundWorkerArtwork.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorkerArtwork_RunWorkerCompleted);
            // 
            // buttonSettings
            // 
            this.buttonSettings.Image = global::CUERipper.Properties.Resources.cog;
            resources.ApplyResources(this.buttonSettings, "buttonSettings");
            this.buttonSettings.Name = "buttonSettings";
            this.buttonSettings.UseVisualStyleBackColor = true;
            this.buttonSettings.Click += new System.EventHandler(this.buttonSettings_Click);
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.panel7);
            this.panel2.Controls.Add(this.panel1);
            this.panel2.Controls.Add(this.groupBoxSettings);
            resources.ApplyResources(this.panel2, "panel2");
            this.panel2.Name = "panel2";
            // 
            // panel7
            // 
            this.panel7.Controls.Add(this.pictureBox1);
            resources.ApplyResources(this.panel7, "panel7");
            this.panel7.Name = "panel7";
            // 
            // panel3
            // 
            this.panel3.Controls.Add(this.bnComboBoxDrives);
            this.panel3.Controls.Add(this.bnComboBoxRelease);
            this.panel3.Controls.Add(this.buttonSettings);
            this.panel3.Controls.Add(this.buttonVA);
            this.panel3.Controls.Add(this.buttonFreedbSubmit);
            this.panel3.Controls.Add(this.buttonTrackMetadata);
            this.panel3.Controls.Add(this.buttonTracks);
            this.panel3.Controls.Add(this.buttonReload);
            this.panel3.Controls.Add(this.buttonEncoding);
            this.panel3.Controls.Add(this.buttonMetadata);
            resources.ApplyResources(this.panel3, "panel3");
            this.panel3.Name = "panel3";
            // 
            // panel4
            // 
            this.panel4.Controls.Add(this.listTracks);
            this.panel4.Controls.Add(this.listMetadata);
            resources.ApplyResources(this.panel4, "panel4");
            this.panel4.Name = "panel4";
            // 
            // panel5
            // 
            this.panel5.Controls.Add(this.bnComboBoxOutputFormat);
            this.panel5.Controls.Add(this.txtOutputPath);
            resources.ApplyResources(this.panel5, "panel5");
            this.panel5.Name = "panel5";
            // 
            // panel6
            // 
            this.panel6.Controls.Add(this.panel4);
            this.panel6.Controls.Add(this.panel5);
            this.panel6.Controls.Add(this.panel2);
            this.panel6.Controls.Add(this.panel3);
            resources.ApplyResources(this.panel6, "panel6");
            this.panel6.Name = "panel6";
            // 
            // frmCUERipper
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.panel6);
            this.Controls.Add(this.statusStrip1);
            this.KeyPreview = true;
            this.Name = "frmCUERipper";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUERipper_FormClosed);
            this.Load += new System.EventHandler(this.frmCUERipper_Load);
            this.ClientSizeChanged += new System.EventHandler(this.frmCUERipper_ClientSizeChanged);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.frmCUERipper_KeyDown);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
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
            ((System.ComponentModel.ISupportInitialize)(this.drivesBindingSource)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.releasesBindingSource)).EndInit();
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.panel2.ResumeLayout(false);
            this.panel7.ResumeLayout(false);
            this.panel3.ResumeLayout(false);
            this.panel4.ResumeLayout(false);
            this.panel5.ResumeLayout(false);
            this.panel5.PerformLayout();
            this.panel6.ResumeLayout(false);
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
		private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusAr;
		private System.Windows.Forms.NumericUpDown numericWriteOffset;
        private System.Windows.Forms.Label lblWriteOffset;
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
		private System.Windows.Forms.TextBox txtOutputPath;
		private System.Windows.Forms.ToolTip toolTip1;
		private CUEControls.ImgComboBox bnComboBoxImage;
		private System.Windows.Forms.BindingSource bindingSourceCR;
		private System.Windows.Forms.ImageList imageListMetadataSource;
		private System.Windows.Forms.BindingSource cUEStylesBindingSource;
		private CUEControls.ImgComboBox bnComboBoxRelease;
		private System.Windows.Forms.BindingSource releasesBindingSource;
		private CUEControls.ImgComboBox bnComboBoxDrives;
		private System.Windows.Forms.BindingSource drivesBindingSource;
		private CUEControls.ImgComboBox bnComboBoxFormat;
		private System.Windows.Forms.BindingSource formatsBindingSource;
		private CUEControls.ImgComboBox bnComboBoxEncoder;
		private System.Windows.Forms.BindingSource encodersBindingSource;
		private CUEControls.ImgComboBox bnComboBoxLosslessOrNot;
		private System.Windows.Forms.BindingSource losslessOrNotBindingSource;
		private CUEControls.ImgComboBox bnComboBoxOutputFormat;
		private System.Windows.Forms.ImageList imageListChecked;
		private System.Windows.Forms.ListView listMetadata;
		private System.Windows.Forms.ColumnHeader columnHeaderName;
		private System.Windows.Forms.ColumnHeader columnHeaderValue;
		private System.Windows.Forms.Button buttonTrackMetadata;
		private System.Windows.Forms.Button buttonMetadata;
		private System.Windows.Forms.Button buttonVA;
		private System.Windows.Forms.Button buttonReload;
		private System.Windows.Forms.Button buttonEncoding;
		private System.Windows.Forms.Button buttonTracks;
		private System.Windows.Forms.ColumnHeader columnHeaderArtist;
		private System.Windows.Forms.Button buttonFreedbSubmit;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.ComponentModel.BackgroundWorker backgroundWorkerArtwork;
        private System.Windows.Forms.Button buttonSettings;
        private System.Windows.Forms.CheckBox checkBoxTestAndCopy;
		private System.Windows.Forms.Panel panel2;
		private System.Windows.Forms.Panel panel3;
		private System.Windows.Forms.Panel panel4;
		private System.Windows.Forms.Panel panel5;
		private System.Windows.Forms.Panel panel6;
		private System.Windows.Forms.Panel panel7;
        private System.Windows.Forms.Button buttonEncoderSettings;
	}
}

