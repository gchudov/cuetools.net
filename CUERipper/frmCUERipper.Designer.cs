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
			this.toolStripStatusAr = new System.Windows.Forms.ToolStripStatusLabel();
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
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
			this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
			this.releaseBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.numericWriteOffset = new System.Windows.Forms.NumericUpDown();
			this.lblWriteOffset = new System.Windows.Forms.Label();
			this.comboBoxEncoder = new System.Windows.Forms.ComboBox();
			this.radioButtonAudioLossy = new System.Windows.Forms.RadioButton();
			this.radioButtonAudioHybrid = new System.Windows.Forms.RadioButton();
			this.radioButtonAudioLossless = new System.Windows.Forms.RadioButton();
			this.checkBoxEACMode = new System.Windows.Forms.CheckBox();
			this.groupBoxSettings = new System.Windows.Forms.GroupBox();
			this.statusStrip1.SuspendLayout();
			this.contextMenuStripRelease.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.releaseBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).BeginInit();
			this.groupBoxSettings.SuspendLayout();
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
            this.toolStripStatusAr,
            this.toolStripProgressBar1,
            this.toolStripProgressBar2,
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
			// toolStripStatusAr
			// 
			this.toolStripStatusAr.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
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
			// toolStripProgressBar2
			// 
			this.toolStripProgressBar2.AutoToolTip = true;
			this.toolStripProgressBar2.Name = "toolStripProgressBar2";
			resources.ApplyResources(this.toolStripProgressBar2, "toolStripProgressBar2");
			this.toolStripProgressBar2.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
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
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioLossless);
			this.groupBoxSettings.Controls.Add(this.comboBoxAudioFormat);
			this.groupBoxSettings.Controls.Add(this.checkBoxEACMode);
			this.groupBoxSettings.Controls.Add(this.lblWriteOffset);
			this.groupBoxSettings.Controls.Add(this.numericWriteOffset);
			this.groupBoxSettings.Controls.Add(this.comboImage);
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioLossy);
			this.groupBoxSettings.Controls.Add(this.comboBoxEncoder);
			this.groupBoxSettings.Controls.Add(this.radioButtonAudioHybrid);
			resources.ApplyResources(this.groupBoxSettings, "groupBoxSettings");
			this.groupBoxSettings.Name = "groupBoxSettings";
			this.groupBoxSettings.TabStop = false;
			// 
			// frmCUERipper
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.groupBoxSettings);
			this.Controls.Add(this.comboRelease);
			this.Controls.Add(this.buttonPause);
			this.Controls.Add(this.listTracks);
			this.Controls.Add(this.buttonAbort);
			this.Controls.Add(this.buttonGo);
			this.Controls.Add(this.statusStrip1);
			this.Controls.Add(this.comboDrives);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "frmCUERipper";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Load += new System.EventHandler(this.frmCUERipper_Load);
			this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.frmCUERipper_FormClosed);
			this.statusStrip1.ResumeLayout(false);
			this.statusStrip1.PerformLayout();
			this.contextMenuStripRelease.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.releaseBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.numericWriteOffset)).EndInit();
			this.groupBoxSettings.ResumeLayout(false);
			this.groupBoxSettings.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.ComboBox comboDrives;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar1;
		private System.Windows.Forms.ToolStripProgressBar toolStripProgressBar2;
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
	}
}

