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
			this.toolStripProgressBar1 = new System.Windows.Forms.ToolStripProgressBar();
			this.toolStripProgressBar2 = new System.Windows.Forms.ToolStripProgressBar();
			this.listTracks = new System.Windows.Forms.ListView();
			this.Title = new System.Windows.Forms.ColumnHeader();
			this.TrackNo = new System.Windows.Forms.ColumnHeader();
			this.Start = new System.Windows.Forms.ColumnHeader();
			this.Length = new System.Windows.Forms.ColumnHeader();
			this.buttonGo = new System.Windows.Forms.Button();
			this.comboLossless = new System.Windows.Forms.ComboBox();
			this.comboCodec = new System.Windows.Forms.ComboBox();
			this.comboImage = new System.Windows.Forms.ComboBox();
			this.buttonAbort = new System.Windows.Forms.Button();
			this.buttonPause = new System.Windows.Forms.Button();
			this.comboRelease = new System.Windows.Forms.ComboBox();
			this.releaseBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.statusStrip1.SuspendLayout();
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
			this.listTracks.DoubleClick += new System.EventHandler(this.listTracks_DoubleClick);
			this.listTracks.PreviewKeyDown += new System.Windows.Forms.PreviewKeyDownEventHandler(this.listTracks_PreviewKeyDown);
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
			// comboLossless
			// 
			this.comboLossless.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboLossless.FormattingEnabled = true;
			this.comboLossless.Items.AddRange(new object[] {
            resources.GetString("comboLossless.Items"),
            resources.GetString("comboLossless.Items1"),
            resources.GetString("comboLossless.Items2")});
			resources.ApplyResources(this.comboLossless, "comboLossless");
			this.comboLossless.Name = "comboLossless";
			// 
			// comboCodec
			// 
			this.comboCodec.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboCodec.FormattingEnabled = true;
			this.comboCodec.Items.AddRange(new object[] {
            resources.GetString("comboCodec.Items"),
            resources.GetString("comboCodec.Items1"),
            resources.GetString("comboCodec.Items2"),
            resources.GetString("comboCodec.Items3")});
			resources.ApplyResources(this.comboCodec, "comboCodec");
			this.comboCodec.Name = "comboCodec";
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
			this.comboRelease.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboRelease.FormattingEnabled = true;
			this.comboRelease.Name = "comboRelease";
			this.comboRelease.SelectedIndexChanged += new System.EventHandler(this.comboRelease_SelectedIndexChanged);
			this.comboRelease.Format += new System.Windows.Forms.ListControlConvertEventHandler(this.comboRelease_Format);
			// 
			// releaseBindingSource
			// 
			this.releaseBindingSource.DataSource = typeof(MusicBrainz.Release);
			// 
			// frmCUERipper
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.comboRelease);
			this.Controls.Add(this.buttonPause);
			this.Controls.Add(this.buttonAbort);
			this.Controls.Add(this.comboImage);
			this.Controls.Add(this.comboCodec);
			this.Controls.Add(this.comboLossless);
			this.Controls.Add(this.buttonGo);
			this.Controls.Add(this.listTracks);
			this.Controls.Add(this.statusStrip1);
			this.Controls.Add(this.comboDrives);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "frmCUERipper";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Load += new System.EventHandler(this.frmCUERipper_Load);
			this.statusStrip1.ResumeLayout(false);
			this.statusStrip1.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.releaseBindingSource)).EndInit();
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
		private System.Windows.Forms.ComboBox comboLossless;
		private System.Windows.Forms.ComboBox comboCodec;
		private System.Windows.Forms.ComboBox comboImage;
		private System.Windows.Forms.Button buttonAbort;
		private System.Windows.Forms.Button buttonPause;
		private System.Windows.Forms.ComboBox comboRelease;
		private System.Windows.Forms.BindingSource releaseBindingSource;
	}
}

