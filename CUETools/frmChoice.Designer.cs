namespace JDP
{
	partial class frmChoice
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmChoice));
			this.buttonOk = new System.Windows.Forms.Button();
			this.listChoices = new System.Windows.Forms.ListView();
			this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
			this.imageList1 = new System.Windows.Forms.ImageList(this.components);
			this.textBox1 = new System.Windows.Forms.TextBox();
			this.listTracks = new System.Windows.Forms.ListView();
			this.Title = new System.Windows.Forms.ColumnHeader();
			this.TrackNo = new System.Windows.Forms.ColumnHeader();
			this.Start = new System.Windows.Forms.ColumnHeader();
			this.Length = new System.Windows.Forms.ColumnHeader();
			this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
			this.listMetadata = new System.Windows.Forms.ListView();
			this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
			this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
			this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
			this.pictureBox1 = new System.Windows.Forms.PictureBox();
			this.backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
			this.tableLayoutPanel1.SuspendLayout();
			this.tableLayoutPanel2.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
			this.SuspendLayout();
			// 
			// buttonOk
			// 
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			resources.ApplyResources(this.buttonOk, "buttonOk");
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.UseVisualStyleBackColor = true;
			// 
			// listChoices
			// 
			this.listChoices.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1});
			resources.ApplyResources(this.listChoices, "listChoices");
			this.listChoices.FullRowSelect = true;
			this.listChoices.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
			this.listChoices.HideSelection = false;
			this.listChoices.MultiSelect = false;
			this.listChoices.Name = "listChoices";
			this.listChoices.ShowItemToolTips = true;
			this.listChoices.SmallImageList = this.imageList1;
			this.listChoices.UseCompatibleStateImageBehavior = false;
			this.listChoices.View = System.Windows.Forms.View.Details;
			this.listChoices.SelectedIndexChanged += new System.EventHandler(this.listChoices_SelectedIndexChanged);
			// 
			// columnHeader1
			// 
			resources.ApplyResources(this.columnHeader1, "columnHeader1");
			// 
			// imageList1
			// 
			this.imageList1.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList1.ImageStream")));
			this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
			this.imageList1.Images.SetKeyName(0, "eac");
			this.imageList1.Images.SetKeyName(1, "freedb");
			this.imageList1.Images.SetKeyName(2, "musicbrainz");
			this.imageList1.Images.SetKeyName(3, "cue");
			this.imageList1.Images.SetKeyName(4, "tags");
			this.imageList1.Images.SetKeyName(5, "local");
			// 
			// textBox1
			// 
			this.textBox1.BorderStyle = System.Windows.Forms.BorderStyle.None;
			resources.ApplyResources(this.textBox1, "textBox1");
			this.textBox1.Name = "textBox1";
			this.textBox1.ReadOnly = true;
			// 
			// listTracks
			// 
			this.listTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.Title,
            this.TrackNo,
            this.Start,
            this.Length});
			resources.ApplyResources(this.listTracks, "listTracks");
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
			// tableLayoutPanel1
			// 
			resources.ApplyResources(this.tableLayoutPanel1, "tableLayoutPanel1");
			this.tableLayoutPanel1.Controls.Add(this.listMetadata, 0, 3);
			this.tableLayoutPanel1.Controls.Add(this.tableLayoutPanel2, 0, 5);
			this.tableLayoutPanel1.Controls.Add(this.textBox1, 0, 1);
			this.tableLayoutPanel1.Controls.Add(this.listTracks, 0, 4);
			this.tableLayoutPanel1.Controls.Add(this.listChoices, 0, 0);
			this.tableLayoutPanel1.Controls.Add(this.pictureBox1, 0, 2);
			this.tableLayoutPanel1.Name = "tableLayoutPanel1";
			// 
			// listMetadata
			// 
			this.listMetadata.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader2,
            this.columnHeader3});
			resources.ApplyResources(this.listMetadata, "listMetadata");
			this.listMetadata.FullRowSelect = true;
			this.listMetadata.GridLines = true;
			this.listMetadata.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
			this.listMetadata.LabelEdit = true;
			this.listMetadata.Name = "listMetadata";
			this.listMetadata.UseCompatibleStateImageBehavior = false;
			this.listMetadata.View = System.Windows.Forms.View.Details;
			this.listMetadata.AfterLabelEdit += new System.Windows.Forms.LabelEditEventHandler(this.listMetadata_AfterLabelEdit);
			this.listMetadata.DoubleClick += new System.EventHandler(this.listMetadata_DoubleClick);
			this.listMetadata.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listMetadata_KeyDown);
			// 
			// columnHeader2
			// 
			resources.ApplyResources(this.columnHeader2, "columnHeader2");
			// 
			// columnHeader3
			// 
			resources.ApplyResources(this.columnHeader3, "columnHeader3");
			// 
			// tableLayoutPanel2
			// 
			resources.ApplyResources(this.tableLayoutPanel2, "tableLayoutPanel2");
			this.tableLayoutPanel2.Controls.Add(this.buttonOk, 3, 0);
			this.tableLayoutPanel2.Name = "tableLayoutPanel2";
			// 
			// pictureBox1
			// 
			resources.ApplyResources(this.pictureBox1, "pictureBox1");
			this.pictureBox1.Name = "pictureBox1";
			this.pictureBox1.TabStop = false;
			this.pictureBox1.DoubleClick += new System.EventHandler(this.pictureBox1_DoubleClick);
			// 
			// backgroundWorker1
			// 
			this.backgroundWorker1.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorker1_DoWork);
			this.backgroundWorker1.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorker1_RunWorkerCompleted);
			// 
			// frmChoice
			// 
			this.AcceptButton = this.buttonOk;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.tableLayoutPanel1);
			this.MaximizeBox = false;
			this.Name = "frmChoice";
			this.ShowIcon = false;
			this.ShowInTaskbar = false;
			this.Load += new System.EventHandler(this.frmChoice_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmChoice_FormClosing);
			this.tableLayoutPanel1.ResumeLayout(false);
			this.tableLayoutPanel1.PerformLayout();
			this.tableLayoutPanel2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ImageList imageList1;
		private System.Windows.Forms.TextBox textBox1;
		private System.Windows.Forms.ListView listChoices;
		private System.Windows.Forms.ListView listTracks;
		private System.Windows.Forms.ColumnHeader Title;
		private System.Windows.Forms.ColumnHeader TrackNo;
		private System.Windows.Forms.ColumnHeader Length;
		private System.Windows.Forms.ColumnHeader Start;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel2;
		private System.Windows.Forms.ListView listMetadata;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.PictureBox pictureBox1;
		private System.ComponentModel.BackgroundWorker backgroundWorker1;
	}
}