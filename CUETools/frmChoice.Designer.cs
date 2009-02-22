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
			this.button1 = new System.Windows.Forms.Button();
			this.listChoices = new System.Windows.Forms.ListView();
			this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
			this.imageList1 = new System.Windows.Forms.ImageList(this.components);
			this.textBox1 = new System.Windows.Forms.TextBox();
			this.listTracks = new System.Windows.Forms.ListView();
			this.Title = new System.Windows.Forms.ColumnHeader();
			this.TrackNo = new System.Windows.Forms.ColumnHeader();
			this.Length = new System.Windows.Forms.ColumnHeader();
			this.btnEdit = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// button1
			// 
			this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
			resources.ApplyResources(this.button1, "button1");
			this.button1.Name = "button1";
			this.button1.UseVisualStyleBackColor = true;
			// 
			// listChoices
			// 
			this.listChoices.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1});
			this.listChoices.FullRowSelect = true;
			this.listChoices.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
			this.listChoices.HideSelection = false;
			resources.ApplyResources(this.listChoices, "listChoices");
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
			this.imageList1.Images.SetKeyName(0, "eac.ico");
			this.imageList1.Images.SetKeyName(1, "freedb.gif");
			this.imageList1.Images.SetKeyName(2, "musicbrainz.ico");
			this.imageList1.Images.SetKeyName(3, "cue.ico");
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
			resources.ApplyResources(this.listTracks, "listTracks");
			this.listTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.Title,
            this.TrackNo,
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
			// Length
			// 
			resources.ApplyResources(this.Length, "Length");
			// 
			// btnEdit
			// 
			resources.ApplyResources(this.btnEdit, "btnEdit");
			this.btnEdit.Name = "btnEdit";
			this.btnEdit.UseVisualStyleBackColor = true;
			this.btnEdit.Click += new System.EventHandler(this.btnEdit_Click);
			// 
			// frmChoice
			// 
			this.AcceptButton = this.button1;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.btnEdit);
			this.Controls.Add(this.listTracks);
			this.Controls.Add(this.textBox1);
			this.Controls.Add(this.listChoices);
			this.Controls.Add(this.button1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "frmChoice";
			this.Load += new System.EventHandler(this.frmChoice_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmChoice_FormClosing);
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ImageList imageList1;
		private System.Windows.Forms.TextBox textBox1;
		private System.Windows.Forms.ListView listChoices;
		private System.Windows.Forms.ListView listTracks;
		private System.Windows.Forms.ColumnHeader Title;
		private System.Windows.Forms.ColumnHeader TrackNo;
		private System.Windows.Forms.ColumnHeader Length;
		private System.Windows.Forms.Button btnEdit;
	}
}