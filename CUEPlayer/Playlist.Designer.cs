namespace CUEPlayer
{
	partial class Playlist
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
			this.listViewTracks = new System.Windows.Forms.ListView();
			this.columnName = new System.Windows.Forms.ColumnHeader();
			this.columnLength = new System.Windows.Forms.ColumnHeader();
			this.contextMenuStripPlaylist = new System.Windows.Forms.ContextMenuStrip(this.components);
			this.exploreToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.removeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.contextMenuStripPlaylist.SuspendLayout();
			this.SuspendLayout();
			// 
			// listViewTracks
			// 
			this.listViewTracks.AllowDrop = true;
			this.listViewTracks.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.listViewTracks.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnName,
            this.columnLength});
			this.listViewTracks.ContextMenuStrip = this.contextMenuStripPlaylist;
			this.listViewTracks.Dock = System.Windows.Forms.DockStyle.Fill;
			this.listViewTracks.FullRowSelect = true;
			this.listViewTracks.GridLines = true;
			this.listViewTracks.Location = new System.Drawing.Point(0, 0);
			this.listViewTracks.Margin = new System.Windows.Forms.Padding(0);
			this.listViewTracks.Name = "listViewTracks";
			this.listViewTracks.Size = new System.Drawing.Size(372, 279);
			this.listViewTracks.TabIndex = 6;
			this.listViewTracks.UseCompatibleStateImageBehavior = false;
			this.listViewTracks.View = System.Windows.Forms.View.Details;
			this.listViewTracks.DragDrop += new System.Windows.Forms.DragEventHandler(this.listViewTracks_DragDrop);
			this.listViewTracks.KeyDown += new System.Windows.Forms.KeyEventHandler(this.listViewTracks_KeyDown);
			this.listViewTracks.ItemDrag += new System.Windows.Forms.ItemDragEventHandler(this.listViewTracks_ItemDrag);
			this.listViewTracks.DragOver += new System.Windows.Forms.DragEventHandler(this.listViewTracks_DragOver);
			// 
			// columnName
			// 
			this.columnName.Text = "Name";
			this.columnName.Width = 256;
			// 
			// columnLength
			// 
			this.columnLength.Text = "Length";
			this.columnLength.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			this.columnLength.Width = 80;
			// 
			// contextMenuStripPlaylist
			// 
			this.contextMenuStripPlaylist.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exploreToolStripMenuItem,
            this.removeToolStripMenuItem});
			this.contextMenuStripPlaylist.Name = "contextMenuStripPlaylist";
			this.contextMenuStripPlaylist.Size = new System.Drawing.Size(118, 48);
			// 
			// exploreToolStripMenuItem
			// 
			this.exploreToolStripMenuItem.Name = "exploreToolStripMenuItem";
			this.exploreToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
			this.exploreToolStripMenuItem.Text = "Explore";
			this.exploreToolStripMenuItem.Click += new System.EventHandler(this.exploreToolStripMenuItem_Click);
			// 
			// removeToolStripMenuItem
			// 
			this.removeToolStripMenuItem.Name = "removeToolStripMenuItem";
			this.removeToolStripMenuItem.Size = new System.Drawing.Size(117, 22);
			this.removeToolStripMenuItem.Text = "Remove";
			this.removeToolStripMenuItem.Click += new System.EventHandler(this.removeToolStripMenuItem_Click);
			// 
			// Playlist
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(372, 279);
			this.ControlBox = false;
			this.Controls.Add(this.listViewTracks);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "Playlist";
			this.Text = "Playlist";
			this.contextMenuStripPlaylist.ResumeLayout(false);
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.ListView listViewTracks;
		private System.Windows.Forms.ColumnHeader columnName;
		private System.Windows.Forms.ColumnHeader columnLength;
		private System.Windows.Forms.ContextMenuStrip contextMenuStripPlaylist;
		private System.Windows.Forms.ToolStripMenuItem exploreToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem removeToolStripMenuItem;
	}
}