namespace CUEPlayer
{
	partial class Browser
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
			this.fileSystemTreeView1 = new CUEControls.FileSystemTreeView();
			this.SuspendLayout();
			// 
			// fileSystemTreeView1
			// 
			this.fileSystemTreeView1.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.fileSystemTreeView1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.fileSystemTreeView1.Location = new System.Drawing.Point(0, 0);
			this.fileSystemTreeView1.Name = "fileSystemTreeView1";
			this.fileSystemTreeView1.ShowLines = false;
			this.fileSystemTreeView1.ShowRootLines = false;
			this.fileSystemTreeView1.Size = new System.Drawing.Size(284, 264);
			this.fileSystemTreeView1.SpecialFolders = new CUEControls.ExtraSpecialFolder[] {
        CUEControls.ExtraSpecialFolder.MyComputer,
        CUEControls.ExtraSpecialFolder.MyMusic,
        CUEControls.ExtraSpecialFolder.CommonMusic};
			this.fileSystemTreeView1.TabIndex = 1;
			this.fileSystemTreeView1.NodeExpand += new CUEControls.FileSystemTreeViewNodeExpandHandler(this.fileSystemTreeView1_NodeExpand);
			// 
			// Browser
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(284, 264);
			this.ControlBox = false;
			this.Controls.Add(this.fileSystemTreeView1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "Browser";
			this.Text = "Browser";
			this.Load += new System.EventHandler(this.Browser_Load);
			this.ResumeLayout(false);

		}

		#endregion

		private CUEControls.FileSystemTreeView fileSystemTreeView1;
	}
}