namespace CUEPlayer
{
	partial class frmCUEPlayer
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
			this.SuspendLayout();
			// 
			// frmCUEPlayer
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(1200, 477);
			this.IsMdiContainer = true;
			this.Name = "frmCUEPlayer";
			this.Text = "CUEPlayer 2.0.7";
			this.Load += new System.EventHandler(this.frmCUEPlayer_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.frmCUEPlayer_FormClosing);
			this.ResumeLayout(false);

		}

		#endregion

	}
}

