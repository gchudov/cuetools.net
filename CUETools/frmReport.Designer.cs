namespace JDP
{
	partial class frmReport
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmReport));
			this.btnClose = new System.Windows.Forms.Button();
			this.txtReport = new System.Windows.Forms.TextBox();
			this.SuspendLayout();
			// 
			// btnClose
			// 
			this.btnClose.AccessibleDescription = null;
			this.btnClose.AccessibleName = null;
			resources.ApplyResources(this.btnClose, "btnClose");
			this.btnClose.BackgroundImage = null;
			this.btnClose.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.btnClose.Font = null;
			this.btnClose.Name = "btnClose";
			this.btnClose.UseVisualStyleBackColor = true;
			// 
			// txtReport
			// 
			this.txtReport.AccessibleDescription = null;
			this.txtReport.AccessibleName = null;
			resources.ApplyResources(this.txtReport, "txtReport");
			this.txtReport.BackColor = System.Drawing.SystemColors.Control;
			this.txtReport.BackgroundImage = null;
			this.txtReport.Name = "txtReport";
			this.txtReport.ReadOnly = true;
			// 
			// frmReport
			// 
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.CancelButton = this.btnClose;
			this.Controls.Add(this.txtReport);
			this.Controls.Add(this.btnClose);
			this.Icon = null;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "frmReport";
			this.ShowInTaskbar = false;
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button btnClose;
		private System.Windows.Forms.TextBox txtReport;
	}
}