namespace JDP
{
	partial class frmPassword
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmPassword));
			this.txtPassword = new System.Windows.Forms.TextBox();
			this.button1 = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// txtPassword
			// 
			this.txtPassword.AccessibleDescription = null;
			this.txtPassword.AccessibleName = null;
			resources.ApplyResources(this.txtPassword, "txtPassword");
			this.txtPassword.BackgroundImage = null;
			this.txtPassword.Font = null;
			this.txtPassword.Name = "txtPassword";
			this.txtPassword.UseSystemPasswordChar = true;
			// 
			// button1
			// 
			this.button1.AccessibleDescription = null;
			this.button1.AccessibleName = null;
			resources.ApplyResources(this.button1, "button1");
			this.button1.BackgroundImage = null;
			this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.button1.Font = null;
			this.button1.Name = "button1";
			this.button1.UseVisualStyleBackColor = true;
			// 
			// frmPassword
			// 
			this.AcceptButton = this.button1;
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.Controls.Add(this.button1);
			this.Controls.Add(this.txtPassword);
			this.Font = null;
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Icon = null;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "frmPassword";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.TopMost = true;
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button button1;
		public System.Windows.Forms.TextBox txtPassword;
	}
}