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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmChoice));
			this.comboRelease = new System.Windows.Forms.ComboBox();
			this.button1 = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// comboRelease
			// 
			resources.ApplyResources(this.comboRelease, "comboRelease");
			this.comboRelease.BackColor = System.Drawing.SystemColors.Control;
			this.comboRelease.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.comboRelease.FormattingEnabled = true;
			this.comboRelease.Name = "comboRelease";
			// 
			// button1
			// 
			this.button1.DialogResult = System.Windows.Forms.DialogResult.OK;
			resources.ApplyResources(this.button1, "button1");
			this.button1.Name = "button1";
			this.button1.UseVisualStyleBackColor = true;
			// 
			// frmChoice
			// 
			this.AcceptButton = this.button1;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Controls.Add(this.button1);
			this.Controls.Add(this.comboRelease);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.Name = "frmChoice";
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Button button1;
		public System.Windows.Forms.ComboBox comboRelease;
	}
}