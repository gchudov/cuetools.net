namespace JDP
{
	partial class frmSubmit
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmSubmit));
			this.textBox1 = new System.Windows.Forms.TextBox();
			this.buttonOk = new System.Windows.Forms.Button();
			this.buttonCancel = new System.Windows.Forms.Button();
			this.checkBoxRemember = new System.Windows.Forms.CheckBox();
			this.SuspendLayout();
			// 
			// textBox1
			// 
			this.textBox1.AccessibleDescription = null;
			this.textBox1.AccessibleName = null;
			resources.ApplyResources(this.textBox1, "textBox1");
			this.textBox1.BackColor = System.Drawing.SystemColors.Control;
			this.textBox1.BackgroundImage = null;
			this.textBox1.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBox1.Font = null;
			this.textBox1.Name = "textBox1";
			this.textBox1.ReadOnly = true;
			this.textBox1.TabStop = false;
			// 
			// buttonOk
			// 
			this.buttonOk.AccessibleDescription = null;
			this.buttonOk.AccessibleName = null;
			resources.ApplyResources(this.buttonOk, "buttonOk");
			this.buttonOk.BackgroundImage = null;
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.buttonOk.Font = null;
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.UseVisualStyleBackColor = true;
			// 
			// buttonCancel
			// 
			this.buttonCancel.AccessibleDescription = null;
			this.buttonCancel.AccessibleName = null;
			resources.ApplyResources(this.buttonCancel, "buttonCancel");
			this.buttonCancel.BackgroundImage = null;
			this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.buttonCancel.Font = null;
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			// 
			// checkBoxRemember
			// 
			this.checkBoxRemember.AccessibleDescription = null;
			this.checkBoxRemember.AccessibleName = null;
			resources.ApplyResources(this.checkBoxRemember, "checkBoxRemember");
			this.checkBoxRemember.BackgroundImage = null;
			this.checkBoxRemember.Checked = true;
			this.checkBoxRemember.CheckState = System.Windows.Forms.CheckState.Checked;
			this.checkBoxRemember.Font = null;
			this.checkBoxRemember.Name = "checkBoxRemember";
			this.checkBoxRemember.UseVisualStyleBackColor = true;
			// 
			// frmSubmit
			// 
			this.AcceptButton = this.buttonOk;
			this.AccessibleDescription = null;
			this.AccessibleName = null;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackgroundImage = null;
			this.CancelButton = this.buttonCancel;
			this.Controls.Add(this.checkBoxRemember);
			this.Controls.Add(this.buttonOk);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.textBox1);
			this.Font = null;
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Icon = null;
			this.Name = "frmSubmit";
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.TextBox textBox1;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.Button buttonCancel;
		internal System.Windows.Forms.CheckBox checkBoxRemember;

	}
}