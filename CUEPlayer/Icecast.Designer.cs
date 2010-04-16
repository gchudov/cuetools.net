namespace CUEPlayer
{
	partial class Icecast
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
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			this.textBoxBytes = new System.Windows.Forms.TextBox();
			this.checkBoxTransmit = new System.Windows.Forms.CheckBox();
			this.buttonSettings = new System.Windows.Forms.Button();
			this.textBoxLatency = new System.Windows.Forms.TextBox();
			this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
			this.SuspendLayout();
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// textBoxBytes
			// 
			this.textBoxBytes.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxBytes.Enabled = false;
			this.textBoxBytes.Location = new System.Drawing.Point(2, 87);
			this.textBoxBytes.Name = "textBoxBytes";
			this.textBoxBytes.ReadOnly = true;
			this.textBoxBytes.Size = new System.Drawing.Size(70, 13);
			this.textBoxBytes.TabIndex = 14;
			this.textBoxBytes.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
			// 
			// checkBoxTransmit
			// 
			this.checkBoxTransmit.Appearance = System.Windows.Forms.Appearance.Button;
			this.checkBoxTransmit.AutoSize = true;
			this.checkBoxTransmit.Image = global::CUEPlayer.Properties.Resources.transmit_blue;
			this.checkBoxTransmit.Location = new System.Drawing.Point(14, 106);
			this.checkBoxTransmit.Name = "checkBoxTransmit";
			this.checkBoxTransmit.Size = new System.Drawing.Size(22, 22);
			this.checkBoxTransmit.TabIndex = 15;
			this.checkBoxTransmit.UseVisualStyleBackColor = true;
			this.checkBoxTransmit.CheckedChanged += new System.EventHandler(this.checkBoxTransmit_CheckedChanged);
			// 
			// buttonSettings
			// 
			this.buttonSettings.AutoSize = true;
			this.buttonSettings.Image = global::CUEPlayer.Properties.Resources.cog;
			this.buttonSettings.Location = new System.Drawing.Point(42, 106);
			this.buttonSettings.Name = "buttonSettings";
			this.buttonSettings.Size = new System.Drawing.Size(22, 22);
			this.buttonSettings.TabIndex = 16;
			this.buttonSettings.UseVisualStyleBackColor = true;
			this.buttonSettings.Click += new System.EventHandler(this.buttonSettings_Click);
			// 
			// textBoxLatency
			// 
			this.textBoxLatency.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxLatency.Enabled = false;
			this.textBoxLatency.Location = new System.Drawing.Point(2, 62);
			this.textBoxLatency.Name = "textBoxLatency";
			this.textBoxLatency.ReadOnly = true;
			this.textBoxLatency.Size = new System.Drawing.Size(70, 13);
			this.textBoxLatency.TabIndex = 17;
			this.textBoxLatency.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
			// 
			// Icecast
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(74, 136);
			this.Controls.Add(this.textBoxLatency);
			this.Controls.Add(this.buttonSettings);
			this.Controls.Add(this.checkBoxTransmit);
			this.Controls.Add(this.textBoxBytes);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Icecast";
			this.Text = "Icecast";
			this.Load += new System.EventHandler(this.Icecast_Load);
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Timer timer1;
		private System.Windows.Forms.TextBox textBoxBytes;
		private System.Windows.Forms.CheckBox checkBoxTransmit;
		private System.Windows.Forms.Button buttonSettings;
		private System.Windows.Forms.TextBox textBoxLatency;
		private System.Windows.Forms.ToolTip toolTip1;
	}
}