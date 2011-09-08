
namespace AudioDataPlugIn
{
    partial class Options
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
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.linkLabel1 = new System.Windows.Forms.LinkLabel();
			this.pictureBox1 = new System.Windows.Forms.PictureBox();
			this.radioButtonMBExtensive = new System.Windows.Forms.RadioButton();
			this.radioButtonMBFast = new System.Windows.Forms.RadioButton();
			this.radioButtonMBDefault = new System.Windows.Forms.RadioButton();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.buttonOk = new System.Windows.Forms.Button();
			this.buttonCancel = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(127, 16);
			this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(182, 17);
			this.label1.TabIndex = 0;
			this.label1.Text = "CUETools DB Plugin V2.1.3";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(127, 82);
			this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(307, 42);
			this.label2.TabIndex = 1;
			this.label2.Text = "Copyright (c) 2011 Gregory S. Chudov";
			// 
			// linkLabel1
			// 
			this.linkLabel1.AutoSize = true;
			this.linkLabel1.Location = new System.Drawing.Point(127, 44);
			this.linkLabel1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
			this.linkLabel1.Name = "linkLabel1";
			this.linkLabel1.Size = new System.Drawing.Size(209, 17);
			this.linkLabel1.TabIndex = 3;
			this.linkLabel1.TabStop = true;
			this.linkLabel1.Text = "http://db.cuetools.net/about.php";
			this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel1_LinkClicked);
			// 
			// pictureBox1
			// 
			this.pictureBox1.Image = global::CUETools.CTDB.EACPlugin.Properties.Resources.ctdb64;
			this.pictureBox1.Location = new System.Drawing.Point(17, 16);
			this.pictureBox1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.pictureBox1.Name = "pictureBox1";
			this.pictureBox1.Size = new System.Drawing.Size(85, 79);
			this.pictureBox1.TabIndex = 4;
			this.pictureBox1.TabStop = false;
			// 
			// radioButtonMBExtensive
			// 
			this.radioButtonMBExtensive.AutoSize = true;
			this.radioButtonMBExtensive.Location = new System.Drawing.Point(8, 20);
			this.radioButtonMBExtensive.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.radioButtonMBExtensive.Name = "radioButtonMBExtensive";
			this.radioButtonMBExtensive.Size = new System.Drawing.Size(89, 21);
			this.radioButtonMBExtensive.TabIndex = 6;
			this.radioButtonMBExtensive.TabStop = true;
			this.radioButtonMBExtensive.Text = "Extensive";
			this.radioButtonMBExtensive.UseVisualStyleBackColor = true;
			// 
			// radioButtonMBFast
			// 
			this.radioButtonMBFast.AutoSize = true;
			this.radioButtonMBFast.Location = new System.Drawing.Point(8, 62);
			this.radioButtonMBFast.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.radioButtonMBFast.Name = "radioButtonMBFast";
			this.radioButtonMBFast.Size = new System.Drawing.Size(56, 21);
			this.radioButtonMBFast.TabIndex = 7;
			this.radioButtonMBFast.TabStop = true;
			this.radioButtonMBFast.Text = "Fast";
			this.radioButtonMBFast.UseVisualStyleBackColor = true;
			// 
			// radioButtonMBDefault
			// 
			this.radioButtonMBDefault.AutoSize = true;
			this.radioButtonMBDefault.Location = new System.Drawing.Point(8, 41);
			this.radioButtonMBDefault.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.radioButtonMBDefault.Name = "radioButtonMBDefault";
			this.radioButtonMBDefault.Size = new System.Drawing.Size(74, 21);
			this.radioButtonMBDefault.TabIndex = 8;
			this.radioButtonMBDefault.TabStop = true;
			this.radioButtonMBDefault.Text = "Default";
			this.radioButtonMBDefault.UseVisualStyleBackColor = true;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.radioButtonMBDefault);
			this.groupBox1.Controls.Add(this.radioButtonMBExtensive);
			this.groupBox1.Controls.Add(this.radioButtonMBFast);
			this.groupBox1.Location = new System.Drawing.Point(16, 156);
			this.groupBox1.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Padding = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.groupBox1.Size = new System.Drawing.Size(191, 107);
			this.groupBox1.TabIndex = 15;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Metadata search mode:";
			// 
			// buttonOk
			// 
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.buttonOk.Location = new System.Drawing.Point(404, 233);
			this.buttonOk.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.Size = new System.Drawing.Size(100, 31);
			this.buttonOk.TabIndex = 17;
			this.buttonOk.Text = "OK";
			this.buttonOk.UseVisualStyleBackColor = true;
			this.buttonOk.Click += new System.EventHandler(this.button2_Click);
			// 
			// buttonCancel
			// 
			this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.buttonCancel.Location = new System.Drawing.Point(404, 194);
			this.buttonCancel.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(100, 31);
			this.buttonCancel.TabIndex = 19;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			// 
			// Options
			// 
			this.AcceptButton = this.buttonOk;
			this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = this.buttonCancel;
			this.ClientSize = new System.Drawing.Size(520, 279);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonOk);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.pictureBox1);
			this.Controls.Add(this.linkLabel1);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Options";
			this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
			this.Text = "Options";
			this.Load += new System.EventHandler(this.Options_Load);
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
			this.groupBox1.ResumeLayout(false);
			this.groupBox1.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
		private System.Windows.Forms.LinkLabel linkLabel1;
		private System.Windows.Forms.PictureBox pictureBox1;
		private System.Windows.Forms.RadioButton radioButtonMBExtensive;
		private System.Windows.Forms.RadioButton radioButtonMBFast;
		private System.Windows.Forms.RadioButton radioButtonMBDefault;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.Button buttonCancel;
    }
}