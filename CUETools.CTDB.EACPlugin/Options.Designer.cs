
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
			this.radioButtonMBHigh = new System.Windows.Forms.RadioButton();
			this.radioButtonMBLow = new System.Windows.Forms.RadioButton();
			this.radioButtonMBMedium = new System.Windows.Forms.RadioButton();
			this.radioButtonMBNone = new System.Windows.Forms.RadioButton();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.radioButtonFDMedium = new System.Windows.Forms.RadioButton();
			this.radioButtonFDHigh = new System.Windows.Forms.RadioButton();
			this.radioButtonFDLow = new System.Windows.Forms.RadioButton();
			this.radioButtonFDNone = new System.Windows.Forms.RadioButton();
			this.buttonOk = new System.Windows.Forms.Button();
			this.label3 = new System.Windows.Forms.Label();
			this.groupBox3 = new System.Windows.Forms.GroupBox();
			this.radioButtonFZMedium = new System.Windows.Forms.RadioButton();
			this.radioButtonFZHigh = new System.Windows.Forms.RadioButton();
			this.radioButtonFZLow = new System.Windows.Forms.RadioButton();
			this.radioButtonFZNone = new System.Windows.Forms.RadioButton();
			this.buttonCancel = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
			this.groupBox1.SuspendLayout();
			this.groupBox2.SuspendLayout();
			this.groupBox3.SuspendLayout();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(95, 13);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(139, 13);
			this.label1.TabIndex = 0;
			this.label1.Text = "CUETools DB Plugin V2.1.2";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(95, 67);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(230, 34);
			this.label2.TabIndex = 1;
			this.label2.Text = "Copyright (c) 2011 Gregory S. Chudov";
			// 
			// linkLabel1
			// 
			this.linkLabel1.AutoSize = true;
			this.linkLabel1.Location = new System.Drawing.Point(95, 36);
			this.linkLabel1.Name = "linkLabel1";
			this.linkLabel1.Size = new System.Drawing.Size(164, 13);
			this.linkLabel1.TabIndex = 3;
			this.linkLabel1.TabStop = true;
			this.linkLabel1.Text = "http://db.cuetools.net/about.php";
			this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel1_LinkClicked);
			// 
			// pictureBox1
			// 
			this.pictureBox1.Image = global::CUETools.CTDB.EACPlugin.Properties.Resources.ctdb64;
			this.pictureBox1.Location = new System.Drawing.Point(13, 13);
			this.pictureBox1.Name = "pictureBox1";
			this.pictureBox1.Size = new System.Drawing.Size(64, 64);
			this.pictureBox1.TabIndex = 4;
			this.pictureBox1.TabStop = false;
			// 
			// radioButtonMBHigh
			// 
			this.radioButtonMBHigh.AutoSize = true;
			this.radioButtonMBHigh.Location = new System.Drawing.Point(6, 16);
			this.radioButtonMBHigh.Name = "radioButtonMBHigh";
			this.radioButtonMBHigh.Size = new System.Drawing.Size(47, 17);
			this.radioButtonMBHigh.TabIndex = 6;
			this.radioButtonMBHigh.TabStop = true;
			this.radioButtonMBHigh.Text = "High";
			this.radioButtonMBHigh.UseVisualStyleBackColor = true;
			// 
			// radioButtonMBLow
			// 
			this.radioButtonMBLow.AutoSize = true;
			this.radioButtonMBLow.Location = new System.Drawing.Point(6, 50);
			this.radioButtonMBLow.Name = "radioButtonMBLow";
			this.radioButtonMBLow.Size = new System.Drawing.Size(45, 17);
			this.radioButtonMBLow.TabIndex = 7;
			this.radioButtonMBLow.TabStop = true;
			this.radioButtonMBLow.Text = "Low";
			this.radioButtonMBLow.UseVisualStyleBackColor = true;
			// 
			// radioButtonMBMedium
			// 
			this.radioButtonMBMedium.AutoSize = true;
			this.radioButtonMBMedium.Location = new System.Drawing.Point(6, 33);
			this.radioButtonMBMedium.Name = "radioButtonMBMedium";
			this.radioButtonMBMedium.Size = new System.Drawing.Size(62, 17);
			this.radioButtonMBMedium.TabIndex = 8;
			this.radioButtonMBMedium.TabStop = true;
			this.radioButtonMBMedium.Text = "Medium";
			this.radioButtonMBMedium.UseVisualStyleBackColor = true;
			// 
			// radioButtonMBNone
			// 
			this.radioButtonMBNone.AutoSize = true;
			this.radioButtonMBNone.Location = new System.Drawing.Point(6, 67);
			this.radioButtonMBNone.Name = "radioButtonMBNone";
			this.radioButtonMBNone.Size = new System.Drawing.Size(51, 17);
			this.radioButtonMBNone.TabIndex = 9;
			this.radioButtonMBNone.TabStop = true;
			this.radioButtonMBNone.Text = "None";
			this.radioButtonMBNone.UseVisualStyleBackColor = true;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.radioButtonMBMedium);
			this.groupBox1.Controls.Add(this.radioButtonMBHigh);
			this.groupBox1.Controls.Add(this.radioButtonMBLow);
			this.groupBox1.Controls.Add(this.radioButtonMBNone);
			this.groupBox1.Location = new System.Drawing.Point(12, 119);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(83, 95);
			this.groupBox1.TabIndex = 15;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Musicbrainz";
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.Add(this.radioButtonFDMedium);
			this.groupBox2.Controls.Add(this.radioButtonFDHigh);
			this.groupBox2.Controls.Add(this.radioButtonFDLow);
			this.groupBox2.Controls.Add(this.radioButtonFDNone);
			this.groupBox2.Location = new System.Drawing.Point(101, 119);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(83, 95);
			this.groupBox2.TabIndex = 16;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Freedb";
			// 
			// radioButtonFDMedium
			// 
			this.radioButtonFDMedium.AutoSize = true;
			this.radioButtonFDMedium.Location = new System.Drawing.Point(6, 33);
			this.radioButtonFDMedium.Name = "radioButtonFDMedium";
			this.radioButtonFDMedium.Size = new System.Drawing.Size(62, 17);
			this.radioButtonFDMedium.TabIndex = 8;
			this.radioButtonFDMedium.TabStop = true;
			this.radioButtonFDMedium.Text = "Medium";
			this.radioButtonFDMedium.UseVisualStyleBackColor = true;
			// 
			// radioButtonFDHigh
			// 
			this.radioButtonFDHigh.AutoSize = true;
			this.radioButtonFDHigh.Location = new System.Drawing.Point(6, 16);
			this.radioButtonFDHigh.Name = "radioButtonFDHigh";
			this.radioButtonFDHigh.Size = new System.Drawing.Size(47, 17);
			this.radioButtonFDHigh.TabIndex = 6;
			this.radioButtonFDHigh.TabStop = true;
			this.radioButtonFDHigh.Text = "High";
			this.radioButtonFDHigh.UseVisualStyleBackColor = true;
			// 
			// radioButtonFDLow
			// 
			this.radioButtonFDLow.AutoSize = true;
			this.radioButtonFDLow.Location = new System.Drawing.Point(6, 50);
			this.radioButtonFDLow.Name = "radioButtonFDLow";
			this.radioButtonFDLow.Size = new System.Drawing.Size(45, 17);
			this.radioButtonFDLow.TabIndex = 7;
			this.radioButtonFDLow.TabStop = true;
			this.radioButtonFDLow.Text = "Low";
			this.radioButtonFDLow.UseVisualStyleBackColor = true;
			// 
			// radioButtonFDNone
			// 
			this.radioButtonFDNone.AutoSize = true;
			this.radioButtonFDNone.Location = new System.Drawing.Point(6, 67);
			this.radioButtonFDNone.Name = "radioButtonFDNone";
			this.radioButtonFDNone.Size = new System.Drawing.Size(51, 17);
			this.radioButtonFDNone.TabIndex = 9;
			this.radioButtonFDNone.TabStop = true;
			this.radioButtonFDNone.Text = "None";
			this.radioButtonFDNone.UseVisualStyleBackColor = true;
			// 
			// buttonOk
			// 
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.buttonOk.Location = new System.Drawing.Point(303, 189);
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.Size = new System.Drawing.Size(75, 25);
			this.buttonOk.TabIndex = 17;
			this.buttonOk.Text = "OK";
			this.buttonOk.UseVisualStyleBackColor = true;
			this.buttonOk.Click += new System.EventHandler(this.button2_Click);
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(12, 101);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(134, 13);
			this.label3.TabIndex = 18;
			this.label3.Text = "Metadata providers priority:";
			// 
			// groupBox3
			// 
			this.groupBox3.Controls.Add(this.radioButtonFZMedium);
			this.groupBox3.Controls.Add(this.radioButtonFZHigh);
			this.groupBox3.Controls.Add(this.radioButtonFZLow);
			this.groupBox3.Controls.Add(this.radioButtonFZNone);
			this.groupBox3.Location = new System.Drawing.Point(190, 119);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.Size = new System.Drawing.Size(83, 95);
			this.groupBox3.TabIndex = 17;
			this.groupBox3.TabStop = false;
			this.groupBox3.Text = "Freedb fuzzy";
			// 
			// radioButtonFZMedium
			// 
			this.radioButtonFZMedium.AutoSize = true;
			this.radioButtonFZMedium.Location = new System.Drawing.Point(6, 33);
			this.radioButtonFZMedium.Name = "radioButtonFZMedium";
			this.radioButtonFZMedium.Size = new System.Drawing.Size(62, 17);
			this.radioButtonFZMedium.TabIndex = 8;
			this.radioButtonFZMedium.TabStop = true;
			this.radioButtonFZMedium.Text = "Medium";
			this.radioButtonFZMedium.UseVisualStyleBackColor = true;
			// 
			// radioButtonFZHigh
			// 
			this.radioButtonFZHigh.AutoSize = true;
			this.radioButtonFZHigh.Location = new System.Drawing.Point(6, 16);
			this.radioButtonFZHigh.Name = "radioButtonFZHigh";
			this.radioButtonFZHigh.Size = new System.Drawing.Size(47, 17);
			this.radioButtonFZHigh.TabIndex = 6;
			this.radioButtonFZHigh.TabStop = true;
			this.radioButtonFZHigh.Text = "High";
			this.radioButtonFZHigh.UseVisualStyleBackColor = true;
			// 
			// radioButtonFZLow
			// 
			this.radioButtonFZLow.AutoSize = true;
			this.radioButtonFZLow.Location = new System.Drawing.Point(6, 50);
			this.radioButtonFZLow.Name = "radioButtonFZLow";
			this.radioButtonFZLow.Size = new System.Drawing.Size(45, 17);
			this.radioButtonFZLow.TabIndex = 7;
			this.radioButtonFZLow.TabStop = true;
			this.radioButtonFZLow.Text = "Low";
			this.radioButtonFZLow.UseVisualStyleBackColor = true;
			// 
			// radioButtonFZNone
			// 
			this.radioButtonFZNone.AutoSize = true;
			this.radioButtonFZNone.Location = new System.Drawing.Point(6, 67);
			this.radioButtonFZNone.Name = "radioButtonFZNone";
			this.radioButtonFZNone.Size = new System.Drawing.Size(51, 17);
			this.radioButtonFZNone.TabIndex = 9;
			this.radioButtonFZNone.TabStop = true;
			this.radioButtonFZNone.Text = "None";
			this.radioButtonFZNone.UseVisualStyleBackColor = true;
			// 
			// buttonCancel
			// 
			this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.buttonCancel.Location = new System.Drawing.Point(303, 158);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(75, 25);
			this.buttonCancel.TabIndex = 19;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			// 
			// Options
			// 
			this.AcceptButton = this.buttonOk;
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = this.buttonCancel;
			this.ClientSize = new System.Drawing.Size(390, 227);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.groupBox3);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.buttonOk);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.pictureBox1);
			this.Controls.Add(this.linkLabel1);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Options";
			this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
			this.Text = "Options";
			this.Load += new System.EventHandler(this.Options_Load);
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
			this.groupBox1.ResumeLayout(false);
			this.groupBox1.PerformLayout();
			this.groupBox2.ResumeLayout(false);
			this.groupBox2.PerformLayout();
			this.groupBox3.ResumeLayout(false);
			this.groupBox3.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
		private System.Windows.Forms.LinkLabel linkLabel1;
		private System.Windows.Forms.PictureBox pictureBox1;
		private System.Windows.Forms.RadioButton radioButtonMBHigh;
		private System.Windows.Forms.RadioButton radioButtonMBLow;
		private System.Windows.Forms.RadioButton radioButtonMBMedium;
		private System.Windows.Forms.RadioButton radioButtonMBNone;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.RadioButton radioButtonFDMedium;
		private System.Windows.Forms.RadioButton radioButtonFDHigh;
		private System.Windows.Forms.RadioButton radioButtonFDLow;
		private System.Windows.Forms.RadioButton radioButtonFDNone;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.GroupBox groupBox3;
		private System.Windows.Forms.RadioButton radioButtonFZMedium;
		private System.Windows.Forms.RadioButton radioButtonFZHigh;
		private System.Windows.Forms.RadioButton radioButtonFZLow;
		private System.Windows.Forms.RadioButton radioButtonFZNone;
		private System.Windows.Forms.Button buttonCancel;
    }
}