
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
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.radioButtonCoversNone = new System.Windows.Forms.RadioButton();
            this.radioButtonCoversPrimary = new System.Windows.Forms.RadioButton();
            this.radioButtonCoversExtensive = new System.Windows.Forms.RadioButton();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.radioButtonCoversSmall = new System.Windows.Forms.RadioButton();
            this.radioButtonCoversLarge = new System.Windows.Forms.RadioButton();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(142, 20);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(205, 20);
            this.label1.TabIndex = 0;
            this.label1.Text = "CUETools DB Plugin V2.2.0";
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(142, 103);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(345, 52);
            this.label2.TabIndex = 1;
            this.label2.Text = "Copyright (c) 2011-2022 Grigory Chudov";
            // 
            // linkLabel1
            // 
            this.linkLabel1.AutoSize = true;
            this.linkLabel1.Location = new System.Drawing.Point(142, 55);
            this.linkLabel1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.linkLabel1.Name = "linkLabel1";
            this.linkLabel1.Size = new System.Drawing.Size(234, 20);
            this.linkLabel1.TabIndex = 3;
            this.linkLabel1.TabStop = true;
            this.linkLabel1.Text = "http://db.cuetools.net/about.php";
            this.linkLabel1.LinkClicked += new System.Windows.Forms.LinkLabelLinkClickedEventHandler(this.linkLabel1_LinkClicked);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = global::CUETools.CTDB.EACPlugin.Properties.Resources.ctdb64;
            this.pictureBox1.Location = new System.Drawing.Point(20, 20);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(96, 98);
            this.pictureBox1.TabIndex = 4;
            this.pictureBox1.TabStop = false;
            // 
            // radioButtonMBExtensive
            // 
            this.radioButtonMBExtensive.AutoSize = true;
            this.radioButtonMBExtensive.Location = new System.Drawing.Point(9, 25);
            this.radioButtonMBExtensive.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonMBExtensive.Name = "radioButtonMBExtensive";
            this.radioButtonMBExtensive.Size = new System.Drawing.Size(102, 24);
            this.radioButtonMBExtensive.TabIndex = 6;
            this.radioButtonMBExtensive.TabStop = true;
            this.radioButtonMBExtensive.Text = "Extensive";
            this.radioButtonMBExtensive.UseVisualStyleBackColor = true;
            // 
            // radioButtonMBFast
            // 
            this.radioButtonMBFast.AutoSize = true;
            this.radioButtonMBFast.Location = new System.Drawing.Point(9, 77);
            this.radioButtonMBFast.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonMBFast.Name = "radioButtonMBFast";
            this.radioButtonMBFast.Size = new System.Drawing.Size(66, 24);
            this.radioButtonMBFast.TabIndex = 7;
            this.radioButtonMBFast.TabStop = true;
            this.radioButtonMBFast.Text = "Fast";
            this.radioButtonMBFast.UseVisualStyleBackColor = true;
            // 
            // radioButtonMBDefault
            // 
            this.radioButtonMBDefault.AutoSize = true;
            this.radioButtonMBDefault.Location = new System.Drawing.Point(9, 51);
            this.radioButtonMBDefault.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonMBDefault.Name = "radioButtonMBDefault";
            this.radioButtonMBDefault.Size = new System.Drawing.Size(86, 24);
            this.radioButtonMBDefault.TabIndex = 8;
            this.radioButtonMBDefault.TabStop = true;
            this.radioButtonMBDefault.Text = "Default";
            this.radioButtonMBDefault.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radioButtonMBFast);
            this.groupBox1.Controls.Add(this.radioButtonMBDefault);
            this.groupBox1.Controls.Add(this.radioButtonMBExtensive);
            this.groupBox1.Location = new System.Drawing.Point(18, 195);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Size = new System.Drawing.Size(204, 134);
            this.groupBox1.TabIndex = 15;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Metadata search mode:";
            // 
            // buttonOk
            // 
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Location = new System.Drawing.Point(648, 291);
            this.buttonOk.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(112, 38);
            this.buttonOk.TabIndex = 17;
            this.buttonOk.Text = "OK";
            this.buttonOk.UseVisualStyleBackColor = true;
            this.buttonOk.Click += new System.EventHandler(this.button2_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Location = new System.Drawing.Point(648, 243);
            this.buttonCancel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(112, 38);
            this.buttonCancel.TabIndex = 19;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.radioButtonCoversNone);
            this.groupBox2.Controls.Add(this.radioButtonCoversPrimary);
            this.groupBox2.Controls.Add(this.radioButtonCoversExtensive);
            this.groupBox2.Location = new System.Drawing.Point(231, 195);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox2.Size = new System.Drawing.Size(204, 134);
            this.groupBox2.TabIndex = 16;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Covers search mode:";
            // 
            // radioButtonCoversNone
            // 
            this.radioButtonCoversNone.AutoSize = true;
            this.radioButtonCoversNone.Location = new System.Drawing.Point(9, 77);
            this.radioButtonCoversNone.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonCoversNone.Name = "radioButtonCoversNone";
            this.radioButtonCoversNone.Size = new System.Drawing.Size(72, 24);
            this.radioButtonCoversNone.TabIndex = 7;
            this.radioButtonCoversNone.TabStop = true;
            this.radioButtonCoversNone.Text = "None";
            this.radioButtonCoversNone.UseVisualStyleBackColor = true;
            // 
            // radioButtonCoversPrimary
            // 
            this.radioButtonCoversPrimary.AutoSize = true;
            this.radioButtonCoversPrimary.Location = new System.Drawing.Point(9, 51);
            this.radioButtonCoversPrimary.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonCoversPrimary.Name = "radioButtonCoversPrimary";
            this.radioButtonCoversPrimary.Size = new System.Drawing.Size(86, 24);
            this.radioButtonCoversPrimary.TabIndex = 8;
            this.radioButtonCoversPrimary.TabStop = true;
            this.radioButtonCoversPrimary.Text = "Primary";
            this.radioButtonCoversPrimary.UseVisualStyleBackColor = true;
            // 
            // radioButtonCoversExtensive
            // 
            this.radioButtonCoversExtensive.AutoSize = true;
            this.radioButtonCoversExtensive.Location = new System.Drawing.Point(9, 25);
            this.radioButtonCoversExtensive.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonCoversExtensive.Name = "radioButtonCoversExtensive";
            this.radioButtonCoversExtensive.Size = new System.Drawing.Size(102, 24);
            this.radioButtonCoversExtensive.TabIndex = 6;
            this.radioButtonCoversExtensive.TabStop = true;
            this.radioButtonCoversExtensive.Text = "Extensive";
            this.radioButtonCoversExtensive.UseVisualStyleBackColor = true;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.radioButtonCoversSmall);
            this.groupBox3.Controls.Add(this.radioButtonCoversLarge);
            this.groupBox3.Location = new System.Drawing.Point(443, 195);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox3.Size = new System.Drawing.Size(163, 134);
            this.groupBox3.TabIndex = 17;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Covers size:";
            // 
            // radioButtonCoversSmall
            // 
            this.radioButtonCoversSmall.AutoSize = true;
            this.radioButtonCoversSmall.Location = new System.Drawing.Point(9, 51);
            this.radioButtonCoversSmall.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonCoversSmall.Name = "radioButtonCoversSmall";
            this.radioButtonCoversSmall.Size = new System.Drawing.Size(73, 24);
            this.radioButtonCoversSmall.TabIndex = 8;
            this.radioButtonCoversSmall.TabStop = true;
            this.radioButtonCoversSmall.Text = "Small";
            this.radioButtonCoversSmall.UseVisualStyleBackColor = true;
            // 
            // radioButtonCoversLarge
            // 
            this.radioButtonCoversLarge.AutoSize = true;
            this.radioButtonCoversLarge.Location = new System.Drawing.Point(9, 25);
            this.radioButtonCoversLarge.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.radioButtonCoversLarge.Name = "radioButtonCoversLarge";
            this.radioButtonCoversLarge.Size = new System.Drawing.Size(75, 24);
            this.radioButtonCoversLarge.TabIndex = 6;
            this.radioButtonCoversLarge.TabStop = true;
            this.radioButtonCoversLarge.Text = "Large";
            this.radioButtonCoversLarge.UseVisualStyleBackColor = true;
            // 
            // Options
            // 
            this.AcceptButton = this.buttonOk;
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonCancel;
            this.ClientSize = new System.Drawing.Size(773, 354);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOk);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.linkLabel1);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
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
		private System.Windows.Forms.RadioButton radioButtonMBExtensive;
		private System.Windows.Forms.RadioButton radioButtonMBFast;
		private System.Windows.Forms.RadioButton radioButtonMBDefault;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.RadioButton radioButtonCoversNone;
        private System.Windows.Forms.RadioButton radioButtonCoversPrimary;
        private System.Windows.Forms.RadioButton radioButtonCoversExtensive;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.RadioButton radioButtonCoversSmall;
        private System.Windows.Forms.RadioButton radioButtonCoversLarge;
    }
}