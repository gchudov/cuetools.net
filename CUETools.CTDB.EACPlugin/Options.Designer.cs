
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
            this.radioButtonCoversSmall = new System.Windows.Forms.RadioButton();
            this.radioButtonCoversLarge = new System.Windows.Forms.RadioButton();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(95, 13);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(139, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "CUETools DB Plugin V2.1.7";
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(95, 67);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(230, 34);
            this.label2.TabIndex = 1;
            this.label2.Text = "Copyright (c) 2011-12 Grigory Chudov";
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
            // radioButtonMBExtensive
            // 
            this.radioButtonMBExtensive.AutoSize = true;
            this.radioButtonMBExtensive.Location = new System.Drawing.Point(6, 16);
            this.radioButtonMBExtensive.Name = "radioButtonMBExtensive";
            this.radioButtonMBExtensive.Size = new System.Drawing.Size(71, 17);
            this.radioButtonMBExtensive.TabIndex = 6;
            this.radioButtonMBExtensive.TabStop = true;
            this.radioButtonMBExtensive.Text = "Extensive";
            this.radioButtonMBExtensive.UseVisualStyleBackColor = true;
            // 
            // radioButtonMBFast
            // 
            this.radioButtonMBFast.AutoSize = true;
            this.radioButtonMBFast.Location = new System.Drawing.Point(6, 50);
            this.radioButtonMBFast.Name = "radioButtonMBFast";
            this.radioButtonMBFast.Size = new System.Drawing.Size(45, 17);
            this.radioButtonMBFast.TabIndex = 7;
            this.radioButtonMBFast.TabStop = true;
            this.radioButtonMBFast.Text = "Fast";
            this.radioButtonMBFast.UseVisualStyleBackColor = true;
            // 
            // radioButtonMBDefault
            // 
            this.radioButtonMBDefault.AutoSize = true;
            this.radioButtonMBDefault.Location = new System.Drawing.Point(6, 33);
            this.radioButtonMBDefault.Name = "radioButtonMBDefault";
            this.radioButtonMBDefault.Size = new System.Drawing.Size(59, 17);
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
            this.groupBox1.Location = new System.Drawing.Point(12, 127);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(136, 87);
            this.groupBox1.TabIndex = 15;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Metadata search mode:";
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
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.radioButtonCoversNone);
            this.groupBox2.Controls.Add(this.radioButtonCoversSmall);
            this.groupBox2.Controls.Add(this.radioButtonCoversLarge);
            this.groupBox2.Location = new System.Drawing.Point(154, 127);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(136, 87);
            this.groupBox2.TabIndex = 16;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Covers search mode:";
            // 
            // radioButtonCoversNone
            // 
            this.radioButtonCoversNone.AutoSize = true;
            this.radioButtonCoversNone.Location = new System.Drawing.Point(6, 50);
            this.radioButtonCoversNone.Name = "radioButtonCoversNone";
            this.radioButtonCoversNone.Size = new System.Drawing.Size(51, 17);
            this.radioButtonCoversNone.TabIndex = 7;
            this.radioButtonCoversNone.TabStop = true;
            this.radioButtonCoversNone.Text = "None";
            this.radioButtonCoversNone.UseVisualStyleBackColor = true;
            // 
            // radioButtonCoversSmall
            // 
            this.radioButtonCoversSmall.AutoSize = true;
            this.radioButtonCoversSmall.Location = new System.Drawing.Point(6, 33);
            this.radioButtonCoversSmall.Name = "radioButtonCoversSmall";
            this.radioButtonCoversSmall.Size = new System.Drawing.Size(50, 17);
            this.radioButtonCoversSmall.TabIndex = 8;
            this.radioButtonCoversSmall.TabStop = true;
            this.radioButtonCoversSmall.Text = "Small";
            this.radioButtonCoversSmall.UseVisualStyleBackColor = true;
            // 
            // radioButtonCoversLarge
            // 
            this.radioButtonCoversLarge.AutoSize = true;
            this.radioButtonCoversLarge.Location = new System.Drawing.Point(6, 16);
            this.radioButtonCoversLarge.Name = "radioButtonCoversLarge";
            this.radioButtonCoversLarge.Size = new System.Drawing.Size(52, 17);
            this.radioButtonCoversLarge.TabIndex = 6;
            this.radioButtonCoversLarge.TabStop = true;
            this.radioButtonCoversLarge.Text = "Large";
            this.radioButtonCoversLarge.UseVisualStyleBackColor = true;
            // 
            // Options
            // 
            this.AcceptButton = this.buttonOk;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonCancel;
            this.ClientSize = new System.Drawing.Size(390, 227);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOk);
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
        private System.Windows.Forms.RadioButton radioButtonCoversSmall;
        private System.Windows.Forms.RadioButton radioButtonCoversLarge;
    }
}