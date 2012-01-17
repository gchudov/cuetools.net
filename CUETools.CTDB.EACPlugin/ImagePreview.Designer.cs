namespace CUETools.CTDB.EACPlugin
{
    partial class ImagePreview
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.Description = new System.Windows.Forms.Label();
            this.Description2 = new System.Windows.Forms.Label();
            this.ImagePanel = new System.Windows.Forms.Panel();
            this.MouseOverPanel = new System.Windows.Forms.Panel();
            this.SaveFile = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // Description
            // 
            this.Description.Location = new System.Drawing.Point(4, 4+ImageSize+4);
            this.Description.Name = "Description";
            this.Description.Size = new System.Drawing.Size(ImageSize - 16, 8);
            this.Description.TabIndex = 1;
            this.Description.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.Description.Font = new System.Drawing.Font("Microsoft Sans Serif", 6F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            // 
            // Description2
            // 
            this.Description2.Location = new System.Drawing.Point(4, 4 + ImageSize + 4 + 8);
            this.Description2.Name = "Description2";
            this.Description2.Size = new System.Drawing.Size(ImageSize - 16, 8);
            this.Description2.TabIndex = 3;
            this.Description2.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.Description2.Font = new System.Drawing.Font("Microsoft Sans Serif", 6F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            // 
            // ImagePanel
            // 
            this.ImagePanel.BackColor = System.Drawing.Color.Transparent;
            this.ImagePanel.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.ImagePanel.Location = new System.Drawing.Point(4, 4);
            this.ImagePanel.Name = "ImagePanel";
            this.ImagePanel.Size = new System.Drawing.Size(ImageSize, ImageSize);
            this.ImagePanel.TabIndex = 0;
            // 
            // MouseOverPanel
            // 
            this.MouseOverPanel.BackColor = System.Drawing.Color.Transparent;
            this.MouseOverPanel.Location = new System.Drawing.Point(ImageSize/2-16, ImageSize/2-16);
            this.MouseOverPanel.Name = "MouseOverPanel";
            this.MouseOverPanel.Size = new System.Drawing.Size(32, 32);
            this.MouseOverPanel.TabIndex = 2;
            // 
            // SaveFile
            // 
            this.SaveFile.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
            this.SaveFile.FlatAppearance.BorderSize = 0;
            this.SaveFile.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.SaveFile.Image = global::CUETools.CTDB.EACPlugin.Properties.Resources.document_save_12;
            this.SaveFile.Location = new System.Drawing.Point(4+ImageSize-16, 4+ImageSize+4);
            this.SaveFile.Name = "SaveFile";
            this.SaveFile.Size = new System.Drawing.Size(16, 16);
            this.SaveFile.TabIndex = 0;
            this.SaveFile.UseVisualStyleBackColor = true;
            // 
            // ImagePreview
            // 
            this.BackColor = System.Drawing.Color.Transparent;
            this.Controls.Add(this.ImagePanel);
            this.Controls.Add(this.Description);
            this.Controls.Add(this.Description2);
            this.Controls.Add(this.MouseOverPanel);
            this.Controls.Add(this.SaveFile);
            this.Size = new System.Drawing.Size(4+ImageSize+4, 4+ImageSize+4+16+4);
            this.ResumeLayout(false);

        }

        private System.Windows.Forms.Label Description;
        private System.Windows.Forms.Label Description2;
        private System.Windows.Forms.Panel ImagePanel;
        private System.Windows.Forms.Panel MouseOverPanel;
        private System.Windows.Forms.Button SaveFile;
        #endregion
    }
}
