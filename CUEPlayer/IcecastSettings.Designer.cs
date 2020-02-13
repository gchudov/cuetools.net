namespace CUEPlayer
{
	partial class IcecastSettings
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(IcecastSettings));
			this.buttonCancel = new System.Windows.Forms.Button();
			this.buttonOk = new System.Windows.Forms.Button();
			this.textBoxServer = new System.Windows.Forms.TextBox();
			this.textBoxPort = new System.Windows.Forms.TextBox();
			this.textBoxPassword = new System.Windows.Forms.TextBox();
			this.textBoxMount = new System.Windows.Forms.TextBox();
			this.textBoxName = new System.Windows.Forms.TextBox();
			this.textBoxDesctiption = new System.Windows.Forms.TextBox();
			this.textBoxWeb = new System.Windows.Forms.TextBox();
			this.comboBoxGenre = new System.Windows.Forms.ComboBox();
			this.labelServer = new System.Windows.Forms.Label();
			this.labelPort = new System.Windows.Forms.Label();
			this.labelPassword = new System.Windows.Forms.Label();
			this.labelMount = new System.Windows.Forms.Label();
			this.labelStationName = new System.Windows.Forms.Label();
			this.labelStationDescription = new System.Windows.Forms.Label();
			this.labelWeb = new System.Windows.Forms.Label();
			this.labelGenre = new System.Windows.Forms.Label();
			this.textBoxMP3Options = new System.Windows.Forms.TextBox();
			this.labelMP3Options = new System.Windows.Forms.Label();
			this.icecastSettingsDataBindingSource = new System.Windows.Forms.BindingSource(this.components);
			((System.ComponentModel.ISupportInitialize)(this.icecastSettingsDataBindingSource)).BeginInit();
			this.SuspendLayout();
			// 
			// buttonCancel
			// 
			this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.buttonCancel.Location = new System.Drawing.Point(197, 272);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(75, 23);
			this.buttonCancel.TabIndex = 0;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			// 
			// buttonOk
			// 
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.buttonOk.Location = new System.Drawing.Point(116, 272);
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.Size = new System.Drawing.Size(75, 23);
			this.buttonOk.TabIndex = 1;
			this.buttonOk.Text = "OK";
			this.buttonOk.UseVisualStyleBackColor = true;
			// 
			// textBoxServer
			// 
			this.textBoxServer.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Server", true));
			this.textBoxServer.Location = new System.Drawing.Point(91, 12);
			this.textBoxServer.Name = "textBoxServer";
			this.textBoxServer.Size = new System.Drawing.Size(181, 20);
			this.textBoxServer.TabIndex = 2;
			// 
			// textBoxPort
			// 
			this.textBoxPort.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Port", true));
			this.textBoxPort.Location = new System.Drawing.Point(91, 38);
			this.textBoxPort.Name = "textBoxPort";
			this.textBoxPort.Size = new System.Drawing.Size(181, 20);
			this.textBoxPort.TabIndex = 3;
			// 
			// textBoxPassword
			// 
			this.textBoxPassword.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Password", true));
			this.textBoxPassword.Location = new System.Drawing.Point(91, 64);
			this.textBoxPassword.Name = "textBoxPassword";
			this.textBoxPassword.Size = new System.Drawing.Size(181, 20);
			this.textBoxPassword.TabIndex = 4;
			this.textBoxPassword.UseSystemPasswordChar = true;
			// 
			// textBoxMount
			// 
			this.textBoxMount.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Mount", true));
			this.textBoxMount.Location = new System.Drawing.Point(91, 90);
			this.textBoxMount.Name = "textBoxMount";
			this.textBoxMount.Size = new System.Drawing.Size(181, 20);
			this.textBoxMount.TabIndex = 5;
			// 
			// textBoxName
			// 
			this.textBoxName.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Name", true));
			this.textBoxName.Location = new System.Drawing.Point(91, 116);
			this.textBoxName.Name = "textBoxName";
			this.textBoxName.Size = new System.Drawing.Size(181, 20);
			this.textBoxName.TabIndex = 6;
			// 
			// textBoxDesctiption
			// 
			this.textBoxDesctiption.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Description", true));
			this.textBoxDesctiption.Location = new System.Drawing.Point(91, 142);
			this.textBoxDesctiption.Name = "textBoxDesctiption";
			this.textBoxDesctiption.Size = new System.Drawing.Size(181, 20);
			this.textBoxDesctiption.TabIndex = 7;
			// 
			// textBoxWeb
			// 
			this.textBoxWeb.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Url", true));
			this.textBoxWeb.Location = new System.Drawing.Point(91, 168);
			this.textBoxWeb.Name = "textBoxWeb";
			this.textBoxWeb.Size = new System.Drawing.Size(181, 20);
			this.textBoxWeb.TabIndex = 8;
			// 
			// comboBoxGenre
			// 
			this.comboBoxGenre.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "Genre", true));
			this.comboBoxGenre.FormattingEnabled = true;
			this.comboBoxGenre.Location = new System.Drawing.Point(91, 194);
			this.comboBoxGenre.Name = "comboBoxGenre";
			this.comboBoxGenre.Size = new System.Drawing.Size(181, 21);
			this.comboBoxGenre.TabIndex = 9;
			// 
			// labelServer
			// 
			this.labelServer.AutoSize = true;
			this.labelServer.Location = new System.Drawing.Point(12, 15);
			this.labelServer.Name = "labelServer";
			this.labelServer.Size = new System.Drawing.Size(38, 13);
			this.labelServer.TabIndex = 10;
			this.labelServer.Text = "Server";
			// 
			// labelPort
			// 
			this.labelPort.AutoSize = true;
			this.labelPort.Location = new System.Drawing.Point(12, 41);
			this.labelPort.Name = "labelPort";
			this.labelPort.Size = new System.Drawing.Size(26, 13);
			this.labelPort.TabIndex = 11;
			this.labelPort.Text = "Port";
			// 
			// labelPassword
			// 
			this.labelPassword.AutoSize = true;
			this.labelPassword.Location = new System.Drawing.Point(12, 67);
			this.labelPassword.Name = "labelPassword";
			this.labelPassword.Size = new System.Drawing.Size(53, 13);
			this.labelPassword.TabIndex = 12;
			this.labelPassword.Text = "Password";
			// 
			// labelMount
			// 
			this.labelMount.AutoSize = true;
			this.labelMount.Location = new System.Drawing.Point(12, 93);
			this.labelMount.Name = "labelMount";
			this.labelMount.Size = new System.Drawing.Size(37, 13);
			this.labelMount.TabIndex = 13;
			this.labelMount.Text = "Mount";
			// 
			// labelStationName
			// 
			this.labelStationName.AutoSize = true;
			this.labelStationName.Location = new System.Drawing.Point(12, 119);
			this.labelStationName.Name = "labelStationName";
			this.labelStationName.Size = new System.Drawing.Size(35, 13);
			this.labelStationName.TabIndex = 14;
			this.labelStationName.Text = "Name";
			// 
			// labelStationDescription
			// 
			this.labelStationDescription.AutoSize = true;
			this.labelStationDescription.Location = new System.Drawing.Point(12, 145);
			this.labelStationDescription.Name = "labelStationDescription";
			this.labelStationDescription.Size = new System.Drawing.Size(60, 13);
			this.labelStationDescription.TabIndex = 15;
			this.labelStationDescription.Text = "Description";
			// 
			// labelWeb
			// 
			this.labelWeb.AutoSize = true;
			this.labelWeb.Location = new System.Drawing.Point(12, 171);
			this.labelWeb.Name = "labelWeb";
			this.labelWeb.Size = new System.Drawing.Size(30, 13);
			this.labelWeb.TabIndex = 16;
			this.labelWeb.Text = "Web";
			// 
			// labelGenre
			// 
			this.labelGenre.AutoSize = true;
			this.labelGenre.Location = new System.Drawing.Point(12, 197);
			this.labelGenre.Name = "labelGenre";
			this.labelGenre.Size = new System.Drawing.Size(36, 13);
			this.labelGenre.TabIndex = 17;
			this.labelGenre.Text = "Genre";
			// 
			// textBoxMP3Options
			// 
			this.textBoxMP3Options.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.icecastSettingsDataBindingSource, "MP3Options", true));
			this.textBoxMP3Options.Location = new System.Drawing.Point(91, 221);
			this.textBoxMP3Options.Name = "textBoxMP3Options";
			this.textBoxMP3Options.Size = new System.Drawing.Size(181, 20);
			this.textBoxMP3Options.TabIndex = 18;
			// 
			// labelMP3Options
			// 
			this.labelMP3Options.AutoSize = true;
			this.labelMP3Options.Location = new System.Drawing.Point(12, 224);
			this.labelMP3Options.Name = "labelMP3Options";
			this.labelMP3Options.Size = new System.Drawing.Size(68, 13);
			this.labelMP3Options.TabIndex = 19;
			this.labelMP3Options.Text = "MP3 Options";
			// 
			// icecastSettingsDataBindingSource
			// 
			this.icecastSettingsDataBindingSource.DataSource = typeof(CUETools.Codecs.Icecast.IcecastSettingsData);
			// 
			// IcecastSettings
			// 
			this.AcceptButton = this.buttonOk;
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = this.buttonCancel;
			this.ClientSize = new System.Drawing.Size(284, 307);
			this.Controls.Add(this.labelMP3Options);
			this.Controls.Add(this.textBoxMP3Options);
			this.Controls.Add(this.labelGenre);
			this.Controls.Add(this.labelWeb);
			this.Controls.Add(this.labelStationDescription);
			this.Controls.Add(this.labelStationName);
			this.Controls.Add(this.labelMount);
			this.Controls.Add(this.labelPassword);
			this.Controls.Add(this.labelPort);
			this.Controls.Add(this.labelServer);
			this.Controls.Add(this.comboBoxGenre);
			this.Controls.Add(this.textBoxWeb);
			this.Controls.Add(this.textBoxDesctiption);
			this.Controls.Add(this.textBoxName);
			this.Controls.Add(this.textBoxMount);
			this.Controls.Add(this.textBoxPassword);
			this.Controls.Add(this.textBoxPort);
			this.Controls.Add(this.textBoxServer);
			this.Controls.Add(this.buttonOk);
			this.Controls.Add(this.buttonCancel);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "IcecastSettings";
			this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
			this.Text = "Icecast Settings";
			this.Load += new System.EventHandler(this.IcecastSettings_Load);
			((System.ComponentModel.ISupportInitialize)(this.icecastSettingsDataBindingSource)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button buttonCancel;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.TextBox textBoxServer;
		private System.Windows.Forms.TextBox textBoxPort;
		private System.Windows.Forms.TextBox textBoxPassword;
		private System.Windows.Forms.TextBox textBoxMount;
		private System.Windows.Forms.TextBox textBoxName;
		private System.Windows.Forms.TextBox textBoxDesctiption;
		private System.Windows.Forms.TextBox textBoxWeb;
		private System.Windows.Forms.ComboBox comboBoxGenre;
		private System.Windows.Forms.Label labelServer;
		private System.Windows.Forms.Label labelPort;
		private System.Windows.Forms.Label labelPassword;
		private System.Windows.Forms.Label labelMount;
		private System.Windows.Forms.Label labelStationName;
		private System.Windows.Forms.Label labelStationDescription;
		private System.Windows.Forms.Label labelWeb;
		private System.Windows.Forms.Label labelGenre;
		private System.Windows.Forms.BindingSource icecastSettingsDataBindingSource;
		private System.Windows.Forms.TextBox textBoxMP3Options;
		private System.Windows.Forms.Label labelMP3Options;
	}
}