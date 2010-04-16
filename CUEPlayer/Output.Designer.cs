namespace CUEPlayer
{
	partial class Output
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
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Output));
			this.mediaSliderVolume = new MediaSlider.MediaSlider();
			this.peakMeterCtrl1 = new Ernzo.WinForms.Controls.PeakMeterCtrl();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			this.checkBoxMute = new System.Windows.Forms.CheckBox();
			this.imageList1 = new System.Windows.Forms.ImageList(this.components);
			this.outputBindingSource = new System.Windows.Forms.BindingSource(this.components);
			((System.ComponentModel.ISupportInitialize)(this.outputBindingSource)).BeginInit();
			this.SuspendLayout();
			// 
			// mediaSliderVolume
			// 
			this.mediaSliderVolume.Animated = false;
			this.mediaSliderVolume.AnimationSize = 0.2F;
			this.mediaSliderVolume.AnimationSpeed = MediaSlider.MediaSlider.AnimateSpeed.Normal;
			this.mediaSliderVolume.AutoScrollMargin = new System.Drawing.Size(0, 0);
			this.mediaSliderVolume.AutoScrollMinSize = new System.Drawing.Size(0, 0);
			this.mediaSliderVolume.BackColor = System.Drawing.SystemColors.Control;
			this.mediaSliderVolume.BackgroundImage = null;
			this.mediaSliderVolume.ButtonAccentColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
			this.mediaSliderVolume.ButtonBorderColor = System.Drawing.Color.Black;
			this.mediaSliderVolume.ButtonColor = System.Drawing.Color.FromArgb(((int)(((byte)(160)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
			this.mediaSliderVolume.ButtonCornerRadius = ((uint)(4u));
			this.mediaSliderVolume.ButtonSize = new System.Drawing.Size(14, 14);
			this.mediaSliderVolume.ButtonStyle = MediaSlider.MediaSlider.ButtonType.GlassOverlap;
			this.mediaSliderVolume.ContextMenuStrip = null;
			this.mediaSliderVolume.LargeChange = 5;
			this.mediaSliderVolume.Location = new System.Drawing.Point(9, 9);
			this.mediaSliderVolume.Margin = new System.Windows.Forms.Padding(0);
			this.mediaSliderVolume.Maximum = 100;
			this.mediaSliderVolume.Minimum = 0;
			this.mediaSliderVolume.Name = "mediaSliderVolume";
			this.mediaSliderVolume.Orientation = System.Windows.Forms.Orientation.Vertical;
			this.mediaSliderVolume.ShowButtonOnHover = false;
			this.mediaSliderVolume.Size = new System.Drawing.Size(37, 101);
			this.mediaSliderVolume.SliderFlyOut = MediaSlider.MediaSlider.FlyOutStyle.None;
			this.mediaSliderVolume.SmallChange = 1;
			this.mediaSliderVolume.SmoothScrolling = true;
			this.mediaSliderVolume.TabIndex = 14;
			this.mediaSliderVolume.TickColor = System.Drawing.Color.DarkGray;
			this.mediaSliderVolume.TickStyle = System.Windows.Forms.TickStyle.None;
			this.mediaSliderVolume.TickType = MediaSlider.MediaSlider.TickMode.Standard;
			this.mediaSliderVolume.TrackBorderColor = System.Drawing.SystemColors.ActiveBorder;
			this.mediaSliderVolume.TrackDepth = 6;
			this.mediaSliderVolume.TrackFillColor = System.Drawing.SystemColors.ActiveBorder;
			this.mediaSliderVolume.TrackProgressColor = System.Drawing.Color.Green;
			this.mediaSliderVolume.TrackShadow = true;
			this.mediaSliderVolume.TrackShadowColor = System.Drawing.SystemColors.ScrollBar;
			this.mediaSliderVolume.TrackStyle = MediaSlider.MediaSlider.TrackType.Progress;
			this.mediaSliderVolume.Value = 0;
			this.mediaSliderVolume.Scrolled += new System.EventHandler(this.mediaSliderVolume_Scrolled);
			// 
			// peakMeterCtrl1
			// 
			this.peakMeterCtrl1.BandsCount = 2;
			this.peakMeterCtrl1.ColorHigh = System.Drawing.Color.Red;
			this.peakMeterCtrl1.ColorHighBack = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(150)))), ((int)(((byte)(150)))));
			this.peakMeterCtrl1.ColorMedium = System.Drawing.Color.Yellow;
			this.peakMeterCtrl1.ColorMediumBack = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(150)))));
			this.peakMeterCtrl1.ColorNormal = System.Drawing.Color.Green;
			this.peakMeterCtrl1.ColorNormalBack = System.Drawing.Color.FromArgb(((int)(((byte)(150)))), ((int)(((byte)(255)))), ((int)(((byte)(150)))));
			this.peakMeterCtrl1.FalloffColor = System.Drawing.Color.Blue;
			this.peakMeterCtrl1.GridColor = System.Drawing.Color.Gainsboro;
			this.peakMeterCtrl1.LEDCount = 25;
			this.peakMeterCtrl1.Location = new System.Drawing.Point(46, 9);
			this.peakMeterCtrl1.Margin = new System.Windows.Forms.Padding(0);
			this.peakMeterCtrl1.Name = "peakMeterCtrl1";
			this.peakMeterCtrl1.Size = new System.Drawing.Size(15, 101);
			this.peakMeterCtrl1.TabIndex = 13;
			this.peakMeterCtrl1.Text = "peakMeterCtrl1";
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Interval = 50;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// checkBoxMute
			// 
			this.checkBoxMute.Appearance = System.Windows.Forms.Appearance.Button;
			this.checkBoxMute.Checked = true;
			this.checkBoxMute.CheckState = System.Windows.Forms.CheckState.Checked;
			this.checkBoxMute.DataBindings.Add(new System.Windows.Forms.Binding("Checked", this.outputBindingSource, "Muted", true, System.Windows.Forms.DataSourceUpdateMode.OnPropertyChanged));
			this.checkBoxMute.DataBindings.Add(new System.Windows.Forms.Binding("ImageIndex", this.outputBindingSource, "VolumeIcon", true, System.Windows.Forms.DataSourceUpdateMode.OnPropertyChanged));
			this.checkBoxMute.FlatAppearance.BorderSize = 0;
			this.checkBoxMute.FlatAppearance.CheckedBackColor = System.Drawing.Color.Transparent;
			this.checkBoxMute.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
			this.checkBoxMute.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
			this.checkBoxMute.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
			this.checkBoxMute.ImageIndex = 0;
			this.checkBoxMute.ImageList = this.imageList1;
			this.checkBoxMute.Location = new System.Drawing.Point(16, 110);
			this.checkBoxMute.Name = "checkBoxMute";
			this.checkBoxMute.Size = new System.Drawing.Size(22, 22);
			this.checkBoxMute.TabIndex = 15;
			this.checkBoxMute.UseVisualStyleBackColor = true;
			// 
			// imageList1
			// 
			this.imageList1.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList1.ImageStream")));
			this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
			this.imageList1.Images.SetKeyName(0, "sound_mute.png");
			this.imageList1.Images.SetKeyName(1, "sound_none.png");
			this.imageList1.Images.SetKeyName(2, "sound_low.png");
			this.imageList1.Images.SetKeyName(3, "sound.png");
			// 
			// outputBindingSource
			// 
			this.outputBindingSource.DataSource = typeof(CUEPlayer.Output);
			// 
			// Output
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(74, 136);
			this.ControlBox = false;
			this.Controls.Add(this.checkBoxMute);
			this.Controls.Add(this.mediaSliderVolume);
			this.Controls.Add(this.peakMeterCtrl1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Output";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Text = "Output";
			this.Load += new System.EventHandler(this.Output_Load);
			((System.ComponentModel.ISupportInitialize)(this.outputBindingSource)).EndInit();
			this.ResumeLayout(false);

		}

		#endregion

		private MediaSlider.MediaSlider mediaSliderVolume;
		private Ernzo.WinForms.Controls.PeakMeterCtrl peakMeterCtrl1;
		private System.Windows.Forms.Timer timer1;
		private System.Windows.Forms.CheckBox checkBoxMute;
		private System.Windows.Forms.ImageList imageList1;
		private System.Windows.Forms.BindingSource outputBindingSource;
	}
}