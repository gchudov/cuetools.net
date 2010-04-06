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
			this.mediaSliderVolume = new MediaSlider.MediaSlider();
			this.peakMeterCtrl1 = new Ernzo.WinForms.Controls.PeakMeterCtrl();
			this.buttonPause = new System.Windows.Forms.Button();
			this.buttonPlay = new System.Windows.Forms.Button();
			this.buttonStop = new System.Windows.Forms.Button();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
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
			this.mediaSliderVolume.Size = new System.Drawing.Size(37, 114);
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
			this.peakMeterCtrl1.Location = new System.Drawing.Point(60, 9);
			this.peakMeterCtrl1.Margin = new System.Windows.Forms.Padding(0);
			this.peakMeterCtrl1.Name = "peakMeterCtrl1";
			this.peakMeterCtrl1.Size = new System.Drawing.Size(15, 109);
			this.peakMeterCtrl1.TabIndex = 13;
			this.peakMeterCtrl1.Text = "peakMeterCtrl1";
			// 
			// buttonPause
			// 
			this.buttonPause.Image = global::CUEPlayer.Properties.Resources.control_pause_blue;
			this.buttonPause.Location = new System.Drawing.Point(59, 130);
			this.buttonPause.Name = "buttonPause";
			this.buttonPause.Size = new System.Drawing.Size(25, 25);
			this.buttonPause.TabIndex = 12;
			this.buttonPause.UseVisualStyleBackColor = true;
			this.buttonPause.Click += new System.EventHandler(this.buttonPause_Click);
			// 
			// buttonPlay
			// 
			this.buttonPlay.BackColor = System.Drawing.Color.Transparent;
			this.buttonPlay.BackgroundImage = global::CUEPlayer.Properties.Resources.control_play_blue;
			this.buttonPlay.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
			this.buttonPlay.FlatAppearance.BorderSize = 0;
			this.buttonPlay.Location = new System.Drawing.Point(9, 130);
			this.buttonPlay.Name = "buttonPlay";
			this.buttonPlay.Size = new System.Drawing.Size(25, 25);
			this.buttonPlay.TabIndex = 10;
			this.buttonPlay.UseVisualStyleBackColor = false;
			this.buttonPlay.Click += new System.EventHandler(this.buttonPlay_Click);
			// 
			// buttonStop
			// 
			this.buttonStop.Image = global::CUEPlayer.Properties.Resources.control_stop_blue;
			this.buttonStop.Location = new System.Drawing.Point(34, 130);
			this.buttonStop.Name = "buttonStop";
			this.buttonStop.Size = new System.Drawing.Size(25, 25);
			this.buttonStop.TabIndex = 11;
			this.buttonStop.UseVisualStyleBackColor = true;
			this.buttonStop.Click += new System.EventHandler(this.buttonStop_Click);
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Interval = 50;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// Output
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(96, 168);
			this.ControlBox = false;
			this.Controls.Add(this.mediaSliderVolume);
			this.Controls.Add(this.peakMeterCtrl1);
			this.Controls.Add(this.buttonPause);
			this.Controls.Add(this.buttonPlay);
			this.Controls.Add(this.buttonStop);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Output";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Text = "Output";
			this.Load += new System.EventHandler(this.Output_Load);
			this.ResumeLayout(false);

		}

		#endregion

		private MediaSlider.MediaSlider mediaSliderVolume;
		private Ernzo.WinForms.Controls.PeakMeterCtrl peakMeterCtrl1;
		private System.Windows.Forms.Button buttonPause;
		private System.Windows.Forms.Button buttonPlay;
		private System.Windows.Forms.Button buttonStop;
		private System.Windows.Forms.Timer timer1;
	}
}