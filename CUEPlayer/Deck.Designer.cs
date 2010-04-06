namespace CUEPlayer
{
	partial class Deck
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
			this.textBoxArtist = new System.Windows.Forms.TextBox();
			this.textBoxAlbum = new System.Windows.Forms.TextBox();
			this.textBoxTitle = new System.Windows.Forms.TextBox();
			this.textBoxDuration = new System.Windows.Forms.TextBox();
			this.pictureBox = new System.Windows.Forms.PictureBox();
			this.buttonPlay = new System.Windows.Forms.Button();
			this.buttonStop = new System.Windows.Forms.Button();
			this.mediaSlider = new MediaSlider.MediaSlider();
			this.buttonPause = new System.Windows.Forms.Button();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			this.mediaSliderVolume = new MediaSlider.MediaSlider();
			this.buttonRewind = new System.Windows.Forms.Button();
			this.buttonNext = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
			this.SuspendLayout();
			// 
			// textBoxArtist
			// 
			this.textBoxArtist.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxArtist.Location = new System.Drawing.Point(130, 12);
			this.textBoxArtist.Name = "textBoxArtist";
			this.textBoxArtist.ReadOnly = true;
			this.textBoxArtist.Size = new System.Drawing.Size(203, 13);
			this.textBoxArtist.TabIndex = 10;
			// 
			// textBoxAlbum
			// 
			this.textBoxAlbum.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxAlbum.Location = new System.Drawing.Point(130, 31);
			this.textBoxAlbum.Name = "textBoxAlbum";
			this.textBoxAlbum.ReadOnly = true;
			this.textBoxAlbum.Size = new System.Drawing.Size(203, 13);
			this.textBoxAlbum.TabIndex = 11;
			// 
			// textBoxTitle
			// 
			this.textBoxTitle.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxTitle.Location = new System.Drawing.Point(130, 50);
			this.textBoxTitle.Name = "textBoxTitle";
			this.textBoxTitle.ReadOnly = true;
			this.textBoxTitle.Size = new System.Drawing.Size(203, 13);
			this.textBoxTitle.TabIndex = 12;
			// 
			// textBoxDuration
			// 
			this.textBoxDuration.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.textBoxDuration.Location = new System.Drawing.Point(130, 69);
			this.textBoxDuration.Name = "textBoxDuration";
			this.textBoxDuration.ReadOnly = true;
			this.textBoxDuration.Size = new System.Drawing.Size(123, 13);
			this.textBoxDuration.TabIndex = 13;
			// 
			// pictureBox
			// 
			this.pictureBox.ErrorImage = global::CUEPlayer.Properties.Resources.ctdb;
			this.pictureBox.Image = global::CUEPlayer.Properties.Resources.ctdb;
			this.pictureBox.ImeMode = System.Windows.Forms.ImeMode.NoControl;
			this.pictureBox.InitialImage = global::CUEPlayer.Properties.Resources.ctdb;
			this.pictureBox.Location = new System.Drawing.Point(12, 12);
			this.pictureBox.Name = "pictureBox";
			this.pictureBox.Size = new System.Drawing.Size(100, 100);
			this.pictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
			this.pictureBox.TabIndex = 16;
			this.pictureBox.TabStop = false;
			// 
			// buttonPlay
			// 
			this.buttonPlay.BackColor = System.Drawing.Color.Transparent;
			this.buttonPlay.BackgroundImage = global::CUEPlayer.Properties.Resources.control_play_blue;
			this.buttonPlay.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Center;
			this.buttonPlay.FlatAppearance.BorderSize = 0;
			this.buttonPlay.Location = new System.Drawing.Point(206, 87);
			this.buttonPlay.Name = "buttonPlay";
			this.buttonPlay.Size = new System.Drawing.Size(25, 25);
			this.buttonPlay.TabIndex = 4;
			this.buttonPlay.UseVisualStyleBackColor = false;
			this.buttonPlay.Click += new System.EventHandler(this.buttonPlay_Click);
			// 
			// buttonStop
			// 
			this.buttonStop.Image = global::CUEPlayer.Properties.Resources.control_stop_blue;
			this.buttonStop.Location = new System.Drawing.Point(231, 87);
			this.buttonStop.Name = "buttonStop";
			this.buttonStop.Size = new System.Drawing.Size(25, 25);
			this.buttonStop.TabIndex = 9;
			this.buttonStop.UseVisualStyleBackColor = true;
			this.buttonStop.Click += new System.EventHandler(this.buttonStop_Click);
			// 
			// mediaSlider
			// 
			this.mediaSlider.Animated = false;
			this.mediaSlider.AnimationSize = 0.2F;
			this.mediaSlider.AnimationSpeed = MediaSlider.MediaSlider.AnimateSpeed.Normal;
			this.mediaSlider.AutoScrollMargin = new System.Drawing.Size(0, 0);
			this.mediaSlider.AutoScrollMinSize = new System.Drawing.Size(0, 0);
			this.mediaSlider.BackColor = System.Drawing.SystemColors.Control;
			this.mediaSlider.BackgroundImage = null;
			this.mediaSlider.ButtonAccentColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))), ((int)(((byte)(64)))));
			this.mediaSlider.ButtonBorderColor = System.Drawing.Color.Black;
			this.mediaSlider.ButtonColor = System.Drawing.Color.FromArgb(((int)(((byte)(160)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
			this.mediaSlider.ButtonCornerRadius = ((uint)(4u));
			this.mediaSlider.ButtonSize = new System.Drawing.Size(10, 20);
			this.mediaSlider.ButtonStyle = MediaSlider.MediaSlider.ButtonType.GlassOverlap;
			this.mediaSlider.ContextMenuStrip = null;
			this.mediaSlider.LargeChange = 2;
			this.mediaSlider.Location = new System.Drawing.Point(0, 115);
			this.mediaSlider.Margin = new System.Windows.Forms.Padding(0);
			this.mediaSlider.Maximum = 1;
			this.mediaSlider.Minimum = 0;
			this.mediaSlider.Name = "mediaSlider";
			this.mediaSlider.Orientation = System.Windows.Forms.Orientation.Horizontal;
			this.mediaSlider.ShowButtonOnHover = true;
			this.mediaSlider.Size = new System.Drawing.Size(362, 25);
			this.mediaSlider.SliderFlyOut = MediaSlider.MediaSlider.FlyOutStyle.None;
			this.mediaSlider.SmallChange = 1;
			this.mediaSlider.SmoothScrolling = true;
			this.mediaSlider.TabIndex = 9;
			this.mediaSlider.TickColor = System.Drawing.Color.DarkGray;
			this.mediaSlider.TickStyle = System.Windows.Forms.TickStyle.None;
			this.mediaSlider.TickType = MediaSlider.MediaSlider.TickMode.Standard;
			this.mediaSlider.TrackBorderColor = System.Drawing.SystemColors.ButtonShadow;
			this.mediaSlider.TrackDepth = 9;
			this.mediaSlider.TrackFillColor = System.Drawing.SystemColors.ButtonFace;
			this.mediaSlider.TrackProgressColor = System.Drawing.Color.FromArgb(((int)(((byte)(5)))), ((int)(((byte)(101)))), ((int)(((byte)(188)))));
			this.mediaSlider.TrackShadow = true;
			this.mediaSlider.TrackShadowColor = System.Drawing.SystemColors.ButtonShadow;
			this.mediaSlider.TrackStyle = MediaSlider.MediaSlider.TrackType.Progress;
			this.mediaSlider.Value = 0;
			this.mediaSlider.ValueChanged += new System.EventHandler(this.mediaSliderA_ValueChanged);
			this.mediaSlider.Scrolled += new System.EventHandler(this.mediaSlider_Scrolled);
			// 
			// buttonPause
			// 
			this.buttonPause.Image = global::CUEPlayer.Properties.Resources.control_pause_blue;
			this.buttonPause.Location = new System.Drawing.Point(256, 87);
			this.buttonPause.Name = "buttonPause";
			this.buttonPause.Size = new System.Drawing.Size(25, 25);
			this.buttonPause.TabIndex = 17;
			this.buttonPause.UseVisualStyleBackColor = true;
			this.buttonPause.Click += new System.EventHandler(this.buttonPause_Click);
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
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
			this.mediaSliderVolume.Location = new System.Drawing.Point(336, 12);
			this.mediaSliderVolume.Margin = new System.Windows.Forms.Padding(0);
			this.mediaSliderVolume.Maximum = 100;
			this.mediaSliderVolume.Minimum = 0;
			this.mediaSliderVolume.Name = "mediaSliderVolume";
			this.mediaSliderVolume.Orientation = System.Windows.Forms.Orientation.Vertical;
			this.mediaSliderVolume.ShowButtonOnHover = false;
			this.mediaSliderVolume.Size = new System.Drawing.Size(26, 100);
			this.mediaSliderVolume.SliderFlyOut = MediaSlider.MediaSlider.FlyOutStyle.None;
			this.mediaSliderVolume.SmallChange = 1;
			this.mediaSliderVolume.SmoothScrolling = true;
			this.mediaSliderVolume.TabIndex = 18;
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
			this.mediaSliderVolume.Value = 100;
			this.mediaSliderVolume.Scrolled += new System.EventHandler(this.mediaSliderVolume_Scrolled);
			// 
			// buttonRewind
			// 
			this.buttonRewind.Image = global::CUEPlayer.Properties.Resources.control_rewind_blue;
			this.buttonRewind.Location = new System.Drawing.Point(281, 87);
			this.buttonRewind.Name = "buttonRewind";
			this.buttonRewind.Size = new System.Drawing.Size(25, 25);
			this.buttonRewind.TabIndex = 19;
			this.buttonRewind.UseVisualStyleBackColor = true;
			this.buttonRewind.Click += new System.EventHandler(this.buttonRewind_Click);
			// 
			// buttonNext
			// 
			this.buttonNext.Image = global::CUEPlayer.Properties.Resources.control_end_blue;
			this.buttonNext.Location = new System.Drawing.Point(306, 87);
			this.buttonNext.Name = "buttonNext";
			this.buttonNext.Size = new System.Drawing.Size(25, 25);
			this.buttonNext.TabIndex = 20;
			this.buttonNext.UseVisualStyleBackColor = true;
			this.buttonNext.Click += new System.EventHandler(this.buttonNext_Click);
			// 
			// Deck
			// 
			this.AllowDrop = true;
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(361, 145);
			this.ControlBox = false;
			this.Controls.Add(this.buttonNext);
			this.Controls.Add(this.buttonRewind);
			this.Controls.Add(this.mediaSliderVolume);
			this.Controls.Add(this.buttonPause);
			this.Controls.Add(this.mediaSlider);
			this.Controls.Add(this.pictureBox);
			this.Controls.Add(this.textBoxDuration);
			this.Controls.Add(this.buttonPlay);
			this.Controls.Add(this.textBoxTitle);
			this.Controls.Add(this.buttonStop);
			this.Controls.Add(this.textBoxAlbum);
			this.Controls.Add(this.textBoxArtist);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "Deck";
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Text = "Deck";
			this.DragDrop += new System.Windows.Forms.DragEventHandler(this.Deck_DragDrop);
			this.DragOver += new System.Windows.Forms.DragEventHandler(this.Deck_DragOver);
			((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Button buttonPlay;
		private System.Windows.Forms.Button buttonStop;
		private System.Windows.Forms.TextBox textBoxArtist;
		private System.Windows.Forms.TextBox textBoxAlbum;
		private System.Windows.Forms.TextBox textBoxTitle;
		private System.Windows.Forms.TextBox textBoxDuration;
		private System.Windows.Forms.PictureBox pictureBox;
		private MediaSlider.MediaSlider mediaSlider;
		private System.Windows.Forms.Button buttonPause;
		private System.Windows.Forms.Timer timer1;
		private MediaSlider.MediaSlider mediaSliderVolume;
		private System.Windows.Forms.Button buttonRewind;
		private System.Windows.Forms.Button buttonNext;

	}
}