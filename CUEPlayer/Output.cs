using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using NAudio.CoreAudioApi;
using CUETools.Codecs;
using CUETools.Codecs.CoreAudio;
using CUETools.Processor;

namespace CUEPlayer
{
	public partial class Output : Form
	{
		private MMDevice _device;

		internal MMDevice Device
		{
			get
			{
				return _device;
			}

			set
			{
				_device = value;
			}
		}

		public Output()
		{
			InitializeComponent();
		}

		public void Init(frmCUEPlayer parent)
		{
			MdiParent = parent;
			_device = WasapiOut.GetDefaultAudioEndpoint();
			_device.AudioEndpointVolume.OnVolumeNotification += new AudioEndpointVolumeNotificationDelegate(AudioEndpointVolume_OnVolumeNotification);
			mediaSliderVolume.Value = (int)(_device.AudioEndpointVolume.MasterVolumeLevelScalar * 100);
			Show();
		}

		void AudioEndpointVolume_OnVolumeNotification(AudioVolumeNotificationData data)
		{
			if (data.EventContext == Guid.Empty)
				return;
			if (this.InvokeRequired)
				this.Invoke((MethodInvoker)delegate() { AudioEndpointVolume_OnVolumeNotification(data); });
			else
				mediaSliderVolume.Value = (int)(data.MasterVolume * 100);
		}

		private void mediaSliderVolume_Scrolled(object sender, EventArgs e)
		{
			try
			{
				_device.AudioEndpointVolume.MasterVolumeLevelScalar = mediaSliderVolume.Value / 100.0f;
			}
			catch (Exception ex)
			{
				Trace.WriteLine(ex.Message);
			}
		}

		private int[] peakValues = null;

		private void timer1_Tick(object sender, EventArgs e)
		{
			if (_device == null)
				return;
			if (peakValues == null || peakValues.Length != _device.AudioMeterInformation.PeakValues.Count)
			{
				peakValues = new int[_device.AudioMeterInformation.PeakValues.Count];
				peakMeterCtrl1.SetMeterBands(peakValues.Length, 25);
			}
			for (int i = 0; i < peakValues.Length; i++)
				peakValues[i] = (int)(_device.AudioMeterInformation.PeakValues[i] * 100);
			//peakValues[0] = (int)(_device.AudioMeterInformation.MasterPeakValue * 100);
			peakMeterCtrl1.SetData(peakValues, 0, peakValues.Length);
		}

		private void buttonPlay_Click(object sender, EventArgs e)
		{
			(MdiParent as frmCUEPlayer).buttonPlay_Click(sender, e);
		}

		private void buttonStop_Click(object sender, EventArgs e)
		{
			(MdiParent as frmCUEPlayer).buttonStop_Click(sender, e);
		}

		private void buttonPause_Click(object sender, EventArgs e)
		{
			(MdiParent as frmCUEPlayer).buttonPause_Click(sender, e);
		}

		private void Output_Load(object sender, EventArgs e)
		{
			peakMeterCtrl1.Start(50);
		}
	}
}
