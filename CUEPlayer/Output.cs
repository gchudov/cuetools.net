using System;
using System.Diagnostics;
using System.Windows.Forms;
using CUETools.Codecs;
using CUETools.Codecs.CoreAudio;
using CUETools.DSP.Mixer;
using CUETools.DSP.Resampler;
using NAudio.CoreAudioApi;

namespace CUEPlayer
{
	public partial class Output : Form
	{
		private IWavePlayer _player;
		private AudioBuffer resampled;
		private SOXResampler _resampler;
		private MMDevice _device;
		private DateTime _pauseTill;
		private bool _muted;

		public int VolumeIcon
		{
			get
			{
				return _muted ? 0 : mediaSliderVolume.Value < 10 ? 1 : mediaSliderVolume.Value < 50 ? 2 : 3;
			}
		}

		public bool Muted
		{
			get
			{
				return _muted;
			}
			set
			{
				_muted = value;
				_pauseTill = DateTime.Now;
			}
		}

		public Output()
		{
			InitializeComponent();
		}

		private void Output_Load(object sender, EventArgs e)
		{
			peakMeterCtrl1.Start(50);
			outputBindingSource.DataSource = this;
		}

		public void Init(frmCUEPlayer parent)
		{
			MdiParent = parent;
			_device = WasapiOut.GetDefaultAudioEndpoint();
			_device.AudioEndpointVolume.OnVolumeNotification += new AudioEndpointVolumeNotificationDelegate(AudioEndpointVolume_OnVolumeNotification);
			mediaSliderVolume.Value = (int)(_device.AudioEndpointVolume.MasterVolumeLevelScalar * 100);
			//mediaSliderVolume.Maximum = (int)(_device.AudioEndpointVolume.VolumeRange);
			Show();

			int delay = 100;
			try
			{
				_player = new WasapiOut(_device, NAudio.CoreAudioApi.AudioClientShareMode.Shared, true, delay, new AudioPCMConfig(32, 2, 44100));
			}
			catch
			{
				_player = null;
			}
			if (_player == null)
			{
				try
				{
					_player = new WasapiOut(_device, NAudio.CoreAudioApi.AudioClientShareMode.Shared, true, delay, new AudioPCMConfig(32, 2, 48000));
					SOXResamplerConfig cfg;
					cfg.Quality = SOXResamplerQuality.Very;
					cfg.Phase = 50;
					cfg.AllowAliasing = false;
					cfg.Bandwidth = 0;
					_resampler = new SOXResampler(parent.Mixer.PCM, _player.PCM, cfg);
					resampled = new AudioBuffer(_player.PCM, parent.Mixer.BufferSize * 2 * parent.Mixer.PCM.SampleRate / _player.PCM.SampleRate);
				}
				catch (Exception ex)
				{
					_player = null;
					Trace.WriteLine(ex.Message);
				}
			}
			parent.Mixer.AudioRead += new EventHandler<AudioReadEventArgs>(Mixer_AudioRead);
			if (_player != null)
				_player.Play();
		}

		void Mixer_AudioRead(object sender, AudioReadEventArgs e)
		{
			if (_muted || _player == null)
			{
				double tosleep = (_pauseTill - DateTime.Now).TotalMilliseconds;
				if (tosleep > 0) System.Threading.Thread.Sleep((int)tosleep);
				_pauseTill = DateTime.Now.AddMilliseconds(1000 * e.buffer.Length / e.buffer.PCM.SampleRate);
				return;
			}
			if (_resampler == null)
				_player.Write(e.buffer);
			else
			{
				//Trace.WriteLine(string.Format("Flow: {0}", result.Length));
				_resampler.Flow(e.buffer, resampled);
				//Trace.WriteLine(string.Format("Play: {0}", resampled.Length));
				if (resampled.Length != 0)
					_player.Write(resampled);
			}
		}

		void AudioEndpointVolume_OnVolumeNotification(AudioVolumeNotificationData data)
		{
			if (data.EventContext == Guid.Empty)
				return;
			if (this.InvokeRequired)
				this.Invoke((MethodInvoker)delegate() { AudioEndpointVolume_OnVolumeNotification(data); });
			else
			{
				mediaSliderVolume.Value = (int)(data.MasterVolume * 100);
				outputBindingSource.ResetBindings(false);
			}
		}

		private void mediaSliderVolume_Scrolled(object sender, EventArgs e)
		{
			try
			{
				outputBindingSource.ResetBindings(false);
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
	}
}
