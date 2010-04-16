using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using System.Threading;
using System.Net;
using CUETools.Codecs;
using CUETools.Codecs.Icecast;

namespace CUEPlayer
{
	public partial class Icecast : Form
	{
		private IcecastWriter _icecastWriter;
		private IcecastSettingsData _icecastSettings;
		private CUETools.DSP.Mixer.MixingSource _mixer;
		private Thread flushThread;
		private AudioPipe buffer;
		private bool close = false;
		private int latency = 0;

		public Icecast()
		{
			InitializeComponent();
		}

		private void Icecast_Load(object sender, EventArgs e)
		{
			if (Properties.Settings.Default.IcecastSettings == null)
				Properties.Settings.Default.IcecastSettings = new IcecastSettingsData();
			_icecastSettings = Properties.Settings.Default.IcecastSettings;
		}
		
		public void Init(frmCUEPlayer parent)
		{
			MdiParent = parent;
			Show();
			_mixer = parent.Mixer;
			buffer = new AudioPipe(_mixer.PCM, _mixer.PCM.SampleRate * 10); // 10 secs
			_mixer.AudioRead += new EventHandler<CUETools.DSP.Mixer.AudioReadEventArgs>(Mixer_AudioRead);
			parent.updateMetadata += new EventHandler<UpdateMetadataEvent>(parent_updateMetadata);

			flushThread = new Thread(FlushThread);
			flushThread.Priority = ThreadPriority.AboveNormal;
			flushThread.IsBackground = true;
			flushThread.Name = "Icecast";
			flushThread.Start();
		}

		void parent_updateMetadata(object sender, UpdateMetadataEvent e)
		{
			if (_icecastWriter != null)
				_icecastWriter.UpdateMetadata(e.artist, e.title);
		}

		private void FlushThread()
		{
			AudioBuffer result = new AudioBuffer(_mixer.PCM, _mixer.BufferSize);
			while (true)
			{
				buffer.Read(result, -1);
				if (_icecastWriter != null && !close)
				{
					try
					{
						_icecastWriter.Write(result);
					}
					catch (Exception ex)
					{
						close = true;
					}
				}
				if (_icecastWriter != null && close)
				{
					_icecastWriter.Delete();
					_icecastWriter = null;
				}
			}
		}

		void Mixer_AudioRead(object sender, CUETools.DSP.Mixer.AudioReadEventArgs e)
		{
			latency = buffer.Write(e.buffer);
			//int bs = buffer.Write(e.buffer);
			//if (bs > 0)
			//{
			//    Trace.WriteLine(string.Format("buffer size {0}", bs));
			//}
		}

		private void timer1_Tick(object sender, EventArgs e)
		{
			textBoxBytes.Text = _icecastWriter == null ? "" : string.Format("{0}K", _icecastWriter.BytesWritten/1024);
			textBoxLatency.Text = (_icecastWriter == null || latency == 0 ) ? "" : string.Format("{0}", 1.0 * latency / buffer.PCM.SampleRate);
		}

		private void checkBoxTransmit_CheckedChanged(object sender, EventArgs e)
		{
			close = !checkBoxTransmit.Checked;
			this.toolTip1.SetToolTip(this.checkBoxTransmit, "");
			if (!close && _icecastWriter == null)
			{
				IcecastWriter icecastWriter = new IcecastWriter(_mixer.PCM, _icecastSettings);
				try
				{
					icecastWriter.Connect();
					if (icecastWriter.Response.StatusCode == HttpStatusCode.OK)
						_icecastWriter = icecastWriter;
					else
					{
						toolTip1.ToolTipIcon = ToolTipIcon.Error;
						toolTip1.ToolTipTitle = icecastWriter.Response.StatusCode.ToString();
						toolTip1.IsBalloon = true;
						//toolTip1.Show(resp.StatusDescription, checkBoxTransmit, 0, 0, 2000);
						toolTip1.SetToolTip(checkBoxTransmit, icecastWriter.Response.StatusDescription);
					}
				}
				catch (Exception ex)
				{
					Trace.WriteLine(ex.Message);
					icecastWriter.Close();
					toolTip1.ToolTipIcon = ToolTipIcon.Error;
					toolTip1.ToolTipTitle = "Exception";
					toolTip1.IsBalloon = true;
					//toolTip1.Show(ex.Message, checkBoxTransmit, 0, 0, 2000);
					toolTip1.SetToolTip(checkBoxTransmit, ex.Message);
				}
			}
		}

		private void buttonSettings_Click(object sender, EventArgs e)
		{
			IcecastSettings frm = new IcecastSettings(_icecastSettings);
			if (frm.ShowDialog(this) == DialogResult.OK)
				Properties.Settings.Default.Save();
		}
	}
}
